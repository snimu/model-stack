"""
Modified from https://github.com/KellerJordan/modded-nanogpt/blob/master/records/101024_Muon/train_gpt2.py
  which is one run from https://github.com/KellerJordan/modded-nanogpt
  where the embedding and lm-head weights are tied. 
"""

import argparse
import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import glob
import time
from dataclasses import dataclass
import random

import polars as pl
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16() / (G.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    """
    Muon: MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                if g.size(0) == 3 * g.size(1): # split grouped QKV parameters
                    g = torch.cat([zeropower_backend(g1, steps=group['backend_steps']) for g1 in g.split(g.size(1))])
                    scale = g.size(1)**0.5
                else:
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    scale = max(g.size(0), g.size(1))**0.5 # scale to have update.square().mean() == 1
                p.data.add_(g, alpha=-lr * scale)

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = (1 / (2 * config.n_layer)**0.5)

    def forward(self, x):
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 50257
    n_layer : int = 12
    n_head : int = 12
    n_embd : int = 768
    use_norm: bool = False
    from_model: str | None = None

class GPT(nn.Module):  # TODO: allow passing of embedding (if not None, no_grad=True)

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.from_model is not None:
            self.lm_head.weight.data = torch.load(config.from_model)["model"]["_orig_mod.lm_head.weight"]
        wte = nn.Embedding(config.vocab_size, config.n_embd)
        wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        self.transformer = nn.ModuleDict(dict(
            wte = wte,
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))

    def forward(self, idx, targets=None, return_logits=True):
        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        if self.config.use_norm:
            x = rmsnorm(x)

        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits.float() # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss


class ModelStack(nn.Module):

    def __init__(self, models: list[GPT], use_first_layer=False, use_last_layer=False, use_norm=False):
        super().__init__()
        self.wte = models[0].transformer.wte
        self.lm_head = models[0].lm_head

        # Stack all the blocks; cut off first and/or last layer if needed
        start = 0 if use_first_layer else 1
        end = None if use_last_layer else -1
        transformer_cores = []
        for i in range(len(models)):
            if i == 0:
                transformer_cores.append([block for block in models[i].transformer.h[:end]])
            elif i == len(models) - 1:
                transformer_cores.append([block for block in models[i].transformer.h[start:]])
            else:
                transformer_cores.append([block for block in models[i].transformer.h[start:end]])
        
        # Save the stack
        self.transformer_cores: list[list[Block]] = transformer_cores

        self.use_first_layer = use_first_layer
        self.use_last_layer = use_last_layer
        self.use_norm = use_norm

    def forward(self, x, targets=None, return_logits=True):
        x = self.wte(x)

        for transformer_core in self.transformer_cores:
            if self.use_norm:
                x = rmsnorm(x)
            for block in transformer_core:
                x = block(x)
        x = rmsnorm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits.float() # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss


# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# int main


def train(
        # data hyperparams
        input_bin : str = 'fineweb100B/fineweb_train_*.bin', # input .bin to train on
        input_val_bin : str = 'fineweb100B/fineweb_val_*.bin', # input .bin to eval validation loss on
        # optimization hyperparams
        batch_size : int = 8*64, # batch size, in sequences, across all devices
        device_batch_size : int = 64, # batch size, in sequences, per device
        sequence_length : int = 1024, # sequence length, in tokens
        num_iterations : int = 6200, # number of iterations to run; batch_size*sequence_length*num_iterations = (8*64)*1024*6200 = ~3.25B tokens
        learning_rate : float = 0.0036,
        warmup_iters : int = 0,
        warmdown_iters : int = 1800, # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
        weight_decay : float = 0,
        # evaluation and logging hyperparams
        val_loss_every : int = 125, # every how many steps to evaluate val loss? 0 for only at the end
        val_tokens : int = 10485760, # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
        save_every : int = 0, # every how many steps to save the checkpoint? 0 for only at the end
        # other
        seed: int = 1234,
        num_vocab: int = 50304,
        model_id: str = str(uuid.uuid4()),
        from_model: str | None = None,
        use_norm: bool = False,
) -> str:
    # set up DDP (distributed data parallel). torchrun sets this env variable
    assert torch.cuda.is_available()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    print(f"using device: {device}")
    master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.

    # convenience variables
    B, T = device_batch_size, sequence_length
    # calculate the number of steps to take in the val loop.
    assert val_tokens % (B * T * ddp_world_size) == 0
    val_steps = val_tokens // (B * T * ddp_world_size)
    # calculate the steps of gradient accumulation required to attain the desired global batch size.
    assert batch_size % (B * ddp_world_size) == 0
    train_accumulation_steps = batch_size // (B * ddp_world_size)

    # load tokens
    train_loader = DistributedDataLoader(input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = DistributedDataLoader(input_val_bin, B, T, ddp_rank, ddp_world_size)
    random.shuffle(train_loader.files)
    if master_process:
        print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
        print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
    x, y = train_loader.next_batch()

    # init the model from scratch
    model = GPT(GPTConfig(
        vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768, from_model=from_model, use_norm=use_norm,
    ))
    model = model.cuda()
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True # suggested by @Chillee
    model = torch.compile(model)
    # here we wrap model into DDP container
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module # always contains the "raw" unwrapped model
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

    # init the optimizer(s)
    if from_model is not None:  # don't optimize loaded embeddings / lm_head parameters
        optimizers = [Muon(raw_model.transformer.h.parameters(), lr=0.1*learning_rate, momentum=0.95)]
    else:
        optimizer1 = torch.optim.AdamW(raw_model.lm_head.parameters(), lr=learning_rate, betas=(0.9, 0.95),
                                    weight_decay=weight_decay, fused=True)
        optimizer2 = Muon(raw_model.transformer.h.parameters(), lr=0.1*learning_rate, momentum=0.95)
        optimizers = [optimizer1, optimizer2]
    # learning rate decay scheduler (linear warmup and warmdown)
    def get_lr(it):
        assert it <= num_iterations
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return (it+1) / warmup_iters
        # 2) constant lr for a while
        elif it < num_iterations - warmdown_iters:
            return 1.0
        # 3) linear warmdown
        else:
            decay_ratio = (num_iterations - it) / warmdown_iters
            return decay_ratio
    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

    # begin logging
    if master_process:
        logdir = 'logs/%s/' % model_id
        os.makedirs(logdir, exist_ok=True)
        logfile = 'logs/%s.txt' % model_id
        # create the log file
        with open(logfile, "w") as f:
            # begin the log by printing this file (the Python code)
            f.write('='*100 + '\n')
            f.write(code)
            f.write('='*100 + '\n')
            # log information about the hardware/software environment this is running on
            # and print the full `nvidia-smi` to file
            f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n")
            import subprocess
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            f.write(f'{result.stdout}\n')
            f.write('='*100 + '\n')

    training_time_ms = 0
    # start the clock
    torch.cuda.synchronize()
    t0 = time.time()
    # begin training
    train_loader.reset()
    for step in range(num_iterations + 1):
        last_step = (step == num_iterations)
        # This effectively ignores timing first 10 steps, which are slower for weird reasons.
        # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
        # steps with dummy data first, and then re-initialize the model and reset the loader.
        if step == 10:
            training_time_ms = 0
            t0 = time.time()
        timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

        # once in a while evaluate the validation dataset
        if (last_step or (val_loss_every > 0 and step % val_loss_every == 0)):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.time() - t0)
            # run validation batches
            model.eval()
            val_loader.reset()
            val_loss = 0.0
            for _ in range(val_steps):
                x_val, y_val = val_loader.next_batch()
                with torch.no_grad(): # of course, we'd like to use ctx here too, but that creates a torch.compile error for some reason
                    _, loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            val_loss /= val_steps
            # log val loss to console and to logfile
            if master_process:
                print(f'step:{step}/{num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
                with open(logfile, "a") as f:
                    f.write(f'step:{step}/{num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
            # start the clock again
            torch.cuda.synchronize()
            t0 = time.time()

        if master_process and (last_step or (save_every > 0 and step % save_every == 0)):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.time() - t0)
            # save the state of the training process
            log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            run_name = f'logs/{model_id}/state_step{step:06d}.pt' if not last_step else f'logs/{model_id}/final_state.pt'
            torch.save(log, run_name)
            # start the clock again
            torch.cuda.synchronize()
            t0 = time.time()

        # bit confusing: we want to make sure to eval on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        for i in range(1, train_accumulation_steps+1):
            # forward pass
            with ctx:
                _, loss = model(x, y, return_logits=False)
                train_loss = loss.detach()
            # advance the dataset for the next batch
            x, y = train_loader.next_batch()
            # backward pass
            if i < train_accumulation_steps:
                with model.no_sync(): # there's no need to sync gradients every accumulation step
                    loss.backward()
            else:
                loss.backward() # just sync on the last step
        for p in model.parameters():
            p.grad /= train_accumulation_steps
        # step the optimizers and schedulers
        for opt, sched in zip(optimizers, schedulers):
            opt.step()
            sched.step()
        # null the gradients
        model.zero_grad(set_to_none=True)
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
        if master_process:
            approx_time = training_time_ms + 1000 * (time.time() - t0)
            print(f"step:{step+1}/{num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
            with open(logfile, "a") as f:
                f.write(f"step:{step+1}/{num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms\n")

    if master_process:
        print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
        with open(f"{model_id}.txt", "w") as f:
            f.write(f"val_loss: {val_loss}\n")
    dist.destroy_process_group()
        

    # -------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train', action='store_true',
        help="If set, train new model; else, stack them.",
    )
    parser.add_argument(
        '--savefile', type=str, default="results",
        help="Save results from stacking in here. "
        "type=str, default=results",
    )
    parser.add_argument(
        '--model-id', type=str, default=str(uuid.uuid4()),
        help="Trained model saved in logs/<model-id>/final_state.pt. "
        "type=str, default=<uuid4 string>",
    )
    parser.add_argument(
        '--model-names', nargs='+', type=str, default=None,
        help="Stack & eval these models. "
        "Example: --model-names logs/model1/final_state.pt logs/model2/final_state.pt --- "
        "type=str, default=None, nargs=+",
    )
    parser.add_argument('--num-iterations', type=int, default=6200, help="type=int, default=6200")
    parser.add_argument('--warmdown-iters', type=int, default=1800, help="type=int, default=1800")
    parser.add_argument(
        '--use-first-layer', action='store_true',
        help="Only relevant when stacking. "
        "If set, use all layers of all models, "
        "else cut off first layer from all models but the first. "
        "type=FLAG",
    )
    parser.add_argument(
        '--use-last-layer', action='store_true',
        help="Only relevant when stacking. "
        "If set, use all layers of all models, "
        "else cut off last layer from all models but the last. "
        "type=FLAG",
    )
    parser.add_argument(
        '--use-norm', action='store_true',
        help="Only relevant when stacking. "
        "If set, use norm after each transformer core. "
        "type=FLAG",
    )
    parser.add_argument('--seed', type=int, default=1234, help="type=int, default=1234")
    parser.add_argument(
        '--from-model', type=str, default=None,
        help="Train new model with embeddings from this one. type=str, default=None",
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0,
        help="Weight decay for AdamW. type=float, default=0",
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.0036,
        help="Learning rate for AdamW. type=float, default=0.0036",
    )
    args = parser.parse_args()
    torch.set_float32_matmul_precision('high')
    if args.train:
        train(
            num_iterations=args.num_iterations,
            warmdown_iters=args.warmdown_iters,
            seed=args.seed,
            model_id=args.model_id,
            from_model=args.from_model,
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            use_norm=args.use_norm,
        )
    else:
        assert args.model_names is not None
        assert torch.cuda.is_available()
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        num_models = len(args.model_names)
        input_val_bin = "fineweb100B/fineweb_val_*.bin"
        val_tokens = 10485760
        device_batch_size = 64
        sequence_length = 1024
        num_vocab = 50304
        dist.init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        print(f"using device: {device}")
        master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.
        B, T = device_batch_size // num_models, sequence_length
        val_loader = DistributedDataLoader(input_val_bin, B, T, ddp_rank, ddp_world_size)
        val_steps = val_tokens // (B * T * ddp_world_size)
        
        models = []
        for model_name in args.model_names:
            model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768))
            state_dict = torch.load(model_name)["model"]
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            model = model.cuda()
            models.append(model)

        model = ModelStack(
            models,
            use_first_layer=args.use_first_layer,
            use_last_layer=args.use_last_layer,
            use_norm=args.use_norm,
        )
        model = model.cuda()
        if hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = True # suggested by @Chillee
        model = torch.compile(model)
        # here we wrap model into DDP container
        model = DDP(model, device_ids=[ddp_local_rank])
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with torch.no_grad(): # of course, we'd like to use ctx here too, but that creates a torch.compile error for some reason
                _, loss = model(x_val, y_val, return_logits=False)
                val_loss += loss
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        if master_process:
            val_loss /= val_steps
            val_losses = []
            for model_name in args.model_names:
                with open(model_name.split("/")[1] + ".txt", "r") as f:
                    val_losses.append(float(f.read().split("val_loss: ")[1].split("\n")[0]))
            results = dict(
                val_loss_stack=[val_loss],
                val_losses=[str(val_losses)],
                use_first_layer=[args.use_first_layer],
                use_last_layer=[args.use_last_layer],
                use_norm=[args.use_norm],
                num_tokens_seen=[int(8*64*1024*args.num_iterations)],  # batch_size*sequence_length*num_iterations
                num_models=[num_models],
                seed=[args.seed],
                model_names=[str(args.model_names)],
                weight_decay=[args.weight_decay],
                learning_rate=[args.learning_rate],
            )
            df = pl.DataFrame(results)
            print(df)
            if os.path.exists(f"{args.savefile}.csv"):
                with open(f"{args.savefile}.csv", "ab") as f:
                    df.write_csv(f, include_header=False)
            else:
                df.write_csv(f"{args.savefile}.csv")
        
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

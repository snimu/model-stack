# TRAIN
torchrun --nproc_per_node=8 01_train.py --train --model-id=m1-shared-emb-1xtoks-norm --seed=399938282
torchrun --nproc_per_node=8 01_train.py --train --model-id=m2-shared-emb-1xtoks-norm --seed=677719191 --from-model=logs/m1-shared-emb-1xtoks-norm/final_state.pt 

# STACK & EVAL WITH NORMALIZATION
## two different models
torchrun --nproc_per_node=8 01_train.py --model-names logs/m1-shared-emb-1xtoks-norm/final_state.pt logs/m2-shared-emb-1xtoks-norm/final_state.pt --use-norm --savefile=results_1xtoks_norm
torchrun --nproc_per_node=8 01_train.py --model-names logs/m1-shared-emb-1xtoks-norm/final_state.pt logs/m2-shared-emb-1xtoks-norm/final_state.pt --use-norm --savefile=results_1xtoks_norm --use-first-layer
torchrun --nproc_per_node=8 01_train.py --model-names logs/m1-shared-emb-1xtoks-norm/final_state.pt logs/m2-shared-emb-1xtoks-norm/final_state.pt --use-norm --savefile=results_1xtoks_norm --use-last-layer
torchrun --nproc_per_node=8 01_train.py --model-names logs/m1-shared-emb-1xtoks-norm/final_state.pt logs/m2-shared-emb-1xtoks-norm/final_state.pt --use-norm --savefile=results_1xtoks_norm --use-first-layer --use-last-layer
## baseline: same model twice
torchrun --nproc_per_node=8 01_train.py --model-names logs/m1-shared-emb-1xtoks-norm/final_state.pt logs/m1-shared-emb-1xtoks-norm/final_state.pt --use-norm --savefile=results_1xtoks_norm
torchrun --nproc_per_node=8 01_train.py --model-names logs/m1-shared-emb-1xtoks-norm/final_state.pt logs/m1-shared-emb-1xtoks-norm/final_state.pt --use-norm --savefile=results_1xtoks_norm --use-first-layer
torchrun --nproc_per_node=8 01_train.py --model-names logs/m1-shared-emb-1xtoks-norm/final_state.pt logs/m1-shared-emb-1xtoks-norm/final_state.pt --use-norm --savefile=results_1xtoks_norm --use-last-layer
torchrun --nproc_per_node=8 01_train.py --model-names logs/m1-shared-emb-1xtoks-norm/final_state.pt logs/m1-shared-emb-1xtoks-norm/final_state.pt --use-norm --savefile=results_1xtoks_norm --use-first-layer --use-last-layer

# STACK & EVAL WITHOUT NORMALIZATION
## two different models
torchrun --nproc_per_node=8 01_train.py --model-names logs/m1-shared-emb-1xtoks-norm/final_state.pt logs/m2-shared-emb-1xtoks-norm/final_state.pt --savefile=results_1xtoks_norm
torchrun --nproc_per_node=8 01_train.py --model-names logs/m1-shared-emb-1xtoks-norm/final_state.pt logs/m2-shared-emb-1xtoks-norm/final_state.pt --savefile=results_1xtoks_norm --use-first-layer
torchrun --nproc_per_node=8 01_train.py --model-names logs/m1-shared-emb-1xtoks-norm/final_state.pt logs/m2-shared-emb-1xtoks-norm/final_state.pt --savefile=results_1xtoks_norm --use-last-layer
torchrun --nproc_per_node=8 01_train.py --model-names logs/m1-shared-emb-1xtoks-norm/final_state.pt logs/m2-shared-emb-1xtoks-norm/final_state.pt --savefile=results_1xtoks_norm --use-first-layer --use-last-layer
## baseline: same model twice
torchrun --nproc_per_node=8 01_train.py --model-names logs/m1-shared-emb-1xtoks-norm/final_state.pt logs/m1-shared-emb-1xtoks-norm/final_state.pt --savefile=results_1xtoks_norm
torchrun --nproc_per_node=8 01_train.py --model-names logs/m1-shared-emb-1xtoks-norm/final_state.pt logs/m1-shared-emb-1xtoks-norm/final_state.pt --savefile=results_1xtoks_norm --use-first-layer
torchrun --nproc_per_node=8 01_train.py --model-names logs/m1-shared-emb-1xtoks-norm/final_state.pt logs/m1-shared-emb-1xtoks-norm/final_state.pt --savefile=results_1xtoks_norm --use-last-layer  
torchrun --nproc_per_node=8 01_train.py --model-names logs/m1-shared-emb-1xtoks-norm/final_state.pt logs/m1-shared-emb-1xtoks-norm/final_state.pt --savefile=results_1xtoks_norm --use-first-layer --use-last-layer

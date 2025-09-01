torchrun --standalone --nproc-per-node=8 2_train_with_model_merging.py \
    --train \
    --savefile results-one \
    --model-id one \
    --seed 229806

torchrun --standalone --nproc-per-node=8 2_train_with_model_merging.py \
    --train \
    --savefile results-two \
    --model-id two \
    --from-model one \
    --seed 122334

torchrun --standalone --nproc-per-node=8 2_train_with_model_merging.py \
    --train \
    --savefile results-three \
    --model-id three \
    --mixin-weight 0.1 \
    --mixin-every 100 \
    --mixin-from two \
    --seed 544332

torchrun --standalone --nproc-per-node=8 2_train_with_model_merging.py \
    --model-names two three \
    --savefile stack-two-three

torchrun --standalone --nproc-per-node=8 2_train_with_model_merging.py \
    --model-names two two \
    --savefile stack-two-two

torchrun --standalone --nproc-per-node=8 2_train_with_model_merging.py \
    --model-names one two \
    --savefile stack-one-two

torchrun --standalone --nproc-per-node=8 2_train_with_model_merging.py \
    --model-names one two three \
    --savefile stack-one-two-three
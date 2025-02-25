# TRAIN
torchrun --nproc_per_node=8 01_train.py --train --model-id=m1-shared-emb-5xtoks --seed=98765432 --num-iterations 31000 --weight-decay=0.01 --warmdown-iters 9000
torchrun --nproc_per_node=8 01_train.py --train --model-id=m2-shared-emb-5xtoks --seed=84759274 --num-iterations 31000 --weight-decay=0.01 --warmdown-iters 9000 --from-model=logs/m1-shared-emb-5xtoks/final_state.pt

# STACK & EVAL
## two different models
torchrun --nproc_per_node=8 01_train.py --num-iterations 31000 --model-names logs/m1-shared-emb-5xtoks/final_state.pt logs/m2-shared-emb-5xtoks/final_state.pt --weight-decay=0.01 --savefile=results_5xtoks
torchrun --nproc_per_node=8 01_train.py --num-iterations 31000 --model-names logs/m1-shared-emb-5xtoks/final_state.pt logs/m2-shared-emb-5xtoks/final_state.pt --weight-decay=0.01 --savefile=results_5xtoks --use-first-layer
torchrun --nproc_per_node=8 01_train.py --num-iterations 31000 --model-names logs/m1-shared-emb-5xtoks/final_state.pt logs/m2-shared-emb-5xtoks/final_state.pt --weight-decay=0.01 --savefile=results_5xtoks --use-last-layer
torchrun --nproc_per_node=8 01_train.py --num-iterations 31000 --model-names logs/m1-shared-emb-5xtoks/final_state.pt logs/m2-shared-emb-5xtoks/final_state.pt --weight-decay=0.01 --savefile=results_5xtoks --use-first-layer --use-last-layer
## baseline: same model twice
torchrun --nproc_per_node=8 01_train.py --num-iterations 31000 --model-names logs/m1-shared-emb-5xtoks/final_state.pt logs/m1-shared-emb-5xtoks/final_state.pt --weight-decay=0.01 --savefile=results_5xtoks
torchrun --nproc_per_node=8 01_train.py --num-iterations 31000 --model-names logs/m1-shared-emb-5xtoks/final_state.pt logs/m1-shared-emb-5xtoks/final_state.pt --weight-decay=0.01 --savefile=results_5xtoks --use-first-layer
torchrun --nproc_per_node=8 01_train.py --num-iterations 31000 --model-names logs/m1-shared-emb-5xtoks/final_state.pt logs/m1-shared-emb-5xtoks/final_state.pt --weight-decay=0.01 --savefile=results_5xtoks --use-last-layer
torchrun --nproc_per_node=8 01_train.py --num-iterations 31000 --model-names logs/m1-shared-emb-5xtoks/final_state.pt logs/m1-shared-emb-5xtoks/final_state.pt --weight-decay=0.01 --savefile=results_5xtoks --use-first-layer --use-last-layer

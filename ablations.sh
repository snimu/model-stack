# torchrun --nproc_per_node=2 01_train.py --train --model-id=model1
# torchrun --nproc_per_node=2 01_train.py --train --model-id=model2
# torchrun --nproc_per_node=2 01_train.py --model-names logs/model1/final_state.pt logs/model2/final_state.pt --savefile=2models-no-first-layer
# torchrun --nproc_per_node=2 01_train.py --model-names logs/model1/final_state.pt logs/model2/final_state.pt --use-first-layer --savefile=2models-first-layer

# WITH SHARED EMBEDDINGS
## TRAIN
torchrun --nproc_per_node=4 01_train.py --train --model-id=model1-shared-embeddings --seed=1234
torchrun --nproc_per_node=4 01_train.py --train --model-id=model2-shared-embeddings --from-model=logs/model1-shared-embeddings/final_state.pt --seed=2345
## STACK & EVAL
### two different models
torchrun --nproc_per_node=4 01_train.py --model-names logs/model1-shared-embeddings/final_state.pt logs/model2-shared-embeddings/final_state.pt --savefile=2models-no-first-layer-shared-embeddings --use-last-layer
torchrun --nproc_per_node=4 01_train.py --model-names logs/model1-shared-embeddings/final_state.pt logs/model2-shared-embeddings/final_state.pt --use-first-layer --savefile=2models-first-layer-shared-embeddings --use-last-layer
### baseline: same model twice
torchrun --nproc_per_node=4 01_train.py --model-names logs/model1-shared-embeddings/final_state.pt logs/model1-shared-embeddings/final_state.pt --savefile=1model-twice-no-first-layer-shared-embeddings --use-last-layer
torchrun --nproc_per_node=4 01_train.py --model-names logs/model1-shared-embeddings/final_state.pt logs/model1-shared-embeddings/final_state.pt --use-first-layer --savefile=1model-twice-first-layer-shared-embeddings --use-last-layer
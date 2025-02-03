# torchrun --nproc_per_node=2 01_train.py --train --model-id=model1
# torchrun --nproc_per_node=2 01_train.py --train --model-id=model2
# torchrun --nproc_per_node=2 01_train.py --model-names logs/model1/final_state.pt logs/model2/final_state.pt --savefile=2models-no-first-layer
# torchrun --nproc_per_node=2 01_train.py --model-names logs/model1/final_state.pt logs/model2/final_state.pt --use-first-layer --savefile=2models-first-layer

torchrun --nproc_per_node=4 01_train.py --train --model-id=model1-shared-embeddings
torchrun --nproc_per_node=4 01_train.py --train --model-id=model2-shared-embeddings --from-model=logs/model1-shared-embeddings/final_state.pt
torchrun --nproc_per_node=4 01_train.py --model-names logs/model1-shared-embeddings/final_state.pt logs/model2/final_state.pt --savefile=2models-no-first-layer-shared-embeddings
torchrun --nproc_per_node=4 01_train.py --model-names logs/model1-shared-embeddings/final_state.pt logs/model2/final_state.pt --use-first-layer --savefile=2models-first-layer-shared-embeddings
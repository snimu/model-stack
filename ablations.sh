torchrun --nproc_per_node=2 01_train.py --train --model-id=model1
torchrun --nproc_per_node=2 01_train.py --train --model-id=model2
torchrun --nproc_per_node=2 01_train.py --model-names logs/model1/final_state.pt logs/model2/final_state.pt --savefile=2models-no-first-layer
torchrun --nproc_per_node=2 01_train.py --model-names logs/model1/final_state.pt logs/model2/final_state.pt --use-first-layer --savefile=2models-first-layer
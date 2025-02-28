# TRAIN
torchrun --nproc_per_node=8 01_train.py --train --norm-wte none --norm-lm-head rms_norm --model-id=m1-nr --seed=99944455
torchrun --nproc_per_node=8 01_train.py --train --norm-wte none --norm-lm-head rms_norm --model-id=m2-nr --seed=12345678 --from-model=m1-nr

torchrun --nproc_per_node=8 01_train.py --train --norm-wte rms_norm --norm-lm-head rms_norm --model-id=m3-rr --seed=33004958
torchrun --nproc_per_node=8 01_train.py --train --norm-wte rms_norm --norm-lm-head rms_norm --model-id=m4-rr --seed=22883717 --from-model=m2-rr

torchrun --nproc_per_node=8 01_train.py --train --norm-wte layer_norm --norm-lm-head layer_norm --model-id=m5-ll --seed=47747474
torchrun --nproc_per_node=8 01_train.py --train --norm-wte layer_norm --norm-lm-head layer_norm --model-id=m6-ll --seed=83838392 --from-model=m6-ll

# STACK & EVAL
# Norm-less models without norm
torchrun --nproc_per_node=8 01_train.py --model-names m1-nr m2-nr --norm-wte none --norm-lm-head rms_norm --norm-inter-model none --savefile=results
torchrun --nproc_per_node=8 01_train.py --model-names m1-nr m2-nr --norm-wte none --norm-lm-head rms_norm --norm-inter-model none --savefile=results --use-first-layer
torchrun --nproc_per_node=8 01_train.py --model-names m1-nr m2-nr --norm-wte none --norm-lm-head rms_norm --norm-inter-model none --savefile=results --use-last-layer
torchrun --nproc_per_node=8 01_train.py --model-names m1-nr m2-nr --norm-wte none --norm-lm-head rms_norm --norm-inter-model none --savefile=results --use-first-layer --use-last-layer

# Models with rms_norm, with rms_norm
torchrun --nproc_per_node=8 01_train.py --model-names m3-rr m4-rr --norm-wte rms_norm --norm-lm-head rms_norm --norm-inter-model rms_norm --savefile=results
torchrun --nproc_per_node=8 01_train.py --model-names m3-rr m4-rr --norm-wte rms_norm --norm-lm-head rms_norm --norm-inter-model rms_norm --savefile=results --use-first-layer
torchrun --nproc_per_node=8 01_train.py --model-names m3-rr m4-rr --norm-wte rms_norm --norm-lm-head rms_norm --norm-inter-model rms_norm --savefile=results --use-last-layer
torchrun --nproc_per_node=8 01_train.py --model-names m3-rr m4-rr --norm-wte rms_norm --norm-lm-head rms_norm --norm-inter-model rms_norm --savefile=results --use-first-layer --use-last-layer

# Models with layer_norm, with layer_norm
torchrun --nproc_per_node=8 01_train.py --model-names m5-ll m6-ll --norm-wte layer_norm --norm-lm-head layer_norm --norm-inter-model layer_norm --savefile=results
torchrun --nproc_per_node=8 01_train.py --model-names m5-ll m6-ll --norm-wte layer_norm --norm-lm-head layer_norm --norm-inter-model layer_norm --savefile=results --use-first-layer
torchrun --nproc_per_node=8 01_train.py --model-names m5-ll m6-ll --norm-wte layer_norm --norm-lm-head layer_norm --norm-inter-model layer_norm --savefile=results --use-last-layer
torchrun --nproc_per_node=8 01_train.py --model-names m5-ll m6-ll --norm-wte layer_norm --norm-lm-head layer_norm --norm-inter-model layer_norm --savefile=results --use-first-layer --use-last-layer

# Baseline: same model twice
torchrun --nproc_per_node=8 01_train.py --model-names m1-nr m1-nr --norm-wte none --norm-lm-head rms_norm --norm-inter-model none --savefile=results
torchrun --nproc_per_node=8 01_train.py --model-names m1-nr m1-nr --norm-wte none --norm-lm-head rms_norm --norm-inter-model none --savefile=results --use-first-layer
torchrun --nproc_per_node=8 01_train.py --model-names m1-nr m1-nr --norm-wte none --norm-lm-head rms_norm --norm-inter-model none --savefile=results --use-last-layer
torchrun --nproc_per_node=8 01_train.py --model-names m1-nr m1-nr --norm-wte none --norm-lm-head rms_norm --norm-inter-model none --savefile=results --use-first-layer --use-last-layer

# Baseline: same model twice, with rms_norm
torchrun --nproc_per_node=8 01_train.py --model-names m3-rr m3-rr --norm-wte rms_norm --norm-lm-head rms_norm --norm-inter-model rms_norm --savefile=results
torchrun --nproc_per_node=8 01_train.py --model-names m3-rr m3-rr --norm-wte rms_norm --norm-lm-head rms_norm --norm-inter-model rms_norm --savefile=results --use-first-layer
torchrun --nproc_per_node=8 01_train.py --model-names m3-rr m3-rr --norm-wte rms_norm --norm-lm-head rms_norm --norm-inter-model rms_norm --savefile=results --use-last-layer
torchrun --nproc_per_node=8 01_train.py --model-names m3-rr m3-rr --norm-wte rms_norm --norm-lm-head rms_norm --norm-inter-model rms_norm --savefile=results --use-first-layer --use-last-layer

# Baseline: same model twice, with layer_norm
torchrun --nproc_per_node=8 01_train.py --model-names m5-ll m5-ll --norm-wte layer_norm --norm-lm-head layer_norm --norm-inter-model layer_norm --savefile=results
torchrun --nproc_per_node=8 01_train.py --model-names m5-ll m5-ll --norm-wte layer_norm --norm-lm-head layer_norm --norm-inter-model layer_norm --savefile=results --use-first-layer
torchrun --nproc_per_node=8 01_train.py --model-names m5-ll m5-ll --norm-wte layer_norm --norm-lm-head layer_norm --norm-inter-model layer_norm --savefile=results --use-last-layer
torchrun --nproc_per_node=8 01_train.py --model-names m5-ll m5-ll --norm-wte layer_norm --norm-lm-head layer_norm --norm-inter-model layer_norm --savefile=results --use-first-layer --use-last-layer

# SAVE DATA
python 01_train.py --save-data --savefile=results --wandb-project=model-stacking

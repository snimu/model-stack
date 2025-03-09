# TRAIN
torchrun --nproc_per_node=8 01_train.py --train --coconut-every 100 --detach-output-latents --norm-wte rms_norm --norm-lm-head rms_norm --model-id=m1-rr-coconut --seed=47747474
torchrun --nproc_per_node=8 01_train.py --train --coconut-every 100 --detach-output-latents --norm-wte rms_norm --norm-lm-head rms_norm --model-id=m2-rr-coconut --seed=83838392 --from-model=m1-rr-coconut

# STACK & EVAL
# Norm-less models without norm
torchrun --nproc_per_node=8 01_train.py --model-names m1-rr-coconut m2-rr-coconut --norm-wte none --norm-lm-head rms_norm --norm-inter-model none --savefile=results_coconut --use-first-layer
torchrun --nproc_per_node=8 01_train.py --model-names m1-rr-coconut m2-rr-coconut --norm-wte none --norm-lm-head rms_norm --norm-inter-model none --savefile=results_coconut --use-first-layer --use-last-layer

# Baseline: same model twice
torchrun --nproc_per_node=8 01_train.py --model-names m1-rr-coconut m2-rr-coconut --norm-wte none --norm-lm-head rms_norm --norm-inter-model none --savefile=results_coconut --use-first-layer
torchrun --nproc_per_node=8 01_train.py --model-names m1-rr-coconut m2-rr-coconut --norm-wte none --norm-lm-head rms_norm --norm-inter-model none --savefile=results_coconut --use-first-layer --use-last-layer

# TRAIN
torchrun --nproc_per_node=8 01_train.py --train --coconut-every 100 --norm-wte rms_norm --norm-lm-head rms_norm --model-id=m1-rr-coconut-every-100 --seed=47747474
torchrun --nproc_per_node=8 01_train.py --train --coconut-every 100 --norm-wte rms_norm --norm-lm-head rms_norm --model-id=m2-rr-coconut-every-100 --seed=83838392 --from-model=m1-rr-coconut-every-100
torchrun --nproc_per_node=8 01_train.py --train --coconut-every 10 --norm-wte rms_norm --norm-lm-head rms_norm --model-id=m3-rr-coconut-every-10 --seed=789363
torchrun --nproc_per_node=8 01_train.py --train --coconut-every 10 --norm-wte rms_norm --norm-lm-head rms_norm --model-id=m4-rr-coconut-every-10 --seed=223002 --from-model=m1-rr-coconut-every-10


# STACK & EVAL
# Two models, coconut every 100 steps
torchrun --nproc_per_node=8 01_train.py --model-names m1-rr-coconut-every-100 m2-rr-coconut-every-100 --norm-wte none --norm-lm-head rms_norm --norm-inter-model none --savefile=results_coconut --use-first-layer
torchrun --nproc_per_node=8 01_train.py --model-names m1-rr-coconut-every-100 m2-rr-coconut-every-100 --norm-wte none --norm-lm-head rms_norm --norm-inter-model none --savefile=results_coconut --use-first-layer --use-last-layer

# Baseline: same model twice
torchrun --nproc_per_node=8 01_train.py --model-names m1-rr-coconut-every-100 m2-rr-coconut-every-100 --norm-wte none --norm-lm-head rms_norm --norm-inter-model none --savefile=results_coconut --use-first-layer
torchrun --nproc_per_node=8 01_train.py --model-names m1-rr-coconut-every-100 m2-rr-coconut-every-100 --norm-wte none --norm-lm-head rms_norm --norm-inter-model none --savefile=results_coconut --use-first-layer --use-last-layer

# Two models, coconut every 10 steps
torchrun --nproc_per_node=8 01_train.py --model-names m1-rr-coconut-every-10 m2-rr-coconut-every-10 --norm-wte none --norm-lm-head rms_norm --norm-inter-model none --savefile=results_coconut --use-first-layer
torchrun --nproc_per_node=8 01_train.py --model-names m1-rr-coconut-every-10 m2-rr-coconut-every-10 --norm-wte none --norm-lm-head rms_norm --norm-inter-model none --savefile=results_coconut --use-first-layer --use-last-layer

# Baseline: same model twice
torchrun --nproc_per_node=8 01_train.py --model-names m1-rr-coconut-every-10 m2-rr-coconut-every-10 --norm-wte none --norm-lm-head rms_norm --norm-inter-model none --savefile=results_coconut --use-first-layer
torchrun --nproc_per_node=8 01_train.py --model-names m1-rr-coconut-every-10 m2-rr-coconut-every-10 --norm-wte none --norm-lm-head rms_norm --norm-inter-model none --savefile=results_coconut --use-first-layer --use-last-layer

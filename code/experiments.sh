#!/bin/bash

# Base directories
base_ckpt_dir="checkpoints"
base_logs_dir="logs"

# Experiment 1: Run training with 10 different seeds. Shuffle is false.
# for seed in {0..9}
# do
#   ckpt_dir="${base_ckpt_dir}_seed_${seed}"
#   logs_dir="${base_logs_dir}_seed_${seed}"
#   shuffle=false
#   subsample=1
  
#   echo "Running Experiment 1 with seed $seed"
#   echo "Checkpoint Directory: $ckpt_dir"
#   echo "Logs Directory: $logs_dir"
  
#   python train.py --ckpt_dir "$ckpt_dir" --logs_dir "$logs_dir" --seed "$seed"
# done

# Experiment 2: Run training with 10 different shuffles and seeds. Shuffle is true.
# for seed in {0..9}
# do
#   ckpt_dir="${base_ckpt_dir}_seed_${seed}_shuffle"
#   logs_dir="${base_logs_dir}_seed_${seed}_shuffle"
#   shuffle=true
#   subsample=1
  
#   echo "Running Experiment 2 with seed $seed and shuffle $shuffle"
#   echo "Checkpoint Directory: $ckpt_dir"
#   echo "Logs Directory: $logs_dir"
  
#   python train.py --ckpt_dir "$ckpt_dir" --logs_dir "$logs_dir" --seed "$seed" --shuffle "$shuffle" --subsample "$subsample"
# done

# Experiment 3: Run training with 10 seeds each for subsample in [0.8, 0.6, 0.4, 0.2]
for subsample in 0.8 0.6 0.4 0.2
do
  for seed in {0..4}
  do
    ckpt_dir="${base_ckpt_dir}_seed_${seed}_subsample_${subsample}"
    logs_dir="${base_logs_dir}_seed_${seed}_subsample_${subsample}"
    shuffle=false
    
    echo "Running Experiment 3 with seed $seed and subsample $subsample"
    echo "Checkpoint Directory: $ckpt_dir"
    echo "Logs Directory: $logs_dir"
    
    python train.py --ckpt_dir "$ckpt_dir" --logs_dir "$logs_dir" --seed "$seed" --subsample "$subsample"
  done
done

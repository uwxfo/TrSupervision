# tmux: 0
workdir: /home/yluo/RLIPVISTR

## task-list
bash train_detant_vidhoi.sh

## worked list

## failed list
bash train_detant_vidhoi.sh  # [failed exceeded max_retries=3 (exit code 1)]
bash train_detant_vidhoi.sh  # [failed exceeded max_retries=3 (exit code 1)]

# tmux: 1
workdir: /home/yluo/RLIPVISTR

## task-list
[workdir=/home/yluo/RLIPVISTR-1] bash train_detant_vidhoi.sh
bash train_detant_vidhoi.sh --seed 126

## worked list


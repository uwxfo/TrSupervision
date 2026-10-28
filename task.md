# tmux: 0
workdir: /home/yluo/RLIPVISTR

## task-list
[workdir=/home/yluo/RLIPVISTR-1] bash train_detant_vidhoi.sh --resume "" --seed 126 --name seed_126
[workdir=/home/yluo/RLIPVISTR-2] bash train_detant_vidhoi.sh --resume "" --seed 126 --name ablation_assignment_seed_21

MASTER_PORT=29502 bash train_detant_vidhoi.sh --seed 126 --resume ""
## worked list

## failed list
bash train_detant_vidhoi.sh  # [failed exceeded max_retries=3 (exit code 1)]
bash train_detant_vidhoi.sh  # [failed exceeded max_retries=3 (exit code 1)]
MASTER_PORT=29502 bash train_detant_vidhoi.sh --seed 126 --resume ""  # [failed manually interrupted (no exit record: process killed or terminal closed)]
  # [failed manually interrupted (no exit record: process killed or terminal closed)]


# tmux: 1
workdir: /home/yluo/RLIPVISTR

## task-list
[workdir=/home/yluo/RLIPVISTR-1] bash train_detant_vidhoi_sensitivity.sh --resume ../RLIPVISTR/output/sensitive_coef0.4_10280328/checkpoint0019.pth
## worked list

## failed list
bash train_detant_vidhoi.sh  # [failed exceeded max_retries=3 (exit code 120)]
[workdir=/home/yluo/RLIPVISTR-1] bash train_detant_vidhoi_sensitivity.sh --resume ""  # [failed exceeded max_retries=3 (exit code 120)]
[workdir=/home/yluo/RLIPVISTR-1] bash train_detant_vidhoi_sensitivity.sh --resume output/sensitive_coef0.4_10280328/checkpoint.pth  # [failed exceeded max_retries=3 (exit code 1)]


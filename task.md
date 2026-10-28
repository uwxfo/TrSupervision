# tmux: run
workdir: /hkfs/work/workspace/scratch/nj9475-store/RLIPVISTR

## task-list
MASTER_PORT=29502 ANTICIPATION_VERB_COEF=1.6 ANTICIPATION_HORIZON_GAMMA=0.7 bash train_detant_vidhoi_sensitivity.sh --name sensitivity_coef_1.6
## worked list
## failed list
MASTER_PORT=29502 ANTICIPATION_VERB_COEF=1.6 bash train_detant_vidhoi_sensitivity.sh  # [failed manually interrupted (no exit record: process killed or terminal closed)]
MASTER_PORT=29502 ANTICIPATION_VERB_COEF=1.6 bash train_detant_vidhoi_sensitivity.sh  # [failed manually interrupted (no exit record: process killed or terminal closed)]
MASTER_PORT=29502 ANTICIPATION_VERB_COEF=1.6 ANTICIPATION_HORIZON_GAMMA=0.7 bash train_detant_vidhoi_sensitivity.sh  # [failed exceeded max_retries=3 (exit code 1)]
MASTER_PORT=29502 ANTICIPATION_VERB_COEF=1.6 ANTICIPATION_HORIZON_GAMMA=0.7 bash train_detant_vidhoi_sensitivity.sh --name sensitivity_coef_1.6  # [failed manually interrupted (no exit record: process killed or terminal closed)]


# tmux: run1
workdir: /hkfs/work/workspace/scratch/nj9475-store/RLIPVISTR

## task-list
bash train_detant_vidhoi_sensitivity.sh
ANTICIPATION_VERB_COEF=0.4 bash train_detant_vidhoi_sensitivity.sh

## worked list

## failed list
bash train_detant_vidhoi_sensitivity.sh  # [failed manually interrupted (no exit record: process killed or terminal closed)]


# tmux: run2
workdir: /hkfs/work/workspace/scratch/nj9475-store/RLIPVISTR

## task-list
MASTER_PORT=29502 TASK_ORTH_COEF=0.2 HORIZON_ORTH_COEF=0.2 ACCUM_STEPS=1 bash train_detant_vidhoi_sensitivity.sh --name sensitivity_coef_1.6 --batch_size 4 

DATASET=${1:-humaneval}

# Self-infilling arguments
# `--use_self_infill False/True` enables self-infilling. 
#       If `False`, vanilla left-to-right generation is used. 
# `--self_infill_tau`: 𝜏 for controlling self-infill interruption
# `--self_infill_max_iters`: N for looping
# `--self_infill_suffix_split_mode`: suffix splitting for looping 
# (check Section 3.4 of paper for details)

###################################################
# starcoder

bash launch.sh -r gen \
    -d $DATASET \
    -m bigcode/starcoder \
    -s results \
    -g greedy \
    -e True \
    --use_self_infill True \
    --self_infill_tau 0.25 \
    --self_infill_max_iters 1 \
    --self_infill_suffix_split_mode extended

# For DS-1000 tasks, make sure a proper
# Python environment is used for evaluation
# source envs/si_eval/bin/activate

bash launch.sh -r eval \
    -d $DATASET \
    -m bigcode/starcoder \
    -s results \
    -g greedy \
    -p 10 \
    -e True \
    --use_self_infill True \
    --self_infill_tau 0.25 \
    --self_infill_max_iters 1 \
    --self_infill_suffix_split_mode extended

# Or using **sample** mode to evaluate pass@10 and pass@100
# bash launch.sh -r gen \
#     -d $DATASET \
#     -m bigcode/starcoder \
#     -s results \
#     -g sample \
#     -e True \
#     --use_self_infill True \
#     --self_infill_tau 0.25 \
#     --self_infill_max_iters 1 \
#     --self_infill_suffix_split_mode extended

# bash launch.sh -r eval \
#     -d $DATASET \
#     -m bigcode/starcoder \
#     -s results \
#     -g sample \
#     -p 10 \
#     -e True \
#     --use_self_infill True \
#     --self_infill_tau 0.25 \
#     --self_infill_max_iters 1 \
#     --self_infill_suffix_split_mode extended

###################################################
###################################################
# starcoderbase

# bash launch.sh -r gen \
#     -d $DATASET \
#     -m bigcode/starcoderbase \
#     -s results \
#     -g greedy \
#     -e True \
#     --use_self_infill True \
#     --self_infill_tau 0.25 \
#     --self_infill_max_iters 1 \
#     --self_infill_suffix_split_mode extended

# bash launch.sh -r eval \
#     -d $DATASET \
#     -m bigcode/starcoderbase \
#     -s results \
#     -g greedy \
#     -p 10 \
#     -e True \
#     --use_self_infill True \
#     --self_infill_tau 0.25 \
#     --self_infill_max_iters 1 \
#     --self_infill_suffix_split_mode extended

###################################################
###################################################
# codellama-7b

# bash launch.sh -r gen \
#     -d $DATASET \
#     -m codellama/CodeLlama-7b-hf \
#     -s results \
#     -g greedy \
#     -e True \
#     --use_self_infill True \
#     --self_infill_tau 0.25 \
#     --self_infill_max_iters 1 \
#     --self_infill_suffix_split_mode vanilla

# bash launch.sh -r eval \
#     -d $DATASET \
#     -m codellama/CodeLlama-7b-hf \
#     -s results \
#     -g greedy \
#     -p 10 \
#     -e True \
#     --use_self_infill True \
#     --self_infill_tau 0.25 \
#     --self_infill_max_iters 1 \
#     --self_infill_suffix_split_mode vanilla


###################################################
###################################################
# codellama-13b

# bash launch.sh -r gen \
#     -d $DATASET \
#     -m codellama/CodeLlama-13b-hf \
#     -s results \
#     -g greedy \
#     -e True \
#     --use_self_infill True \
#     --self_infill_tau 0.25 \
#     --self_infill_max_iters 1 \
#     --self_infill_suffix_split_mode extended

# bash launch.sh -r eval \
#     -d $DATASET \
#     -m codellama/CodeLlama-13b-hf \
#     -s results \
#     -g greedy \
#     -p 10 \
#     -e True \
#     --use_self_infill True \
#     --self_infill_tau 0.25 \
#     --self_infill_max_iters 1 \
#     --self_infill_suffix_split_mode extended



export CUDA_VISIBLE_DEVICES=0,1,2,3
export HYDRA_FULL_ERROR=1
export MASKAD_ROOT=/mnt/pai-pdc-nas/tianle_DPR/MaskAD
###################################
# User Configuration Section
###################################
# Set environment variables
export NUPLAN_DEVKIT_ROOT="/mnt/pai-pdc-nas/tianle_DPR/Diffusion-Planner/nuplan-devkit"  # nuplan-devkit absolute path (e.g., "/home/user/nuplan-devkit")
export NUPLAN_DATA_ROOT="/mnt/pai-pdc-nas/tianle_DPR/nuplan/dataset"  # nuplan dataset absolute path (e.g. "/data")
export NUPLAN_MAPS_ROOT="/mnt/pai-pdc-nas/tianle_DPR/nuplan/dataset/maps" # nuplan maps absolute path (e.g. "/data/nuplan-v1.1/maps")
export NUPLAN_EXP_ROOT="/mnt/pai-pdc-nas/tianle_DPR/nuplan" # nuplan experiment absolute path (e.g. "/data/nuplan-v1.1/exp")

# Dataset split to use
# Options: 
#   - "test14-random"
#   - "test14-hard"
#   - "val14"
SPLIT="test14-hard"  # e.g., "val14"

# Challenge type
# Options: 
#   - "closed_loop_nonreactive_agents"
#   - "closed_loop_reactive_agents"
CHALLENGE="closed_loop_nonreactive_agents"  # e.g., "closed_loop_nonreactive_agents"
###################################


BRANCH_NAME=mask_planner_release
CKPT_FILE=/mnt/pai-pdc-nas/tianle_DPR/MaskAD/train_logs/20251209-161525/nuplan_experiment_IL/checkpoints/last.ckpt
ARGS_FILE=/mnt/pai-pdc-nas/tianle_DPR/MaskAD/checkpoints/args.json

if [ "$SPLIT" == "val14" ]; then
    SCENARIO_BUILDER="nuplan"
else
    SCENARIO_BUILDER="nuplan_challenge"
fi
echo "Processing $CKPT_FILE..."
FILENAME=$(basename "$CKPT_FILE")
FILENAME_WITHOUT_EXTENSION="${FILENAME%.*}"

PLANNER=maskplanner

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    planner.maskplanner.config.args_file=$ARGS_FILE \
    planner.maskplanner.ckpt_path=$CKPT_FILE \
    scenario_builder=$SCENARIO_BUILDER \
    scenario_filter=$SPLIT \
    experiment_uid=$PLANNER/$SPLIT/$BRANCH_NAME/${FILENAME_WITHOUT_EXTENSION}_$(date "+%Y-%m-%d-%H-%M-%S") \
    verbose=true \
    worker=ray_distributed \
    worker.threads_per_node=128 \
    distributed_mode='SINGLE_NODE' \
    number_of_gpus_allocated_per_simulation=0.04 \
    enable_simulation_progress_bar=true \
    hydra.searchpath="[file://$MASKAD_ROOT/config/scenario_filter, file://$MASKAD_ROOT/config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
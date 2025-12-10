###################################
# User Configuration Section
###################################
NUPLAN_DATA_PATH="/mnt/pai-pdc-nas/tianle_DPR/nuplan/dataset/nuplan-v1.1/trainval1" # nuplan training data path (e.g., "/data/nuplan-v1.1/trainval")
NUPLAN_MAP_PATH="/mnt/pai-pdc-nas/tianle_DPR/nuplan/dataset/maps" # nuplan map path (e.g., "/data/nuplan-v1.1/maps")

TRAIN_SET_PATH="/mnt/pai-pdc-nas/tianle_DPR/nuplan/dataset/processed_data" # preprocess training data
###################################

python /mnt/pai-pdc-nas/tianle_DPR/MaskAD/MaskAD/data_process/data_nuplan/data_process.py \
--data_path $NUPLAN_DATA_PATH \
--map_path $NUPLAN_MAP_PATH \
--save_path $TRAIN_SET_PATH \
--total_scenarios 1000000 \


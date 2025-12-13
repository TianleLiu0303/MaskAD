import gzip
import pickle as pkl
def read_gz_file_with_fallback(file_path):
    """

    """
        # 尝试以 UTF-8 编码读取文件
    with open(file_path, 'rb') as f:
        x = pkl.load(f)

        for k,v in x.items():
            print(f"{k}: {type(v)}")
            print(v.shape if hasattr(v, 'shape') else v)
        # print("lanes_valid", x["lanes_valid"])
    print(x["agents_type"])
    print(x["agents_id"])
    # print(x["agents_history"][10,:,:])
    print(x["agents_interested"])
    # print(x["route_lanes"])
    # print(x["route_lanes_valid"])
gz_file="/mnt/pai-pdc-nas/tianle_DPR/waymo/data_waymo/testing_module_processed/processed/scenario_80b6930491b32cb0.pkl"

content = read_gz_file_with_fallback(gz_file)
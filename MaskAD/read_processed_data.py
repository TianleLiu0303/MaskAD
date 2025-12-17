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
            # print(v.shape)
            print(v.shape if hasattr(v, 'shape') else v)
    print(x["agents_slot"])
    print(x["agents_object_id"])
gz_file="/mnt/pai-pdc-nas/tianle_DPR/waymo/data_waymo/testing_module_processed/processed/scenario_41f633d89d641255.pkl"

content = read_gz_file_with_fallback(gz_file)


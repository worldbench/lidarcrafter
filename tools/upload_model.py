from huggingface_hub import upload_folder, upload_file
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# export HF_ENDPOINT=https://hf-mirror.com 
local_file = "/home/alan/AlanLiang/Projects/AlanLiang/LidarGen4D/data/infos/nuscenes_infos_val.pkl"

repo_id = "ALanLiangC/Alan"

upload_file(
    path_or_fileobj=local_file,
    path_in_repo='nuscenes_infos_val.pkl',
    repo_id=repo_id,
    repo_type="model",
)
from huggingface_hub import snapshot_download
import transformers
print(transformers.__version__)
snapshot_download(repo_id="meta-llama/Llama-2-7b-hf",
 local_dir="/data/140-0/users/liying/Sharing/Models/meta-llama/Llama-2-7b-hf",
 local_dir_use_symlinks=False, max_workers=1 )

# meta-llama/Llama-3.2-1B

# snapshot_download(repo_id="meta-llama/Llama-3.2-1B",
#  local_dir="/data/140-0/users/liying/Sharing/Models/meta-llama/Llama-3.2-1B",
#  local_dir_use_symlinks=False, max_workers=1 )

# snapshot_download(repo_id="stanford-oval/llama-7b-wikiwebquestions",
#  local_dir="/disk1/users/liying/AA_DataSharing/initial_model/llama-13b/",
#  local_dir_use_symlinks=False, max_workers=1 )


# T5-large
# snapshot_download(repo_id="google-t5/t5-large",
#  local_dir="/data/140-0/users/liying/Sharing/Models/google-t5/t5-large",
#  local_dir_use_symlinks=False, max_workers=8 )

# T5-large google-t5/t5-large
# snapshot_download(repo_id="meta-llama/Llama-2-7b-chat-hf",
#  local_dir="/disk1/users/liying/AA_DataSharing/initial_model/meta-llama/Llama-2-7b-chat-hf",
#  local_dir_use_symlinks=False, max_workers=1 )


# T5-small 
# snapshot_download(repo_id="google-t5/t5-small",
#  local_dir="/data/140-0/users/liying/Sharing/Models/google-t5/t5-small",
#  local_dir_use_symlinks=False, max_workers=8 )

# snapshot_download(repo_id="google/vit-base-patch16-224",
#  local_dir="initial_model/vit-base-patch16-224/",
#  local_dir_use_symlinks=False, max_workers=1 )






# vit-base-patch16-224-in21k
# snapshot_download(repo_id="google/vit-base-patch16-224-in21k",
#  local_dir="initial_model/vit-base-patch16-224-in21k/",
#  local_dir_use_symlinks=False, max_workers=1 )


#  openai/clip-vit-base-patch16
# snapshot_download(repo_id="openai/clip-vit-base-patch16",
#  local_dir="initial_model/clip-vit-base-patch16/",
#  local_dir_use_symlinks=False, max_workers=1 )

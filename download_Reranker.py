from huggingface_hub import snapshot_download

model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"

local_dir = "hugging_face_model/ms-marco-MiniLM-L-6-v2"

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)

print("Reranker model downloaded successfully.")
print(f"Saved at: {local_dir}")
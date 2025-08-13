from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    local_dir="./mistralai_base_model",
    local_dir_use_symlinks=False,
    token=True  # Only if it's private
)

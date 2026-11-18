"""Upload the trained weights (artifacts/) to a Hugging Face Hub model repo, so that they are
not stored in git (requirement R7).

    huggingface-cli login            # or: export HF_TOKEN=...
    python upload_artifacts.py selimyilm/support-ticket-router
"""
import sys

from huggingface_hub import HfApi

repo = sys.argv[1]
api = HfApi()
api.create_repo(repo, repo_type="model", exist_ok=True)
api.upload_folder(folder_path="artifacts", repo_id=repo, repo_type="model",
                  ignore_patterns=["scratch_tok/*"])
print(f"uploaded → https://huggingface.co/{repo}")

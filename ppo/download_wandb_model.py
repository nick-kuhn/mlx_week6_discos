import wandb
import os

# The artifact path you provided
ARTIFACT_PATH = "ntkuhn/summarization-finetuning/best_finetuned_model_qwen_summarization_20250716_170600:v57"

# The local directory where you want to save it
LOCAL_DIR = "./qwen_finetuned_local"

print(f"Downloading artifact: {ARTIFACT_PATH}")

# Initialize a wandb run to use the API
run = wandb.init(project="model-downloader", job_type="download")

# Download the artifact
artifact = run.use_artifact(ARTIFACT_PATH)
downloaded_path = artifact.download(root=LOCAL_DIR)

print(f"Model downloaded to: {downloaded_path}")

run.finish()
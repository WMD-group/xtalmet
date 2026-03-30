"""Upload pre-computed journal evaluation results to HuggingFace dataset."""

from pathlib import Path

from huggingface_hub import HfApi

api = HfApi()
base = Path("examples/journal/results/mp20/")

for model_dir in sorted(base.iterdir()):
	if not model_dir.is_dir():
		continue
	print(f"Uploading {model_dir.name}...")
	api.upload_folder(
		repo_id="masahiro-negishi/xtalmet",
		repo_type="dataset",
		folder_path=str(model_dir),
		path_in_repo=f"journal/results/mp20/{model_dir.name}",
	)
	print(f"Done: {model_dir.name}")

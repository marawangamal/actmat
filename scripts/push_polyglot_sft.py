import os, glob
from huggingface_hub import HfApi

token = open(os.path.join(os.environ["HF_HOME"], "token")).read().strip()
api = HfApi(token=token)
me = api.whoami()["name"]

langs = ["ar", "es", "de", "cs"]
hub = os.path.join(os.environ["HF_HOME"], "hub")

for lang in langs:
    src = f"ljvmiranda921--Polyglot-OLMo3-7B-SFT-{lang}"
    snap_glob = os.path.join(hub, f"models--{src}", "snapshots", "*")
    snap = sorted(glob.glob(snap_glob))[0]
    repo_id = f"{me}/Polyglot-OLMo3-7B-SFT-{lang}"
    print(f"\n=== {lang}: {snap} -> {repo_id} ===", flush=True)
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=snap,
        commit_message=f"Add Polyglot-OLMo3-7B-SFT-{lang} (mirror of ljvmiranda921)",
    )
    print(f"=== done {lang}: https://huggingface.co/{repo_id} ===", flush=True)

print("\nALL DONE", flush=True)

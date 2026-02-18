import argparse
import subprocess
import sys
from pathlib import Path


BEAVERTAILS_URL = (
    "https://huggingface.co/datasets/anonymous4486/booster_dataset/resolve/main/"
    "beavertails_with_refusals_train.json"
)


def run_command(command, cwd=None):
    result = subprocess.run(command, cwd=cwd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}")


def download_beavertails_json(output_path: Path):
    if output_path.exists():
        print(f"[skip] already exists: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id="anonymous4486/booster_dataset",
            repo_type="dataset",
            filename="beavertails_with_refusals_train.json",
            local_dir=str(output_path.parent),
            local_dir_use_symlinks=False,
        )
        print(f"[ok] downloaded via huggingface_hub: {downloaded}")
        return
    except Exception as exc:
        print(f"[warn] huggingface_hub download failed, fallback to URL. reason: {exc}")

    try:
        import urllib.request

        urllib.request.urlretrieve(BEAVERTAILS_URL, str(output_path))
        print(f"[ok] downloaded via url: {output_path}")
    except Exception as exc:
        raise RuntimeError(
            "Failed to download beavertails_with_refusals_train.json. "
            f"Please download manually from {BEAVERTAILS_URL} to {output_path}."
        ) from exc


def build_sst2(repo_root: Path):
    print("[run] building sst2 dataset -> data/sst2.json")
    run_command([sys.executable, "build_dataset.py"], cwd=repo_root / "sst2")


def main():
    parser = argparse.ArgumentParser(description="Prepare Booster reproduction datasets.")
    parser.add_argument(
        "--repo-root",
        type=str,
        default=".",
        help="Path to Booster repo root.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    data_dir = repo_root / "data"
    beavertails_json = data_dir / "beavertails_with_refusals_train.json"

    print(f"[info] repo root: {repo_root}")
    download_beavertails_json(beavertails_json)
    build_sst2(repo_root)
    print("[done] data preparation finished.")


if __name__ == "__main__":
    main()

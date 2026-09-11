"""Weight-file resolution.

Every weight file is resolved with the same precedence:

1. an explicit ``weights_dir`` argument (CLI ``--weights-dir``),
2. the ``CUPID_WEIGHTS_DIR`` environment variable,
3. automatic download from the HuggingFace Hub (cached under the standard
   HF cache, ``~/.cache/huggingface`` by default).

The CUPID MAE checkpoint lives in the CUPID HF repo. The four third-party
face-analysis weights are downloaded from the 3DDFA_V3 authors' own HF
dataset repo and are subject to their respective licenses (see README).
"""

import os
from pathlib import Path

CUPID_HF_REPO = "heyGio/CUPID"
# Upstream 3DDFA_V3 assets (dataset repo maintained by the 3DDFA_V3 authors).
TDDFA_HF_REPO = "Zidu-Wang/3DDFA-V3"

CUPID_CHECKPOINT_FILE = "cupid_mae.pth"
RETINAFACE_FILE = "retinaface_resnet50_2020-07-20_old_torch.pth"
LANDMARK_FILE = "large_base_net.pth"
FACE_MODEL_FILE = "face_model.npy"
NET_RECON_FILE = "net_recon.pth"

TDDFA_ASSET_FILES = (RETINAFACE_FILE, LANDMARK_FILE, FACE_MODEL_FILE, NET_RECON_FILE)


def _local_weights_dir(weights_dir=None):
    return weights_dir or os.environ.get("CUPID_WEIGHTS_DIR")


def _resolve_local(filename, local_dir):
    path = Path(local_dir) / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Weight file '{filename}' not found in weights dir '{local_dir}'. "
            f"Either place the file there, or unset --weights-dir/CUPID_WEIGHTS_DIR "
            f"to download weights automatically from the HuggingFace Hub."
        )
    return str(path)


def get_tddfa_asset(filename, weights_dir=None):
    """Return a local path for one of the 3DDFA_V3 asset files."""
    local_dir = _local_weights_dir(weights_dir)
    if local_dir:
        return _resolve_local(filename, local_dir)

    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(
            repo_id=TDDFA_HF_REPO, repo_type="dataset", filename=f"assets/{filename}"
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not download '{filename}' from https://huggingface.co/datasets/{TDDFA_HF_REPO}. "
            f"Check your internet connection, or download the file manually and point "
            f"--weights-dir/CUPID_WEIGHTS_DIR at the folder containing it."
        ) from e


def get_cupid_checkpoint(weights_dir=None):
    """Return a local path for the CUPID MAE checkpoint."""
    local_dir = _local_weights_dir(weights_dir)
    if local_dir:
        return _resolve_local(CUPID_CHECKPOINT_FILE, local_dir)

    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(repo_id=CUPID_HF_REPO, filename=CUPID_CHECKPOINT_FILE)
    except Exception as e:
        raise RuntimeError(
            f"Could not download '{CUPID_CHECKPOINT_FILE}' from https://huggingface.co/{CUPID_HF_REPO}. "
            f"Check your internet connection, or download the file manually and point "
            f"--weights-dir/CUPID_WEIGHTS_DIR at the folder containing it."
        ) from e

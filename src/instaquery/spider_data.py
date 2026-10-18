"""Download and load the Spider 1.0 Text-to-SQL benchmark (Yu et al., EMNLP 2018).

Spider is cross-database: the 20 dev databases never appear in training, so the
dev score measures generalisation to *unseen schemas*.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

OFFICIAL_GDRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"  # yale-lily.github.io/spider
HF_MIRROR = ("HAL-9001/spider-databases", "spider_data.zip")
HF_MIRROR_SHA256 = "00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b"


@dataclass
class Example:
    question: str
    query: str
    db_id: str
    split: str = ""
    idx: int = 0

    def to_dict(self) -> dict:
        return {"question": self.question, "query": self.query, "db_id": self.db_id,
                "split": self.split, "idx": self.idx}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_root(base: Path) -> Path | None:
    """Return the folder containing dev.json + tables.json + database/."""
    for dev in base.rglob("dev.json"):
        root = dev.parent
        if (root / "tables.json").exists() and (root / "database").is_dir():
            return root
    return None


def download_spider(dest: Path) -> Path:
    """Download + unzip Spider into ``dest``; returns the data root. Idempotent."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    root = _find_root(dest)
    if root:
        print(f"[spider] already available at {root}")
        return root

    zip_path = dest / "spider_data.zip"
    errors = []
    # 1) official Google Drive release
    if not zip_path.exists():
        try:
            import gdown  # noqa: F401
            print("[spider] downloading official release from Google Drive ...")
            subprocess.run(["gdown", "--fuzzy", "-O", str(zip_path),
                            f"https://drive.google.com/uc?id={OFFICIAL_GDRIVE_ID}"], check=True)
            if not zipfile.is_zipfile(zip_path):
                raise RuntimeError("downloaded file is not a zip (Drive quota page?)")
        except Exception as e:  # pragma: no cover - network dependent
            errors.append(f"gdrive: {e}")
            zip_path.unlink(missing_ok=True)
    # 2) Hugging Face mirror (checksum-verified)
    if not zip_path.exists():
        try:
            from huggingface_hub import hf_hub_download
            print("[spider] downloading Hugging Face mirror ...")
            p = hf_hub_download(repo_id=HF_MIRROR[0], filename=HF_MIRROR[1], repo_type="dataset",
                                local_dir=str(dest))
            if _sha256(Path(p)) != HF_MIRROR_SHA256:
                print("[spider] WARNING: checksum differs from the value recorded in this repo")
            if Path(p) != zip_path:
                shutil.move(p, zip_path)
        except Exception as e:  # pragma: no cover
            errors.append(f"hf: {e}")
    if not zip_path.exists():
        raise RuntimeError("Could not download Spider. Errors: " + " | ".join(errors))

    print("[spider] extracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    root = _find_root(dest)
    if root is None:
        raise RuntimeError("Spider zip extracted but dev.json/tables.json/database not found")
    print(f"[spider] ready at {root}")
    return root


def load_split(root: Path, split: str) -> list[Example]:
    """split in {'train_spider', 'train_others', 'dev'}."""
    path = Path(root) / f"{split}.json"
    if not path.exists():
        return []
    rows = json.loads(path.read_text())
    return [Example(question=r["question"].strip(), query=r["query"].strip(), db_id=r["db_id"],
                    split=split, idx=i) for i, r in enumerate(rows)]


def db_path(root: Path, db_id: str) -> Path:
    return Path(root) / "database" / db_id / f"{db_id}.sqlite"


def load_train(root: Path) -> list[Example]:
    return load_split(root, "train_spider") + load_split(root, "train_others")


def load_dev(root: Path) -> list[Example]:
    return load_split(root, "dev")

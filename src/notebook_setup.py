"""
src/notebook_setup.py

Bootstrap helpers for Colab notebooks. Three functions, each replacing one
duplicated cell from the old per-notebook bootstrap:

    setup_environment(repo_url, project_folder_name="Senior_Project")
        Mount Drive, ensure repo at /content/senior_project is current via
        git pull --ff-only (or clone if missing), put repo on sys.path.
        Returns (DRIVE_ROOT, REPO_ROOT).

    copy_to_local(drive_root, datasets, local_root="/content/Senior_Project_local")
        Copy data/<dataset>/ from Drive to local SSD scratch, create empty
        outputs/ on local SSD, chdir to local_root. Returns LOCAL_ROOT.

    sync_outputs_to_drive(drive_root, local_root, task, dataset,
                          experiment_name, categories=...)
        End-of-run sync: batched copytree per category from local SSD back
        to Drive.

Why local SSD: the dev/run doc §1 documents the "fold-4 freeze" pattern
where sustained writes to Drive FUSE hang. We avoid it by writing only to
local SSD during a run and batching one Drive copy at the end.

Cell-2 pattern in notebooks (see dev/run doc §5):

    import os, sys
    if not os.path.exists("/content/senior_project"):
        from google.colab import userdata
        try:
            token = userdata.get("GITHUB_TOKEN")
        except Exception:
            token = None
        url = "https://github.com/salemaker47/senior_project.git"
        if token:
            url = url.replace("https://", f"https://{token}@", 1)
        os.system(f"git clone {url} /content/senior_project")
    if "/content/senior_project" not in sys.path:
        sys.path.insert(0, "/content/senior_project")

    from src.notebook_setup import setup_environment, copy_to_local
    DRIVE_ROOT, REPO_ROOT = setup_environment(
        repo_url="https://github.com/salemaker47/senior_project.git",
    )
    LOCAL_ROOT = copy_to_local(DRIVE_ROOT, datasets=["figshare"])
    PROJECT_ROOT = LOCAL_ROOT
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile as _zipfile
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

PathLike = Union[str, Path]

DEFAULT_LOCAL_ROOT  = "/content/Senior_Project_local"
DEFAULT_REPO_ROOT   = "/content/senior_project"
DEFAULT_DRIVE_MOUNT = "/content/drive"
DEFAULT_CATEGORIES  = ("checkpoints", "logs", "tables", "figures", "predictions")


# --------------------------------------------------------------------------- #
# Drive mount
# --------------------------------------------------------------------------- #
def _mount_drive(mountpoint: str = DEFAULT_DRIVE_MOUNT) -> None:
    """Idempotent Google Drive mount via google.colab.drive."""
    if os.path.exists(f"{mountpoint}/MyDrive"):
        return
    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError(
            "google.colab is not available. notebook_setup is meant to be "
            "imported from inside a Google Colab runtime."
        ) from exc
    drive.mount(mountpoint)


# --------------------------------------------------------------------------- #
# GitHub auth (private repos)
# --------------------------------------------------------------------------- #
def _get_github_token() -> Optional[str]:
    """Return GITHUB_TOKEN from Colab Secrets if available; None otherwise."""
    try:
        from google.colab import userdata
    except ImportError:
        return None
    try:
        return userdata.get("GITHUB_TOKEN")
    except Exception:
        return None


def _authenticated_url(repo_url: str, token: Optional[str]) -> str:
    """Inject token into HTTPS URL if both present; else return url unchanged."""
    if not token:
        return repo_url
    if repo_url.startswith("https://"):
        return repo_url.replace("https://", f"https://{token}@", 1)
    return repo_url


def _redact(cmd: Sequence[str]) -> str:
    """Hide token-bearing args before printing for error messages."""
    return " ".join("***REDACTED***" if "@github.com" in arg else arg for arg in cmd)


def _run(cmd: Sequence[str]) -> None:
    """Run a subprocess silently. On failure, raise with redacted cmd + stderr."""
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Command failed (exit {exc.returncode}): {_redact(cmd)}\n"
            f"stderr: {exc.stderr}"
        ) from exc


def _clone_repo(repo_url: str, dest: Path) -> None:
    """Clone repo. Strips token from .git/config after clone."""
    token = _get_github_token()
    auth_url = _authenticated_url(repo_url, token)
    _run(["git", "clone", auth_url, str(dest)])
    if token:
        _run(["git", "-C", str(dest), "remote", "set-url", "origin", repo_url])


def _pull_repo(repo_url: str, dest: Path) -> None:
    """git pull --ff-only origin main. Token injected then immediately stripped."""
    token = _get_github_token()
    auth_url = _authenticated_url(repo_url, token)
    if token:
        _run(["git", "-C", str(dest), "remote", "set-url", "origin", auth_url])
    try:
        _run(["git", "-C", str(dest), "pull", "--ff-only", "origin", "main"])
    finally:
        if token:
            _run(["git", "-C", str(dest), "remote", "set-url", "origin", repo_url])


# --------------------------------------------------------------------------- #
# Public: setup_environment
# --------------------------------------------------------------------------- #
def setup_environment(
    repo_url: str,
    project_folder_name: str = "Senior_Project",
) -> Tuple[Path, Path]:
    """
    Mount Google Drive, ensure /content/senior_project/ is current with origin,
    add it to sys.path so `from src.X import Y` works.

    Returns:
        (DRIVE_ROOT, REPO_ROOT)
            DRIVE_ROOT = /content/drive/MyDrive/<project_folder_name>
            REPO_ROOT  = /content/senior_project

    Note: cell-2 of every notebook prepends a small "bootstrap shim" that
    handles the very first clone (because this function can't be imported
    until the repo exists). After the first call, this function manages
    pulls. See module docstring for the shim pattern.
    """
    _mount_drive()

    drive_root = Path(DEFAULT_DRIVE_MOUNT) / "MyDrive" / project_folder_name
    if not drive_root.exists():
        raise FileNotFoundError(
            f"Drive folder not found: {drive_root}\n"
            f"Create it manually in Drive (M0 Stage 9): "
            f"MyDrive/{project_folder_name}/{{data,outputs}}/"
        )

    repo_root = Path(DEFAULT_REPO_ROOT)
    if (repo_root / ".git").exists():
        # Strip any token left in .git/config by the bootstrap shim before pulling.
        _run(["git", "-C", str(repo_root), "remote", "set-url", "origin", repo_url])
        _pull_repo(repo_url, repo_root)
    else:
        _clone_repo(repo_url, repo_root)

    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    return drive_root, repo_root


# --------------------------------------------------------------------------- #
# Public: copy_to_local
# --------------------------------------------------------------------------- #
def copy_to_local(
    drive_root: PathLike,
    datasets: Iterable[str],
    local_root: PathLike = DEFAULT_LOCAL_ROOT,
) -> Path:
    """
    Copy data/<dataset>/ from Drive to local SSD for each dataset, create an
    empty outputs/ on local SSD, and chdir to local_root.

    Idempotent: if local_root/data/<dataset>/ already exists, that dataset
    is skipped (re-runs in the same Colab session don't recopy).

    Returns:
        LOCAL_ROOT (Path)
    """
    drive_root = Path(drive_root)
    local_root = Path(local_root)
    local_root.mkdir(parents=True, exist_ok=True)

    local_data = local_root / "data"
    local_data.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        src = drive_root / "data" / dataset
        dst = local_data / dataset

        zip_src = drive_root / "data" / f"{dataset}.zip"

        if not src.exists() and not zip_src.exists():
            print(f"[copy_to_local] WARNING: source not found, skipping: {src}")
            continue
        if dst.exists():
            print(f"[copy_to_local] already present, skipping: {dst}")
            continue

        if zip_src.exists():
            local_zip = local_data / f"{dataset}.zip"
            size_mb = zip_src.stat().st_size / 1_048_576
            print(f"[copy_to_local] copying zip {zip_src.name} ({size_mb:.0f} MB) ...")
            shutil.copy2(str(zip_src), str(local_zip))
            print(f"[copy_to_local] extracting to {local_data} ...")
            r = subprocess.run(
                ["unzip", "-q", str(local_zip), "-d", str(local_data)],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                with _zipfile.ZipFile(str(local_zip)) as zf:
                    zf.extractall(str(local_data))
            local_zip.unlink(missing_ok=True)
            print(f"[copy_to_local] extracted {dataset}")
        else:
            print(f"[copy_to_local] copying {src} -> {dst}")
            shutil.copytree(src, dst)

    (local_root / "outputs").mkdir(parents=True, exist_ok=True)
    os.chdir(local_root)
    print(f"[copy_to_local] cwd is now {os.getcwd()}")
    return local_root


# --------------------------------------------------------------------------- #
# Public: sync_outputs_to_drive
# --------------------------------------------------------------------------- #
def sync_outputs_to_drive(
    drive_root: PathLike,
    local_root: PathLike,
    task: str,
    dataset: str,
    experiment_name: str,
    categories: Iterable[str] = DEFAULT_CATEGORIES,
) -> None:
    """
    Copy outputs/<cat>/<task>/<dataset>/<experiment_name>/ from local SSD
    back to Drive, one batched copytree per category.

    Categories not present locally are skipped silently (e.g. predictions
    won't exist after a training-only run; figures may not exist after a
    test-only run).
    """
    drive_root = Path(drive_root)
    local_root = Path(local_root)
    suffix = Path(task) / dataset / experiment_name

    synced, skipped = [], []
    for category in categories:
        src = local_root / "outputs" / category / suffix
        dst = drive_root / "outputs" / category / suffix

        if not src.exists():
            skipped.append(str(category))
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        synced.append(str(category))

    if synced:
        print(f"[sync_outputs_to_drive] synced: {', '.join(synced)}")
    if skipped:
        print(f"[sync_outputs_to_drive] skipped (not present locally): {', '.join(skipped)}")
    if not synced and not skipped:
        print("[sync_outputs_to_drive] nothing to sync (no categories specified)")
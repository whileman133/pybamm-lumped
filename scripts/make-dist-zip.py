"""
make-dist-zip.py

Package this project into a zip to share with teammates, leaving out the virtual
environment, git history, caches, and OS clutter. Recipients unzip, cd into the
folder, and run `uv sync` (see README).

Usage: uv run scripts/make-dist-zip.py [--out-dir DIR]
Writes <out-dir>/<YYYY-MM-DD>-pybamm-lumped-<version>.zip (default out-dir: dist/).
"""
import argparse
import datetime
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOP_DIR = "pybamm-lumped"  # folder name inside the zip

# Directory names skipped anywhere in the tree.
EXCLUDE_DIRS = {
    ".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    ".idea", ".vscode", "dist", "build", "wheels",
}
# File names / suffixes skipped anywhere in the tree.
EXCLUDE_FILES = {".DS_Store", "Thumbs.db", "desktop.ini"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".egg-info"}


def included_files(root: Path) -> list[Path]:
    """Files under root to ship, as paths relative to root."""
    files = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in EXCLUDE_DIRS or part.endswith(".egg-info") for part in rel.parts[:-1]):
            continue
        if not path.is_file() or path.is_symlink():
            continue
        if rel.parts[0] in EXCLUDE_DIRS or path.name in EXCLUDE_FILES or path.suffix in EXCLUDE_SUFFIXES:
            continue
        files.append(rel)
    return files


def make_zip(root: Path, out_dir: Path, date: datetime.date | None = None) -> Path:
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    date = date or datetime.date.today()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{date.isoformat()}-{TOP_DIR}-{version}.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in included_files(root):
            zf.write(root / rel, (Path(TOP_DIR) / rel).as_posix())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "dist", help="where to write the zip (default: dist/)")
    args = parser.parse_args()
    out = make_zip(ROOT, args.out_dir)
    with zipfile.ZipFile(out) as zf:
        n = len(zf.namelist())
    print(f"Wrote {out} ({n} files, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

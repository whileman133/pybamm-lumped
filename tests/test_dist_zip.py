"""
scripts/make-dist-zip.py ships what `uv sync` needs and nothing machine-specific.
"""
import datetime
import importlib.util
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("make_dist_zip", ROOT / "scripts" / "make-dist-zip.py")
make_dist_zip = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_dist_zip)


def test_excludes_environment_vcs_caches_and_os_clutter(tmp_path):
    src = tmp_path / "proj"
    keep = ["pyproject.toml", "uv.lock", ".python-version", ".gitignore", "src/pkg/mod.py", "src/pkg/data.xlsx"]
    drop = [".venv/lib/x.py", ".git/HEAD", "src/pkg/__pycache__/mod.cpython-312.pyc", ".DS_Store",
            "src/.DS_Store", ".pytest_cache/v", "dist/old.zip", ".idea/ws.xml", "src/pkg/stray.pyc"]
    for rel in keep + drop:
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text('[project]\nversion = "9.9"\n' if rel == "pyproject.toml" else "x")

    out = make_dist_zip.make_zip(src, tmp_path / "out", date=datetime.date(2026, 9, 23))

    assert out.name == "2026-09-23-pybamm-lumped-9.9.zip"
    names = set(zipfile.ZipFile(out).namelist())
    assert names == {f"pybamm-lumped/{rel}" for rel in keep}


def test_real_project_zip_has_what_uv_sync_needs(tmp_path):
    names = set(zipfile.ZipFile(make_dist_zip.make_zip(ROOT, tmp_path)).namelist())
    for rel in ["pyproject.toml", "uv.lock", ".python-version", "README.md",
                "src/lumped/__init__.py", "src/lumped/lib-defaults.json",
                "src/lumped/_vendor/pybammeis/LICENSE.md"]:
        assert f"pybamm-lumped/{rel}" in names
    assert not any("/.venv/" in n or "/.git/" in n or "__pycache__" in n for n in names)

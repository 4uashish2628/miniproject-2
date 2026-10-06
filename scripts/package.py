"""Create a portable source/document archive, excluding recordings and binaries."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(__file__).resolve().parents[1]
output = root / "deliverables" / "PoseLab_Project.zip"
root_files = {
    "README.md", "THIRD_PARTY_NOTICES.md", "requirements.txt",
    "requirements-dev.txt", "run.py", ".gitignore",
}
folders = {"posehar", "static", "scripts", "tests", "docs", "deliverables"}

with ZipFile(output, "w", ZIP_DEFLATED) as archive:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if rel.name not in root_files and rel.parts[0] not in folders:
            continue
        if "__pycache__" in rel.parts or path.suffix in {".pyc", ".zip"}:
            continue
        if rel.parts[:2] in {("static", "vendor"), ("static", "models")}:
            continue
        if rel.name == "asset_manifest.json":
            continue
        archive.write(path, Path("PoseLab") / rel)
    for folder in ("data/clips", "models", "results"):
        archive.writestr(f"PoseLab/{folder}/.gitkeep", "")

with ZipFile(output) as archive:
    assert archive.testzip() is None
    print(f"Created {output.name}: {len(archive.namelist())} files")

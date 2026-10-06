"""Verify the saved project without downloading weights or using a GPU."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".dev", ".git", ".venv", "models", "cache", "build", "dist", "local", "private", "__pycache__"}


def main():
    files = [p for p in ROOT.rglob("*") if p.is_file() and not (set(p.relative_to(ROOT).parts) & EXCLUDED)]
    for file in files:
        # A pinned numeric GEMM tuning table is not a private benchmark dataset.
        tuning_table = ROOT / "experimental/rocm/tunable-rx580-2026-10-04.csv"
        if file == tuning_table:
            rows = list(csv.reader(file.read_text().splitlines()))
            expected = {"PT_VERSION", "HIP_VERSION", "HIPBLASLT_VERSION", "GCN_ARCH_NAME", "ROCBLAS_VERSION"}
            assert {row[1] for row in rows if row[0] == "Validator"} == expected
            for row in rows:
                if row[0] == "Validator":
                    assert len(row) == 3 and re.fullmatch(r"[0-9A-Za-z._-]+", row[2])
                else:
                    assert len(row) == 4 and re.fullmatch(r"Gemm(?:StridedBatched)?TunableOp_(?:Half|float)_[NT]{2}", row[0])
                    assert re.fullmatch(r"[0-9A-Za-z_]+", row[1])
                    assert re.fullmatch(r"Default|Gemm_Rocblas_-?[0-9]+", row[2]) and float(row[3]) >= 0
        else:
            assert file.suffix not in {".safetensors", ".gguf", ".onnx", ".pem", ".key", ".csv", ".jsonl", ".zip"}, file
        if file.suffix == ".py":
            ast.parse(file.read_text(), filename=str(file))
        if file.suffix == ".json":
            json.loads(file.read_text())
        if file.suffix == ".md":
            for link in re.findall(r"\]\(([^)]+)\)", file.read_text()):
                if not link.startswith(("http:", "https:", "#")):
                    assert (file.parent / link.split("#")[0]).exists(), (file, link)
    provenance = json.loads((ROOT / "docs/provenance.json").read_text())
    assert hashlib.sha256((ROOT / "vendor/cloudflare/joint_schema_model.py").read_bytes()).hexdigest() == provenance["upstream_wrapper_sha256"]
    pack = ROOT / "artifacts/triton-host"
    manifest = json.loads((pack / "manifest.json").read_text())
    allowed = {"manifest.json", "LICENSE"}
    for record in manifest["modules"]:
        binary = (pack / record["file"]).resolve()
        assert binary.is_relative_to(pack.resolve()) and binary.is_file()
        assert hashlib.sha256(binary.read_bytes()).hexdigest() == record["sha256"]
        allowed.add(record["file"])
    assert {str(p.relative_to(pack)) for p in pack.rglob("*") if p.is_file()} == allowed, "Unmanifested runtime artifacts"
    html = (ROOT / "web/index.html").read_text()
    js = (ROOT / "web/app.js").read_text()
    assert 'id="model-name"' in html and "model: apiModel" in js and "apiModel = health.model" in js
    assert 'Up to 16 images' in html and 'images.length > 16' in js
    assert 'value="compact" selected' in html and 'type="file"' in html and "paste" in js
    assert "/ui/image-input.js" in html and (ROOT / "web/image-input.js").is_file()
    assert "request.media_kwargs = {images_kwargs: {do_resize: true, min_pixels: 1024, max_pixels: 20000000}}" in js
    example = json.loads((ROOT / "examples/request.json").read_text())
    assert {q["type"] for q in example["questions"].values()} == {"noul", "choice", "score"}
    for profile in ["full", "flash"]:
        config = json.loads((ROOT / f"profiles/{profile}.json").read_text())
        wrapper = (ROOT / f"builds/{profile}/clef_app.py").read_text()
        assert 'clef_service.app' in wrapper
        assert f'"{profile}"' in wrapper
        app = (ROOT / "clef_service/app.py").read_text()
        assert 'max_length=16,' in app and '"max_images": 16' in app
        assert 'SOURCE_REVISION = PROFILE["revision"]' in app
        assert 'UI_DIR = PROJECT_ROOT / "web"' in app
        assert 'str(PROJECT_ROOT / "vendor" / "cloudflare")' in app
        assert (ROOT / f"requirements/{profile}.txt").is_file()
    from benchmark_emails import load_dataset
    sample, _, _, _ = load_dataset(ROOT / "benchmarks/email-sample")
    print(f"PASS: {len(files)} project files; Python syntax, profiles, provenance, UI, links, and {len(sample)} synthetic emails")


if __name__ == "__main__":
    main()

"""CPU-only checks of native request options and deprecated fidelity compatibility."""
import ast
from pathlib import Path
import sys
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
from image_request import media_options

assert media_options(None, None, 1048576) is None
native = {"images_kwargs": {"do_resize": False}}
assert media_options(native, None, 1048576) is native
native = {"images_kwargs": {"min_pixels": 1024, "max_pixels": 4194304}}
assert media_options(native, None, 1048576) is native
assert media_options(None, "standard", 1048576)["images_kwargs"]["max_pixels"] == 65536
for options, fidelity, limit in [(native, "high", 1048576), (None, "high", 65536)]:
    try:
        media_options(options, fidelity, limit)
    except ValueError:
        pass
    else:
        raise AssertionError("Ambiguous or unsupported legacy options accepted")

# Exercise the real adapter schemas without importing/loading GPU model weights.
for profile in ("full", "flash"):
    source = ast.parse((ROOT / "clef_service/app.py").read_text())
    schema = ast.Module(body=[n for n in source.body if isinstance(n, ast.ClassDef)
                             and n.name in {"Question", "DecisionRequest"}], type_ignores=[])
    namespace = dict(BaseModel=BaseModel, ConfigDict=ConfigDict, Field=Field,
                     Literal=Literal, POOLING_DEFAULT=False, MODEL_ID="clef" if profile == "full" else "clef-flash", __name__=__name__)
    exec(compile(schema, str(ROOT / "clef_service/app.py"), "exec"), namespace)
    request_type = namespace["DecisionRequest"]
    for options in (None, {"images_kwargs": {"do_resize": False}}):
        request = request_type(state="Inspect the image.", media_kwargs=options,
                              questions={"visible": {"type": "noul"}})
        assert request.image_fidelity is None
        assert request.media_kwargs == options
        dumped = request.model_dump(exclude_none=True)
        assert "image_fidelity" not in dumped
    assert request_type.model_json_schema()["properties"]["image_fidelity"]["deprecated"]
print("PASS: native defaults/options, no-resize input, legacy limits/conflicts and both request schemas")

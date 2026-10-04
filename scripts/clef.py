"""Compatibility launcher; all profiles use the unified implementation."""
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    # Preserve `scripts/clef.py serve flash/full` and legacy default ports.
    args = sys.argv[1:]
    if len(args) >= 2 and args[1] in {"full", "flash"}:
        profile = args.pop(1)
        args.extend(["--model", profile])
        if args[0] == "serve" and "--checkpoint" not in args and not os.environ.get("CLEF_MODEL_DIR"):
            root = Path(__file__).resolve().parents[1]
            data = Path(args[args.index("--data-dir")+1]) if "--data-dir" in args else Path(
                os.environ.get("CLEF_DATA_DIR", root / "models" / profile))
            config = json.loads((root / "profiles" / f"{profile}.json").read_text())
            checkpoint = data / config["checkpoint"]
            if checkpoint.is_dir() and not (checkpoint / "clef-checkpoint.json").exists():
                args.extend(["--checkpoint", str(checkpoint.resolve())])

        if "--port" not in args:
            args.extend(["--port", "8083" if profile == "full" else "8082"])
        if "--host" not in args:
            args.extend(["--host", "127.0.0.1"])
    from clef_service.__main__ import main as unified_main
    unified_main(args)


if __name__ == "__main__":
    main()

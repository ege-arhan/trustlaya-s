from __future__ import annotations
import importlib.metadata as md
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def version(name):
    try: return md.version(name)
    except md.PackageNotFoundError: return "not installed"
def main():
    import torch
    ram = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
    data = {
        "macOS": subprocess.check_output(["sw_vers", "-productVersion"], text=True).strip(),
        "architecture": platform.machine(), "python": sys.version.split()[0],
        "ram_gib": round(ram / 2**30, 2), "free_disk_gib": round(shutil.disk_usage(ROOT).free / 2**30, 2),
        "mps_available": torch.backends.mps.is_available(), "device": "mps" if torch.backends.mps.is_available() else "cpu",
        "packages": {p: version(p) for p in ["torch", "transformers", "datasets", "onnxruntime", "optimum", "tokenizers", "safetensors"]},
    }
    (ROOT / "environment_report.md").write_text("# Environment report\n\n```json\n" + json.dumps(data, indent=2) + "\n```\n")
    print(json.dumps(data, indent=2))
if __name__ == "__main__": main()

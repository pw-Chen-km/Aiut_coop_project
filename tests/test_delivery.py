import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_checksum_command_rejects_lfs_pointer(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:dummy\n")
    output = tmp_path / "checksum.json"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/record_model.py"),
                             str(model), "--output", str(output)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "not a GGUF binary" in result.stderr
    assert not output.exists()


def test_checksum_command_records_exact_bytes_without_loading_model(tmp_path):
    model = tmp_path / "fixture.gguf"
    data = b"GGUF" + b"synthetic checksum fixture; not a usable model"
    model.write_bytes(data)
    output = tmp_path / "checksum.json"
    subprocess.run([sys.executable, str(ROOT / "scripts/record_model.py"), str(model),
                    "--output", str(output)], check=True, capture_output=True)
    report = json.loads(output.read_text())
    assert report == {"filename": model.name, "size_bytes": len(data),
                      "sha256": hashlib.sha256(data).hexdigest()}

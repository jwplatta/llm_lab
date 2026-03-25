import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_lab.entry import run


def test_run_prints_ready_message(capsys):
    run()
    captured = capsys.readouterr()
    assert "llm-lab Python workspace is ready" in captured.out

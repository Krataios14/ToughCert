import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUBCOMMANDS = ["ingest", "prepare", "qualify", "certify", "screen"]


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "src.cli", *args],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )


def test_top_level_help_lists_all_subcommands():
    result = run_cli("--help")
    assert result.returncode == 0, result.stderr
    for name in SUBCOMMANDS:
        assert name in result.stdout


def test_qualify_help_exits_zero():
    result = run_cli("qualify", "--help")
    assert result.returncode == 0, result.stderr


def test_set_seed_works():
    from src.utils import set_seed

    set_seed(0)


def test_utils_has_no_top_level_torch_import():
    # torch must only be imported inside functions, so src.utils stays
    # importable without it. Top-level imports start at column 0.
    source = (ROOT / "src" / "utils.py").read_text(encoding="utf-8")
    for line in source.splitlines():
        assert not line.startswith("import torch"), line
        assert not line.startswith("from torch"), line

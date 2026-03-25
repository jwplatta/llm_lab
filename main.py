import sys
from pathlib import Path


def main():
    """Run the python workspace entrypoint."""
    root = Path(__file__).resolve().parent
    src_root = root / "python" / "src"
    sys.path.insert(0, str(src_root))
    from llm_lab.entry import run

    run()


if __name__ == "__main__":
    main()

"""Execute the lab notebook locally and save outputs after a successful run."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent


def main():
    path = HERE / "algebra.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    client = NotebookClient(notebook, timeout=300, kernel_name="python3",
                            resources={"metadata": {"path": str(HERE)}})
    client.execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print(f"Executed and saved {path}")


if __name__ == "__main__":
    main()

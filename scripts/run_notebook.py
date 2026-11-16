"""Execute all notebook cells in a fresh, isolated kernel and verify outputs."""
from __future__ import annotations

import argparse
from contextlib import chdir
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager


def execute_inprocess(nb: nbformat.NotebookNode, root: Path) -> None:
    """Use a real fresh ipykernel and its messages without opening sockets.

    This script itself is launched in a new Python process. No execution state
    is reused. Notebook outputs are constructed only from actual IOPub messages.
    This mode is synchronous; the caller can enforce a process-level timeout.
    """
    from ipykernel.inprocess.manager import InProcessKernelManager
    manager = InProcessKernelManager()
    manager.start_kernel()
    client = manager.client()
    client.start_channels()
    try:
        with chdir(root):
            for number, cell in enumerate(nb.cells):
                if cell.cell_type != "code":
                    continue
                msg_id = client.execute(cell.source, store_history=True, allow_stdin=False)
                reply = client.get_shell_msg()
                cell.execution_count = reply["content"]["execution_count"]
                for msg in client.iopub_channel.get_msgs():
                    parent_id = msg.get("parent_header", {}).get("msg_id")
                    # InProcess stdout messages have an empty parent header.
                    # Execution is synchronous and this queue is drained after
                    # every cell, so these messages belong to the current cell.
                    unparented_stream = msg["msg_type"] == "stream" and parent_id is None
                    if parent_id != msg_id and not unparented_stream:
                        continue
                    if msg["msg_type"] in {"stream", "display_data", "execute_result", "error"}:
                        cell.outputs.append(nbformat.v4.output_from_msg(msg))
                    elif msg["msg_type"] == "clear_output":
                        cell.outputs.clear()
                if reply["content"]["status"] != "ok":
                    raise RuntimeError(f"Notebook cell {number} failed: {reply['content']}")
    finally:
        client.stop_channels()
        manager.shutdown_kernel()


def main() -> None:
    """Run from any directory using the current Python, not a global kernel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=["tcp", "ipc", "inprocess"], default="inprocess")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cache = root / ".cache"
    cache.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
    os.environ["IPYTHONDIR"] = str(cache / "ipython")
    os.environ["JUPYTER_PLATFORM_DIRS"] = "1"
    os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
    notebook_path = root / "analysis.ipynb"
    nb = nbformat.read(notebook_path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    start = time.monotonic()
    if args.transport == "inprocess":
        execute_inprocess(nb, root)
    else:
        execute_external_kernel(nb, root, args.transport)
    code = [c for c in nb.cells if c.cell_type == "code"]
    errors = [o for c in code for o in c.outputs if o.output_type == "error"]
    images = [o for c in code for o in c.outputs if "image/png" in o.get("data", {})]
    if errors or not code or not all(c.execution_count for c in code):
        raise AssertionError("Notebook execution is incomplete")
    if [c.execution_count for c in code] != list(range(1, len(code) + 1)):
        raise AssertionError("Cells were not executed sequentially from a fresh kernel")
    if len(images) < 10:
        raise AssertionError("Expected at least ten actual inline PNG figure outputs")
    nbformat.validate(nb)
    exporter = HTMLExporter()
    html, _ = exporter.from_notebook_node(nb)
    html_images = len(re.findall(r'<img\b[^>]*\bsrc="data:image/png;base64,', html))
    if html_images != len(images):
        nbformat.write(nb, cache / "executed_for_review.ipynb")
        (cache / "export_for_review.html").write_text(html, encoding="utf-8")
        raise AssertionError(f"HTML figure count {html_images} "
                             f"differs from the notebook {len(images)}")
    nbformat.write(nb, notebook_path)
    (root / "reports/analysis.html").write_text(html, encoding="utf-8")
    versions = {p: importlib.metadata.version(p) for p in
                ["numpy", "pandas", "scipy", "statsmodels", "matplotlib", "openpyxl", "nbclient", "nbformat"]}
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "transport": args.transport, "fresh_kernel": True,
              "code_cells": len(code), "inline_png_figures": len(images), "errors": len(errors),
              "elapsed_seconds": round(time.monotonic() - start, 2), "versions": versions,
              "data_source": "provided_original_xlsx", "network_download_verified": False}
    (root / "reports/verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def execute_external_kernel(nb: nbformat.NotebookNode, root: Path, transport: str) -> None:
    """Run nbclient against a separate ordinary kernel when sockets are allowed."""
    with tempfile.TemporaryDirectory(prefix="retail-kernel-") as temporary:
        folder = Path(temporary)
        spec_dir = folder / "kernels/retail"
        spec_dir.mkdir(parents=True)
        spec = {"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Retail study (isolated)", "language": "python"}
        (spec_dir / "kernel.json").write_text(json.dumps(spec))
        manager = KernelSpecManager(kernel_dirs=[str(folder / "kernels")])
        km = AsyncKernelManager(kernel_name="retail", kernel_spec_manager=manager,
                                 connection_file=str(folder / "connection.json"),
                                 transport=transport,
                                 ip=str(folder / "ipc") if transport == "ipc" else "127.0.0.1")
        client = NotebookClient(nb, km=km, timeout=300, allow_errors=False,
                                 resources={"metadata": {"path": str(root)}})
        client.execute()


if __name__ == "__main__":
    main()

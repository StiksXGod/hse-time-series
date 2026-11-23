"""Check actual Jupyter stream/display/result capture in a new process."""
import os
from pathlib import Path
import subprocess
import sys


def test_inprocess_kernel_captures_real_streams_figures_and_execution_counts(tmp_path):
    root = Path(__file__).resolve().parents[1]
    program = r'''
from pathlib import Path
import nbformat
from scripts.run_notebook import execute_inprocess
nb = nbformat.v4.new_notebook(cells=[
    nbformat.v4.new_code_cell("%matplotlib inline\nimport matplotlib.pyplot as plt\nprint('captured stdout')\nx = 7\nplt.plot([1, 2]); plt.show()"),
    nbformat.v4.new_code_cell("x + 1"),
])
execute_inprocess(nb, Path.cwd())
assert [c.execution_count for c in nb.cells] == [1, 2]
assert any(o.output_type == 'stream' and 'captured stdout' in o.text for o in nb.cells[0].outputs)
assert any('image/png' in o.get('data', {}) for o in nb.cells[0].outputs)
assert nb.cells[1].outputs[0].data['text/plain'] == '8'
nbformat.validate(nb)
'''
    env = {**os.environ, "MPLCONFIGDIR": str(tmp_path / "mpl"),
           "IPYTHONDIR": str(tmp_path / "ipy"), "PYDEVD_DISABLE_FILE_VALIDATION": "1"}
    result = subprocess.run([sys.executable, "-c", program], cwd=root, env=env,
                             capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr

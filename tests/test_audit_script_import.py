import os
import subprocess
import sys
import unittest
from pathlib import Path


class AuditScriptImportTests(unittest.TestCase):
    def test_script_imports_shared_core_when_launched_from_scripts_directory(self):
        root = Path(__file__).resolve().parent.parent
        code = (
            "import os, runpy, sys; "
            "root=os.getcwd(); scripts=os.path.join(root, 'scripts'); "
            "sys.path=[scripts]+[p for p in sys.path "
            "if os.path.abspath(p or '.') not in {root, scripts}]; "
            "runpy.run_path('scripts/auditar_reconciliacion.py', run_name='audit_import_smoke')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=root,
            env={**os.environ, "PYTHONPATH": ""},
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("ModuleNotFoundError", result.stderr)


if __name__ == "__main__":
    unittest.main()

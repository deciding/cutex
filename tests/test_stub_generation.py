import pathlib
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_stub_files.py"
STUB_FILES = [
    ROOT / "cutex" / "__init__.pyi",
    ROOT / "cutex" / "pipeline.pyi",
    ROOT / "cutex" / "runtime.pyi",
    ROOT / "cutex" / "testing.pyi",
    ROOT / "cutex" / "utils" / "__init__.pyi",
    ROOT / "cutex" / "utils" / "sm100.pyi",
]


class StubGenerationTests(unittest.TestCase):
    def test_stub_generator_keeps_checked_in_files_in_sync(self) -> None:
        before = {path: path.read_text() for path in STUB_FILES}

        subprocess.run(
            ["python", str(SCRIPT)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        after = {path: path.read_text() for path in STUB_FILES}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()

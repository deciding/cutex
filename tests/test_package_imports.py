import importlib
import pathlib
import unittest


class PublicImportSurfaceTests(unittest.TestCase):
    def test_import_cutex_exposes_expected_top_level_names(self) -> None:
        cutex = importlib.import_module("cutex")
        cute = importlib.import_module("cutlass.cute")
        cute_nvgpu = importlib.import_module("cutlass.cute.nvgpu")
        cute_runtime = importlib.import_module("cutlass.cute.runtime")

        for name in [
            "cpasync",
            "tcgen05",
            "from_dlpack",
            "nvgpu",
            "runtime",
            "testing",
        ]:
            with self.subTest(name=name):
                self.assertTrue(hasattr(cutex, name), msg=f"cutex missing {name}")

        for name in ["Tensor", "Layout"]:
            with self.subTest(name=name):
                self.assertIs(getattr(cutex, name), getattr(cute, name))

        self.assertIs(cutex.cpasync, cute_nvgpu.cpasync)
        self.assertIs(cutex.tcgen05, cute_nvgpu.tcgen05)
        self.assertIs(cutex.from_dlpack, cute_runtime.from_dlpack)
        self.assertEqual(cutex.dummy(), "dummy")
        self.assertIs(cute.dummy, cutex.dummy)
        self.assertIs(cute.runtime, cutex.runtime)
        self.assertIs(cute.testing, cutex.testing)
        self.assertIs(cute.nvgpu, cutex.nvgpu)

        cute_all = set(getattr(cute, "__all__", ()))
        cutex_all = set(cutex.__all__)
        self.assertTrue(
            cutex_all.issuperset(
                {
                    "nvgpu",
                    "runtime",
                    "testing",
                    "pipeline",
                    "utils",
                    "sm100",
                    "cpasync",
                    "tcgen05",
                    "from_dlpack",
                    "dummy",
                }
            )
        )
        for name in cutex.__all__:
            with self.subTest(module="cutex", exported_name=name):
                self.assertTrue(
                    hasattr(cutex, name), msg=f"cutex.__all__ includes missing {name}"
                )

    def test_import_cutex_pipeline_exposes_expected_helpers(self) -> None:
        cutex = importlib.import_module("cutex")
        pipeline = cutex.pipeline
        upstream_pipeline = importlib.import_module("cutlass.pipeline")

        for name in ["pipeline_init_arrive", "pipeline_init_wait"]:
            with self.subTest(name=name):
                self.assertTrue(
                    hasattr(pipeline, name), msg=f"cutex.pipeline missing {name}"
                )
                self.assertIs(getattr(pipeline, name), getattr(upstream_pipeline, name))

    def test_import_cutex_utils_and_sm100_reexport_representative_names(self) -> None:
        cutex = importlib.import_module("cutex")
        utils = cutex.utils
        upstream_utils = importlib.import_module("cutlass.utils")
        sm100 = cutex.sm100
        upstream_sm100 = importlib.import_module("cutlass.utils.blackwell_helpers")

        for name in ["get_smem_capacity_in_bytes", "SmemAllocator"]:
            with self.subTest(module="cutex.utils", name=name):
                self.assertTrue(hasattr(utils, name), msg=f"cutex.utils missing {name}")
                self.assertIs(getattr(utils, name), getattr(upstream_utils, name))

        for name in ["compute_epilogue_tile_shape", "make_smem_layout_a"]:
            with self.subTest(module="cutex.utils.sm100", name=name):
                self.assertTrue(
                    hasattr(sm100, name), msg=f"cutex.utils.sm100 missing {name}"
                )
                self.assertIs(getattr(sm100, name), getattr(upstream_sm100, name))

        self.assertTrue(
            set(utils.__all__).issubset(set(getattr(upstream_utils, "__all__", ())))
        )
        for name in utils.__all__:
            with self.subTest(module="cutex.utils", exported_name=name):
                self.assertEqual(
                    hasattr(utils, name),
                    hasattr(upstream_utils, name),
                    msg=f"cutex.utils export mismatch for {name}",
                )

        self.assertTrue(
            set(sm100.__all__).issubset(set(getattr(upstream_sm100, "__all__", ())))
        )
        for name in sm100.__all__:
            with self.subTest(module="cutex.utils.sm100", exported_name=name):
                self.assertTrue(
                    hasattr(sm100, name),
                    msg=f"cutex.utils.sm100.__all__ includes missing {name}",
                )

    def test_import_cutex_runtime_and_testing_modules(self) -> None:
        cutex = importlib.import_module("cutex")
        runtime = cutex.runtime
        upstream_runtime = importlib.import_module("cutlass.cute.runtime")
        testing = cutex.testing
        upstream_testing = importlib.import_module("cutlass.cute.testing")

        self.assertIs(runtime.from_dlpack, upstream_runtime.from_dlpack)
        self.assertIs(runtime.make_fake_stream, upstream_runtime.make_fake_stream)
        self.assertIs(testing.benchmark, upstream_testing.benchmark)

        upstream_runtime_public = {
            name for name in dir(upstream_runtime) if not name.startswith("_")
        }
        runtime_public_names = [
            name for name in dir(runtime) if not name.startswith("_")
        ]
        self.assertTrue(set(runtime_public_names).issubset(upstream_runtime_public))

        upstream_testing_public = {
            name for name in dir(upstream_testing) if not name.startswith("_")
        }
        testing_public_names = [
            name for name in dir(testing) if not name.startswith("_")
        ]
        self.assertTrue(set(testing_public_names).issubset(upstream_testing_public))

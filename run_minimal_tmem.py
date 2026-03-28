"""Run minimal TMEM shape test via Modal"""

from modal import Image, App, Volume
import pathlib
import subprocess

root_dir = pathlib.Path(__file__).parent
GPU_model = "B200"

app = App(name="minimal-tmem-shape-test")

VOLUME_NAME = "fa4-dump"
volume = Volume.from_name(VOLUME_NAME, create_if_missing=True)

tmem_test_image = (
    Image.debian_slim(python_version="3.12")
    .apt_install("wget", "curl", "gnupg", "git")
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
    )
    .apt_install("cuda-toolkit-12-6")
    .workdir("/workspace")
    .pip_install("torch", "pytest", "einops")
    .pip_install("nvidia-cutlass-dsl>=4.4.1")
    .pip_install("quack-kernels>=0.2.10")
    .pip_install("apache-tvm-ffi>=0.1.5,<0.2")
    .pip_install("torch-c-dlpack-ext")
    .pip_install("triton==3.5.1")
    .add_local_dir(root_dir / "blackwell", remote_path="/workspace/blackwell")
)


@app.function(
    gpu=GPU_model,
    image=tmem_test_image,
    timeout=300,
    volumes={"/workspace/dump": volume},
)
def run_minimal_test():
    """Run the minimal TMEM shape test"""
    result = subprocess.run(
        ["python", "/workspace/blackwell/minimal_tmem_shape.py"],
        capture_output=True,
        text=True,
    )
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr[:2000] if len(result.stderr) > 2000 else result.stderr)
    return result.returncode, result.stdout, result.stderr


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("Minimal TMEM Store Shape Test")
    print("=" * 70)
    run_minimal_test.remote()

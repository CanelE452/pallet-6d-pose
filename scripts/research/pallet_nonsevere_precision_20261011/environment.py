"""CPU-only environment inventory for a clean replay; never overwrites a receipt.

Added after the original inline inventory. Its existence does not mean it was
run in the original experiment. No model, optimizer, CUDA context or PnP runs.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import cv2
import torch


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--public-dir",required=True)
    args=parser.parse_args();out=Path(args.public_dir)/"ENVIRONMENT.json"
    assert not out.exists(),"Existing environment receipt is retained"
    assert not torch.cuda.is_initialized(),"Inventory must not initialize CUDA"
    packages={name:importlib.metadata.version(name) for name in
              ("torch","torchvision","numpy","matplotlib","scipy","ultralytics")}
    inventory=subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version,memory.total",
                                      "--format=csv,noheader"],text=True).strip()
    result=dict(status="ACTUALLY_INSPECTED_CPU_ONLY",python=platform.python_version(),
                platform=platform.platform(),packages=packages,opencv=cv2.__version__,
                torch_compiled_CUDA=torch.version.cuda,GPU_readonly_inventory=inventory,
                GPU_context_initialized=torch.cuda.is_initialized(),inspection_model_forwards=0,
                inspection_optimizer_updates=0,inspection_PnP=0,
                precision_contract=dict(model_arithmetic="FP32",AMP=False,matmul_TF32=False,
                    cudnn_TF32=False,
                    neck_capture="Inherited frozen p3/p4 capture rounds to FP16, then original GenericPointRefiner samples features.float(); same for all feature-based routes",
                    PoseFix_native_RGB="FP32",
                    reference="SOURCE_BINDINGS.json contains frozen feature source hash; actual training and inference receipts record runtime flags"),
                dependency_requirement="Public code requires the identical pre-existing private data, RGB and checkpoint inputs in REPRO_DEPENDENCIES.json; these are not bundled.")
    assert result["GPU_context_initialized"] is False
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x",encoding="utf-8") as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write("\n")
    print("CPU_ENVIRONMENT_RECORDED",out.name)


if __name__=="__main__":main()

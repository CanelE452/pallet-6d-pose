#!/usr/bin/env python3
"""SAM2.1 image API smoke using only a generated constant RGB array.

This entry point intentionally has no RGB/data path argument. Loading the pinned
official checkpoint and generated-image CPU/CUDA inference are its full scope.
"""
from pathlib import Path
from contextlib import nullcontext
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time

import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

ROOT=Path(__file__).resolve().parent
CHECKPOINT_SHA='a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device',choices=('cpu','cuda'),required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    assert not args.out.exists(),'Preserve previous generated smoke output'
    ckpt=ROOT/'checkpoints/sam2.1_hiera_base_plus.pt'
    assert sha(ckpt)==CHECKPOINT_SHA and ckpt.stat().st_size==323606802
    assert Path(sys.prefix).resolve()==ROOT/'venv','Use the isolated runtime only'
    assert sys.prefix!=sys.base_prefix and sys.version_info[:2]==(3,10)
    assert torch.__version__=='2.5.1+cu121'
    assert importlib.metadata.version('torchvision')=='0.20.1+cu121'
    assert importlib.metadata.version('triton')=='3.1.0'
    torch.set_num_threads(4);torch.manual_seed(20261023)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    def forbidden_compile(*a,**kw):
        raise AssertionError('torch.compile is prohibited for this smoke')
    torch.compile=forbidden_compile
    if args.device=='cuda':
        assert torch.cuda.is_available()
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    model=build_sam2('configs/sam2.1/sam2.1_hiera_b+.yaml',str(ckpt),device=args.device,mode='eval',
                     apply_postprocessing=False,hydra_overrides_extra=['++model.compile_image_encoder=false'])
    model.requires_grad_(False);assert not model.training
    assert all(not p.requires_grad for p in model.parameters())
    predictor=SAM2ImagePredictor(model,mask_threshold=0.,max_hole_area=0.,max_sprinkle_area=0.)
    image=np.full((384,512,3),127,np.uint8)
    box=np.array([128,96,384,288],dtype=np.float32)
    image_sha=hashlib.sha256(image.tobytes()).hexdigest()
    context=torch.autocast('cuda',dtype=torch.bfloat16) if args.device=='cuda' else nullcontext()
    with torch.inference_mode(),context:
        predictor.set_image(image)
        masks,scores,low=predictor.predict(box=box,multimask_output=True,return_logits=True)
    if args.device=='cuda':torch.cuda.synchronize()
    assert masks.shape==(3,384,512) and scores.shape==(3,) and low.shape==(3,256,256)
    assert np.isfinite(masks).all() and np.isfinite(scores).all() and np.isfinite(low).all()
    assert sha(ckpt)==CHECKPOINT_SHA
    args.out.mkdir(parents=True)
    np.savez_compressed(args.out/'GENERATED_BLANK_OUTPUTS.npz',mask_logits=masks,quality=scores,lowres_logits=low)
    report=dict(schema='ISOLATED_SAM21_BASE_PLUS_GENERATED_BLANK_API_SMOKE_V1',PASS=True,
        device=args.device,python=sys.executable,python_version=platform.python_version(),venv=str(Path(sys.prefix).resolve()),
        versions={name:importlib.metadata.version(name) for name in ('torch','torchvision','triton','numpy','pillow','hydra-core','omegaconf','iopath')},
        CUDA_runtime=torch.version.cuda,GPU=torch.cuda.get_device_name(0) if args.device=='cuda' else None,
        CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else 0,
        CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.device=='cuda' else 0,
        seconds=time.monotonic()-started,model_parameters=sum(p.numel() for p in model.parameters()),
        checkpoint=dict(path=str(ckpt),sha256=CHECKPOINT_SHA,bytes=ckpt.stat().st_size),
        code=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
        generated_input=dict(shape=list(image.shape),constant_RGB=127,bytes_sha256=image_sha,box=box.tolist()),
        output_shapes=dict(mask_logits=list(masks.shape),quality=list(scores.shape),lowres_logits=list(low.shape)),
        output_all_finite=True,quality_values=scores.tolist(),image_quality_evaluation=False,
        masks_are_unthresholded_logits=True,mask_threshold_default=0.,multimask=True,
        compile_called=False,CUDA_extension_built=False,apply_postprocessing=False,
        max_hole_area=0,max_sprinkle_area=0,cuda_autocast='bfloat16' if args.device=='cuda' else None,
        model_eval=True,all_parameters_frozen=True,optimizer_updates=0,training_calls=0,PnP_calls=0,
        source_RGB_data_reads=0,current_RGB_GT_scores_inputs_read=False,
        dataset_inference_allowed_by_this_smoke=False,
        caveat='Generated constant-image API/finite-value check only; no segmentation accuracy or dataset conclusion.')
    op=args.out/'GENERATED_BLANK_OUTPUTS.npz'
    report['outputs']=[dict(path=str(op.resolve()),sha256=sha(op),bytes=op.stat().st_size)]
    (args.out/'SMOKE.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,allow_nan=False),flush=True)


if __name__=='__main__':main()

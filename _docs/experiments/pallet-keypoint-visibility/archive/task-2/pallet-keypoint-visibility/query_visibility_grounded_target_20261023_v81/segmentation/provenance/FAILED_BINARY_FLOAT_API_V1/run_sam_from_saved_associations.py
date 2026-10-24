"""Source-only SAM stage. No source/model I/O at import; root entry required.

One saved detector association -> one native bool SAM mask. Neither a mask-zero
nor an unassociated input is a semantic external/self-occlusion prediction.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import time
import numpy as np
from sam_stage_contract import (require, validate_input, validate_association,
    single_native_mask, query_mask_sample, INPUT_FIELDS, RGB_FIELDS)

R=Path(__file__).resolve().parent
V=R.parent
P=V.parent
RT=P/'foundation_runtime_sam21_20261023'
DECISIONS=V/'EXPERIMENT_DECISIONS_BEFORE_ASSETS_AND_SOURCE.json'
DECISION_SHA='34e406371c3452f7013d6f871568f218f69d9e668e1ea8115919b11d0472040e'
RUNTIME=RT/'RUNTIME_SEAL.json'
RUNTIME_SHA='e5655309c2c99ee18e76fa0ebe85541194445895673d9b64f915384130bcbf82'
CHECKPOINT=RT/'checkpoints/sam2.1_hiera_base_plus.pt'
CHECKPOINT_SHA='a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5'
CONFIG='configs/sam2.1/sam2.1_hiera_b+.yaml'
REPO=RT/'sam2_repo'
OUT=R/'outputs'
STATIC={
 'code':Path(__file__).resolve(), 'helper':R/'sam_stage_contract.py',
 'generated':R/'GENERATED_SAM_STAGE_TESTS_V3.json',
 'decisions':DECISIONS,'runtime':RUNTIME,'checkpoint':CHECKPOINT,
 'sam_build':REPO/'sam2/build_sam.py',
 'sam_predictor':REPO/'sam2/sam2_image_predictor.py',
 'sam_transforms':REPO/'sam2/utils/transforms.py',
 'sam_config':REPO/'sam2'/CONFIG,
 'precision_precedent_code':P/'query_visibility_sam21_frozen_20261023_v68/extract_source_masks.py',
 'precision_precedent_protocol':P/'query_visibility_sam21_frozen_20261023_v68/PROTOCOL.json',
}

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def bind(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s]
def check(b):
    require(set(b)>={'path','sha256'},'Path/SHA binding required')
    require(sha(b['path'])==b['sha256'],'Binding changed: '+b['path'])
    return Path(b['path'])
def write(path,obj):
    with open(path,'x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')

def entry_guard(entry_path):
    """Must fail before any source file or model module is opened."""
    e=read(entry_path)
    require(e.get('approved') is True and e.get('allow_source_SAM_forward') is True,'Root-approved SAM source entry required')
    require(e.get('current223') is False and e.get('fit') is False and e.get('PnP')==0,'Source-only/no-fit scope')
    require(e.get('expected_targets')==1035 and e.get('expected_unique_RGB')==435,'Frozen cohort count')
    require(Path(sys.executable).absolute()==(RT/'venv/bin/python').absolute(),'Use sealed SAM Python')
    require(Path(e['output_directory']).resolve()==OUT,'Fixed new output directory')
    require(not OUT.exists(),'Preserve previous outputs; no automatic retry')
    require(set(e['bindings'])==set(STATIC)|{'detector_output_seal'},'Exact entry binding whitelist')
    for name,path in STATIC.items():
        b=e['bindings'][name];require(Path(b['path']).resolve()==path,'Unexpected static path '+name);check(b)
    require(e['bindings']['decisions']['sha256']==DECISION_SHA,'Frozen decisions')
    require(e['bindings']['runtime']['sha256']==RUNTIME_SHA,'Sealed runtime')
    require(e['bindings']['checkpoint']['sha256']==CHECKPOINT_SHA,'Official checkpoint')
    proof=read(STATIC['generated']);require(proof['PASS'] is True,'Generated tests required')
    require(proof['bindings']['runner']==e['bindings']['code'] and proof['bindings']['helper']==e['bindings']['helper'],'Generated proof must bind current runner/helper')
    rt=read(RUNTIME)
    require(rt['READY'] is True and rt['config']==CONFIG and rt['repo_revision']=='2b90b9f5ceec907a1c18123530e92e794ad901a4','Runtime scope')
    check(e['bindings']['detector_output_seal'])
    return e

def saved_metadata(e):
    """Read only bound GT-free source metadata/detections, never RGB or GT."""
    seal=read(e['bindings']['detector_output_seal']['path'])
    require(seal['completed'] is True and seal['PASS'] is True,'Completed detector seal required')
    require(seal['decisions']==e['bindings']['decisions'],'Detector/SAM share frozen source protocol')
    inputs=seal['source_inputs']
    require(set(inputs)=={'manifest','unique_manifest','seal'},'Source input binding schema')
    for b in inputs.values():check(b)
    check(seal['per_RGB']);check(seal['associations'])
    targets=lines(inputs['manifest']['path']);unique=lines(inputs['unique_manifest']['path'])
    assocs=lines(seal['associations']['path']);rgbs=lines(seal['per_RGB']['path'])
    require(len(targets)==len(assocs)==1035 and len(unique)==len(rgbs)==435,'Retain entire frozen source cohort')
    require(len({r['input_id'] for r in targets})==1035,'Unique target identities')
    require(len({r['rgb_input_id'] for r in rgbs})==435,'Unique RGB identities')
    unique_by={r['input_id']:r for r in unique};rgb_by={r['rgb_input_id']:r for r in rgbs}
    require(len(unique_by)==435 and set(unique_by)==set(rgb_by),'Unique RGB join')
    dets={};prompts=[]
    for rgb in rgbs:
        require(set(rgb)==RGB_FIELDS,'PerRGB whitelist')
        u=unique_by[rgb['rgb_input_id']]
        require(set(u)=={'input_id','image','image_sha256','raw_hw'},'Unique RGB input whitelist')
        require(all(u[k]==rgb[k] for k in ('image','image_sha256','raw_hw')),'Unique RGB changed')
        check(rgb['detections'])
        with np.load(rgb['detections']['path'],allow_pickle=False) as z:dets[rgb['rgb_input_id']]={k:z[k] for k in z.files}
    for original,a in zip(targets,assocs):
        validate_input(original)
        require(a['rgb_input_id'] in rgb_by,'Unknown association RGB')
        prompts.append(validate_association(a,original,rgb_by[a['rgb_input_id']],dets[a['rgb_input_id']]))
    require({a['rgb_input_id'] for a in assocs}==set(rgb_by),'All unique RGB accounted for')
    return seal,targets,assocs,rgbs,prompts

def mask_row(association,mask_binding,quality):
    """Export no-match as U, never as a synthetic all-zero mask."""
    known=association['status']=='ASSOCIATED'
    require(known==(mask_binding is not None),'Association/mask status consistency')
    return dict(association,mask_status='OK' if known else 'U',
        mask_reason='SINGLE_BOX_NATIVE_VISIBLE_MASK' if known else 'NO_ASSOCIATED_BOX:'+association['reason'],
        mask=mask_binding,predicted_quality=quality if known else None,
        selected_mask_index=0 if known else None)

def state_digest(model):
    h=hashlib.sha256()
    for k,t in sorted(model.state_dict().items()):h.update(k.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def run(entry_path):
    e=entry_guard(entry_path)
    seal,targets,assocs,rgbs,prompts=saved_metadata(e)
    # Everything above is metadata/array validation. Only this explicitly invoked
    # function after a root-approved entry imports the model or reads source RGB.
    import torch
    import PIL
    from PIL import Image
    import sam2.build_sam as builder
    import sam2.sam2_image_predictor as predictor_module
    require(torch.__version__=='2.5.1+cu121' and np.__version__=='1.26.4','Sealed Torch/NumPy versions')
    require(torch.cuda.is_available(),'Sealed CUDA runtime required')
    require(Path(builder.__file__).resolve()==STATIC['sam_build'],'Official SAM import path')
    require(Path(predictor_module.__file__).resolve()==STATIC['sam_predictor'],'Official predictor import path')
    torch.manual_seed(20261023);np.random.seed(20261023);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    def no_compile(*args,**kwargs):raise RuntimeError('torch.compile forbidden')
    torch.compile=no_compile
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    OUT.mkdir();(OUT/'masks').mkdir()
    write(OUT/'RUN_STARTED.json',dict(entry=bind(entry_path),source_only=True,fit=False,current223=False,PnP=0))
    start=time.monotonic()
    model=builder.build_sam2(CONFIG,str(CHECKPOINT),device='cuda',mode='eval',
        apply_postprocessing=False,hydra_overrides_extra=['++model.compile_image_encoder=false'])
    model.requires_grad_(False)
    require(not model.training and all(not p.requires_grad for p in model.parameters()),'Frozen eval weights')
    require({str(p.dtype) for p in model.parameters()}=={'torch.float32'},'FP32 weights')
    predictor=predictor_module.SAM2ImagePredictor(model,mask_threshold=0.,max_hole_area=0.,max_sprinkle_area=0.)
    before=state_digest(model);rows=[None]*1035;query_rows=[];embedding_calls=0;mask_calls=0;zero_masks=0
    by_rgb={r['rgb_input_id']:[] for r in rgbs}
    for i,a in enumerate(assocs):by_rgb[a['rgb_input_id']].append(i)
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for rgb in rgbs:
                indices=by_rgb[rgb['rgb_input_id']]
                associated=[i for i in indices if prompts[i] is not None]
                if associated:
                    require(sha(rgb['image'])==rgb['image_sha256'],'Source RGB file changed')
                    image=np.asarray(Image.open(rgb['image']).convert('RGB'),np.uint8)
                    require(list(image.shape[:2])==rgb['raw_hw'],'Native image H/W changed')
                    require(hashlib.sha256(image.tobytes()).hexdigest()==rgb['decoded_RGB_SHA256'],'RGB decode differs from detector')
                    predictor.set_image(image);embedding_calls+=1
                for i in indices:
                    mask=None;quality=None;mb=None
                    if prompts[i] is not None:
                        masks,quality_array,_=predictor.predict(point_coords=None,point_labels=None,
                            box=prompts[i],multimask_output=False,return_logits=False)
                        mask,quality=single_native_mask(masks,quality_array,assocs[i]['raw_hw'])
                        path=OUT/'masks'/('%04d.npz'%i)
                        np.savez_compressed(path,visible_mask=mask);mb=bind(path);mask_calls+=1;zero_masks+=int(not mask.any())
                    rows[i]=mask_row(assocs[i],mb,quality)
                    for k,q in enumerate(assocs[i]['initial_points']):
                        query_rows.append(dict(input_index=i,input_id=assocs[i]['input_id'],native_corner=k,
                            query_x=q[0],query_y=q[1],mask_status=rows[i]['mask_status'],
                            **query_mask_sample(mask,q,assocs[i]['raw_hw'])))
                print(json.dumps(dict(processed_RGB=embedding_calls,mask_calls=mask_calls,elapsed=time.monotonic()-start)),flush=True)
        after=state_digest(model);require(before==after,'SAM state mutated')
        require(all(r is not None for r in rows) and len(query_rows)==8280,'Full source/order retention')
        query_rows.sort(key=lambda r:(r['input_index'],r['native_corner']))
        with open(OUT/'OUTPUT_MASKS.jsonl','x') as f:
            for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
        with open(OUT/'QUERY_MASK_SAMPLES.csv','x') as f:
            writer=csv.DictWriter(f,fieldnames=list(query_rows[0]));writer.writeheader();writer.writerows(query_rows)
        summary=dict(completed=True,PASS=True,targets=1035,unique_RGB=435,query_slots=8280,
            associated=mask_calls,unassociated=1035-mask_calls,embedding_calls=embedding_calls,
            zero_visible_masks=zero_masks,
            query_predictions={s:sum(r['mask_prediction']==s for r in query_rows) for s in ('VISIBLE_MASK','NOT_VISIBLE_MASK','U')},
            state_before=before,state_after=after,seconds=time.monotonic()-start,
            weights='FP32',autocast='CUDA BF16',mask_threshold=0.,multimask_output=False,return_logits=False,
            selected_mask_index=0,quality_selection=False,points=False,postprocessing=False,compile=False,
            mask_semantics='Visible segmentation proxy; mask0 is NOT_VISIBLE_MASK, no match is U, neither establishes semantic E/S.',
            Torch=torch.__version__,NumPy=np.__version__,Pillow=PIL.__version__,current223=False,fit=False,PnP=0)
        write(OUT/'SUMMARY.json',summary)
        outputs=[bind(OUT/'OUTPUT_MASKS.jsonl'),bind(OUT/'QUERY_MASK_SAMPLES.csv'),bind(OUT/'SUMMARY.json')]+[r['mask'] for r in rows if r['mask'] is not None]
        write(OUT/'SAM_OUTPUT_SEAL.json',dict(completed=True,PASS=True,entrylock=bind(entry_path),
            decisions=e['bindings']['decisions'],detector_output_seal=e['bindings']['detector_output_seal'],
            source_inputs=seal['source_inputs'],associations=seal['associations'],per_RGB=seal['per_RGB'],
            runtime=e['bindings']['runtime'],checkpoint=e['bindings']['checkpoint'],code=e['bindings']['code'],
            helper=e['bindings']['helper'],generated=e['bindings']['generated'],
            output_masks=bind(OUT/'OUTPUT_MASKS.jsonl'),summary=bind(OUT/'SUMMARY.json'),outputs=outputs,
            source_only=True,current223=False,fit=False,PnP=0))
    except BaseException as ex:
        write(OUT/'FAILED.json',dict(type=type(ex).__name__,error=str(ex),no_retry=True,current223=False));raise

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['validate','run']);parser.add_argument('--entry',required=True)
    args=parser.parse_args()
    if args.mode=='validate':
        e=entry_guard(args.entry);_,targets,_,rgbs,_=saved_metadata(e)
        print(json.dumps(dict(PASS=True,targets=len(targets),unique_RGB=len(rgbs),model_imports=0,RGB_reads=0)))
    else:run(args.entry)

if __name__=='__main__':main()

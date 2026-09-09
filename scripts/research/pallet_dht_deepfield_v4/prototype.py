"""Three SHA-selected synthetic images; exact field-only/reference comparison.

No pallet labels, real-image forward, new training or full field cache.
The official reference import has a fail-closed pytlsd placeholder; its call
count must remain zero because detect_lines=False. No official file is edited.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np
import torch

from .field_model import FieldOnlyDeepLSD, EXPECTED_COMMIT, fp32_inference, sha


def read(path):
    return json.loads(Path(path).read_text())


def official_reference(repo, checkpoint):
    """Import untouched official inference code; reject optional LSD calls."""
    calls = {'pytlsd_lsd':0}
    def unavailable(*args, **kwargs):
        calls['pytlsd_lsd'] += 1
        raise RuntimeError('pytlsd is intentionally unavailable for field-only inference')
    previous = sys.modules.get('pytlsd')
    stub = types.ModuleType('pytlsd'); stub.lsd = unavailable
    sys.modules['pytlsd'] = stub
    sys.path.insert(0, str(repo))
    try:
        module = importlib.import_module('deeplsd.models.deeplsd_inference')
        model = module.DeepLSD({'detect_lines':False, 'multiscale':False})
        model.load_state_dict(torch.load(checkpoint,map_location='cpu')['model'],strict=True)
    finally:
        sys.path.remove(str(repo))
        if previous is None:
            sys.modules.pop('pytlsd',None)
        else:
            sys.modules['pytlsd'] = previous
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model,calls


def prototype(run_dir, gray_run, device='cuda:0'):
    import hashlib
    run, gray_run = Path(run_dir).resolve(), Path(gray_run).resolve()
    if not (run/'PURPOSE.md').exists():
        raise ValueError('PURPOSE.md required')
    repo, checkpoint = run/'external/DeepLSD',run/'deeplsd_md.tar'
    commit = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if commit != EXPECTED_COMMIT:
        raise ValueError('Official reviewed repository commit differs')
    folder = run/'prototype'
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('Preserve existing prototype before explicit rerun')
    folder.mkdir(parents=True,exist_ok=True)
    source = {str(p):sha(p) for p in [Path(__file__).resolve(),Path(__file__).with_name('field_model.py').resolve()]}
    before = {str(p):sha(p) for p in repo.rglob('*.py')}
    completion = read(gray_run/'CACHE_COMPLETION.json')
    for name,key in [('CACHE_RECORDS.json','cache_records_sha256'),('CACHE_ARRAYS.json','cache_arrays_sha256')]:
        if sha(gray_run/name) != completion[key]:raise ValueError('Source gray metadata changed')
    metadata = read(gray_run/'CACHE_RECORDS.json')
    desc = read(gray_run/'CACHE_ARRAYS.json')['arrays']['inputs']
    gray_path = Path(desc['image_gray']['path'])
    if sha(gray_path) != completion['array_sha256'][str(gray_path)]:raise ValueError('Source gray cache changed')
    gray = np.load(gray_path,mmap_mode='r')
    eligible = metadata['populations']['train']
    indices = sorted(eligible,key=lambda i:hashlib.sha256(('pallet_dht_deepfield_v4:prototype:'+metadata['records'][i]['id']).encode()).hexdigest())[:3]
    if any(metadata['records'][i]['population']=='real_dev' for i in indices):raise ValueError('Synthetic-only prototype required')
    inputs = {str(gray_run/'CACHE_COMPLETION.json'):sha(gray_run/'CACHE_COMPLETION.json'),
        str(gray_run/'CACHE_RECORDS.json'):sha(gray_run/'CACHE_RECORDS.json'),str(gray_path):sha(gray_path)}
    torch.set_num_threads(2)
    model = FieldOnlyDeepLSD(repo,checkpoint).to(device)
    reference,calls = official_reference(repo,checkpoint)
    reference = reference.to(device)
    rows=[]
    for index in indices:
        record=metadata['records'][index];h,w=record['input_shape_hw']
        image=torch.from_numpy(np.array(gray[index:index+1],copy=True)).to(device).float()/255
        actual=image[:,:,:h,:w]
        started=time.perf_counter()
        with fp32_inference():
            ours=model(actual);official=reference({'image':actual})
        if str(device).startswith('cuda'):torch.cuda.synchronize()
        comparison={}
        for key in ['df_norm','df','line_level']:
            if not torch.isfinite(ours[key]).all():raise ValueError('Nonfinite field')
            comparison[key]={'bit_exact':torch.equal(ours[key],official[key]),
                'max_abs_delta':float((ours[key]-official[key]).abs().max())}
        if not all(v['bit_exact'] for v in comparison.values()):raise ValueError('Field subset differs from official reference')
        df=np.full((640,640),5.,np.float32);angle=np.zeros((640,640),np.float32);extent=np.zeros((640,640),bool)
        df[:h,:w]=ours['df'][0].cpu().numpy();angle[:h,:w]=ours['line_level'][0].cpu().numpy();extent[:h,:w]=True
        path=folder/f'{index:06d}.npz'
        np.savez_compressed(path,df=df,line_level=angle,input_extent_mask=extent,input_shape_hw=np.array([h,w]),
            raw_to_input_affine=np.array(record['raw_to_input_affine']),index=np.array(index),id=np.array(record['id']))
        rows.append({'id':record['id'],'index':index,'population':'train','source':record['source'],
            'image':record['image'],'image_sha256':record['image_sha256'],'input_shape_hw':[h,w],
            'df_range':[float(df[:h,:w].min()),float(df[:h,:w].max())],
            'line_level_range':[float(angle[:h,:w].min()),float(angle[:h,:w].max())],
            'df_quantiles':np.percentile(df[:h,:w],[1,10,50,90,99]).tolist(),
            'official_parity':comparison,'fields':str(path),'fields_sha256':sha(path),
            'elapsed_with_reference_seconds':time.perf_counter()-started})
        print(record['id'],comparison,flush=True)
    if calls['pytlsd_lsd'] != 0:raise ValueError('Optional detector was called')
    for path,digest in {**inputs,**before,**source}.items():
        if sha(path)!=digest:raise ValueError('Input or official source changed')
    receipt={'schema':'pallet_deeplsd_field_prototype_v1','complete':True,'PASS':True,
        'repo_commit':commit,'pretrained_model':model.provenance,'source_sha256':source,'input_sha256':inputs,
        'records':rows,'selection_rule':'First3 IDs by SHA256(pallet_dht_deepfield_v4:prototype:+id) within fixedtrain1792',
        'official_reference':'Unmodified deeplsd_inference.DeepLSD(detect_lines=False,multiscale=False), strict144-entry load; missing pytlsd import replaced by never-called fail-closed placeholder',
        'optional_detector_calls':calls,'official_python_sources_unchanged':True,'original_official_source_sha256':before,
        'n_actual_synthetic_images':3,'n_actual_real_images':0,'model_forward_calls':6,
        'GT_used':False,'GT_annotation_files_opened':0,'training_performed':False,'full_cache_packed':False,
        'device':device,'precision':'FP32; TF32 disabled; B1 actual rectangle',
        'latency_claim':False,'scope':'Generic pretrained field feasibility and exact implementation parity, not pallet accuracy or semantic edge validation'}
    (folder/'PROTOTYPE.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--gray-run',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    result=prototype(args.run_dir,args.gray_run,args.device)
    print(json.dumps({'complete':result['complete'],'PASS':result['PASS'],'n_actual_synthetic_images':3}))

"""Gate B posterior capture: one unchanged N3 head forward per seed/frame."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import importlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import torch

from .preflight import DOC, WORKTREE, digest, finite, read, sha, write

sys.dont_write_bytecode=True


def field(line,key):
    # Project only prediction identity/coordinates. No GT/pose/error fields parsed.
    match=re.search(r'"'+re.escape(key)+r'"\s*:',line)
    assert match is not None,key
    return json.JSONDecoder().raw_decode(line[match.end():].lstrip())[0]


def baseline_points(path):
    result={}
    with gzip.open(path,'rt') as stream:
        for line in stream:
            if field(line,'method')!='N3_DIM_SYM':continue
            result[(int(field(line,'seed')),field(line,'id'))]=np.asarray(field(line,'qFinal'),dtype=float)
    assert len(result)==957
    return result


def gpu_quiet():
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True).splitlines()
    import os
    foreign=[r for r in processes if r and r.split(',')[0].strip()!=str(os.getpid()) and '/usr/share/rustdesk/rustdesk' not in r]
    assert not foreign,'GPU foreign compute present; no process will be stopped'
    return dict(foreign_compute_count=len(foreign),desktop_rustdesk_preserved=any('/usr/share/rustdesk/rustdesk' in r for r in processes))


def source_modules(source):
    # Exact original definitions; importing these files performs no model/GT load.
    sys.path[:0]=[str(source/'scripts/research/pallet_dim_conditioned_p_v1'),str(source/'scripts/research/pallet_final_ml_contribution_test_v1'),str(source)]
    inference=importlib.import_module('inference')
    features=inference.E.old('features')
    assert Path(inference.__file__).resolve()==source/'scripts/research/pallet_dim_conditioned_p_v1/inference.py'
    assert Path(features.__file__).resolve()==source/'scripts/research/pallet_line_pose_v1/features.py'
    return inference,features


def run(source,private):
    started=time.monotonic()
    lock=read(DOC/'INPUT_LOCK.json');assert lock['status']=='PASS'
    assert sha(DOC/'FUSION_METHOD_LOCK.json')==lock['fusion_method_lock_sha256']
    output=DOC/'POSTERIOR_CAPTURE.jsonl.gz'
    assert not output.exists(),'No outcome-dependent capture reruns'
    assert torch.cuda.is_available(),'Existing CUDA interpreter/device required'
    before=gpu_quiet()
    torch.set_num_threads(4)
    inference,features=source_modules(source)
    norm=read(source/'_docs/experiments/pallet_dim_conditioned_p_v1/DIM_NORMALIZATION_LOCK.json')
    pathmap={r['id']:r for r in read(private/'INPUT_PATHS.json')['frames']}
    expected=baseline_points(WORKTREE/lock['baseline_coordinate_rows']['path'])
    rule=dict(lam=1.,max_move_image_diagonal_fraction=.01)
    seed_summary={}
    forwards=0
    rows_written=0
    buffer=None
    with gzip.open(output,'wt',encoding='utf-8',compresslevel=6) as stream:
        for m in lock['models']:
            seed=m['seed'];T=m['temperature']
            assert sha(source/m['checkpoint']['path'])==m['checkpoint']['sha256']
            head,checkpoint=inference.load_head('N3_DIM_SYM',seed)
            assert not head.training and not any(p.requires_grad for p in head.parameters())
            actual_buffer=head.displacements.detach().cpu().numpy()
            assert actual_buffer.shape==(222,2) and np.array_equal(actual_buffer[-1],np.zeros(2))
            if buffer is None:buffer=actual_buffer.copy()
            else:assert np.array_equal(buffer,actual_buffer)
            cached=read(source/m['prediction']['path'])
            cached={r['id']:r for r in cached['records']}
            maxdiff=0.;maxcache=0.;gain_range=[];null_range=[];nonfinite=0;supports=Counter();seedforward=0
            for index,b in enumerate(lock['input_manifest']):
                fid=b['id'];cachepath=source/pathmap[fid]['cache']
                assert sha(cachepath)==b['cache_sha256']
                packet=torch.load(cachepath,map_location='cpu',weights_only=False)
                captured=packet['captured'];selected=captured['selected_index']
                inp=features.branch_inputs(captured)
                assert inp is not None and selected==b['selected_index']
                assert inp['gain']>0 and np.isfinite(inp['gain'])
                cap={**captured,'p3':captured['p3'].cuda(),'p4':captured['p4'].cuda()}
                harvested=[]
                forward_original=inference.forward
                def harvest(*args,**kwargs):
                    out=forward_original(*args,**kwargs);harvested.append(out);return out
                inference.forward=harvest
                try:
                    result,diagnostic=inference.predict_captured(head,'N3_DIM_SYM',cap,packet['dimensions'],packet['order'],T,rule,packet['raw_hw'],norm)
                finally:inference.forward=forward_original
                assert len(harvested)==1
                forwards+=1;seedforward+=1
                out=harvested[0]
                logits=out['logits'][0].detach().cpu().numpy()
                candidate=out['candidate_displacements'][0].detach().cpu().numpy()
                point_support=out['point_support'][0].detach().cpu().numpy()
                diagonal=float(out['box_diagonal'][0])
                assert logits.shape==(8,222) and candidate.shape==(222,2)
                assert point_support.shape==(8,) and np.array_equal(candidate[-1],np.zeros(2))
                assert np.allclose(candidate,actual_buffer*diagonal,rtol=0,atol=1e-6)
                assert np.isfinite(logits).all() and np.isfinite(candidate).all()
                qn=np.asarray(result['candidates'][selected]['keypoints_xy'],dtype=float)
                saved=np.asarray(cached[fid]['candidates'][selected]['keypoints_xy'],dtype=float)
                diff=float(np.max(np.abs(qn-expected[(seed,fid)])))
                diffcache=float(np.max(np.abs(qn-saved)))
                maxdiff=max(maxdiff,diff);maxcache=max(maxcache,diffcache)
                assert diff<=1e-3 and diffcache<=1e-3,(seed,fid,diff,diffcache)
                q0=np.asarray(b['q0']);support=np.asarray(b['prediction_support'],dtype=bool)
                assert np.array_equal(qn[8],q0[8],equal_nan=True)
                assert np.array_equal(qn[~support],q0[~support],equal_nan=True)
                inference.preservation(captured['candidates'],result['candidates'],selected)
                z=logits.astype(np.float64)/T;z-=z.max(-1,keepdims=True)
                probability=np.exp(z);probability/=probability.sum(-1,keepdims=True)
                assert np.max(np.abs(probability.sum(-1)-1))<1e-12
                native=candidate.astype(np.float64)/inp['gain']
                mean=probability@native
                residual=native[None]-mean[:,None]
                covariance=np.einsum('kj,kja,kjb->kab',probability,residual,residual)
                entropy=-np.sum(probability*np.log(np.maximum(probability,np.finfo(float).tiny)),axis=-1)
                row=dict(seed=seed,id=fid,session=b['session'],grade=b['grade'],temperature=T,
                    q0=q0,qN=saved,forward_qN=qn,prediction_support=support,point_support=point_support,
                    logits=logits,candidate_displacements_network=candidate,gain=inp['gain'],network_box_diagonal=diagonal,
                    candidate_native_mean=mean,candidate_native_covariance=covariance,
                    posterior_probability_sum=probability.sum(-1),posterior_null_probability=probability[:,-1],posterior_entropy=entropy,
                    qN_parity_max_abs_px=diff,qN_source_cache_parity_max_abs_px=diffcache,
                    checkpoint_sha256=m['checkpoint']['sha256'],detector_feature_cache_sha256=b['cache_sha256'],raw_hw=b['raw_hw'],selected_index=selected,
                    GT_inference_inputs=False,covariance_note='Uncapped candidate-distribution spread; prior mean is saved capped native qN.')
                stream.write(json.dumps(finite(row),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
                rows_written+=1;gain_range.append(inp['gain']);null_range.extend(probability[:,-1].tolist());supports.update(point_support.tolist())
                if index%80==0 or index==318:
                    stream.flush();print('CAPTURE',seed,index+1,319,'forward',forwards,'seconds',round(time.monotonic()-started,2),flush=True)
            seed_summary[str(seed)]=dict(frames=319,forward_calls=seedforward,max_abs_qN_public_baseline_px=maxdiff,max_abs_qN_source_prediction_px=maxcache,
                gain_range=[min(gain_range),max(gain_range)],null_probability_range=[min(null_range),max(null_range)],point_supported_corners=supports[True],point_unsupported_corners=supports[False])
            del head
            torch.cuda.empty_cache()
    torch.cuda.synchronize()
    after=gpu_quiet()
    parity=dict(schema='feature_gradient_joint_posterior_parity_v1',status='PASS',rows=rows_written,frames=319,seeds=[1,2,3],
        input_lock_sha256=sha(DOC/'INPUT_LOCK.json'),fusion_method_lock_sha256=sha(DOC/'FUSION_METHOD_LOCK.json'),
        capture=dict(path=output.name,sha256=sha(output),bytes=output.stat().st_size),per_seed=seed_summary,
        candidate_buffer=buffer,candidate_buffer_sha256=digest(buffer),candidate_order='original head.displacements;221 nonnull +lastzero/null',
        checks=dict(shape_8_222=True,all_222_including_null=True,gain_and_bbox_diagonal_applied=True,probability_sum_1=True,
            all_957_native_public_baseline_parity=True,absolute_qN_tolerance_px=1e-3,center_sentinel_unsupported_metadata_preserved=True,
            posterior_GT_input=False,baseline_projection_fields=['seed','id','method','qFinal'],pose_or_DEV_error_fields_parsed=False),
        environment=dict(torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__,device=torch.cuda.get_device_name(),
            torch_threads=torch.get_num_threads(),cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic),
        interference_checks=dict(before=before,after=after),
        execution=dict(N3_forward_calls=forwards,detector_backbone_calls=0,F_calls=0,optimizer_updates=0,new_synthetic_RGB=0,
            elapsed_seconds=time.monotonic()-started,latency_is_end_to_end_benchmark=False),
        code=[dict(path=str(p.relative_to(WORKTREE)),sha256=sha(p)) for p in [Path(__file__),Path(__file__).with_name('preflight.py')]])
    assert rows_written==forwards==957
    write(DOC/'POSTERIOR_PARITY.json',parity)
    print('POSTERIOR_GATE_B_PASS',forwards,'rows',rows_written,'gzip_bytes',output.stat().st_size,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--private-dir',type=Path,required=True)
    a=p.parse_args();run(a.source_root.resolve(),a.private_dir)


if __name__=='__main__':main()

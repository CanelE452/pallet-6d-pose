"""Read-only join of original selected detector boxes; no GT/model/solver I/O."""
from __future__ import annotations
import argparse,gzip,hashlib,json,time,resource
from collections import Counter
from pathlib import Path
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb')as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with gzip.open(p,'rt')as f:
        for line in f:
            if line.strip():yield json.loads(line)
def write(p,value):
    with Path(p).open('x')as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def run(repo,archive,output):
    repo=Path(repo).resolve();archive=Path(archive).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();cpu=time.process_time();artifacts={};proof=[];result=[];counts=Counter()
    upstream=archive/'d1_upstream_v1';seal=read(upstream/'UPSTREAM_SEAL.json')
    sourcebase=repo/'data/pallet/results/pallet_dim_conditioned_p_v1/predictions'
    codebase=Path(__file__).resolve().parents[1]
    evidence=[codebase/'pallet_dim_conditioned_p_v1/paper_evaluate.py',codebase/'pallet_dim_conditioned_p_v1/dev_evaluate.py',codebase/'pallet_dim_conditioned_p_v1/inference.py',codebase/'pallet_final_ml_contribution_test_v1/point_inference.py']
    for p in evidence:artifacts[str(p)]=sha(p)
    for domain,group,upname,expected_n in [('real','REAL_DEV','REAL238',238),('synthetic','SYNTH_HELDOUT','SYNTHETIC1003',1003)]:
        predpath=sourcebase/group/'N3_DIM_SYM_seed1.json';basepath=sourcebase/group/'N0_BASE_REPLAY_seed1.json';uppath=upstream/(upname+'_UPSTREAM.jsonl.gz')
        for p in(predpath,basepath,uppath):artifacts[str(p)]=sha(p)
        if artifacts[str(uppath)]!=seal['artifacts'][uppath.name]:raise ValueError('Original D1 upstream hash mismatch')
        source=read(predpath);baseline=read(basepath)
        if source.get('GT_input')is not False or baseline.get('GT_input')is not False:raise ValueError('Explicit GT-free source prediction required')
        records=source['records'];baselines=baseline['records'];by={r['id']:r for r in records};bb={r['id']:r for r in baselines}
        if len(by)!=len(records)or len(bb)!=len(baselines):raise ValueError('Repeated original prediction ID')
        up=list(rows(uppath));featureids=[r['id']for r in rows(archive/f'd1_{domain}_candidates_v1/HEAD_FEATURES.jsonl.gz')]
        if len(up)!=expected_n or [r['id']for r in up]!=featureids:raise ValueError('D1 upstream/action ID order mismatch')
        stats=Counter()
        for row in up:
            fid=row['id'];entry=by.get(fid);before=bb.get(fid)
            if entry is None or before is None:raise ValueError('Missing detector prediction ID: '+fid)
            idx=entry['selected_index'];baseidx=before['selected_index'];stats['joined']+=1
            if idx is None:
                result.append(dict(id=fid,partition=row['partition'],detector_box_xyxy_raw=None,box_source='ORIGINAL_N3_SELECTED_DETECTOR',transform=dict(source_to_raw_offset_xy=[0.,0.],clipping=False,raw_hw=row['raw_hw']),source_sha256=artifacts[str(predpath)],availability='MISSING_ORIGINAL_SELECTED_BOX'))
                stats['missing_original_detection']+=1;continue
            if idx!=baseidx or idx!=int(np.argmax([c['score']for c in entry['candidates']])):raise ValueError('Original detector selection changed or is not maximum confidence')
            candidate=entry['candidates'][idx];original=before['candidates'][baseidx]
            box=np.asarray(candidate['box_xyxy'],float);basebox=np.asarray(original['box_xyxy'],float)
            if box.shape!=(4,)or not np.isfinite(box).all()or not np.all(box[2:]>box[:2]):raise ValueError('Invalid original detector box')
            if not np.array_equal(box,basebox):raise ValueError('N3 changed original detector box')
            q=np.asarray(candidate['keypoints_xy'],float);initial=np.asarray(row['initial_points'],float)
            if not np.array_equal(q[8],initial[8]):raise ValueError('Original N3 center is not in D1 raw pixel coordinates')
            if np.array_equal(q[8]-100.,initial[8]):raise ValueError('Ambiguous duplicate unpadding')
            h,w=row['raw_hw'];outside=bool(box[0]<0 or box[1]<0 or box[2]>w-1 or box[3]>h-1)
            stats['available_finite_positive_box']+=1;stats['N3_box_equals_original_detector']+=1;stats['raw_center_exact']+=1
            stats['subtract100_center_would_be_wrong']+=1;stats['off_canvas_box_retained_without_clipping']+=int(outside)
            result.append(dict(id=fid,partition=row['partition'],detector_box_xyxy_raw=box.tolist(),box_source='ORIGINAL_N3_SELECTED_DETECTOR',
                transform=dict(source_to_raw_offset_xy=[0.,0.],source_coordinates='ALREADY_ORIGINAL_RAW_PIXELS',
                    synthetic_reflect_pad100_removed_once_by_original_baseline_export=domain=='synthetic',
                    selected_N3_center_equals_D1_raw_center=True,clipping=False,raw_hw=row['raw_hw']),
                source_sha256=artifacts[str(predpath)],availability='AVAILABLE'))
        counts.update({domain+'/'+k:v for k,v in stats.items()})
        proof.append(dict(domain=domain,source_path=str(predpath),source_sha256=artifacts[str(predpath)],source_field='records[id].candidates[selected_index].box_xyxy',baseline_path=str(basepath),baseline_sha256=artifacts[str(basepath)],
            source_records=len(records),exact_D1_ID_join=len(up),counts=dict(stats),upstream_path=str(uppath),upstream_sha256=artifacts[str(uppath)],
            raw_coordinate_basis='Original synthetic paper_evaluate.baseline subtracts100 from BOTH keypoints and boxes. N3 replace_selected modifies only corner keypoints and preserves box+center. Original REAL_DEV cache uses raw BGR prediction directly. All selected centers equal D1 raw initial_points[8] exactly; subtracting100 again would be incorrect.',
            original_model_source_evidence=[dict(path=str(p),sha256=artifacts[str(p)])for p in evidence]))
    if len(result)!=1241 or len({r['id']for r in result})!=1241:raise ValueError('Expected exact1241 unique IDs')
    outputpath=output/'V_BOX_INPUTS.jsonl.gz'
    with gzip.open(outputpath,'xt')as f:
        for row in result:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    for p,digest in artifacts.items():
        if sha(p)!=digest:raise ValueError('Read-only source changed during join')
    receipt=dict(schema='RISK_V_ORIGINAL_DETECTOR_BOX_JOIN_SEAL_V1',status='PASS',rows=1241,real_rows=238,synthetic_rows=1003,
        output_sha256=sha(outputpath),source_bindings=artifacts,domain_proofs=proof,counts=dict(counts),
        reference_or_GT_bbox_reads=0,new_detector_N3_SubPix_SAM_forwards=0,new_PnP_optimizer_solves=0,original_archives_modified=False,
        no_clipping=True,source_sha256=sha(__file__),cost=dict(wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu,threads=1,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    write(output/'V_BOX_SEAL.json',receipt);return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run(a.repo,a.archive,a.output);print(json.dumps({k:r[k]for k in('status','rows','counts','cost')}))

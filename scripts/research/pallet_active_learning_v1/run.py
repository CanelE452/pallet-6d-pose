"""Label-blind acquisition pilot. No student training or performance claim."""
import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pallet_paper_contribution_screen_v1'))
from common.contracts import ROOT,R0,R0_SHA,sha,write
from track_c.selection_diagnostic import Capture
from common.gpu_snapshot import snapshot

RAW=ROOT/'data/pallet/results/pallet_active_learning_v1'
DOC=ROOT/'_docs/experiments/pallet_active_learning_v1'
POOL=ROOT/'data/evaluation/pallet_eval_v1/adaptation/MAIN_UNLABELED_BALANCED.csv'
PREVIOUS=ROOT/'data/pallet/results/pallet_paper_contribution_screen_v1/C_geometry_preserving_da/dataset/BINDINGS.json'
SENSOR=ROOT/'data/pallet/results/paper_depth_selftrain_v1/sensor_validation_v1/SENSOR_VALIDATION_POPULATION.csv'
METHODS=('random','diversity','instability_only','geometry_weighted_diversity')

def read(p):return json.loads(p.read_text())

def eligible(rows,training,evaluation):
    seen=set();kept=[];excluded=[]
    for r in rows:
        h=r['image_sha256'];reasons=[]
        if h in training:reasons.append('previous_C_training')
        if h in evaluation:reasons.append('existing_evaluation_or_manual_reference')
        if h in seen:reasons.append('duplicate_image_hash')
        seen.add(h)
        if reasons:excluded.append(dict(**r,reasons=reasons))
        else:kept.append(r)
    return kept,excluded

def initialize():
    assert sha(R0)==R0_SHA
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='main'
    rows=list(csv.DictReader(POOL.open()));assert len(rows)==1000
    bindings={str(p.relative_to(ROOT)):sha(p) for p in (POOL,PREVIOUS,SENSOR,R0)}
    for r in rows:assert sha(ROOT/r['image_path'])==r['image_sha256']
    eval_hash=set();eval_names=set();sources=[]
    for p in sorted((ROOT/'challenge/real_gt_v2/manifests').glob('*.json')):
        data=read(p);items=data.get('items',[])
        for item in items:
            image=ROOT/(item['image_path'] if 'image_path' in item else item['image']);eval_hash.add(sha(image));eval_names.add(image.name)
        sources.append(dict(path=str(p.relative_to(ROOT)),count=len(items)))
        bindings[str(p.relative_to(ROOT))]=sha(p)
    sensor=list(csv.DictReader(SENSOR.open()))
    for r in sensor:
        # This manifest binds RGB paths; never open the annotation file.
        image=ROOT/r['image_path'] if 'image_path' in r else ROOT/r['rgb_path']
        if not image.exists():image=ROOT/'data/evaluation/pallet_eval_v1'/r.get('image_path',r.get('rgb_path',''))
        assert image.is_file(),image
        eval_hash.add(sha(image));eval_names.add(image.name)
    previous=read(PREVIOUS)
    training={r['image_sha256'] for domain in previous.values() for r in domain}
    keep,excluded=eligible(rows,training,eval_hash)
    assert keep,'No eligible development acquisition pool'
    for r in keep:
        r['timestamp_ns']=int(Path(r['image_path']).stem)
    related={r['capture_session'] for r in rows if Path(r['image_path']).name in eval_names}
    write(DOC/'POOL_AUDIT.json',dict(status='PASS',input_frames=len(rows),eligible_frames=len(keep),
        excluded_frames=len(excluded),excluded_reasons=dict(Counter(x for r in excluded for x in r['reasons'])),
        eligible_sessions=dict(Counter(r['capture_session'] for r in keep)),
        eligible_conditions=dict(Counter(r['paper_condition'] for r in keep)),
        exact_training_hash_overlap=0,exact_evaluation_hash_overlap=0,
        known_pool_sessions_with_evaluation_same_basename=sorted(related),
        session_independence_established=False,previous_pool_development_use=True,
        scope='Development acquisition pool only; not a new untouched confirmation population',
        GT_coordinate_files_opened=0,checked_manifests=sources,source_bindings=bindings,
        previous_training_audit_scope='Exact known C synthetic/real exposure; not exhaustive proof of every historical training run'))
    write(RAW/'POOL.json',dict(eligible=keep,excluded=excluded))
    write(DOC/'PROTOCOL_LOCK.json',dict(status='ACQUISITION_PILOT_LOCKED',
        start_main=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        checkpoint_sha256=R0_SHA,methods=list(METHODS),preview_budget_per_method=30,
        budget_status='Nonbinding preview; actual annotation/training budget requires user confirmation',
        random_seed=20260913,min_selected_temporal_gap_seconds=2,
        features='Frozen R0 P3/P4/P5 global mean; per-level L2 normalization then concatenation and L2 normalization',
        instability='Mean nine-corner selected-output displacement / preprocessed image diagonal under fixed0.8 RGB intensity; original versus perturbed top grid candidate',
        candidate_index='Unchanged stock one2one grid/top-k; record top index changes and detection-floor availability',
        perturbation='Single fixed photometric perturbation; no geometric or symmetry reassignment, no sweep',
        diversity='Greedy k-center squared Euclidean distance, shared centroid-nearest initial sample',
        geometry_weight='Multiply nearest-selected squared distance by1+empirical midrank of instability (range1..2)',
        temporal_constraint='All methods skip already selected frames within2seconds in the same capture; no relaxation',
        missing_detection='Retain and flag; no positive-only pseudo-label filter; human may annotate no-object',
        compared_baseline_note='Diversity k-center and uncertainty-only controls; not a claimed reproduction of CLUE',
        GT_in_selection=False,student_updates=0,annotation_source='Human required; predictions are not ground truth',
        evidence='ACQUISITION_PILOT_ONLY; label efficiency and pose improvement NOT_RUN',
        next_stage='Freeze budget/splits/training protocol before revealing new labels; same R0, loss and exposures; independent sessions needed for confirmation'))
    print('Eligible development pool',len(keep),flush=True)

def norm(x):return x/max(float(np.linalg.norm(x)),1e-12)

def extract():
    audit=read(DOC/'POOL_AUDIT.json')
    for p,h in audit['source_bindings'].items():assert sha(ROOT/p)==h
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
    if gpu:
        write(DOC/'RESOURCE_UNAVAILABLE.json',dict(status='NOT_RUN',processes=gpu));raise SystemExit('No waiting or process mutation')
    assert not (RAW/'FEATURES.npz').exists(),'Existing features must be verified, not overwritten'
    rows=read(RAW/'POOL.json')['eligible'];torch.set_num_threads(4)
    capture=Capture(R0,ROOT/rows[0]['image_path']);features=[];details=[]
    cv2=capture.p.cv2
    with torch.inference_mode():
        for i,r in enumerate(rows):
            image=ROOT/r['image_path'];pred,ids=capture.predict(image)
            base=capture.dense.clone();anchors=capture.anchors.clone();strides=capture.strides.clone()
            f=norm(np.concatenate([norm(v.mean((2,3))[0].cpu().numpy()) for v in capture.features]))
            original_index=int(base[0,4].argmax());original_points=base[0,5:].reshape(9,3,-1)[:,:2,original_index]
            raw=cv2.imread(str(image));dark=(raw.astype(np.float32)*.8).astype(np.uint8)
            padded=cv2.copyMakeBorder(dark,100,100,100,100,cv2.BORDER_REFLECT_101)
            capture.p.model.predict(padded,conf=.001,imgsz=640,device='0',verbose=False)
            assert torch.equal(anchors,capture.anchors) and torch.equal(strides,capture.strides)
            changed=capture.dense;index=int(changed[0,4].argmax())
            points=changed[0,5:].reshape(9,3,-1)[:,:2,index]
            same=changed[0,5:].reshape(9,3,-1)[:,:2,original_index]
            diagonal=float(np.hypot(*capture.input.shape[-2:]))
            drift=float(torch.linalg.vector_norm(points-original_points,dim=1).mean())/diagonal
            same_drift=float(torch.linalg.vector_norm(same-original_points,dim=1).mean())/diagonal
            assert np.isfinite(f).all() and np.isfinite(drift)
            features.append(f);details.append(dict(image_sha256=r['image_sha256'],
                selected_pose_instability=drift,same_candidate_instability=same_drift,
                top_index=original_index,perturbed_top_index=index,top_index_changed=original_index!=index,
                original_top_score=float(base[0,4,original_index]),perturbed_top_score=float(changed[0,4,index]),
                original_above_floor=bool(pred),perturbed_above_floor=float(changed[0,4,index])>.001))
            if (i+1)%100==0:print(f'Label-blind feature extraction {i+1}/{len(rows)}',flush=True)
    with (RAW/'FEATURES.npz').open('xb') as handle:np.savez_compressed(handle,features=np.stack(features))
    write(RAW/'ACQUISITION_SIGNALS.json',details)
    write(DOC/'EXTRACTION_AUDIT.json',dict(status='PASS',frames=len(rows),feature_dimension=len(features[0]),
        neural_image_forwards=2*len(rows),setup_warmup_excluded=True,optimizer_updates=0,
        GT_coordinates_read=0,feature_sha256=sha(RAW/'FEATURES.npz'),signals_sha256=sha(RAW/'ACQUISITION_SIGNALS.json'),
        selection_signal='Photometric instability, not calibrated error probability',
        source_checkpoint_unchanged=sha(R0)==R0_SHA))

def midranks(values):
    values=np.asarray(values);return np.array([(np.sum(values<v)+.5*np.sum(values==v))/len(values) for v in values])

def select(features,uncertainty,rows,budget,method,seed=20260913):
    assert method in METHODS and budget>0
    x=np.asarray(features,float);u=np.asarray(uncertainty,float)
    assert len(x)==len(u)==len(rows) and np.isfinite(x).all() and np.isfinite(u).all()
    weight=1+midranks(u);available=np.ones(len(x),bool);chosen=[];nearest=np.full(len(x),np.inf)
    random_order=np.random.default_rng(seed).permutation(len(x));random_rank=np.empty(len(x));random_rank[random_order]=np.arange(len(x),0,-1)
    initial=int(np.argmin(((x-x.mean(0))**2).sum(1)))
    while len(chosen)<budget and available.any():
        if method=='random':score=random_rank.copy()
        elif method=='instability_only':score=u.copy()
        elif not chosen:score=-((x-x[initial])**2).sum(1)
        else:score=nearest*(weight if method=='geometry_weighted_diversity' else 1)
        score[~available]=-np.inf;idx=int(np.argmax(score));chosen.append(idx)
        nearest=np.minimum(nearest,((x-x[idx])**2).sum(1))
        for j,r in enumerate(rows):
            if r['capture_session']==rows[idx]['capture_session'] and abs(r['timestamp_ns']-rows[idx]['timestamp_ns'])<2_000_000_000:available[j]=False
        available[idx]=False
    return chosen

def acquire():
    lock=read(DOC/'PROTOCOL_LOCK.json');budget=lock['preview_budget_per_method']
    rows=read(RAW/'POOL.json')['eligible'];signals=read(RAW/'ACQUISITION_SIGNALS.json')
    x=np.load(RAW/'FEATURES.npz')['features'];u=np.array([r['selected_pose_instability'] for r in signals])
    assert [r['image_sha256'] for r in rows]==[r['image_sha256'] for r in signals]
    chosen={m:select(x,u,rows,budget,m) for m in METHODS};union=sorted(set(i for v in chosen.values() for i in v))
    method_rows={m:[dict(**rows[i],**{k:v for k,v in signals[i].items() if k!='image_sha256'},rank=j+1) for j,i in enumerate(v)] for m,v in chosen.items()}
    write(DOC/'PREVIEW_SELECTIONS.json',dict(status='LABELING_PREVIEW_NOT_TRAINED',budget_per_method=budget,selections=method_rows))
    queue=[dict(**rows[i],methods=[m for m in METHODS if i in chosen[m]],annotation_status='NOT_ANNOTATED',
        annotation=None) for i in union]
    write(DOC/'ANNOTATION_QUEUE_PREVIEW.json',dict(status='AWAITING_USER_LABELING_BUDGET',
        union_images=len(queue),note='Union cost is not30 total. Blind annotation should hide acquisition method and model predictions.',frames=queue))
    order=np.random.default_rng(20260913).permutation(len(queue))
    blind=[dict(task_id=f'AL{i+1:03}',image_path=queue[j]['image_path'],image_sha256=queue[j]['image_sha256'],
        annotation_status='NOT_ANNOTATED',annotation=None) for i,j in enumerate(order)]
    write(DOC/'BLIND_ANNOTATION_QUEUE_PREVIEW.json',dict(status='PREVIEW_AWAITING_APPROVAL',
        frames=blind,model_predictions_included=False,acquisition_method_hidden=True))
    write(DOC/'ACQUISITION_AUDIT.json',dict(status='PASS',pool_size=len(rows),budget_per_method=budget,
        actual_counts={m:len(v) for m,v in chosen.items()},union_annotation_cost=len(queue),
        pairwise_overlap={a+'__'+b:len(set(chosen[a])&set(chosen[b])) for j,a in enumerate(METHODS) for b in METHODS[j+1:]},
        sessions={m:dict(Counter(rows[i]['capture_session'] for i in v)) for m,v in chosen.items()},
        ground_truth_access_for_selection=False,student_optimizer_updates=0,performance_evaluation='NOT_RUN',
        scientific_verdict='UNTESTED: selected frames are not evidence of label efficiency or pose improvement'))
    print('Selection preview complete; union annotation cost',len(queue),flush=True)

def verify():
    pool=read(RAW/'POOL.json')['eligible'];signals=read(RAW/'ACQUISITION_SIGNALS.json')
    x=np.load(RAW/'FEATURES.npz')['features'];u=[r['selected_pose_instability'] for r in signals]
    selected=read(DOC/'PREVIEW_SELECTIONS.json');budget=selected['budget_per_method']
    assert sha(RAW/'FEATURES.npz')==read(DOC/'EXTRACTION_AUDIT.json')['feature_sha256']
    assert sha(RAW/'ACQUISITION_SIGNALS.json')==read(DOC/'EXTRACTION_AUDIT.json')['signals_sha256']
    for p,h in read(DOC/'POOL_AUDIT.json')['source_bindings'].items():assert sha(ROOT/p)==h
    preserved=read(ROOT/'_docs/experiments/pallet_paper_contribution_screen_v1/PRESERVED_SOURCE_SHA.json')
    assert all(sha(ROOT/p)==h for p,h in preserved.items())
    for m,actual in selected['selections'].items():
        expected=select(x,u,pool,budget,m)
        assert [pool[i]['image_sha256'] for i in expected]==[r['image_sha256'] for r in actual]
        assert len(actual)==budget and len({r['image_sha256'] for r in actual})==budget
        for i,a in enumerate(actual):
            for b in actual[i+1:]:
                assert a['capture_session']!=b['capture_session'] or abs(a['timestamp_ns']-b['timestamp_ns'])>=2_000_000_000
    union={r['image_sha256'] for rr in selected['selections'].values() for r in rr}
    blind=read(DOC/'BLIND_ANNOTATION_QUEUE_PREVIEW.json')['frames']
    assert len(blind)==len(union) and {r['image_sha256'] for r in blind}==union
    assert all(r['annotation'] is None and set(r)=={'task_id','image_path','image_sha256','annotation_status','annotation'} for r in blind)
    write(DOC/'FINAL_AUDIT.json',dict(execution_status='ACQUISITION_PILOT_COMPLETE',integrity='PASS',
        acquisition_pool=len(pool),methods=4,preview_budget_per_method=budget,union_images=len(union),
        deterministic_saved_selection_recomputation=True,temporal_gap_verified=True,
        source_bindings_unchanged=True,old_sources_preserved=len(preserved),
        student_training='NOT_RUN',student_updates=0,performance_evaluation='NOT_RUN',
        scientific_verdict='UNTESTED',human_annotations_available=False,
        pending='Confirm manual-label availability and budget, then freeze downstream protocol before opening labels',
        prior_C_diagnostic_reopened=False,paper_final_modified=False))
    print('Saved acquisitions, blind queue and source preservation PASS',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['initialize','extract','acquire','verify']);args=ap.parse_args();globals()[args.phase]()

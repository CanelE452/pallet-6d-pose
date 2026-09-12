"""No-training, same-grid score/box intervention; immutable development evidence."""
import argparse
import csv
import inspect
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.contracts import ROOT, RAW, DOC, R0, R0_SHA, sha, write
from track_c.verify_saved import detection_state_keys
from track_c.aggregate import geometry
sys.path.insert(0,str(ROOT/'scripts/research/pallet_line_pose_v1'))
import paper_evaluation as P
from scripts.paper.framing_closure_v1.static_missing_stat_audit import ranking

BASE=RAW/'C_score_box_selection_v1'
OUT=DOC/'C_score_box_selection_v1'
KINDS=('R0','C2','R0box_C2score','C2box_R0score')

def read(p):return json.loads(p.read_text())
def checkpoint(seed):return RAW/f'C_geometry_preserving_da/C2_seed{seed}/last.pt'
def old_cache(seed):
    return (ROOT/'data/pallet/results/pallet_line_pose_v1/baseline/FULL_CANDIDATES.json' if seed==0
        else RAW/f'C_geometry_preserving_da/C2_seed{seed}/evaluation_pose/PREDICTIONS.json')

def combine(base, adapted, kind):
    assert kind in KINDS,'Unknown intervention'
    assert base.shape==adapted.shape and base.shape[1]==32
    assert torch.equal(base[:,5:],adapted[:,5:]),'Decoded dense pose changed'
    boxes=base[:,:4] if kind in ('R0','R0box_C2score') else adapted[:,:4]
    scores=base[:,4:5] if kind in ('R0','C2box_R0score') else adapted[:,4:5]
    return torch.cat((boxes,scores,base[:,5:]),1)

def initialize():
    assert sha(R0)==R0_SHA
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='main'
    start=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    bindings={str(p.relative_to(ROOT)):sha(p) for p in [R0,*[checkpoint(s) for s in (1,2,3)],
        *[old_cache(s) for s in (0,1,2,3)],P.POS,P.NEG,Path(P.E.__file__),
        ROOT/'scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py',
        DOC/'C_geometry_preserving_da/RESUME_COMPLETE/VERDICT.json']}
    baseline=torch.load(R0,map_location='cpu')['model'].float()
    allowed=detection_state_keys(baseline);frozen=[]
    for s in (1,2,3):
        model=torch.load(checkpoint(s),map_location='cpu')['model'].float()
        assert set(model.state_dict())==set(baseline.state_dict())
        names=[n for n in baseline.state_dict() if n not in allowed]
        assert all(torch.equal(baseline.state_dict()[n],model.state_dict()[n]) for n in names)
        assert model.model[-1].end2end and model.model[-1].nc==1
        frozen.append(dict(seed=s,frozen_keys=len(names),equal=True,
            branches=list(model.model[-1].one2one),one2many_branches=list(model.model[-1].one2many)))
    write(OUT/'PROTOCOL_LOCK.json',dict(start_main=start,status='LOCKED',optimizer_updates=0,
        seeds=[1,2,3],arms=list(KINDS),population='Repeated DEV319 + NEG2689; no independent confirmation',
        bindings=bindings,frozen_audit=frozen,parity='All original postprocessed candidates bit-exact to saved cache',
        candidate_contract='Same pre-postprocess multiscale grid index, never posthoc box matching',
        runtime_path='Unchanged canonical fused end2end one2one inference. Stock fusion removes unused one2many; unfused branch ownership audited above.',
        geometry_contract='Bit-exact feature maps, grid/strides and decoded dense keypoints on every image',
        selection='All seeds retained; no winner, training, threshold tuning, adapter or checkpoint modification',
        metrics=['AP50/AP75/AP by IoU/AP50-95','Det','negative FP at existing floor .001, AUROC/FPR95',
            'top candidate-index changes','four-arm common-frame geometry','same top-candidate geometry',
            'canonical MAIN pose with full319 denominator'],
        interpretation='Posthoc diagnostic only; original C FAIL unchanged; no novelty or independent superiority claim'))

class Capture:
    def __init__(self,path,first):
        self.p=P.E._UltralyticsPredictor(path,'0')
        self.p.predict(first)  # Canonical AutoBackend setup/fusion and warmup.
        self.head=self.p.model.predictor.model.model.model[-1]
        assert self.head.end2end and self.head.nc==1 and self.head.nk==27
        self.head.register_forward_pre_hook(self.pre)
        self.head.register_forward_hook(self.post)
        self.p.model.predictor.model.register_forward_pre_hook(self.input_hook)
    def input_hook(self,m,args):self.input=args[0].clone()
    def pre(self,m,args):self.features=[v.clone() for v in args[0]]
    def post(self,m,args,out):
        raw=out[1]['one2one'];self.dense=m._inference(raw).clone()
        self.anchors=m.anchors.clone();self.strides=m.strides.clone()
        assert torch.equal(m.postprocess(self.dense.permute(0,2,1)),out[0])
    def ids(self,dense):
        scores,_,idx=self.head.get_topk_index(dense[:,4:5].permute(0,2,1),self.head.max_det)
        keep=scores[0,:,0]>P.E.INFERENCE_CONFIDENCE_FLOOR
        return idx[0,:,0][keep].cpu().tolist()
    def predict(self,image):
        values=self.p.predict(image)
        return self.pack(values,self.ids(self.dense))
    def replay(self,image,dense):
        predictor=self.p.model.predictor;original=predictor.inference
        output=self.head.postprocess(dense.permute(0,2,1))
        def inference(im,*args,**kwargs):
            assert torch.equal(im,self.input),'Hybrid preprocessing changed'
            return (output.clone(),{})
        predictor.inference=inference
        try:values=self.p.predict(image)
        finally:predictor.inference=original
        return self.pack(values,self.ids(dense))
    @staticmethod
    def pack(values,ids):
        assert len(values)==len(ids),'Postprocess/index filter mismatch'
        return [dict(score=float(s),box_xyxy=b.tolist(),keypoints_xy=k.tolist()) for s,b,k in values],ids

def infer():
    lock=read(OUT/'PROTOCOL_LOCK.json')
    for p,h in lock['bindings'].items():assert sha(ROOT/p)==h
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
    if gpu:
        write(OUT/'RESOURCE_UNAVAILABLE.json',dict(status='NOT_RUN',processes=gpu,action='No wait or mutation'))
        raise SystemExit('GPU unavailable')
    assert not (BASE/'INFERENCE_AUDIT.json').exists(),'Inference already complete; use evaluate'
    pair=P.population();items=[*pair.positive.items,*pair.negative.items]
    first=ROOT/items[0].image
    original={s:read(old_cache(s))['frames'] for s in (0,1,2,3)}
    models={s:Capture(R0 if s==0 else checkpoint(s),first) for s in (0,1,2,3)}
    frames={f'{k}_seed{s}':{} for s in (1,2,3) for k in KINDS};indices={k:{} for k in frames}
    with torch.inference_mode():
        for i,item in enumerate(items):
            image=ROOT/item.image;key=P.canonical_key(item.image)
            r,ri=models[0].predict(image)
            assert r==original[0][key],f'R0 cache parity: {key}'
            for seed in (1,2,3):
                model=models[seed];c,ci=model.predict(image)
                assert c==original[seed][key],f'C2 cache parity: {seed} {key}'
                assert torch.equal(models[0].input,model.input)
                assert len(models[0].features)==len(model.features)
                assert all(torch.equal(a,b) for a,b in zip(models[0].features,model.features))
                assert torch.equal(models[0].anchors,model.anchors) and torch.equal(models[0].strides,model.strides)
                for kind in KINDS:
                    mixed=combine(models[0].dense,model.dense,kind)
                    if kind=='R0':value,idx=r,ri
                    elif kind=='C2':value,idx=c,ci
                    else:value,idx=model.replay(image,mixed)
                    name=f'{kind}_seed{seed}';frames[name][key]=value;indices[name][key]=idx
                    assert idx==(ri if kind in ('R0','C2box_R0score') else ci)
                    by_id=dict(zip(ri,r))
                    for candidate,anchor in zip(value,idx):
                        if anchor in by_id:assert candidate['keypoints_xy']==by_id[anchor]['keypoints_xy']
            if (i+1)%250==0:print(f'Candidate intervention {i+1}/3008; all three seeds parity PASS',flush=True)
    for name,value in frames.items():
        seed=int(name[-1]);path=BASE/name/'PREDICTIONS.json'
        write(path,dict(schema_version='paper_cached_predictions_v1',complete=True,model=name,
            weights_sha256=sha(checkpoint(seed)),frames=value,role='POSTHOC_DEVELOPMENT_DIAGNOSTIC',
            source_checkpoints={str(R0):R0_SHA,str(checkpoint(seed)):sha(checkpoint(seed))}))
        write(BASE/name/'CANDIDATE_INDICES.json',indices[name])
    write(BASE/'INFERENCE_AUDIT.json',dict(status='PASS',frames=3008,C2_seeds=[1,2,3],
        original_candidate_cache_bit_exact=True,input_feature_grid_dense_pose_bit_exact=True,
        same_candidate_points_bit_exact=True,score_source_determines_selected_indices=True,
        fresh_neural_image_forwards=4*3008,warmup_setup_excluded=True,hybrid_neural_forwards=0,optimizer_updates=0))
    write(OUT/'INFERENCE_AUDIT.json',read(BASE/'INFERENCE_AUDIT.json'))

def evaluate():
    assert read(BASE/'INFERENCE_AUDIT.json')['status']=='PASS'
    for seed in (1,2,3):
        for kind in KINDS:
            name=f'{kind}_seed{seed}';d=BASE/name;p=d/'PREDICTIONS.json'
            P.paper_2d(d,p,str(R0 if kind=='R0' else checkpoint(seed)))
            P.paper_pose(d,read(p)['frames'],name)
            print(name,'canonical evaluation complete',flush=True)

def summarize():
    result=[];paired=[]
    for seed in (1,2,3):
        rows={};ids={}
        for kind in KINDS:
            name=f'{kind}_seed{seed}';d=BASE/name
            rows[kind]={r['frame_id']:r for r in csv.DictReader((d/'PAPER_2D_per_frame.csv').open())}
            ids[kind]=read(d/'CANDIDATE_INDICES.json')
            m=read(d/'PAPER_2D.json')['metrics']['box_and_keypoint_2d']
            pose=read(d/f'POSE_EVALUATION_{name}.json')['paths']['MAIN']
            pos=[r for r in rows[kind].values() if r['kind']=='POSITIVE'];neg=[r for r in rows[kind].values() if r['kind']=='NEGATIVE']
            scores=lambda rr:np.array([float(r['top_score'] or 0) for r in rr])
            result.append(dict(seed=seed,arm=kind,ap50=m['box_ap50'],ap75=m['box_ap_by_iou']['0.75'],
                ap50_95=m['box_ap50_95'],ap_by_iou=m['box_ap_by_iou'],
                Det=sum(r['top_iou50_match']=='True' for r in pos)/319,
                negative_frames=2689,negative_frames_with_detection=sum(int(r['candidate_count'])>0 for r in neg),
                negative_candidate_count=sum(int(r['candidate_count']) for r in neg),**ranking(scores(pos),scores(neg)),
                pose=pose,pose_denominator=319))
        common=set.intersection(*[{k for k,r in rr.items() if r['kind']=='POSITIVE' and r['top_iou50_match']=='True' and r['top_keypoint_supervised_errors_px']} for rr in rows.values()])
        same=[];changes={}
        for kind in KINDS:
            both=[k for k,v in ids['R0'].items() if v and ids[kind][k]]
            poskeys=[P.canonical_key(r['image']) for r in rows['R0'].values() if r['kind']=='POSITIVE']
            p=[k for k in both if k in poskeys]
            changes[kind]=dict(both_detected=len(both),changed=sum(ids['R0'][k][0]!=ids[kind][k][0] for k in both),
                positive_both_detected=len(p),positive_changed=sum(ids['R0'][k][0]!=ids[kind][k][0] for k in p),
                missing_transition=sum(bool(ids['R0'][k])!=bool(ids[kind][k]) for k in ids['R0']))
        for fid in sorted(common):
            key=P.canonical_key(rows['R0'][fid]['image'])
            if len({ids[k][key][0] for k in KINDS})==1:same.append(fid)
        paired.append(dict(seed=seed,common_frames=len(common),same_top_candidate_frames=len(same),
            common_frame_geometry={k:geometry(rows[k],sorted(common)) for k in KINDS},
            same_top_candidate_geometry={k:geometry(rows[k],same) for k in KINDS},selection=changes))
    lock=read(OUT/'PROTOCOL_LOCK.json')
    for p,h in lock['bindings'].items():assert sha(ROOT/p)==h
    preserved=read(DOC/'PRESERVED_SOURCE_SHA.json')
    assert all(sha(ROOT/p)==h for p,h in preserved.items())
    write(OUT/'RESULTS.json',dict(status='COMPLETE',evidence='POSTHOC_DEVELOPMENT_DIAGNOSTIC',
        optimizer_updates=0,per_seed=result,paired=paired,original_C_verdict_changed=False,
        negative_threshold='Existing canonical confidence floor0.001; no operating threshold selected',
        uncertainty='Descriptive three-seed results, no new independent population or significance claim',
        pose_reference='Canonical geometry-reconstructed reference, not external sensor ground truth'))
    report(result,paired)
    write(OUT/'FINAL_AUDIT.json',dict(status='PASS',optimizer_updates=0,new_checkpoints=0,
        seeds=[1,2,3],arms_per_seed=4,evaluations=12,population_positive=319,population_negative=2689,
        original_artifact_bindings_unchanged=True,additional_preserved_sources=len(preserved),
        original_C_verdict='FAIL unchanged',paper_final_modified=False,
        candidate_index_definition='Index in flattened pre-postprocess multiscale feature grid',
        original_candidate_parity='All complete candidate lists exactly equal on all3008 images',
        raw_caches={f'{k}_seed{s}':sha(BASE/f'{k}_seed{s}/PREDICTIONS.json') for s in (1,2,3) for k in KINDS},
        diagnostic_script_sha256=sha(Path(__file__)),confirmation=False,new_method_frozen=False))
    print('Diagnostic results complete',flush=True)

def report(results,paired):
    means={}
    for kind in KINDS:
        rr=[r for r in results if r['arm']==kind]
        means[kind]={k:float(np.mean([r[k] for r in rr])) for k in ('ap50','ap75','ap50_95','Det','auroc','fpr95','negative_frames_with_detection')}
        means[kind]['geometry']={k:float(np.mean([p['common_frame_geometry'][kind][k] for p in paired])) for k in ('median_px','p90_px','gross20')}
        means[kind]['pose']={k:float(np.mean([r['pose']['ALL'][k] for r in rr])) for k in ('translation_median_cm','yaw_median_deg','rotation_median_deg','iou3d_median','add_sym_auc')}
    write(OUT/'SEED_MEANS.json',means)
    lines=['# R0/C2 score–box–selection diagnostic','',
        'POSTHOC DEVELOPMENT DIAGNOSTIC. Training0; no new checkpoint, threshold selection, adapter or winner. Original C FAIL and original reports remain unchanged.',
        '', '## Scope and integrity','',
        'Immutable R0 and all three final C2 seeds; four source combinations per seed; same DEV319 + NEG2689. R0 is repeated as a shared reference, not three independent R0 fits.',
        'Before intervention, original full candidate lists are bit-exact to their saved caches on all3008 images. All image tensors, backbone/neck feature maps, feature-grid anchors/strides and dense decoded pose tensors are bit-exact between R0/C2.',
        'Canonical fusion and one2one decoding/top-k/postprocessing are unchanged; unused one2many removal is the existing stock inference behavior. Both branches were checked in unfused checkpoint ownership. Hybrid replay uses the original postprocessing pipeline without a new neural forward.',
        'Common-candidate identity means the pre-postprocess grid index, never IoU-based correspondence between final boxes. All jointly retained identical indices have bit-exact keypoints.',
        '', '## Three-seed means','',
        '| Box / score source | AP50 | AP75 | AP50-95 | Det | AUROC | FPR95 | common kp median/P90 px | translation cm | yaw deg |',
        '|---|---:|---:|---:|---:|---:|---:|---|---:|---:|']
    for k,m in means.items():
        g=m['geometry'];p=m['pose']
        lines.append(f"| {k} | {m['ap50']:.6f} | {m['ap75']:.6f} | {m['ap50_95']:.6f} | {m['Det']:.6f} | {m['auroc']:.6f} | {m['fpr95']:.6f} | {g['median_px']:.4f}/{g['p90_px']:.4f} | {p['translation_median_cm']:.4f} | {p['yaw_median_deg']:.4f} |")
    lines+=['','AP at every IoU0.50:0.05:0.95 and negative candidate counts are in RESULTS.json. Negative FP counts use only the existing0.001 inference floor, not a newly tuned operating point.',
        'Geometry is pooled supervised0..8 raw-pixel error on the four-arm common matched frames within each seed. Pose uses the original MAIN selector and all319-frame denominator; failures/missing cases are not removed. Reference pose is geometry-reconstructed, not external sensor ground truth.',
        '', '## Per seed','',
        '| Seed | Arm | AP50-95 | negative frames with candidates /2689 | pose coverage | translation cm | yaw deg |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in results:
        p=r['pose'];a=p['ALL']
        lines.append(f"| {r['seed']} | {r['arm']} | {r['ap50_95']:.6f} | {r['negative_frames_with_detection']} | {p['coverage']:.6f} | {a['translation_median_cm']:.4f} | {a['yaw_median_deg']:.4f} |")
    lines+=['', '## Selection and common-candidate controls','',
        '| Seed | Four-arm matched frames | Same top-grid-index frames | C2 score top-index changes /positive both detected | Missing transitions /all3008 |',
        '|---|---:|---:|---|---:|']
    for p in paired:
        c=p['selection']['C2']
        lines.append(f"| {p['seed']} | {p['common_frames']} | {p['same_top_candidate_frames']} | {c['positive_changed']}/{c['positive_both_detected']} | {c['missing_transition']} |")
        same=p['same_top_candidate_geometry']
        assert all(same[k]==same['R0'] for k in KINDS),'Same-candidate metric identity failed'
    lines+=['','All four arms have identical geometry statistics on the same-top-candidate subsets. For every image, changing only boxes preserves the selected index list; changing scores uses exactly the score-source index list. Thus dense invariance does not imply selected-output invariance.',
        '', '## Diagnostic contrasts (AP50-95 percentage points)','']
    for a,b in [('C2','R0'),('R0box_C2score','R0'),('C2box_R0score','R0'),('R0box_C2score','C2')]:
        lines.append(f"- {a} minus {b}: {100*(means[a]['ap50_95']-means[b]['ap50_95']):+.4f} pp.")
    lines+=['', '## Interpretation limits','',
        'Score and box swaps identify output-path effects for these fixed checkpoints. They do not alone prove that frozen features are insufficient, nor isolate the training cause of head changes. Do not automatically add an adapter or adopt a hybrid as a novel method.',
        'Box changes may change IoU matching/metric inclusion even when selected keypoints are identical. Compare common-frame and common-candidate statistics, not separately selected error pools.',
        'Original C0 is replay-finetuned, not immutable R0. Original pose_safe allows up to10% translation/yaw regression and is not a no-regression guarantee.',
        'The review also correctly identifies that D used19 scalar features without the requested explicit local appearance descriptor. Its FAIL covers that restricted feature set and risk rule, not full visual trust or student performance. D/B/E are not rerun here.',
        'This repeated development population supplies no independent generalization, statistical significance or novelty claim. A remains frozen pending independent confirmation. No paper-final document is edited.']
    path=OUT/'REPORT.md';value='\n'.join(lines)+'\n'
    if path.exists():assert path.read_text()==value
    else:path.write_text(value)

def verify():
    checks=[]
    for seed in (1,2,3):
        for kind in KINDS:
            name=f'{kind}_seed{seed}';d=BASE/name
            m=read(d/'PAPER_2D.json')['metrics']['box_and_keypoint_2d']
            rows=list(csv.DictReader((d/'PAPER_2D_per_frame.csv').open()))
            errors=np.array([float(v) for r in rows for v in r['top_keypoint_supervised_errors_px'].split(';') if v])
            assert len(rows)==3008
            assert abs(float(np.median(errors))-m['keypoint_location_median_px'])<1e-5
            assert abs(float(np.quantile(errors,.9))-m['keypoint_location_p90_px'])<1e-5
            assert abs(np.mean(list(m['box_ap_by_iou'].values()))-m['box_ap50_95'])<1e-12
            pose=read(d/f'POSE_EVALUATION_{name}.json')['paths']['MAIN']
            if kind in ('R0','C2'):
                old2d=P.OLD2D if kind=='R0' else RAW/f'C_geometry_preserving_da/C2_seed{seed}/evaluation_pose/PAPER_2D.json'
                old=read(old2d)['metrics']['box_and_keypoint_2d']
                for k in ('box_ap50','box_ap50_95','keypoint_location_median_px','keypoint_location_p90_px'):
                    assert abs(m[k]-old[k])<1e-10,(name,k,m[k],old[k])
            else:
                source='C2' if kind=='R0box_C2score' else 'R0'
                sd=BASE/f'{source}_seed{seed}'
                assert pose==read(sd/f'POSE_EVALUATION_{source}_seed{seed}.json')['paths']['MAIN']
                a,b=read(d/'PREDICTIONS.json')['frames'],read(sd/'PREDICTIONS.json')['frames']
                assert read(d/'CANDIDATE_INDICES.json')==read(sd/'CANDIDATE_INDICES.json')
                for key in a:
                    assert len(a[key])==len(b[key])
                    for x,y in zip(a[key],b[key]):
                        assert x['score']==y['score'] and x['keypoints_xy']==y['keypoints_xy']
            checks.append(dict(arm=name,status='PASS',CSV_frames=len(rows),pose_coverage=pose['coverage']))
    write(OUT/'INDEPENDENT_RECOMPUTATION.json',dict(status='PASS',checks=checks,
        CSV_median_p90_tolerance_px=1e-5,original_summary_tolerance=1e-10,
        hybrid_score_source_candidate_points_and_MAIN_pose_equal=True,optimizer_updates=0))
    print('All12 saved-result independent checks PASS',flush=True)

if __name__=='__main__':
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['initialize','infer','evaluate','summarize','verify'])
    globals()[parser.parse_args().phase]()

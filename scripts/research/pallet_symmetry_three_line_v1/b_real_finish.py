"""GT-free shuffle and serialization, then separate canonical DEV scoring."""
import copy,csv
import numpy as np
import torch,cv2
import env as E
from b_model import FrozenHeads,MODES
from b_evaluate import predict,csvout
from metrics import measure,summarize,contrast

def main():
    assert E.read(E.DOC/'B/DEV_CACHE_COMPLETE.json')['complete'];torch.set_num_threads(4);cv2.setNumThreads(1)
    selected=E.read(E.DOC/'B/selection_lock.json')['selected_eta'];selection=E.read(E.C.B/'P_SELECTION.json')
    paths=sorted((E.RAW/'B/cache/DEV').glob('*.pt'));records=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
    usable=[i for i,c in enumerate(records) if c['P']];rng=np.random.default_rng(20260916)
    buckets={i:tuple(np.floor(np.log2(np.maximum(np.array(records[i]['box_raw'][2:]-records[i]['box_raw'][:2]),1))).astype(int)) for i in usable}
    plan=[]
    for i in usable:
        pool=[j for j in usable if j!=i and buckets[j]==buckets[i]];fallback=not pool
        if not pool:pool=[j for j in usable if j!=i]
        j=int(rng.choice(pool));plan.append(dict(receiver=i,donor=j,receiver_id=records[i]['frame_id'],donor_id=records[j]['frame_id'],size_bucket_fallback=fallback))
    E.freeze(E.DOC/'B/DEV_SHUFFLE_LOCK.json',dict(seed=20260916,GT_input=False,plan=plan,
      bucket_policy='predicted ROI size only inside the fixed paper task; GT object-type/physical asset not used as runtime metadata'))
    E.gpu();heads=FrozenHeads()
    for r in plan:
        i,j=r['receiver'],r['donor'];dst=E.RAW/f'B/shuffle/DEV/{i:04d}.pt'
        if dst.exists():continue
        c,d=records[i],records[j];out={k:v.cuda() if torch.is_tensor(v) else v for k,v in c['P'][1]['output'].items()}
        score=heads.score(out,d['line_logits'].cuda(),d['line_valid'].cuda(),c['box_raw'].cuda(),c['base_points'].cuda(),
          c['point_valid'].cuda(),c['perms'],c['gain'],torch.tensor(c['offset'],device='cuda',dtype=torch.float32))
        dst.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(bias=score['biases']['B3_THREE_AMBIG'],alignment=score['alignment'],donor=r),dst)
    del heads;torch.cuda.empty_cache()
    # No GT opened yet. Copy every candidate and change only selected corner coordinates.
    newpaths={};baseline=E.read(E.C.LINE/'baseline/FULL_CANDIDATES.json')
    negative_motion={};invariance={}
    for arm in ['B0_P',*MODES,'B4_SHUFFLE_DIAG']:
        eta=0 if arm=='B0_P' else selected['B3_THREE_AMBIG'] if arm=='B4_SHUFFLE_DIAG' else selected[arm]
        for seed in [1,2,3]:
            frames={};negchanged=0
            for i,c in enumerate(records):
                key=c['image_key'];candidates=copy.deepcopy(baseline['frames'][key])
                if c['P']:
                    shuffle=torch.load(E.RAW/f'B/shuffle/DEV/{i:04d}.pt',map_location='cpu',weights_only=False)['bias'] if arm=='B4_SHUFFLE_DIAG' else None
                    points=predict(c,seed,arm,eta,selection,shuffle);k=c['selected_index']
                    assert np.isfinite(points).all();assert np.array_equal(points[8],np.array(candidates[k]['keypoints_xy'])[8])
                    if baseline['frame_metadata'][key]['kind']=='negative':negchanged+=int(not np.array_equal(points,np.array(candidates[k]['keypoints_xy'])))
                    candidates[k]['keypoints_xy']=points.tolist()
                for index,(new,old) in enumerate(zip(candidates,baseline['frames'][key])):
                    assert new['score']==old['score'] and new['box_xyxy']==old['box_xyxy']
                    if index!=c['selected_index']:assert new==old
                frames[key]=candidates
            name=f'{arm}_seed{seed}';dst=E.RAW/f'B/DEV_predictions/{name}.json'
            E.write(dst,dict(frames=frames,GT_input=False,arm=arm,seed=seed,eta=eta))
            newpaths[(arm,seed)]=dst;negative_motion[name]=negchanged
            invariance[name]=dict(detection_list_length_order_box_score=True,unselected_candidates=True,center8=True)
            print('DEV SERIALIZED',name,flush=True)
    # All predictions now serialized; GT appears only in this scoring stage.
    pe=E.C.old('paper_evaluation');population=pe.population();meta={r['frame_id']:r for r in E.read(pe.POS)['items']}
    targets={};hw={}
    for item in population.positive.items:
        targets[item.frame_id]=pe.E._legacy_forbidden_target(item)
        hw[item.frame_id]=cv2.imread(str(E.ROOT/item.image)).shape[:2]
    perms=np.array(E.read(E.DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'][0]['permutations'])
    summary={};stores={};flat=[]
    for (arm,seed),path in newpaths.items():
        frames=E.read(path)['frames'];rows=[]
        for item in population.positive.items:
            t=targets[item.frame_id];key=pe.canonical_key(item.image);candidates=frames[key]
            top=max(candidates,key=lambda c:c['score']) if candidates else None
            matched=top is not None and top['keypoints_xy'] is not None and pe.E._box_iou(np.array(top['box_xyxy']),t.box_xyxy)>=.5
            pred=np.array(top['keypoints_xy']) if matched else np.full((9,2),np.nan)
            m=measure(pred,t.keypoints_xy,t.keypoint_supervision_mask,perms,hw[item.frame_id])
            m.update(frame_id=item.frame_id,session_id=meta[item.frame_id]['session_id'],matched=matched)
            rows.append(m)
        stores.setdefault(arm,{})[seed]=rows
        summary.setdefault(arm,{})[seed]=dict(**summarize(rows),matched_frames=sum(r['matched'] for r in rows))
        for r in rows:flat.append(dict(arm=arm,seed=seed,**{k:v for k,v in r.items() if not isinstance(v,list)}))
    sessions=[r['session_id'] for r in stores['B0_P'][1]];pairs={}
    for arm in ['B0_P','B1_ONE_AMBIG','B2_THREE_EQUAL','B4_SHUFFLE_DIAG']:
        pairs['B3-'+arm]=contrast([[r['E_sym'] for r in stores['B3_THREE_AMBIG'][s]] for s in [1,2,3]],
          [[r['E_sym'] for r in stores[arm][s]] for s in [1,2,3]],sessions)
    damage={}
    for seed in [1,2,3]:
        b=stores['B0_P'][seed];n=stores['B3_THREE_AMBIG'][seed];be=np.concatenate([r['errors'] for r in b]);ne=np.concatenate([r['errors'] for r in n]);good=be<5
        delta=np.array([x['frame_mean_px']-y['frame_mean_px'] for x,y in zip(n,b)])
        damage[seed]=dict(good_baseline_points=int(good.sum()),good5_to_bad10=int((good&(ne>10)).sum()),
          improved_frames=int((delta< -1e-9).sum()),harmed_frames=int((delta>1e-9).sum()),unchanged_frames=int((abs(delta)<=1e-9).sum()),
          delta_mean_px=float(delta.mean()),delta_median_px=float(np.median(delta)),delta_P90_px=float(np.quantile(delta,.9)),
          eval_symmetry_changed_frames=sum(x['branch']!=y['branch'] for x,y in zip(n,b)))
    csvout(E.DOC/'B/DEV_per_frame_results.csv',flat)
    E.write(E.RAW/'B/DEV_FULL_METRICS.json',stores)
    lo,hi=pairs['B3-B0_P']['CI95'];verdict='LINE_AUXILIARY_DEV_SIGNAL' if hi<0 and pairs['B3-B0_P']['improved_seeds']>=2 else 'LINE_AUXILIARY_HARM' if lo>0 else 'LINE_AUXILIARY_NOT_ESTABLISHED'
    E.write(E.DOC/'B/DEV_RESULTS.json',dict(status='COMPLETE_2D',verdict=verdict,summary=summary,contrasts=pairs,damage=damage,
      positive=319,negative=2689,GT_free_inference_all_images=True,invariance=invariance,
      negative_selected_corner_motion=negative_motion,negative_detection_metrics_unchanged=True,
      negative_note='No GT-positive gate: selected corner refinement also runs on negatives. Candidate identities/order/box/score/center and unselected candidates are unchanged; selected negative corners can move.',
      secondary_CI='exploratory, multiplicity unadjusted',new_training_updates=0))
    print('DEV RESULT',verdict,pairs['B3-B0_P'],flush=True)
if __name__=='__main__':main()

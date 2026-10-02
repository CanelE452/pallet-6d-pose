"""Preserve PnP outputs/hypotheses and fixed-branch tail diagnostics."""
from collections import Counter
import time
import numpy as np
from .compute import C,R,M,ROOT,DOC,RAW,SOURCE,write,bind,clean
from scripts.research.pallet_n3_completion_v3 import evaluation as E

def main():
    started=time.time();rows,sources=R.load_dev_context(SOURCE,include_pose=True)
    for arm in ('OLD_P',)+R.DCP_ARMS:
        for seed in (1,2,3):R.attach_dcp(rows,SOURCE,arm,seed)
    collections={'yolo':rows};scores={'yolo':C.read(RAW/'YOLO_SCORES.json')}
    for b in ['dope','resnet18']:
        p=C.read(C.RAW/f'predictions/{b}_DEV319.json');norm=E.normalize_prediction_payload(p)['methods'];rr=[]
        for r in rows:
            t={k:v for k,v in r.items() if k not in ['predictions','detected','matched','_candidates']};t.update(predictions={},detected={},matched={})
            for method,records in norm.items():
                pred=records[r['id']];t['predictions'][method]=pred['points'];t['detected'][method]=pred['detected'];t['matched'][method]=bool(pred['detected'] and E._iou(pred['bbox'],r['box'])>=.5)
            rr.append(t)
        collections[b]=rr
        scores[b]={k:{'pose_scores':v['result']['pose_rows'],'corner_scores':v['result']['corner_rows']} for k,v in C.read(C.RAW/f'evaluation/{b}.json')['methods'].items()}
    module=M._pose_contract();traces={};changes=[];checks={};tail={}
    for backbone,rr in collections.items():
        traces[backbone]={};base_name='R0' if backbone=='yolo' else 'base'
        for method in rr[0]['predictions']:
            out=[];expected={r['id']:r for r in scores[backbone][method]['pose_scores']}
            for row in rr:
                spec=row['pose'];points=M._prediction(row,method) if row['detected'][method] else None
                inferred=module.infer(points,np.array(spec['K']),np.array(spec['xyz']),source=False)
                metric=module.metric((row['id'],inferred,spec['truth']))
                assert all(clean(v)==clean(expected[row['id']][k]) for k,v in metric.items())
                reason=None
                if not inferred['available']:
                    reason='NO_DETECTION' if points is None else 'FEWER_THAN_SIX_FINITE_CORNERS' if np.isfinite(points[:8]).all(-1).sum()<6 else 'LOCKED_SELECTOR_OR_SOLVER_UNAVAILABLE (legacy solver does not expose a finer reason)'
                sym=None
                if inferred['available']:
                    G=np.array(spec['truth']['R']);Q=np.array(inferred['R_physical'])
                    angle=[float(np.degrees(np.arccos(np.clip((np.trace((G@s).T@Q)-1)/2,-1,1)))) for s in module.rotations(spec['truth']['order'])]
                    sym=int(np.argmin(angle))
                out.append({'id':row['id'],'session':row['session'],'occlusion':row['occlusion'],'pose':inferred,'metric':metric,'failure_reason':reason,'evaluation_rotation_symmetry_index':sym})
            traces[backbone][method]=out;checks[backbone+':'+method]=True
        base={r['id']:r for r in traces[backbone][base_name]}
        for seed in (1,2,3):
            method=f'N3_DIM_SYM_seed{seed}' if backbone=='yolo' else f'n3_seed{seed}'
            for r,row in zip(traces[backbone][method],rr):
                b=base[r['id']];available=b['pose']['available'] and r['pose']['available']
                delta=None if not available else r['metric']['rotation_deg']-b['metric']['rotation_deg']
                p=M._prediction(row,base_name);a=M._prediction(row,method);valid=np.isfinite(p[:8]).all(-1)&np.isfinite(a[:8]).all(-1)
                movement=np.linalg.norm(a[:8][valid]-p[:8][valid],axis=1)
                changes.append({'backbone':backbone,'seed':seed,'id':r['id'],'occlusion':row['occlusion'],
                    'delta_rotation_deg':delta,'hypothesis_before':b['pose'].get('selected_hypothesis'),'hypothesis_after':r['pose'].get('selected_hypothesis'),
                    'WD_hypothesis_changed':available and b['pose']['selected_hypothesis']!=r['pose']['selected_hypothesis'],
                    'evaluation_symmetry_changed':available and b['evaluation_rotation_symmetry_index']!=r['evaluation_rotation_symmetry_index'],
                    'max_corner_movement_px':float(movement.max()) if len(movement) else None,'cause':'Descriptive co-occurrence only; no causal attribution'})
        print('Pose traces',backbone,flush=True)
    # Complete original table XII at fixed 1% cap only. No new cap search.
    for arm in ['OLD_P','N3_DIM_SYM']:
        tail[arm]={}
        for seed in (1,2,3):
            method=f'{arm}_seed{seed}';summary,frames=R.fixed_branch_damage(rows,'R0',method,.01)
            bins={key:Counter() for key in ['inside','outside']}
            for row in rows:
                if not row['matched']['R0']:continue
                p=np.array(row['predictions']['R0']);a=np.array(row['predictions'][method]);base=M.score_corner_rows([row],'R0')[0]
                perm=np.array(row['permutations'])[base['branch']];gt=np.array(row['gt'])[perm];valid=np.array(row['valid'])[perm][:8]
                eb=np.linalg.norm(p[:8]-gt[:8],axis=1);ea=np.linalg.norm(a[:8]-gt[:8],axis=1);cap=.01*np.linalg.norm(row['hw'])
                for x,y in zip(eb[valid],ea[valid]):bins['inside' if x<=cap else 'outside']['improved' if y-x< -1e-9 else 'worsened' if y-x>1e-9 else 'unchanged']+=1
            bp={r['id']:r for r in scores['yolo']['R0']['pose_scores']};ap={r['id']:r for r in scores['yolo'][method]['pose_scores']}
            joint=Counter()
            for f in frames:
                b,a=bp[f['id']],ap[f['id']]
                if f['delta_px']< -1e-9 and b['available'] and a['available']:
                    dt=a['translation_cm']-b['translation_cm'];dr=a['rotation_deg']-b['rotation_deg']
                    if dt< -1e-9 and dr< -1e-9:joint['2d_improved_both_TR_improved']+=1
                    if dt>1e-9 or dr>1e-9:joint['2d_improved_either_TR_worsened']+=1
            tail[arm][str(seed)]={'fixed_branch':summary,'inside_outside':bins,'joint':joint}
    write(RAW/'PNP_TRACES.json',traces);write(RAW/'POSE_HYPOTHESIS_CHANGES.json',changes)
    write(DOC/'POSE_TRACE_AND_TAIL.json',{'regression':checks,'tail':tail,'source':sources,'seconds':time.time()-started,
        'hypothesis_change_counts':dict(Counter(r['backbone'] for r in changes if r['WD_hypothesis_changed'])),
        'evaluation_symmetry_change_counts':dict(Counter(r['backbone'] for r in changes if r['evaluation_symmetry_changed'])),
        'large_rotation_analysis':'All frame deltas retained; sort absolute change for descriptive review without a new threshold. Hypothesis/symmetry changes are not ground truth or causal proof.'})

if __name__=='__main__':main()

"""Diagnose source-VAL candidate opportunity after the completed fixed screen.

The already failed VAL gate is preserved. This is not a new checkpoint search,
real-data route, or permission to waive the original gate.
"""
from scripts.research.pallet_pose_union_selection_20261001_v1 import common as U
READS=U.source_guard(allow_source_targets=True)
from scripts.research.pallet_pose_union_selection_20261001_v1 import source_features as F
from pathlib import Path
import csv
import io
import json
import numpy as np

DOC=U.ROOT/'_docs/experiments/pallet_pose_selector_objective_audit_20261001_v1'
RAW=U.ROOT/'data/pallet/results/pallet_pose_selector_objective_audit_20261001_v1'


def save(p,x):
    p=Path(p).resolve();assert p.is_relative_to(DOC) or p.is_relative_to(RAW)
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:f.write(x if isinstance(x,str) else json.dumps(U.clean(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    publication=U.read(U.DOC/'PUBLICATION_MANIFEST.json')
    expected={b['path']:b for b in publication['files']}
    def public(name):
        path=U.DOC/name;b=expected[str(path.relative_to(U.ROOT))];U.verify(b);return U.read(path)
    gate=public('SOURCE_VAL_GATE.json');assert gate['complete'] and not gate['PASS'] and not gate['real_routing_authorized']
    train_gate=public('SOURCE_TRAIN_GATE.json');assert train_gate['PASS']
    lock=public('SOURCE_VAL_ROUTING_LOCK.json');feature_lock=public('SOURCE_FEATURE_LOCK.json')
    contract=public('SOURCE_CONTRACT.json');complete=public('TRAINING_COMPLETE.json');assert complete['fit_count']==6
    for b in [gate['metrics'],lock['choices'],feature_lock['poses'],feature_lock['features'],feature_lock['metadata'],*gate['source_reference_bindings']]:U.verify(b)
    U.verify(gate['source_history_binding'])
    assert gate['routing_lock']==U.bind(U.DOC/'SOURCE_VAL_ROUTING_LOCK.json')
    scale=np.array([train_gate['source_scale']['sT_cm'],train_gate['source_scale']['sR_deg']])
    assert np.isfinite(scale).all() and (scale>0).all()
    save(DOC/'VAL_ORACLE_PROTOCOL.json',dict(created_at=U.now(),phase='Post-fit source-VAL diagnostic only',
        previous_publication=U.bind(U.DOC/'PUBLICATION_MANIFEST.json'),previous_failed_gate=U.bind(U.DOC/'SOURCE_VAL_GATE.json'),
        feature_lock=U.bind(U.DOC/'SOURCE_FEATURE_LOCK.json'),source_contract=U.bind(U.DOC/'SOURCE_CONTRACT.json'),
        scope='All1024 existing proper-C2 source VAL. No source split/model/runtime/frozen decisions changed.',
        queries=['Same TRAIN-scale min-max whole-pose oracle opportunity on VAL for all3seeds',
            'Frozen learned route versus oracle candidate regret; exact target mismatch and cost-margin severity',
            'Preserve previous 45-gate result and zero new real routing'],
        scale=scale.tolist(),code=U.bind(__file__),new_fits=0,new_image_forwards=0,real_reference_reads=0,
        warning='VAL is reused for diagnosis; this does not constitute an independent holdout or new learned result. No best-seed selection.'))
    with np.load(U.ROOT/gate['metrics']['path']) as z:
        ids=z['ids'].tolist();indices=z['source_table_index'];saved={m:z[m] for m in z['models'].tolist()}
    assert len(ids)==1024 and set(ids)==set(contract['fit_eligibility']['eligible_ids']['VAL'])
    geometry=next(b for b in gate['source_reference_bindings'] if b['path'].endswith('GEOMETRY_SIDETABLE.npz'))
    with np.load(U.ROOT/geometry['path']) as z:
        assert z['stems'][indices].tolist()==ids
        reference_R=z['R'][indices]@F.S;reference_t=z['t'][indices]
    poses=U.read(U.ROOT/feature_lock['poses']['path'])['records']
    metadata=U.read(U.ROOT/feature_lock['metadata']['path']);positions={r['id']:j for j,r in enumerate(metadata)}
    with np.load(U.ROOT/feature_lock['features']['path']) as z:
        valid={m:z[m+'_valid'][[positions[fid] for fid in ids]] for m in U.MODELS}
    error={m:np.full((1024,2,2),np.inf) for m in U.MODELS}
    parity=0
    for m in U.MODELS:
        for j,fid in enumerate(ids):
            h={h['name']:h['pose'] for h in poses[m][fid]['hypotheses']}
            for k,name in enumerate(F.HYP):
                if valid[m][j,k]:error[m][j,k]=F.error_pair(h[name],reference_R[j],reference_t[j])
            op=F.error_pair(poses[m][fid]['GEO_pose'],reference_R[j],reference_t[j])
            np.testing.assert_array_equal(op,saved[m+'_GEO'][j]);parity+=1
    routes=U.read(U.ROOT/lock['choices']['path'])['records'];rows=[];summaries={};oracles={}
    def oracle(errors,names):
        picks=[F.choose_cost(e,scale,names)[0] for e in errors]
        selected=np.array([e[p] if p>=0 else [np.inf,np.inf] for e,p in zip(errors,picks)])
        return np.array(picks),selected
    names0=['R0:'+h for h in F.HYP];base_picks,base=oracle(error['R0'],names0)
    oracles['R0_ONLY']=F.summary(base)
    for seed in [1,2,3]:
        m=f'DIVERSE251_s{seed}';name=f'UNION_s{seed}';names=names0+[m+':'+h for h in F.HYP]
        errors=np.concatenate([error['R0'],error[m]],axis=1);picks,best=oracle(errors,names)
        learned=np.array([routes[name][fid]['candidate_index'] for fid in ids]);assert (learned>=0).all()
        chosen=np.array([e[p] for e,p in zip(errors,learned)]);np.testing.assert_array_equal(chosen,saved[name]);parity+=1024
        allcost=np.max(errors/scale,axis=-1);bestcost=allcost[np.arange(1024),picks];cost=allcost[np.arange(1024),learned]
        regret=cost-bestcost;assert np.isfinite(regret).all() and (regret>=0).all()
        difference=chosen-best
        summaries[name]=dict(oracle_gate_vs_R0_ONLY=F.gate(base,best,1.05),oracle=F.summary(best),learned=F.summary(chosen),
            target_disagreement=int(np.sum(learned!=picks)),positive_regret=int(np.sum(regret>0)),
            regret_mean=float(regret.mean()),regret_P90=float(np.quantile(regret,.9)),regret_max=float(regret.max()),
            regret_concentration_largest10_fraction=float(np.sort(regret)[-10:].sum()/regret.sum()) if regret.sum()>0 else 0.,
            learned_vs_oracle_T_lower_R_higher=int(np.sum((difference[:,0]<0)&(difference[:,1]>0))),
            learned_vs_oracle_R_lower_T_higher=int(np.sum((difference[:,1]<0)&(difference[:,0]>0))),
            selection_counts={n:int(np.sum(learned==k)) for k,n in enumerate(names)})
        for j,fid in enumerate(ids):rows.append(dict(id=fid,seed=seed,learned=names[learned[j]],oracle=names[picks[j]],
            learned_T_cm=chosen[j,0],learned_R_deg=chosen[j,1],oracle_T_cm=best[j,0],oracle_R_deg=best[j,1],
            learned_cost=cost[j],oracle_cost=bestcost[j],regret=regret[j],diagnostic_only=True))
    out=io.StringIO(newline='');writer=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    save(DOC/'VAL_ORACLE_ROWS.csv',out.getvalue())
    save(DOC/'VAL_ORACLE.json',dict(complete=True,created_at=U.now(),protocol=U.bind(DOC/'VAL_ORACLE_PROTOCOL.json'),
        previous_gate_preserved=U.bind(U.DOC/'SOURCE_VAL_GATE.json'),source_reference_bindings=gate['source_reference_bindings'],
        frames=1024,parity_pose_checks=parity,scale=scale.tolist(),baseline_oracle=oracles['R0_ONLY'],seeds=summaries,
        rows=U.bind(DOC/'VAL_ORACLE_ROWS.csv'),new_fits=0,new_image_forwards=0,new_real_routes=0,
        goal_achieved=False,real_reference_reads=0,read_paths=sorted(set(READS))))
    print('VAL_ORACLE_DIAGNOSTIC_COMPLETE',summaries,flush=True)


if __name__=='__main__':main()

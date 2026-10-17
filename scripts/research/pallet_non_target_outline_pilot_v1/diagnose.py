"""Post-seal explanatory joins only; these reference flags never select a pose."""
import argparse
from collections import defaultdict
from pathlib import Path
import json

from .evaluate import sha, read, rows, csvwrite, metric, write


def run(run_folder,neutral_package,state_private,output):
    folder,package,state,out=map(Path,(run_folder,neutral_package,state_private,output))
    seal=read(folder/'PREDICTION_SEAL.json')
    for name,value in seal['artifacts'].items():
        if sha(folder/name)!=value:raise ValueError('Prediction seal mismatch')
    pkg=read(package);frames=rows(package.parent/pkg['frames']['path'])
    candidates=rows(package.parent/pkg['candidates']['path'])
    costs={(c['id'],c['stage'],c['family'],c['candidate_id']):c for c in rows(folder/'CANDIDATE_COSTS.jsonl.gz')}
    refs_path=state/'score_v2/FIXED_REFERENCES.json'
    if sha(refs_path)!=read(state/'score_v2/SCORE_SEAL.json')['artifacts']['FIXED_REFERENCES.json']:raise ValueError('Reference changed')
    refs=read(refs_path)['references'];frame={f['id']:f for f in frames};byid=defaultdict(list)
    measured={}
    for c in candidates:
        f=frame[c['id']];pose=dict(available=True,centroid=c['pose']['centroid'],R_physical=c['pose']['R_physical'])
        measured[(c['id'],c['candidate_id'])]=metric(pose,refs[f['partition']][c['id']]);byid[c['id']].append(c)
    predictions=rows(folder/'PREDICTIONS.jsonl.gz');g10=[];g13=[];components=[]
    for p in predictions:
        fid,variant=p['id'],p['variant'];families=p['stage_winner_ids']['G10']
        for fam,cid in families.items():
            if cid is None:continue
            cost=costs[(fid,'G10',fam,cid)];m=measured[(fid,cid)]
            g10.append(dict(id=fid,variant=variant,family=fam,candidate_id=cid,score=cost['scores'][variant],
                frozen_family=p['selected_family']==fam,translation_cm=m['translation_cm'],rotation_deg=m['rotation_deg'],
                reference_success_after_seal=m['success_le5']))
        fam=p['selected_family'];selected=p['candidate_id']
        own=[c for c in byid[fid]if c['stage']=='G13'and c['family']==fam]
        good=[c for c in own if measured[(fid,c['candidate_id'])]['success_le5']]
        # This is a post-hoc diagnostic comparator, never the selected output.
        good.sort(key=lambda c:(costs[(fid,'G13',fam,c['candidate_id'])]['scores'][variant],c['original_record']['initialization'],c['original_outcome_order']))
        best=good[0]if good else None;scost=costs[(fid,'G13',fam,selected)]if selected else None
        bcost=costs[(fid,'G13',fam,best['candidate_id'])]if best else None
        g13.append(dict(id=fid,variant=variant,family=fam,selected_candidate_id=selected,
            selected_reference_success=measured[(fid,selected)]['success_le5']if selected else False,
            selected_score=scost['scores'][variant]if scost else None,
            successful_retained_same_family=len(good),diagnostic_success_candidate_id=best['candidate_id']if best else None,
            diagnostic_success_score=bcost['scores'][variant]if best else None,
            success_minus_selected_score=bcost['scores'][variant]-scost['scores'][variant]if best and scost else None,
            postseal_oracle_diagnostic_only=True))
        for role,cost in [('ACTUAL_SELECTED',scost),('POSTSEAL_SUCCESS_COMPARATOR',bcost)]:
            if cost is None:continue
            parts=cost['original_weighted_components']
            outline=cost['original_outline_cost']if variant=='BASELINE_E'else cost['support_outline_cost']if variant=='SUPPORT_AWARE_RERANK'else cost['mass_control_outline_cost']
            components.append(dict(id=fid,variant=variant,role=role,candidate_id=cost['candidate_id'],family=fam,
                score=cost['scores'][variant],full_BI=parts['full_BI'],partial_normal=parts['partial_normal'],prior=parts['prior'],
                outline=outline,postseal_reference_comparator_not_input=True))
    out.mkdir(parents=True,exist_ok=False)
    csvwrite(out/'G10_FAMILY_DIAGNOSTICS.csv',g10);csvwrite(out/'G13_SELECTION_DIAGNOSTICS.csv',g13)
    csvwrite(out/'G13_SELECTED_COMPONENTS.csv',components)
    write(out/'DIAGNOSTIC_RECEIPT.json',dict(prediction_seal_sha256=sha(folder/'PREDICTION_SEAL.json'),
        fixed_candidates_scored_after_seal=len(candidates),new_selection=False,new_NN=False,new_pose=False,
        references_used_only_after_seal=True,artifacts={f.name:sha(f)for f in out.iterdir()}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('run-folder','neutral-package','state-private','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();run(a.run_folder,a.neutral_package,a.state_private,a.output)

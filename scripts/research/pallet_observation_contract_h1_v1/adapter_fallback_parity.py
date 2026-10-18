"""Read-only postseal technical adapter parity; no model or reference input.

The immutable old predictor snapshot reconstructs the executed finite path.
Its complete returned dictionaries and the repaired adapter are both compared
with all already-sealed active/trial outputs. No prediction file is rewritten.
"""
from __future__ import annotations
import argparse,importlib.util,json,resource,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from scripts.research.pallet_boundary_action_repair_v1 import action_data as D
from scripts.research.pallet_risk_information_v1.common import read,rows,sha,write
from . import predict
from .policy import decode_tau

def run(root,features,original_source,output):
    root,features,original_source,output=map(Path,(root,features,original_source,output));start,cpu=time.monotonic(),time.process_time()
    source=Path(predict.__file__);manifest=root/'PREDICTION_MANIFEST.json';mm=read(manifest)
    if sha(original_source)!=mm['source_sha256']['predict.py']:raise ValueError('Original executed adapter bytes missing')
    if mm['artifacts']['PREDICTIONS.jsonl.gz']!=sha(root/'PREDICTIONS.jsonl.gz')or mm['real_features_sha256']!=sha(features):raise ValueError('Frozen prediction/input binding differs')
    policy=read(root/'POLICY_CALIBRATION.json');ledger=list(rows(root/'PREDICTIONS.jsonl.gz'));fs=list(rows(features))
    if len(ledger)!=238 or [r['id']for r in ledger]!=[r['id']for r in fs]:raise ValueError('Original ordered238 rows required')
    fixed_paths=[manifest,root/'PREDICTIONS.jsonl.gz',root/'POLICY_CALIBRATION.json',root/'POLICY_CONFIRMATION.json',root/'POLICY_SPLIT_LOCK_v2.json',root/'EXPERIMENT_PLAN.json',features]
    evseal=root/'evaluation_v1/EVALUATION_SEAL.json';ee=read(evseal);fixed_paths.extend([evseal]+[evseal.parent/n for n in ee['artifacts']])
    before={str(p):sha(p)for p in fixed_paths}
    spec=importlib.util.spec_from_file_location('scripts.research.pallet_observation_contract_h1_v1._frozen_predict_snapshot',original_source)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    checks=dict(actual_rows=0,archived_H1_exact_pose_index_bindings=0,bitwise_finite_score_arrays=0,original_adapter_whole_dict_matches=0,repaired_adapter_whole_dict_matches=0)
    def forbidden(*args,**kwargs):raise AssertionError('NN/weight loading is prohibited in stored-output parity')
    torch.set_num_threads(1)
    with patch.object(torch.nn.Module,'_call_impl',forbidden),patch.object(torch,'load',forbidden):
        for frame,row in zip(fs,ledger):
            entries=D.actions(frame);pred=row['predictions'];hs,rs=predict.bind_scores(frame,entries,pred);oh,orr=old.bind_scores(frame,entries,pred)
            checks['actual_rows']+=1
            for seed in predict.SEEDS:
                hh=pred['D1_H1_MLP_s'+str(seed)];index=predict.archived_h1_index(entries,hh)
                if index is None or entries[index]['actual_pose']!=hh['actual_pose']:raise ValueError('Actual archived H1 index missing')
                checks['archived_H1_exact_pose_index_bindings']+=1
                for a,b in((hs[seed],oh[seed]),(rs[seed],orr[seed])):
                    if not np.isfinite(a).all()or not np.array_equal(a,b):raise ValueError('Finite actual score array changed')
                    checks['bitwise_finite_score_arrays']+=1
                pp=policy['seeds'][str(seed)]
                for group,key,trial in(('H1_GUARDED_SWITCH','active_tau',False),('H1_GUARD_CALIBRATED_TRIAL','calibrated_trial_tau',True)):
                    expected=pred[group+'_s'+str(seed)];tau=decode_tau(pp[key])
                    original=old.guarded_entry(frame,entries,oh[seed],orr[seed],tau,pp['status'],trial)
                    repaired=predict.guarded_entry(frame,entries,hs[seed],rs[seed],tau,pp['status'],trial,h1_selected_index=index)
                    if original!=expected or repaired!=expected:raise ValueError('Finite original/repaired whole dictionary parity failed: '+row['id'])
                    checks['original_adapter_whole_dict_matches']+=1;checks['repaired_adapter_whole_dict_matches']+=1
    preserved={p:sha(p)==h for p,h in before.items()}
    if not all(preserved.values()):raise ValueError('Immutable campaign bytes changed during read-only parity')
    result=dict(schema='ARCHIVED_H1_FALLBACK_POSTSEAL_TECHNICAL_AMENDMENT_V1',status='PASS',
        scope='Connect existing select_guarded h1_selected_index fallback to exact archived H1 action when scores/context/alternative are missing or nonfinite; finite order/pose contradictions still rejected.',
        original_source=dict(path=str(original_source),sha256=sha(original_source),bound_prediction_manifest_sha256=sha(manifest)),
        repaired_source=dict(path=str(source),sha256=sha(source)),
        unchanged_policy_source_sha256=sha(Path(predict.__file__).with_name('policy.py')),
        fixture_receipt=dict(path=str(root/'execution/archived_H1_fallback_contract_tests_v1/RECEIPT.json'),sha256=sha(root/'execution/archived_H1_fallback_contract_tests_v1/RECEIPT.json')),
        checks=checks,all_original_scientific_inputs_and_output_bytes_preserved=True,immutable_file_hashes=before,immutable_file_preservation=preserved,
        no_prediction_or_evaluation_regeneration=True,new_NN_forwards=0,weights_loaded=0,new_reference_decodes=0,new_PnP_or_optimizer_fits=0,
        current_source_does_not_replace_original_execution_source=True,adapter_scope='Future unavailable-score paths; executed actual finite outputs are unchanged. No new performance evaluation.',
        source_sha256=sha(__file__),cost=dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,threads=1))
    write(output,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--original-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=run(a.run_dir,a.features,a.original_source,a.output);print(json.dumps({k:r[k]for k in('status','checks','cost')}))

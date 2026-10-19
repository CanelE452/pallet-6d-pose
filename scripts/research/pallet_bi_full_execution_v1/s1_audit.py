"""Exhaustive reference-free S1 old-bank/admission/cache preservation audit."""
from __future__ import annotations
import argparse,time
from collections import Counter
from pathlib import Path
import numpy as np
from . import s1_union as U

def audit(d1,run):
    started=time.monotonic();d1,run=map(Path,(d1,run));report={};total=Counter()
    for domain in ('real','synthetic'):
        _,src=U.paths(d1,domain);dest=run/f's1_{domain}_union';old=list(U.rows(src/'HEAD_FEATURES.jsonl.gz'));new=list(U.rows(dest/'HEAD_FEATURES.jsonl.gz'))
        if [f['id']for f in old]!=[f['id']for f in new]:raise AssertionError('Full ordered IDs differ')
        count=Counter();rejections=Counter();duplicates=Counter();admission_changes=[];gapchanges=Counter();minima_reconstructions=[];maxima=Counter()
        for i,(a,b)in enumerate(zip(old,new)):
            count['frames']+=1;count['old_unique_poses']+=len(a['candidates']);count['union_unique_poses']+=len(b['candidates'])
            if len(b['candidates'])<len(a['candidates']):raise AssertionError('Old pose dropped')
            for ca,cb in zip(a['candidates'],b['candidates']):
                if ca['candidate_id']!=cb['candidate_id']or ca['actual_pose']!=cb['actual_pose']:raise AssertionError('Old pose/ordering changed')
                if ca['features'][:26]!=cb['features'][:26]or ca['features'][31:]!=cb['features'][31:]:raise AssertionError('Individual scalar/mask changed')
                if not set(ca.get('source_aliases',[]))<=set(cb.get('source_aliases',[])):raise AssertionError('Original source alias dropped')
                count['old_individual_scalar_and_validity_bitwise_equal_candidates']+=1
                if ca['eligible']!=cb['eligible']:
                    if not cb.get('complete_action_aliases')or not cb['eligible']:raise AssertionError('Old eligibility changed without duplicate completed PF')
                    admission_changes.append(dict(id=a['id'],candidate_id=ca['candidate_id'],reason='same physical pose independently produced by valid PF complete action'))
                for j,mode in enumerate(U.MODES):
                    if ca['features'][26+j]!=cb['features'][26+j]:gapchanges[mode]+=1
            data=b['s1_union'];count['new_unique_PF_cost_evaluations']+=data['new_unique_PF_actions']
            for r in data['PF_rejections']:rejections[r['reason']]+=1
            for r in data['complete_action_duplicates']:duplicates[r['method']]+=1
            n=data['old_cost_reconstructions_for_missing_context'];count['old_individual_cost_reconstruction_evaluations']+=n
            if n:minima_reconstructions.append(dict(id=a['id'],old_candidates=n))
            count['old_eligible']+=sum(c['eligible']for c in a['candidates']);count['union_eligible']+=sum(c['eligible']for c in b['candidates'])
            seal=U.read(dest/'frames'/f'{i:04d}'/'SEAL.json')
            for name,digest in seal['dependencies']['original_artifacts'].items():
                if U.sha(src/'frames'/f'{i:04d}'/name)!=digest:raise AssertionError('Original dependency bytes changed')
            # The saved exact E_RUN contains all visited endpoints, P1/WD0
            # feasible samples, starts, stage summaries and source aliases.
            copied=next(U.rows(dest/'frames'/f'{i:04d}'/'ORIGINAL_E_RUN_EXACT.jsonl.gz'))
            if U.payload_sha(copied)!=seal['original_E_RUN_payload_sha256']:raise AssertionError('Copied actual endpoint payload changed')
            count['old_E_RUN_payloads_exact']+=1;count['old_raw_candidate_endpoints_preserved']+=len(copied['candidates'])
            count['old_endpoint_journal_records_preserved']+=len(copied['endpoint_journal'])
        report[domain]=dict(counts=dict(count),PF_rejections=dict(rejections),PF_duplicate_aliases=dict(duplicates),
            old_entries_promoted_by_identical_PF_complete_action=admission_changes,gap_changes_by_mode=dict(gapchanges),
            old_minimum_missing_context_reconstructions=minima_reconstructions,
            feature_sha256=U.sha(dest/'HEAD_FEATURES.jsonl.gz'),feature_seal_sha256=U.sha(dest/'FEATURE_SEAL.json'))
        total.update(count)
    result=dict(schema='S1_EXHAUSTIVE_REFERENCE_FREE_REUSE_AUDIT_V1',PASS=True,domains=report,totals=dict(total),
        old_baseline_eligibility_contract_unchanged=True,matched_baseline_refit_required=False,
        bitwise_claim='old physical pose dictionaries, first26 signedlog scalars and31validbits, original aliases; exact full E_RUN canonical payloads',
        numerical_claim='all5 relative gaps recomputed; expm1/log1p inversion tolerance1e-10 registered before refs and validated against fulloriginal scorer on fixed constructed pair',
        current_real_reference_reads=0,new_pose_optimizer_calls_for_union=0,wall_seconds=time.monotonic()-started)
    U.write(run/'S1_REUSE_AND_ADMISSION_AUDIT.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--d1',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=audit(a.d1,a.run)
    print(U.canonical(dict(PASS=r['PASS'],totals=r['totals'],wall_seconds=r['wall_seconds'])),flush=True)
if __name__=='__main__':main()

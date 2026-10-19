"""Postseal synthetic positive-candidate coverage by frozen proxy status.

Only previously saved training/validation labels are consumed. No pose,
reference projection, matcher, fit, forward, or hyperparameter search is run.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from pathlib import Path
from . import image_batch as I,material_assets as A


def _labels(folder):
    folder=Path(folder);seal=I.read(folder/'LABEL_SEAL.json');path=folder/'ACTION_TARGETS.jsonl.gz'
    if not seal['complete']or seal['rows']!=1003 or seal['labels_sha256']!=A.sha(path):
        raise ValueError('Complete original1003 label binding required')
    rows={row['id']:row for row in I.rows(path)}
    if len(rows)!=1003:raise ValueError('Unique original1003 label IDs required')
    return rows,dict(path=str(path),sha256=A.sha(path),label_seal_sha256=A.sha(folder/'LABEL_SEAL.json'))


def _positive(row):
    return dict(any_labelled_bank_candidate_positive=any(c['success_5cm5deg']for c in row['candidate_metrics']),
        any_eligible_positive=any(c['eligible']and c['success_5cm5deg']for c in row['candidate_metrics']),
        any_H1_action_positive=any(c['success_5cm5deg']for c in row['action_metrics']),
        N0_positive=bool(row['n0_metric']['success_5cm5deg']),
        positive_eligible_candidate_count=sum(c['eligible']and c['success_5cm5deg']for c in row['candidate_metrics']),
        positive_H1_action_count=sum(c['success_5cm5deg']for c in row['action_metrics']))


def report(run,output):
    run,output=map(Path,(run,output));finish=run/'s3_s4_postprocess/FINISH_RECEIPT.json'
    if not I.read(finish)['complete']:raise ValueError('All five arm prediction seals required')
    assetpath=run/'s3_s4_archive_report/PROXY_ASSET_STATUS_ROWS1241.jsonl.gz'
    assets={r['id']:r for r in I.rows(assetpath)if r['domain']=='synthetic'}
    base,basebinding=_labels(run/'s1_labels')
    if set(assets)!=set(base):raise ValueError('Asset and original S1 label identity sets differ')
    armresults={};bindings={}
    for arm in ('S1_D1_PLUS_PF',)+I.ARMS:
        labels,binding=(base,basebinding)if arm=='S1_D1_PLUS_PF'else _labels(run/'s3_s4_labels'/arm)
        if set(labels)!=set(assets):raise ValueError('Arm synthetic label IDs differ')
        bindings[arm]=binding;groups=defaultdict(Counter);changes=defaultdict(lambda:dict(gained_ids=[],lost_ids=[]))
        for ident,row in labels.items():
            asset=assets[ident]
            if row['split']!=asset['original_split']:raise ValueError('Original split mismatch')
            positive,original=_positive(row),_positive(base[ident])
            groupkeys=[(row['split'],asset['asset_status'],'ALL_NAMED_ASSETS'),
                (row['split'],asset['asset_status'],asset['named_source_asset'])]
            for key in groupkeys:
                groups[key]['frames']+=1;groups[key].update({k:int(v)for k,v in positive.items()})
                if positive['any_eligible_positive']and not original['any_eligible_positive']:changes[key]['gained_ids'].append(ident)
                if original['any_eligible_positive']and not positive['any_eligible_positive']:changes[key]['lost_ids'].append(ident)
        armresults[arm]=[dict(original_split=k[0],asset_status=k[1],named_source_asset=k[2],**dict(v),
            eligible_coverage_rate=v['any_eligible_positive']/v['frames'],
            gain_vs_S1_count=len(changes[k]['gained_ids']),loss_vs_S1_count=len(changes[k]['lost_ids']),**changes[k])
            for k,v in sorted(groups.items())]
    result=dict(schema='S3_S4_POSTSEAL_SYNTHETIC_ASSET_STRATUM_POSITIVE_COVERAGE_V1',complete=True,
        arms=armresults,label_bindings=bindings,asset_status_rows_sha256=A.sha(assetpath),
        finish_receipt_sha256=A.sha(finish),report_source_sha256=A.sha(Path(__file__)),
        reference_scope='saved original synthetic train768/validation235 labels only; no real references decoded',
        candidate_scope='saved feature-bank candidate_metrics and eligible H1 actions; full raw generated/guard-rejected endpoint attribution is a separate root post_bank analysis',
        all_validation_uses_missing_named_GLB_proxy=True,present_named_USD_stratum_absent_from_validation=True,
        coverage_is_posthoc_descriptive_not_new_independent_test=True,new_production_fits_renders_forwards_or_pose_updates=0,
        current_real_reference_reads=0,threshold='original joint translation<=5cm and C2-aware rotation<=5deg labels')
    output.mkdir(parents=True,exist_ok=True);I.write(output/'SYNTHETIC_ASSET_POSITIVE_COVERAGE.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(I.canonical(report(a.run,a.output)),flush=True)

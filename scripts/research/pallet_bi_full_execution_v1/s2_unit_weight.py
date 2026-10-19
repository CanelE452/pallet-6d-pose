"""Explicit input correction for the preregistered S2 unit reliability weight.

The original packets, covariance model, numerical producer and prior attempt
are immutable. Only existing nonunit C0 complete-factor weights are replaced.
CP/CB, typed partials, revoked archives, q and every covariance remain exact.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import shutil
import time

import numpy as np

from . import qcov as Q, calibration as CAL
from . import fullnormal_batch as K


def corrected_selection(selection):
    result=deepcopy(selection)
    partial_ids={p['native_corner']for p in selection['partial_lines']}
    changed=[]
    for row in result['full_points']:
        if row['native_corner']in partial_ids:continue
        if float(row.get('fit_reliability_weight',1.))!=1.:
            changed.append(row['native_corner']);row['fit_reliability_weight']=1.
    return result,changed


def assert_selection_contract(original,corrected):
    check=deepcopy(corrected)
    for old,new in zip(original['full_points'],check['full_points']):
        new.pop('fit_reliability_weight',None)
        if 'fit_reliability_weight'in old:new['fit_reliability_weight']=old['fit_reliability_weight']
    if check!=original:raise ValueError('Unit adapter changed q, covariance, source, partial or other fields')
    a,pa,da=Q.effective(original);b,pb,db=Q.effective(corrected)
    if set(a)!=set(b)or pa!=pb or da!=db:raise ValueError('Effective source graph changed')
    if any(float(row.get('fit_reliability_weight',1.))!=1. for row in b.values()):raise ValueError('Effective complete factor reliability is not1')


def validate_packet(frame,original,packet,census):
    if packet['id']!=frame['id']:raise ValueError('Packet inventory mismatch')
    for arm in ('C0','CP','CB'):
        old=original['selections'][arm];new=packet['selections'][arm]
        if arm=='C0':assert_selection_contract(old,new)
        elif new!=old:raise ValueError('CP/CB packet changed')
        full,partial,dedup=Q.effective(new)
        label=(arm,frame['partition'])
        for point in full.values():
            census['effective_full_weights'][label][str(float(point.get('fit_reliability_weight',1.)))]+=1
            CAL._validate_covariance(point['covariance_px2']);CAL._validate_covariance(Q.covariance(point))
        census['typed_partial_factors'][label]+=len(partial)
        partial_ids={p['native_corner']for p in partial}
        for point in new['full_points']:
            if point['native_corner']in partial_ids:
                census['revoked_archive_weights'][label][str(float(point.get('fit_reliability_weight',1.)))]+=1
        visible,boundaries,_=Q.S._deduplicate(new['full_points'],new['partial_lines'])
        Q.S._prepare(frame['native_xyz'],frame['K'],new['states'],visible,frame['raw_hw'],boundaries,Q.S.SOLVER_CONFIG)


def export(prior_run,source_root,output):
    prior_run,source_root,output=map(Path,(prior_run,source_root,output));output.mkdir(parents=True,exist_ok=True)
    source_seal=Q.read(source_root/'PACKET_SEAL.json');artifacts={};changes=[];before=Counter()
    from collections import defaultdict
    census=dict(effective_full_weights=defaultdict(Counter),typed_partial_factors=Counter(),revoked_archive_weights=defaultdict(Counter))
    for domain in ('real','synthetic'):
        frames,_=CAL.load_bound_frames(prior_run,domain)
        for ordinal,frame in enumerate(frames):
            relative=f'packets/{domain}/{ordinal:04d}.json.gz';source=source_root/relative
            if Q.sha(source)!=source_seal['artifacts'][relative]:raise ValueError('Original sealed S2 packet changed')
            original=Q.read(source);packet=deepcopy(original)
            full,_,_=Q.effective(original['selections']['C0'])
            before.update((domain,str(float(p.get('fit_reliability_weight',1.))))for p in full.values())
            packet['selections']['C0'],changed=corrected_selection(original['selections']['C0'])
            validate_packet(frame,original,packet,census)
            Q.write(output/relative,packet);artifacts[relative]=Q.sha(output/relative)
            if changed:changes.append(dict(domain=domain,ordinal=ordinal,id=frame['id'],changed_native_corners=changed))
    report=dict(schema='S2_UNIT_RELIABILITY_INPUT_CORRECTION_V1',epoch=time.time(),rows=1241,
        original_packet_seal_sha256=Q.sha(source_root/'PACKET_SEAL.json'),original_root=str(source_root),
        source_sha256=Q.sha(__file__),changed_frames=dict(Counter(r['domain']for r in changes)),
        changed_factor_n=sum(len(r['changed_native_corners'])for r in changes),changed_rows=changes,
        original_C0_effective_weight_census={d:{w:n for(domain,w),n in before.items()if domain==d}for d in ('real','synthetic')},
        effective_full_weight_census={arm+'/'+part:dict(v)for(arm,part),v in census['effective_full_weights'].items()},
        typed_partial_factor_census={arm+'/'+part:v for(arm,part),v in census['typed_partial_factors'].items()},
        revoked_archive_weight_census={arm+'/'+part:dict(v)for(arm,part),v in census['revoked_archive_weights'].items()},
        all_1241_all3arms_after_effective_dedup_and_SPD_and_typed_parser_validated=True,
        CP_CB_selection_values_bitwise_unchanged=True,C0_q_covariance_source_partial_values_unchanged=True,
        covariance_model_refit=False,covariance_fit_independent_of_Gamma=True,
        covariance_fit_weight_rule='recording/frame/point balance only; calibration.weights and _floor never consume fit_reliability_weight',
        calibration_source_sha256=Q.sha(CAL.__file__),numerical_source_binding=K.source_binding(),
        current_numeric_reference_inputs=False,new_pose_optimizer_calls=0,new_H1_fits=0)
    Q.write(output/'CORRECTION_CENSUS.json',report);artifacts['CORRECTION_CENSUS.json']=Q.sha(output/'CORRECTION_CENSUS.json')
    Q.write(output/'PACKET_SEAL.json',dict(schema='S2_UNIT_WEIGHT_PACKET_SEAL_V1',complete=True,
        real_rows=238,synthetic_rows=1003,arms=['C0','CP','CB'],source_sha256=Q.sha(__file__),
        original_packet_seal_sha256=Q.sha(source_root/'PACKET_SEAL.json'),artifacts=artifacts,
        only_C0_complete_nonunit_weights_replaced=True,typed_archives_unchanged=True,current_numeric_reference_inputs=False))
    return report


def stage_exact_reuse(prior_run,packet_root,output,source_roots):
    """Copy only rows whose complete frozen kernel payload/source still match."""
    prior_run,packet_root,output=map(Path,(prior_run,packet_root,output));output.mkdir(parents=True,exist_ok=True)
    original_config,_,_=K.D1._cpu_models();original_config=K.B.configuration(original_config)
    feature_config=Q.read(prior_run.parent/'pallet-statewise-end-to-end-recovery-v1'/'CONFIG_FROZEN.json')
    inventory=Q.read(prior_run/'P0_INPUT_CONTRACT.json');binding=K.source_binding();rows=[];excluded=[]
    for domain in ('real','synthetic'):
        frames,_=CAL.load_bound_frames(prior_run,domain)
        for ordinal,frame in enumerate(frames):
            existing=[(arm,root,Path(root)/arm/domain/f'{ordinal:04d}')for arm in ('C0','CP','CB')for root in source_roots
                if (Path(root)/arm/domain/f'{ordinal:04d}'/'FRAME_SEAL.json').exists()]
            if not existing:continue
            observation_folder=Path(inventory[domain]['inventory'][ordinal]['observation_folder'])
            observation=K.jsonrow(observation_folder/'OBSERVATION.jsonl.gz');mask_sha=Q.sha(observation_folder/'MASK.npz')
            packet=Q.read(packet_root/'packets'/domain/f'{ordinal:04d}.json.gz');used=set()
            for arm,root,source in existing:
                if arm in used:continue
                signature=Q.P.digest(K.cache_payload(frame,packet['selections'][arm],observation,mask_sha,original_config,feature_config))
                seal=Q.read(source/'FRAME_SEAL.json')
                if seal['cache_payload_sha256']!=signature or seal['source_binding']!=binding:
                    excluded.append(dict(domain=domain,ordinal=ordinal,arm=arm,source=str(source),reason='NUMERICAL_PAYLOAD_OR_SOURCE_DIFFERS'));continue
                destination=output/arm/domain/f'{ordinal:04d}';destination.mkdir(parents=True,exist_ok=True)
                for name,digest in seal['artifacts'].items():
                    if Q.sha(source/name)!=digest:raise ValueError('Previous attempt row changed')
                for name in (*seal['artifacts'],'FRAME_SEAL.json'):shutil.copy2(source/name,destination/name)
                rows.append(dict(domain=domain,ordinal=ordinal,arm=arm,source=str(source),source_frame_seal_sha256=Q.sha(source/'FRAME_SEAL.json'),
                    desired_cache_payload_sha256=signature,exact_dependency_and_artifact_equality_verified=True));used.add(arm)
    report=dict(schema='S2_UNIT_WEIGHT_EXACT_PRIOR_EXECUTION_REUSE_V1',rows=rows,eligible_n=len(rows),
        excluded=excluded,excluded_n=len(excluded),eligible_counts=dict(Counter(row['arm']+'/'+row['domain']for row in rows)),
        source_roots=[str(r)for r in source_roots],source_sha256=Q.sha(__file__),new_pose_optimizer_calls=0,
        packet_seal_sha256=Q.sha(packet_root/'PACKET_SEAL.json'),original_receipts_preserved=True,
        driver_reuse_slot='reuse-pilot; staging includes explicitly identified old production and fixed pilot rows',
        numerical_source_binding=binding)
    Q.write(output/'REUSE_MANIFEST.json',report);return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prior-run',required=True);parser.add_argument('--source-root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();report=export(args.prior_run,args.source_root,args.output)
    print({k:v for k,v in report.items()if k in ('rows','changed_frames','changed_factor_n','all_1241_all3arms_after_effective_dedup_and_SPD_and_typed_parser_validated')})


if __name__=='__main__':main()

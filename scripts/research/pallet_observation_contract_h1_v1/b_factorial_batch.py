"""Saved-reference-only B query factorial, never production candidates/features."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import gzip
import json
from pathlib import Path
import resource
import time
import numpy as np
from scripts.research.pallet_risk_information_v1.common import read,rows,sha,write
from .b_query import QUERY_VARIANTS,factorial_query


def plain(value):
    if isinstance(value,(np.bool_,np.integer)):return value.item()
    if isinstance(value,(float,np.floating)):return float(value)if np.isfinite(value)else None
    if isinstance(value,np.ndarray):return plain(value.tolist())
    if isinstance(value,dict):return {k:plain(v)for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v)for v in value]
    return value


def fixed_profiles(bundle,expected_ids):
    lookup={};eligible=set()
    for group in (bundle or{}).get('groups',[]):
        for q in group['queries']:
            if q['support_id']in lookup and q!=lookup[q['support_id']]:raise ValueError('Conflicting original profile ID')
            lookup[q['support_id']]=q
            if group['supported']and q['supported']:eligible.add(q['support_id'])
    if eligible!=set(expected_ids):raise ValueError('Original fixed query universe changed')
    return [lookup[sid]for sid in expected_ids]


def summaries(reports):
    out={}
    for variant in QUERY_VARIANTS:
        rr=[r for r in reports if r['query_variant']==variant];by=defaultdict(list)
        for r in rr:by[r['native_corner']].append(r)
        corners={str(k):dict(profile_count=len(v),all_profiles_finite=all(x['finite_support']for x in v),
            all_profiles_geometry_domain_supported=all(x['geometry_domain_joint_support']for x in v))for k,v in by.items()}
        out[variant]=dict(profiles=len(rr),finite_profiles=sum(x['finite_support']for x in rr),
            status_counts=dict(Counter(x['geometry_status']for x in rr)),
            query_domain_overrun_profiles=sum(x['query_domain_overrun']for x in rr),
            center_off_canvas_profiles=sum(not x['center_in_canvas']for x in rr),
            sample_line_off_canvas_profiles=sum(not x['sample_line_in_canvas']for x in rr),
            all_profiles_finite=all(x['finite_support']for x in rr)if rr else None,
            all_profiles_geometry_domain_supported=all(x['geometry_domain_joint_support']for x in rr)if rr else None,
            corner_groups=corners)
    return out


def frame_diagnosis(row,bundle,e_run):
    initial=np.asarray(bundle['initial_projection_xy'],float)if bundle else None
    spec=e_run['packet']['solver_input'];anchors={int(m['native_corner']):m for m in spec['visible']}
    expected=[p['support_id']for p in row['registered_phases'][0]['families'][0]['raw_physical_finite_support']['profiles']]
    queries=fixed_profiles(bundle,expected);families=[];q00_differences=[]
    for phase in row['registered_phases']:
        R=np.asarray(phase['reference_pose']['R_physical']);center=np.asarray(phase['reference_pose']['centroid'])
        for f in phase['families']:
            old=f['raw_physical_finite_support']['profiles']
            if [x['support_id']for x in old]!=expected:raise ValueError('Whole registered phase/family profile identity changed')
            uv=np.asarray(f['raw_physical_projection_native'],float);xyz=np.asarray(f['native_xyz']);origin=np.asarray(f['native_origin_mean']);Q=np.asarray(f['physical_registration_Q'])
            cam=(xyz-origin)@(R@Q.T).T+center;depth=cam[:,2]
            hom=cam@np.asarray(spec['K']).T;check_uv=hom[:,:2]/hom[:,2:]
            reports=[]
            for q,original in zip(queries,old):
                k=int(q['native_corner'])
                if k not in anchors:raise ValueError('Original observed B anchor missing from original packet')
                rr=factorial_query(q,initial,uv,anchors[k]['xy'],pixel_scale=bundle['pixel_scale'],image_hw=bundle['image_hw'],depth=depth)
                q00=rr[0]
                q00_differences.append(dict(status_equal=q00['geometry_status']==original['geometry_status'],
                    lambda_abs_difference=None if q00['lambda_segment']is None or original['lambda_segment']is None else abs(q00['lambda_segment']-original['lambda_segment'])))
                for r in rr:
                    r.update(original_corner_state=bundle['original_states'][k],original_neighbor_state=bundle['original_states'][int(q['native_neighbor'])],
                        original_profile_id_unchanged=True,observation_exists_originally=True,
                        changed_query_RGB_or_mask_sampled=False)
                reports.extend(rr)
            families.append(dict(phase_index=phase['phase_index'],family=f['family'],representation='ORIGINAL_RAW_REFERENCE_R_UNCHANGED',
                raw_projection_reexpression_max_abs=float(np.max(np.abs(check_uv-uv))),
                original_raw_backend_mapping_audit=f['physical_mapping_audit'],
                minimum_raw_depth_m=float(np.min(depth)),profiles=reports,summaries=summaries(reports)))
    if any(not x['status_equal']for x in q00_differences):raise ValueError('Q00 failed exact original finite predicate replay')
    anyphase={}
    for name in ('ORIGINAL_WD','SWAPPED_WD'):
        fs=[f for f in families if f['family']==name]
        anyphase[name]={q:dict(phase0_finite=fs[0]['summaries'][q]['all_profiles_finite'],
            any_whole_phase_finite=any(f['summaries'][q]['all_profiles_finite']for f in fs)if expected else None,
            any_whole_phase_geometry_domain_supported=any(f['summaries'][q]['all_profiles_geometry_domain_supported']for f in fs)if expected else None)
            for q in QUERY_VARIANTS}
    return dict(id=row['id'],split=row['split'],symmetry_order=row['symmetry_order'],original_B_input_present=bool(expected),
        original_profile_count=len(expected),original_profile_ids=expected,families=families,whole_phase_results=anyphase,
        Q00_original_status_replay_exact=True,Q00_max_lambda_difference=max((x['lambda_abs_difference'] or 0 for x in q00_differences),default=0.),
        diagnostic_only=True,automatic_performance_result=False,production_bank_insertions=0,new_pose_solves=0,new_optical_cost_evaluations=0)


def aggregate(records):
    result={}
    for split in ('all','train','validation'):
        rr=[r for r in records if split=='all'or r['split']==split];present=[r for r in rr if r['original_B_input_present']]
        groups={}
        for family in ('ORIGINAL_WD','SWAPPED_WD'):
            gs={}
            for q in QUERY_VARIANTS:
                cells=[f['summaries'][q]for r in present for f in r['families']if f['family']==family]
                phase0=[r['whole_phase_results'][family][q]['phase0_finite']for r in present]
                anyp=[r['whole_phase_results'][family][q]['any_whole_phase_finite']for r in present]
                original=[r['whole_phase_results'][family]['Q00_ORIGINAL']['any_whole_phase_finite']for r in present]
                flat=[p for r in present for f in r['families']if f['family']==family for p in f['profiles']if p['query_variant']==q]
                originalflat=[p for r in present for f in r['families']if f['family']==family for p in f['profiles']if p['query_variant']=='Q00_ORIGINAL']
                corners=[g['all_profiles_finite']for c in cells for g in c['corner_groups'].values()]
                gs[q]=dict(B_frame_denominator=len(present),phase0_frame_all_finite=sum(phase0),any_whole_phase_frame_all_finite=sum(anyp),
                    any_whole_phase_frame_geometry_domain_supported=sum(r['whole_phase_results'][family][q]['any_whole_phase_geometry_domain_supported']for r in present),
                    frame_recovery_vs_Q00_anyphase=sum(not a and b for a,b in zip(original,anyp)),frame_damage_vs_Q00_anyphase=sum(a and not b for a,b in zip(original,anyp)),
                    phase_expanded_profiles=len(flat),finite_profiles=sum(p['finite_support']for p in flat),
                    profile_recovery_vs_Q00=sum(not a['finite_support']and b['finite_support']for a,b in zip(originalflat,flat)),
                    profile_damage_vs_Q00=sum(a['finite_support']and not b['finite_support']for a,b in zip(originalflat,flat)),
                    phase_expanded_corner_groups=len(corners),all_profiles_finite_corner_groups=sum(corners),
                    status_counts=dict(Counter(p['geometry_status']for p in flat)),
                    query_domain_overrun_profiles=sum(p['query_domain_overrun']for p in flat),
                    center_off_canvas_profiles=sum(not p['center_in_canvas']for p in flat),sample_line_off_canvas_profiles=sum(not p['sample_line_in_canvas']for p in flat))
            groups[family]=gs
        result[split]=dict(frames=len(rr),B_input_frames=len(present),B_input_absent_frames=len(rr)-len(present),families=groups)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('diagnostic-rows','diagnostic-seal','cache','plan','output'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--diagnostic-GO',action='store_true');a=p.parse_args()
    if not a.diagnostic_GO:raise PermissionError('Root P2 diagnostic GO required')
    seal=read(a.diagnostic_seal)
    if not seal.get('complete')or sha(a.diagnostic_rows)!=seal['rows_sha256']:raise ValueError('Original1003 reference diagnostic changed')
    out=a.output;out.mkdir(parents=True,exist_ok=False);start,cpu=time.monotonic(),time.process_time()
    index={}
    for path in (a.cache/'frames').glob('*/FRAME_SEAL.json'):
        s=read(path)
        if s['id']in index:raise ValueError('Duplicate original frame ID')
        index[s['id']]=(path.parent,s)
    records=[];inputs={};csv_rows=[]
    for row in rows(a.diagnostic_rows):
        folder,s=index[row['id']];inputs[row['id']]={}
        for name,key in [('BOUNDARY_BUNDLE.jsonl.gz','boundary_bundle_sha256'),('E_RUN.jsonl.gz','e_run_sha256')]:
            if sha(folder/name)!=s[key]:raise ValueError('Original packet/observation bytes changed')
            inputs[row['id']][name]=s[key]
        bundle=next(rows(folder/'BOUNDARY_BUNDLE.jsonl.gz'));e=next(rows(folder/'E_RUN.jsonl.gz'))
        r=frame_diagnosis(row,bundle,e);records.append(r)
        for f in r['families']:
            for q in f['profiles']:
                q=plain(q)
                csv_rows.append(dict(id=r['id'],split=r['split'],phase_index=f['phase_index'],family=f['family'],representation=f['representation'],
                    **{k:json.dumps(v)if isinstance(v,(list,dict))else v for k,v in q.items()}))
    if len(records)!=1003 or sum(r['original_B_input_present']for r in records)!=673:raise ValueError('Original1003/673 fixed denominator changed')
    with gzip.open(out/'B_QUERY_FACTORIAL_ROWS.jsonl.gz','wt')as stream:
        for r in records:stream.write(json.dumps(plain(r),allow_nan=False)+'\n')
    with gzip.open(out/'B_QUERY_FACTORIAL.csv.gz','wt',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(plain(csv_rows))
    write(out/'B_QUERY_FACTORIAL_SUMMARY.json',dict(schema='B_REFERENCE_QUERY_FACTORIAL_V1',status='COMPLETE',groups=aggregate(records),
        Q00_replay_exact=all(r['Q00_original_status_replay_exact']for r in records),
        Q00_max_lambda_difference=max(r['Q00_max_lambda_difference']for r in records),
        new_optical_cost_evaluations=0,RGB_distributions_moved_or_reused_as_new_cost=0,production_bank_insertions=0,
        raw_reference_R_modified=False,original_strict_roundtrip_failures_preserved_in_input=True,
        independent_physical_edge_owner='UNVERIFIED',phase_units='whole original C1/C2 representations; never independent per-corner permutations',
        sampling_scope='Geometry-only query line canvas tests, no changed-location RGB/mask sample',
        statistical_units='Profiles/corners/expanded phases are clustered within original frames; not independent samples'))
    write(out/'B_QUERY_FACTORIAL_SEAL.json',dict(status='COMPLETE',rows=1003,B_input_frames=673,profile_variant_rows=len(csv_rows),
        original_diagnostic_rows_sha256=sha(a.diagnostic_rows),original_diagnostic_seal_sha256=sha(a.diagnostic_seal),
        plan_sha256=sha(a.plan),source_sha256={name:sha(Path(__file__).with_name(name))for name in ('b_query.py','b_factorial_batch.py')},
        original_frame_bindings=inputs,artifacts={path.name:sha(path)for path in out.iterdir()if path.is_file()},
        wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        new_solve_calls=0,new_PnP_calls=0,new_model_forward_calls=0,new_candidates=0,GT_diagnostic_only=True))


if __name__=='__main__':main()

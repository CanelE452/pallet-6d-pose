"""Three frozen extreme outputs: fixed-pose costs and finite-edge support.

No candidate generation, fitting or selector change. All pose solvers and
optimizers are poisoned before importing the original fixed-cost routines.
The segment parameter is diagnostic only and never changes an original cost.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import resource
import time
import numpy as np
from .common import read,rows,sha,write,canonical

CASES=(('B_FULL','wood_day_01:005249','B'),
       ('CTRL_EDGE_BOUNDARY','plastic_day_01:012545','edge'),
       ('AB_FULL','wood_184309:000432','AB'))


def prohibit_fitting():
    import cv2
    import scipy.optimize as opt
    attempted=[]
    def denied(name):
        def call(*args,**kwargs):
            attempted.append(name);raise RuntimeError('Forensic fixed-pose evaluation forbids fitting: '+name)
        return call
    names=('minimize','minimize_scalar','least_squares','leastsq','root','differential_evolution','basinhopping','dual_annealing')
    for name in names:
        if hasattr(opt,name):setattr(opt,name,denied('scipy.optimize.'+name))
    for name in ('solvePnP','solvePnPGeneric','solvePnPRansac','solvePnPRefineLM','solvePnPRefineVVS'):
        if hasattr(cv2,name):setattr(cv2,name,denied('cv2.'+name))
    return attempted,denied


def query_segments(bundle,projected,permutation):
    q=np.asarray(projected)[list(permutation)];records=[]
    cross=lambda a,b:float(a[0]*b[1]-a[1]*b[0])
    from . import boundary_model as B
    for group in bundle['groups']:
        if not group['supported']:continue
        unique={v['support_id']:v for v in group['queries'] if v['supported']}
        for p in unique.values():
            k,j=p['native_corner'],p['native_neighbor'];a,b=q[k],q[j];v=b-a
            c=np.asarray(p['center_xy']);n=np.asarray(p['outward_normal_xy']);den=cross(n,v)
            valid=not(np.linalg.norm(v)<=1e-10 or abs(den)<=1e-10*max(1.,np.linalg.norm(v)))
            offset,_=B._intersection_offset_and_gradient(p,q,bundle['pixel_scale'])
            if valid:
                t=cross(a-c,n)/den;d=cross(a-c,v)/den;point=c+d*n
                inside=bool(0.<=t<=1.);gap=float(np.linalg.norm(point-(a+t*v)))
                outside_distance=max(-t,t-1.,0.)*float(np.linalg.norm(v))
                offset_gap=abs(offset-d/bundle['pixel_scale'])
            else:t=d=point=inside=gap=outside_distance=offset_gap=None
            records.append(dict(support_id=p['support_id'],native_corner=k,native_neighbor=j,
                owner_corner=int(permutation[k]),owner_neighbor=int(permutation[j]),candidate_geometry_valid=valid,
                projected_endpoint_a=a.tolist(),projected_endpoint_b=b.tolist(),edge_length_px=float(np.linalg.norm(v)),
                query_center_xy=c.tolist(),normal_xy=n.tolist(),segment_parameter_t=t,inside_finite_segment=inside,
                intersection_xy=None if point is None else point.tolist(),outside_segment_distance_px=outside_distance,
                original_offset_base_px=offset,intersection_binding_gap_px=gap,original_offset_formula_gap=offset_gap))
    valid=[r for r in records if r['candidate_geometry_valid']];ts=[r['segment_parameter_t'] for r in valid]
    return dict(supported_unique_queries=len(records),valid_intersections=len(valid),
                finite_segment_inside=sum(r['inside_finite_segment'] for r in valid),
                finite_segment_outside=sum(not r['inside_finite_segment'] for r in valid),
                invalid_geometry=len(records)-len(valid),t_min=min(ts) if ts else None,t_max=max(ts) if ts else None,
                edge_length_px_min=min((r['edge_length_px'] for r in records),default=None),
                edge_length_px_max=max((r['edge_length_px'] for r in records),default=None),
                maximum_intersection_binding_gap_px=max((r['intersection_binding_gap_px'] for r in valid),default=None),
                queries=records)


def run(run_dir,*,forensic_GO=False):
    if forensic_GO is not True:raise PermissionError('Explicit root fixed-pose forensic GO required before IO')
    start,cpu=time.monotonic(),time.process_time();run=Path(run_dir);batch=run/'real_batch_all_available_v1'
    ps=read(run/'PREDICTION_SEAL.json');bs=read(batch/'BATCH_SEAL.json')
    if ps.get('complete') is not True or sha(batch/'BATCH_SEAL.json')!=ps['batch_seal_sha256']:raise ValueError('Original complete batch binding changed')
    source=Path(__file__).parent
    for name in ('solver.py','boundary_model.py','owner_model.py'):
        if sha(source/name)!=bs['source_sha256'][name]:raise ValueError('Frozen scientific source changed')
    config=read(run/'CONFIG_FROZEN.json');ip=run/'inputs_all_available_v1/PREDICTION_INPUTS238.jsonl.gz'
    if sha(ip)!=bs['input_sha256']:raise ValueError('Original input packet changed')
    wanted={fid for _,fid,_ in CASES};frames={r['id']:r for r in rows(ip) if r['id']in wanted}
    scored_path=run/'evaluation_v1/SCORED_ROWS.jsonl.gz';scored_sha=sha(scored_path)
    scored={r['id']:r for r in rows(scored_path) if r['id']in wanted}
    if sha(scored_path)!=scored_sha:raise ValueError('Evaluation snapshot changed while reading; retry explicitly')
    attempted,denied=prohibit_fitting()
    from . import solver as S
    S.generate_candidates=denied('solver.generate_candidates');S.solve_one=denied('solver.solve_one');S.ransac_control=denied('solver.ransac_control')
    records=[];bindings={};global_recheck_gap=0.
    for arm,fid,mode in CASES:
        i=ps['ordered_ids'].index(fid);folder=batch/'frames'/f'{i:04d}';seal=read(folder/'FRAME_SEAL.json')
        rp=folder/'RESULT.jsonl.gz';bp=folder/'BOUNDARY_BUNDLE.jsonl.gz'
        if sha(rp)!=seal['result_sha256'] or hashlib.sha256(canonical(frames[fid]).encode()).hexdigest()!=seal['input_row_sha256']:
            raise ValueError('Original per-frame result/input binding changed')
        row=next(rows(rp));bundle=next(rows(bp))
        if sha(bp)!=row['boundary']['bundle_sha256']:raise ValueError('Original fixed RGB query bundle changed')
        pred=row['predictions'][arm];family=pred['selected_family'];cid=pred['selected_candidate_id']
        if pred['actual_pose']!=scored[fid]['poses'][arm]:raise ValueError('Final evaluated pose differs from original selected candidate')
        idx=next(i for i,c in enumerate(row['banks'][mode]) if c['candidate_id']==cid)
        selected=row['banks'][mode][idx];stored_selected=row['bank_scores'][mode][idx]
        models=S.compile_frame(frames[fid],config,bundle=bundle);scale=models[family].ctx.scale
        def match(c,p):return bool(np.max(np.abs(np.asarray(c['R_physical'])-p['R_physical']))<=1e-7 and np.max(np.abs(np.asarray(c['centroid'])-p['centroid']))<=1e-7*scale)
        alternatives={'SELECTED':selected};stored={'SELECTED':stored_selected}
        for name in ('N0','E0'):
            p=scored[fid]['poses'][name];alternatives[name]=dict(candidate_id=name,R_physical=p['R_physical'],centroid=p['centroid'])
            j=next((j for j,c in enumerate(row['carried']) if match(c,p)),None)
            stored[name]=None if j is None else row['carried_scores'][j]
        details={}
        for name,candidate in alternatives.items():
            fresh=S.candidate_scores(candidate,models)
            if stored[name] is not None:
                gap=max(abs(fresh[f]['scores'][m]-stored[name][f]['scores'][m]) for f in models for m in ('fixed','A','B','AB','edge'))
                global_recheck_gap=max(global_recheck_gap,gap)
            else:gap=None
            model=models[family];p=S._native_pose(candidate,model.ctx)
            projected,depth,_,_=S.J.projection_and_jacobian(p,model.ctx.xyz,model.ctx.K,model.ctx.scale)
            hvalues=range(4) if mode=='AB' else (0,);segments={}
            for h in hvalues:
                perm=model.owner['permutations'][h] if mode=='AB' else tuple(range(8))
                x=query_segments(bundle,projected,perm)
                x['owner_permutation']=list(perm)
                if mode=='AB':
                    owner=fresh[family]['audits'][mode]['owner']
                    x['stored_owner_responsibility']=None if stored[name] is None else stored[name][family]['audits'][mode]['owner']['global_owner_responsibilities'][h]
                    x['reevaluated_owner_responsibility']=owner['global_owner_responsibilities'][h]
                    x['stored_jointly_feasible_h']=None if stored[name] is None else stored[name][family]['audits'][mode]['jointly_feasible_global_h'][h]
                segments[str(h)]=x
            details[name]=dict(actual_pose=deepcopy(candidate),stored_scores=stored[name],fixed_pose_reevaluated_scores=fresh,
                maximum_stored_cost_recheck_difference=gap,selected_family=family,
                projected_bbox_extent_px=np.ptp(projected,axis=0).tolist(),min_depth_m=float(np.min(depth)),max_depth_m=float(np.max(depth)),
                finite_edge_by_owner_h=segments,baseline_metric_snapshot=scored[fid]['metrics'].get(name if name!='SELECTED' else arm))
        origin=selected.get('original_native_family');solve=selected.get('solve',{})
        optimizer_context=None
        if mode=='AB' and origin in models:
            model=models[origin];p=S._native_pose(selected,model.ctx);q=S.J.projection_and_jacobian(p,model.ctx.xyz,model.ctx.K,model.ctx.scale)[0]
            h=solve['global_owner_h'];optimizer_context=dict(family=origin,conditional_optimizer_h=h,
                differs_from_selected_family=origin!=family,selected_family=family,
                segment_diagnosis=query_segments(bundle,q,model.owner['permutations'][h]),
                own_family_stored_AB_audit=stored_selected[origin])
        records.append(dict(id=fid,partition=row['partition'],arm=arm,mode=mode,metric_snapshot=scored[fid]['metrics'][arm],
            selected_candidate_id=cid,selected_family=family,selected_energy=pred['selected_energy'],
            optimizer_origin_family=origin,optimizer_record=deepcopy(solve),
            optimizer_success_is_not_a_selection_requirement='solver.choose uses feasible+finite score; solve_one records optimizer_success separately',
            original_consumed_group_ids=bundle['consumed_group_ids'],original_states=bundle['original_states'],
            poses=details,AB_optimizer_context=optimizer_context,
            h_context_caveat='B/edge evaluate identity ownership; candidateID hN labels a proposal/solve slot. AB conditional optimization h is relative to origin family; selected family posterior h is separately archived.'))
        bindings[fid]=dict(frame_seal=sha(folder/'FRAME_SEAL.json'),result=sha(rp),bundle=sha(bp),input_row=seal['input_row_sha256'])
    if attempted:raise RuntimeError('Forbidden fitting was attempted')
    out=run/'boundary_extreme_forensics_v1';out.mkdir(exist_ok=False)
    value=dict(schema='FROZEN_BOUNDARY_EXTREME_FIXED_POSE_FORENSICS_V1',complete=True,cases=records,
        maximum_all_family_all_mode_stored_cost_recheck_difference=global_recheck_gap,
        source_findings=dict(intersection='boundary_model327-344 computes supporting infinite-line intersection; no candidate-segment t guard',
            initial_query_gate='boundary_model241-244 rejects along-query beyond INITIAL edge only',
            replacement='solver199-202 removes consumed B full/partial and old hull; remaining factors are retained',
            formal_status='No explicit t-in[0,1] constraint exists in frozen config/solver. Finite physical-edge wording in METHOD_B overstates what infinite supporting-line implementation certifies. This is a representation/domain limitation and documentation gap, not evidence of changed observations or hidden optimization.',
            no_B2='No corrected B2, new candidate, tuned threshold, pose or replacement selector is constructed'),
        fitting_guard=dict(attempted_calls=attempted,forbidden_attempt_count=0,new_PnP_calls=0,new_optimizer_calls=0,new_NN_calls=0),
        evaluation_snapshot_caveat='Metric/status aggregation strict0 boundary audit may be repaired independently; original selected poses, bundles, score archives and scientific model are unchanged',
        input_bindings=dict(prediction_seal=sha(run/'PREDICTION_SEAL.json'),batch_seal=sha(batch/'BATCH_SEAL.json'),
            evaluation_scored_snapshot=scored_sha,input_packet_archive=sha(ip),per_case=bindings),
        source_sha256=sha(__file__),cost=dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    write(out/'BOUNDARY_EXTREME_FORENSICS.json',value)
    lines=['# 고정 B/edge 극단 출력의 유한 변 교차 포렌식','',
           '동결된 최악3사례의 저장 pose·비용·RGB query만 검사했다. 새 추론·PnP·최적화·튜닝·선택 변경은0회다. '
           '구간 t는 사후 진단이며 원 비용이나 feasible 판정을 바꾸지 않는다.','',
           '| 사례 | arm | 위치오차cm | optimizer success | solver feasible | 최저slack | 선택 family |','|---|---|---:|---|---|---:|---|']
    for r in records:
        z=r['optimizer_record'];lines.append(f"| {r['id']} | {r['arm']} | {r['metric_snapshot']['translation_cm']:.6g} | {z.get('optimizer_success')} | {z.get('feasible')} | {z.get('minimum_constraint_slack')} | {r['selected_family']} |")
    for r in records:
        lines += ['',f"## {r['id']} / {r['arm']}",'',
            f"생성 family={r['optimizer_origin_family']}, 생성 conditional h={r['optimizer_record'].get('global_owner_h')}, 실제 선택 family={r['selected_family']}.",
            '', '| pose | 같은 선택family의 저장비용 | mode feasible/slack | projected bbox W/H px | h | owner질량 | 유한변 안/유효교차 | t범위 |',
            '|---|---:|---|---|---:|---:|---:|---|']
        for name,p in r['poses'].items():
            st=p['stored_scores'];family=r['selected_family'];mode=r['mode'];s=st[family] if st else None
            for h,x in p['finite_edge_by_owner_h'].items():
                lines.append(f"| {name} | {s['scores'][mode] if s else 'NA'} | {s['feasible'][mode] if s else 'NA'} / {s['h_min_slack'][mode] if s else 'NA'} | {p['projected_bbox_extent_px']} | {h} | {x.get('stored_owner_responsibility','identity')} | {x['finite_segment_inside']}/{x['valid_intersections']} | [{x['t_min']}, {x['t_max']}] |")
    lines += ['',
        '원 구현은 두 투영 끝점으로 만든 무한 지지선과 고정 normal의 교차를 평가하며 t∈[0,1]을 확인하지 않는다. 초기 RGB query 생성에서만 초기 edge 길이를 검사한다. '
        '따라서 finite segment가 작아지거나 교차점에서 멀어져도 지지선이 같은 profile을 설명할 수 있다. 이 진단은 저장된 관측을 잃었다는 뜻이 아니라, 관측과 후보를 연결하는 기하 표현이 유한 변의 범위를 보존하지 않는다는 뜻이다.','',
        'METHOD_B의 “projected native edge, including both projected endpoints”와 코드의 “physical projected edge”는 유한 변 검증으로 읽힐 수 있으나, 동결 config/제약에 명시적인 segment bound는 없었다. '
        '따라서 새로 선언한 제약 위반으로 단정하지 않고 표현 가정의 한계와 문서의 불충분한 명시를 구분한다. 실제 material owner도 원래 독립 인증되지 않았다.','',
        'AB는 생성한 family/h와 최종 재평가에서 선택한 family의 posterior h를 구분한다. h의 질량은 교정된 물리 pose 확률이 아니며, 한 h의 유리한 비용을 다른 h의 feasible과 섞지 않는다. '
        'optimizer_success와 feasible은 원래 별도 필드였고 selector는 finite score+feasible을 사용한다. 실패 flag를 지우거나 수치적 전역 최적성을 주장하지 않는다.','',
        '같은 고정 pose에서 원 비용·잔차·제약만 재평가했다. 새 B2나 개선 선택기를 만들거나 이3사례에서 새 방법의 성능을 추정하지 않았다.']
    (out/'BOUNDARY_EXTREME_FORENSICS_KO.md').write_text('\n'.join(lines)+'\n')
    write(out/'FORENSIC_SEAL.json',dict(complete=True,input_bindings=value['input_bindings'],source_sha256=sha(__file__),
          artifacts={p.name:sha(p) for p in out.iterdir()if p.is_file()},fitting_guard=value['fitting_guard'],cost=value['cost']))
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--forensic-GO',action='store_true');a=p.parse_args()
    r=run(a.run_dir,forensic_GO=a.forensic_GO);print(r['maximum_all_family_all_mode_stored_cost_recheck_difference'])


if __name__=='__main__':main()

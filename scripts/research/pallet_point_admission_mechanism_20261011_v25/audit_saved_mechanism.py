"""Audit saved corner construction and its existing point-admission gates.

Uses only saved line/corner outputs and source POS wire witnesses. No model,
line fit, pose fit, new projection, threshold tuning or truth-pose scoring.
"""
import argparse
import ast
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path

READY='_docs/experiments/pallet_kp_supervision_repair_20261010_v1/READY_SOURCE_TARGET_ROWS.jsonl.gz'
COMMON='scripts/research/pallet_boundary_corner_refiner_20261010_v2/common.py'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def read(p):
    with gzip.open(p,'rt') as f:return [json.loads(s) for s in f if s.strip()]


def dot(a,b):return math.fsum(x*y for x,y in zip(a,b))


def save(p,r):
    with p.open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def write_rows(p,rr):
    with p.open('xb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as z:
            for r in rr:z.write((json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n').encode())


def cap_from_source(p):
    t=ast.parse(p.read_text());values=[]
    for n in t.body:
        if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='CORNER_OBSERVATION_UNCERTAINTY_CAP_PX' for x in n.targets):
            assert isinstance(n.value,ast.Constant);values.append(n.value.value)
    assert values==[8.0]
    return values[0]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--prior',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();src=a.source.resolve();prior=a.prior.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    paths={'ready':src/READY,'cap_code':src/COMMON,
        'admission_code':src/'scripts/research/pallet_boundary_corner_refiner_20261010_v2/pipeline.py',
        'selection_code':src/'scripts/research/pallet_cornerwise_independent_20261010_v4/selection.py',
        'point_score_code':src/'scripts/research/pallet_finite_edge_consensus_repair_20261011_v13/solver.py',
        'point_rank_code':src/'scripts/research/pallet_factor_subset_consensus_20261011_v15/solver.py',
        'decoded':prior/'decoder/DECODED_FRAMES.jsonl.gz','decisions':prior/'decoder/DECISIONS.jsonl.gz',
        'necessary':prior/'output/ROWS.jsonl.gz','decoder_checks':prior/'decoder/CHECKS.json'}
    bindings={k:dict(path=str(v),sha256=sha(v),bytes=v.stat().st_size) for k,v in paths.items()}
    save(out/'PROTOCOL.json',dict(schema='frozen_saved_point_admission_and_corner_mechanism_v25',
        inputs=bindings,code_sha256=sha(Path(__file__)),
        purpose='Separate decoder candidate selection from existing final-line/radius point admission; decompose saved corner error.',
        source_query_anchor_not_native_N3=True,source_N3_H_LOO='not saved/not reconstructed',
        exact_existing_cap_px=8.,thresholds_changed=False,new_model=0,new_line_fit=0,new_PnP=0,
        new_projection=0,new_pose_accuracy=0,new_training=0,new_RGB=0,
        actual_wire_reference_scope='only saved certified source POS supports; UNKNOWN/NONE never assigned a POS error'))
    oldcheck=json.loads(paths['decoder_checks'].read_text());assert oldcheck['passed']
    assert oldcheck['frames_sha256']==bindings['decoded']['sha256'] and oldcheck['decisions_sha256']==bindings['decisions']['sha256']
    cap=cap_from_source(paths['cap_code'])
    ready={r['id']:r for r in read(paths['ready']) if r['partition']=='calibration'};assert len(ready)==128
    frames=read(paths['decoded']);decisions=read(paths['decisions']);necessary=read(paths['necessary'])
    assert len(frames)==384 and len(decisions)==len(necessary)==573
    candidates={(r['arm'],r['id'],r['corner'],tuple(r['edges'])):r for r in necessary}
    line_lookup={};line_rows=[];admission_lookup={};counts={arm:collections.Counter() for arm in ('GEOMETRY_ONLY','IMAGE_NO_ROLE','IMAGE_ROLE')}
    frame_rows=[]
    for frame in frames:
        arm=frame['arm'];fid=frame['id'];c=counts[arm];c['frames']+=1;source=ready[fid]
        good_edges=set()
        for line in frame['lines']:
            key=(arm,fid,line['edge']);assert key not in line_lookup;line_lookup[key]=line
            residual=[dot(p,line['normal'])-line['offset'] for p in line['support_points']]
            consistent=all(math.isfinite(x) and math.isfinite(r) and r>=0 and abs(x)<=r+1e-9
                           for x,r in zip(residual,line['query_radii_px']))
            if consistent:good_edges.add(line['edge'])
            c['lines']+=1;c['final_line_consistent']+=consistent
            targets=[source['queries'][q] for q in line['queries']]
            states=collections.Counter(t['proposed_target'] for t in targets)
            row=dict(arm=arm,id=fid,edge=line['edge'],support_queries=line['queries'],source_support_states=dict(states),
                final_line_consistent=consistent,predicted_support_rms_px=line['residual_rms_px'],
                source_certified_all_POS=all(t['proposed_target']=='POSITIVE' for t in targets),
                physical_line_ownership='UNKNOWN')
            if row['source_certified_all_POS']:
                assert all(t['ownership']=='NONCOPLANAR_ACTUAL_SHARED_WIRE' and t['actual_target_uv'] is not None for t in targets)
                weights=[1/max(.5,r)**2 for r in line['query_radii_px']]
                uv=[t['actual_target_uv'] for t in targets]
                center=[math.fsum(w*p[j] for w,p in zip(weights,uv))/math.fsum(weights) for j in (0,1)]
                bias=dot(line['normal'],center)-line['offset'];sigma=math.sqrt(line['parameter_covariance'][1][1]);assert sigma>0
                row.update(certified_weighted_support_center_xy=center,
                    certified_center_bias_px=bias,predicted_center_offset_sigma_px=sigma,
                    center_bias_over_recorded_sigma=abs(bias)/sigma)
                c['certified_POS_lines']+=1
            else:
                row.update(certified_center_bias_px=None,center_bias_over_recorded_sigma=None,
                    error_not_defined_reason='At least one NONE or IGNORE source support; not a POS location target.')
            line_rows.append(row)
        admitted=[];c['decoder_selected_corners']+=len(frame['corners'])
        for corner in frame['corners']:
            radius=corner['radius_px']
            reason=('FINAL_LINE_CONSENSUS_INCONSISTENT' if not set(corner['edges'])<=good_edges else
                    'UNCERTAINTY_EXCEEDS_PNP_OBSERVATION_RADIUS' if not math.isfinite(radius) or radius<0 or radius>cap else 'accepted')
            k=(arm,fid,corner['id'],tuple(corner['edges']));admission_lookup[k]=reason
            c['point_admission:'+reason]+=1
            if reason=='accepted':admitted.append(corner['id'])
        c['frames_ge4_point_admitted_before_H']+=len(admitted)>=4
        frame_rows.append(dict(arm=arm,id=fid,decoder_corner_ids=[z['id'] for z in frame['corners']],
            point_admitted_corner_ids_before_H=admitted,source_native_N3_H_LOO='NOT_EVALUATED',
            point_admission_does_not_prove_new_pose=True))
    source_line_rows={(r['arm'],r['id'],r['edge']):r for r in line_rows};corner_rows=[];maxidentity=0.
    for d in decisions:
        key=(d['arm'],d['id'],d['corner'],tuple(d['edges']));n=candidates[key];z=d['final_decoder_candidate']
        c=counts[d['arm']];c['conditional_reference_rows']+=1
        reason=admission_lookup[key] if d['selected_decoder_corner'] else 'NOT_SELECTED_BY_DECODER'
        c['conditional_point_admission:'+reason]+=1
        rows=[];rref=[]
        for edge in d['edges']:
            line=line_lookup[(d['arm'],d['id'],edge)];audit=source_line_rows[(d['arm'],d['id'],edge)]
            assert audit['source_certified_all_POS']
            center=audit['certified_weighted_support_center_xy'];bias=audit['certified_center_bias_px']
            component=dot(line['normal'],[x-y for x,y in zip(n['reference_xy'],center)])
            residual=dot(line['normal'],n['reference_xy'])-line['offset']
            delta=abs(bias+component-residual);assert delta<1e-9;maxidentity=max(maxidentity,delta)
            rows.append(dict(edge=edge,certified_support_center_xy=center,normal=line['normal'],offset=line['offset'],
                center_bias_px=bias,orientation_and_corner_distance_component_px=component,
                normal_residual_at_reference_corner_px=residual,
                predicted_support_rms_px=line['residual_rms_px'],queries=line['queries'],
                recorded_center_offset_sigma_px=audit['predicted_center_offset_sigma_px']))
            rref.append(residual)
        aa,bb=[r['normal'] for r in rows];det=aa[0]*bb[1]-aa[1]*bb[0]
        assert abs(det)>1e-6
        correction=[(bb[1]*rref[0]-aa[1]*rref[1])/det,(-bb[0]*rref[0]+aa[0]*rref[1])/det]
        observed=[x-y for x,y in zip(n['reference_xy'],n['predicted_xy'])]
        delta=max(abs(x-y) for x,y in zip(correction,observed));assert delta<1e-7;maxidentity=max(maxidentity,delta)
        refnorm=math.sqrt(dot(rref,rref));amplification=n['error_px']/refnorm if refnorm>0 else None
        cosine=abs(dot(aa,bb));assert cosine<1
        condition=math.sqrt((1+cosine)/(1-cosine))
        row=dict(arm=d['arm'],id=d['id'],corner=d['corner'],edges=d['edges'],
            reference_xy=n['reference_xy'],predicted_xy=n['predicted_xy'],saved_corner_error_px=d['saved_error_px'],
            decoder_selected=d['selected_decoder_corner'],existing_point_admission_reason=reason,
            existing_point_admission_accepted=reason=='accepted',recorded_radius_px=z['radius_px'],existing_cap_px=cap,
            normal_residuals_px=rref,normal_residual_norm_px=refnorm,error_amplification_ratio=amplification,
            normal_matrix_condition=condition,absolute_normal_determinant=abs(det),
            exact_linear_correction_xy=correction,line_components=rows,
            support_query_max_error_px=max(v['error_px'] for v in n['support_query_errors']),
            source_native_N3_H_LOO='NOT_EVALUATED',physical_ownership='UNKNOWN')
        if d['saved_error_px']>8:
            c['corner_gt8px:'+reason]+=1
        corner_rows.append(row)
    four=[r for r in corner_rows if r['arm']=='IMAGE_ROLE' and r['decoder_selected'] and r['saved_corner_error_px']>8]
    assert len(four)==4 and sum(r['existing_point_admission_accepted'] for r in four)==1
    write_rows(out/'LINE_ROWS.jsonl.gz',line_rows);write_rows(out/'CORNER_ROWS.jsonl.gz',corner_rows);write_rows(out/'FRAME_ROWS.jsonl.gz',frame_rows)
    assert bindings=={k:dict(path=str(v),sha256=sha(v),bytes=v.stat().st_size) for k,v in paths.items()}
    save(out/'SUMMARY.json',dict(schema='actual_saved_point_admission_corner_error_mechanism_v25',complete=True,
        counts={a:dict(c) for a,c in counts.items()},four_decoder_admitted_large_error_cases=four,
        remaining_source_native_N3_H_LOO_not_reconstructed=True,
        fixed_source_query_anchor_never_substituted_for_N3=True,
        new_line_fit=0,new_PnP=0,new_projection=0,new_pose_accuracy=0,new_training=0,new_RGB=0))
    save(out/'CHECKS.json',dict(passed=True,saved_frame_rows=384,saved_line_rows=len(line_rows),conditional_corner_rows=573,
        all_original_inputs_unchanged=True,existing8px_cap_parsed_from_original_AST=True,
        all573_exact_error_decomposition_verified=True,max_abs_identity_error=maxidentity,
        four_decoder_large_cases_three_point_rejected=True,remaining_one_has_no_LOO_or_pose_claim=True,
        output_hashes={n:sha(out/n) for n in ('LINE_ROWS.jsonl.gz','CORNER_ROWS.jsonl.gz','FRAME_ROWS.jsonl.gz')},
        new_line_fit=0,new_PnP=0,new_projection=0,new_pose_accuracy=0))
    print(json.dumps(dict(passed=True,counts={a:dict(c) for a,c in counts.items()},
        four=[dict(id=r['id'],corner=r['corner'],error=r['saved_corner_error_px'],point_admission=r['existing_point_admission_reason']) for r in four])))


if __name__=='__main__':
    main()

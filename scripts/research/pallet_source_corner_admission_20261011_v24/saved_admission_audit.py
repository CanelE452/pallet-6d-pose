"""Join existing source CAL rows without fitting lines, poses or models.

Only necessary image/radius gates are replayable from these saved rows.
Passing them does not establish final decoder acceptance.
"""
import argparse
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path

ARMS = ('GEOMETRY_ONLY', 'IMAGE_NO_ROLE', 'IMAGE_ROLE')
EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
BASE = '_docs/experiments/pallet_three_head_observation_20261010_v7/source_calibration'
READY = '_docs/experiments/pallet_kp_supervision_repair_20261010_v1/READY_SOURCE_TARGET_ROWS.jsonl.gz'
CONTRACT = '_docs/experiments/pallet_corner_mechanism_audit_20261010_v1/CAL_CORNER_CONTRACT_ROWS.jsonl.gz'
CURVES = '_docs/experiments/pallet_kp_corrected_supervision_20261010_v1/SOURCE_CURVES.json'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def rows(p):
    with gzip.open(p, 'rt') as f:
        return [json.loads(s) for s in f if s.strip()]


def save(p, value):
    with p.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def describe(v):
    v = sorted(v)
    if not v:
        return dict(n=0)
    def quant(q):
        x=(len(v)-1)*q; i=int(x); j=min(i+1,len(v)-1)
        return v[i]+(v[j]-v[i])*(x-i)
    mean=math.fsum(v)/len(v)
    var=math.fsum((x-mean)**2 for x in v)/(len(v)-1) if len(v)>1 else None
    return dict(n=len(v), mean=mean, sample_variance=var,
                sample_std=math.sqrt(var) if var is not None else None,
                median=quant(.5), P90=quant(.9), maximum=max(v), unit='px', ddof=1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); root=a.source.resolve(); out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    paths=[READY,CONTRACT,CURVES]
    paths += [f'{BASE}/{arm}/{name}' for arm in ARMS for name in
              ('CALIBRATION.json','CALIBRATION_GEOMETRY_ROWS.jsonl.gz','CALIBRATION_QUERY_DIAGNOSTICS.jsonl.gz')]
    paths += ['scripts/research/pallet_boundary_corner_refiner_20261010_v2/observations.py',
              'scripts/research/pallet_boundary_corner_refiner_20261010_v2/calibration.py',
              'scripts/research/pallet_corner_mechanism_audit_20261010_v1/cal_corner_contract.py',
              'scripts/research/pallet_kp_supervision_repair_20261010_v1/retrain.py']
    bindings={n:dict(sha256=sha(root/n),bytes=(root/n).stat().st_size) for n in paths}
    save(out/'PROTOCOL.json',dict(schema='frozen_saved_source_corner_necessary_gate_audit_v24',
        inputs=bindings,code_sha256=sha(Path(__file__)),source_CAl_frames_per_arm=128,
        source_TEST='Only already saved step3000 aggregate query metrics; no new TEST inference.',
        rules='Original image bounds and radius/nominal-spacing<=1 only. Do not invent an error threshold for admission.',
        diagnostic_error_bins='Existing 2px query CAL criterion and existing 8px real point proxy boundary; descriptive only.',
        final_acceptance='UNKNOWN for rows passing necessary gates: predicted extrapolation and corner-minimum selection are not saved.',
        new_detector=0,new_head=0,new_line_fit=0,new_PnP=0,new_pose_accuracy=0,
        new_training=0,new_RGB=0,thresholds_changed=False))
    ready={r['id']:r for r in rows(root/READY)}
    contract={(r['id'],r['corner'],tuple(r['edges'])):r for r in rows(root/CONTRACT)}
    curves=json.loads((root/CURVES).read_text())
    assert curves['complete'] is True
    output=[]; summaries={}
    for arm in ARMS:
        cal=json.loads((root/f'{BASE}/{arm}/CALIBRATION.json').read_text())
        diag={r['id']:r for r in rows(root/f'{BASE}/{arm}/CALIBRATION_QUERY_DIAGNOSTICS.jsonl.gz')}
        assert len(diag)==128
        geom=rows(root/f'{BASE}/{arm}/CALIBRATION_GEOMETRY_ROWS.jsonl.gz')
        counts=collections.Counter(); this=[]
        for g in geom:
            r=ready[g['id']]; d=diag[g['id']]
            assert r['index']==g['index']==d['index'] and r['partition']=='calibration'
            assert g['reference_from_certified_actual_wire_queries'] is True
            assert g['physical_corner_ownership_independently_proved'] is False
            supports=[]
            for edge,ids in zip(g['edges'],g['supporting_query_ids']):
                assert g['corner'] in EDGES[edge] and len(ids)>=3
                assert max(i%7 for i in ids)-min(i%7 for i in ids)>=2
                for i in ids:
                    t=r['queries'][i]
                    assert i//7==edge and t['query']==i and t['edge']==edge
                    assert t['proposed_target']=='POSITIVE' and t['actual_target_uv'] is not None
                    assert t['ownership']=='NONCOPLANAR_ACTUAL_SHARED_WIRE'
                    assert d['accepted'][i] and d['known_eligible'][i]
                    error=abs(d['candidate'][i]-(t['lo']+t['weight']))
                    assert (error<=2)==d['correct_within2px'][i]
                    supports.append(dict(query=i,edge=edge,target_bin=t['lo']+t['weight'],
                                         predicted_bin=d['candidate'][i],error_px=error))
            spacing=min(math.dist(r['frozen_selected_points'][aa],r['frozen_selected_points'][bb])/8
                        for aa,bb in [EDGES[e] for e in g['edges']])
            radius=g['sigma_px']*cal['uncertainty']['corner_scale']
            h,w=r['raw_hw']; x,y=g['predicted_xy']
            inside=0<=x<w and 0<=y<h
            normalized=radius/max(spacing,1e-6)
            state=('DEFINITELY_REJECTED_IMAGE' if not inside else
                   'DEFINITELY_REJECTED_RADIUS' if normalized>1 else 'NECESSARY_GATES_PASS_FINAL_UNKNOWN')
            err=math.dist(g['predicted_xy'],g['reference_xy'])
            assert math.isclose(err,g['error_px'],rel_tol=0,abs_tol=1e-8)
            row=dict(arm=arm,id=g['id'],index=g['index'],corner=g['corner'],edges=g['edges'],
                raw_hw=r['raw_hw'],reference_xy=g['reference_xy'],predicted_xy=g['predicted_xy'],
                error_px=err,recorded_radius_px=radius,nominal_query_spacing_px=spacing,
                normalized_uncertainty=normalized,in_image=inside,necessary_gate_state=state,
                final_adopted_corner='UNKNOWN',support_query_errors=supports,
                support_query_error_mean_px=math.fsum(z['error_px'] for z in supports)/len(supports),
                support_query_error_max_px=max(z['error_px'] for z in supports),
                all_support_queries_within2px=all(z['error_px']<=2 for z in supports),
                predicted_support_extrapolation_ratio='NOT_SAVED',
                saved_true_reference_extrapolation_ratio=g['true_corner_support_extrapolation_ratio'],
                physical_corner_ownership='UNKNOWN')
            if arm=='IMAGE_ROLE':
                c=contract[(g['id'],g['corner'],tuple(g['edges']))]
                assert math.isclose(radius,c['recorded_radius_px'],rel_tol=0,abs_tol=1e-9)
                assert c['saved_predicted_intersection_xy']==g['predicted_xy']
                row['virtual_vs_native_error_px']=c['virtual_vs_native_error_px']
                row['predicted_vs_native_error_px']=c['predicted_vs_native_error_px']
            counts['rows']+=1; counts[state]+=1
            if err>8:
                counts['corner_error_gt8px']+=1
                counts['corner_error_gt8px:'+state]+=1
                if row['all_support_queries_within2px']:
                    counts['all_support_within2px_but_corner_gt8px']+=1
                    counts['all_support_within2px_but_corner_gt8px:'+state]+=1
            output.append(row);this.append(row)
        assert len({(r['id'],r['corner'],tuple(r['edges'])) for r in this})==len(this)
        summaries[arm]=dict(counts=dict(counts),frames_with_conditional_reference=len({r['id'] for r in this}),
            candidate_error_px=describe([r['error_px'] for r in this]),
            necessary_gates_pass_error_px=describe([r['error_px'] for r in this if r['necessary_gate_state'].startswith('NECESSARY')]),
            definitely_rejected_error_px=describe([r['error_px'] for r in this if r['necessary_gate_state'].startswith('DEFINITELY')]),
            largest_saved_candidates=sorted(this,key=lambda r:r['error_px'],reverse=True)[:8])
    assert len(output)==573 and len(contract)==348
    with (out/'ROWS.jsonl').open('x') as f:
        for r in output:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    save(out/'SUMMARY.json',dict(schema='saved_source_corner_necessary_gate_summary_v24',complete=True,
        arm_summaries=summaries,saved_step3000_query_TEST_metrics=curves['final_source_test'],
        saved_TEST_metrics_are_not_corner_or_pose_metrics=True,
        final_admission_not_claimed=True,new_detector=0,new_head=0,new_line_fit=0,new_PnP=0,
        new_pose_accuracy=0,new_training=0,new_RGB=0))
    assert bindings=={n:dict(sha256=sha(root/n),bytes=(root/n).stat().st_size) for n in paths}
    save(out/'CHECKS.json',dict(passed=True,stored_rows=573,source_inputs_unchanged=True,
        all_geometry_supports_certified_POS_and_CAL_selected=True,
        IMAGE_ROLE348_radius_and_coordinates_match_existing_contract=True,
        all_source_CAL_IDs_unique=True,final_admission_not_claimed=True,
        rows_sha256=sha(out/'ROWS.jsonl'),new_line_fit=0,new_PnP=0,new_head=0,new_training=0))
    print(json.dumps({arm:summaries[arm]['counts'] for arm in ARMS},ensure_ascii=False))


if __name__=='__main__':
    main()

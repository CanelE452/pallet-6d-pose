"""Verify the already evaluated source CAL decoder from sealed logits.

No model, image, pose, truth fitting or recalibration is allowed. Line and
intersection dispatches reproduce the frozen decoder and are counted.
"""
import argparse
import base64
import collections
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import zlib

ARMS=('GEOMETRY_ONLY','IMAGE_NO_ROLE','IMAGE_ROLE')
BASE='_docs/experiments/pallet_three_head_observation_20261010_v7/source_calibration'
READY='_docs/experiments/pallet_kp_supervision_repair_20261010_v1/READY_SOURCE_TARGET_ROWS.jsonl.gz'
OBS='scripts/research/pallet_boundary_corner_refiner_20261010_v2/observations.py'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def rows(p):
    with gzip.open(p,'rt') as f:return [json.loads(s) for s in f if s.strip()]


def save(p,r):
    with p.open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--necessary-rows',type=Path,required=True)
    a=p.parse_args();root=a.source.resolve();out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    paths=[READY,OBS]+[f'{BASE}/{arm}/{name}' for arm in ARMS for name in
        ('CALIBRATION.json','CALIBRATION_ROWS.jsonl.gz','CALIBRATION_GEOMETRY_ROWS.jsonl.gz','CALIBRATION_QUERY_DIAGNOSTICS.jsonl.gz')]
    paths += [f'{BASE}/IMAGE_ROLE/CALIBRATION_CHECKS.json']
    binding={n:dict(sha256=sha(root/n),bytes=(root/n).stat().st_size) for n in paths}
    necessary_hash=sha(a.necessary_rows)
    save(out/'PROTOCOL.json',dict(schema='frozen_original_source_CAL_decoder_replay_v24',
        inputs=binding,code_sha256=sha(Path(__file__)),necessary_rows_sha256=necessary_hash,
        purpose='Record missing final decoder admission fields of already evaluated CAL outputs; no new pose evaluation.',
        arms=ARMS,frames_per_arm=128,partition='existing calibration768:896; no TEST or real inference',
        fixed_decoder=OBS,thresholds_changed=False,new_head=0,new_detector=0,new_RGB=0,
        new_PnP=0,new_pose_scoring=0,new_training=0,new_rays=0,
        line_fits='Original fixed decoder arithmetic replay only; actual dispatch counts will be reported.'))
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    def guard(event,args):
        if event=='import' and args[0].split('.')[0] in ('torch','cv2','scipy','ultralytics','PIL'):
            raise RuntimeError('Forbidden model/pose/image import: '+args[0])
    sys.addaudithook(guard)
    import numpy as np
    assert np.__version__=='1.26.4'
    spec=importlib.util.spec_from_file_location('frozen_observations',root/OBS)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    calls=collections.Counter()
    for name in ('query_geometry','logits_statistics','fit_line','build_lines','intersection','decode'):
        original=getattr(module,name)
        def wrapped(*args,_fn=original,_name=name,**kwargs):
            calls[_name]+=1
            return _fn(*args,**kwargs)
        setattr(module,name,wrapped)
    ready={r['id']:r for r in rows(root/READY)}
    necessary=[json.loads(s) for s in a.necessary_rows.read_text().splitlines()]
    necessary_lookup={(r['arm'],r['id'],r['corner'],tuple(r['edges'])):r for r in necessary}
    assert len(necessary_lookup)==573
    decisions=[];frames=[];arm_counts={};maxdelta=0.
    for arm in ARMS:
        cal=json.loads((root/f'{BASE}/{arm}/CALIBRATION.json').read_text())
        raw=rows(root/f'{BASE}/{arm}/CALIBRATION_ROWS.jsonl.gz')
        geometry=rows(root/f'{BASE}/{arm}/CALIBRATION_GEOMETRY_ROWS.jsonl.gz')
        geo=collections.defaultdict(list)
        for g in geometry:geo[g['id']].append(g)
        diagnostics={r['id']:r for r in rows(root/f'{BASE}/{arm}/CALIBRATION_QUERY_DIAGNOSTICS.jsonl.gz')}
        assert len(raw)==128 and [r['index'] for r in raw]==list(range(768,896))
        counts=collections.Counter()
        for r in raw:
            sr=ready[r['id']];d=diagnostics[r['id']]
            assert sr['index']==r['index']==d['index'] and sr['partition']=='calibration'
            semantic=hashlib.sha256(json.dumps(sr,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            assert semantic==r['source_ready_row_semantic_sha256']
            b=zlib.decompress(base64.b64decode(r['logits_fp32_zlib_base64']))
            assert hashlib.sha256(b).hexdigest()==r['raw_logits_sha256'] and r['logits_shape']==[84,66]
            z=np.frombuffer(b,dtype='<f4').reshape(84,66)
            query=module.query_geometry(sr['frozen_selected_points']);query['raw_hw']=sr['raw_hw']
            decoded=module.decode(query,z,cal)
            assert decoded['raw_logits_sha256']==r['raw_logits_sha256']
            selected={(c['id'],tuple(c['edges'])) for c in decoded['corners']}
            candidates={(c['id'],tuple(c['edges'])):c for c in decoded['diagnostics']['corner_candidates']}
            for q in decoded['queries']:
                i=q['query'];assert q['chosen_candidate']==d['candidate'][i]
                assert abs(q['confidence']-d['confidence'][i])<=1e-12
                assert abs(q['sigma_mode_px']-d['sigma_mode_px'][i])<=1e-12
                if d['known_eligible'][i]:assert (q['selected_xy'] is not None)==d['accepted'][i]
            for g in geo[r['id']]:
                c=candidates[(g['corner'],tuple(g['edges']))]
                delta=max(float(np.max(np.abs(np.array(c['xy'])-g['predicted_xy']))),abs(c['sigma_px']-g['sigma_px']))
                assert delta<=1e-9;maxdelta=max(maxdelta,delta)
                n=necessary_lookup[(arm,r['id'],g['corner'],tuple(g['edges']))]
                assert abs(c['radius_px']-n['recorded_radius_px'])<=1e-9
                assert abs(c['nominal_query_spacing_px']-n['nominal_query_spacing_px'])<=1e-9
                assert abs(c['normalized_uncertainty']-n['normalized_uncertainty'])<=1e-9
                if n['necessary_gate_state'].startswith('DEFINITELY'):assert not c['accepted']
                decision=dict(arm=arm,id=r['id'],index=r['index'],corner=g['corner'],edges=g['edges'],
                    saved_error_px=g['error_px'],necessary_gate_state=n['necessary_gate_state'],
                    final_decoder_candidate=c,selected_decoder_corner=(g['corner'],tuple(g['edges'])) in selected,
                    support_query_errors=n['support_query_errors'],
                    physical_corner_ownership='UNKNOWN',real_LOO_or_pose_result='NOT_EVALUATED')
                decisions.append(decision)
                counts['conditional_rows']+=1;counts['conditional_accepted_pair']+=c['accepted']
                counts['conditional_selected_corner']+=decision['selected_decoder_corner']
                if g['error_px']>8:
                    counts['error_gt8px:'+c['reason']]+=1
                    counts['error_gt8px:selected_decoder_corner']+=decision['selected_decoder_corner']
            counts['frames']+=1;counts['selected_queries']+=decoded['selected_queries']
            counts['lines']+=len(decoded['lines']);counts['accepted_corners']+=len(decoded['corners'])
            counts['frames_ge4']+=len(decoded['corners'])>=4
            counts['all_candidate_pairs']+=len(candidates)
            counts['line_pair_hypotheses']+=decoded['diagnostics']['line_pair_hypotheses']
            counts['unknown_IGNORE_accepted']+=sum(q['selected_xy'] is not None and sr['queries'][q['query']]['proposed_target']=='IGNORE' for q in decoded['queries'])
            frames.append(dict(arm=arm,id=r['id'],index=r['index'],raw_logits_sha256=r['raw_logits_sha256'],
                corners=decoded['corners'],lines=decoded['lines'],corner_candidates=decoded['diagnostics']['corner_candidates'],
                line_rejections=decoded['diagnostics']['line_rejections'],
                selected_query_ids=[q['query'] for q in decoded['queries'] if q['selected_xy'] is not None],
                line_pair_hypotheses=decoded['diagnostics']['line_pair_hypotheses']))
        arm_counts[arm]=dict(counts)
    assert len(decisions)==573 and len(frames)==384
    old=json.loads((root/f'{BASE}/IMAGE_ROLE/CALIBRATION_CHECKS.json').read_text())['counts']
    matches=('selected_queries','lines','accepted_corners','frames_ge4','line_pair_hypotheses','unknown_IGNORE_accepted')
    assert all(arm_counts['IMAGE_ROLE'][k]==old[k] for k in matches)
    for name,data in [('DECISIONS.jsonl.gz',decisions),('DECODED_FRAMES.jsonl.gz',frames)]:
        with (out/name).open('xb') as f:
            with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as zf:
                for r in data:zf.write((json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n').encode())
    assert binding=={n:dict(sha256=sha(root/n),bytes=(root/n).stat().st_size) for n in paths}
    assert necessary_hash==sha(a.necessary_rows)
    formerly_unknown=[r for r in decisions if r['arm']=='IMAGE_ROLE' and r['saved_error_px']>8 and r['necessary_gate_state'].startswith('NECESSARY')]
    save(out/'SUMMARY.json',dict(schema='actual_frozen_saved_CAL_decoder_admission_replay_v24',complete=True,
        counts=arm_counts,previously_unknown_four_rows=formerly_unknown,
        actual_decoder_dispatches=dict(calls),new_head=0,new_detector=0,new_PnP=0,new_pose_scoring=0,
        new_training=0,new_RGB=0,new_rays=0,source_TEST_or_real_inference=0,
        numpy=np.__version__,python=sys.version,
        final_decoder_does_not_include_real_H_LOO_or_PnP=True,thresholds_changed=False))
    save(out/'CHECKS.json',dict(passed=True,decoded_source_frames=384,conditional_reference_rows=573,
        all_saved_query_statistics_match=True,all573_saved_candidate_math_match=True,
        candidate_max_abs_difference=maxdelta,IMAGE_ROLE_historical_aggregate_fields_match=list(matches),
        definite_necessary_rejections_verified=True,all_source_inputs_unchanged=True,
        decisions_sha256=sha(out/'DECISIONS.jsonl.gz'),frames_sha256=sha(out/'DECODED_FRAMES.jsonl.gz'),
        no_model_pose_image_imports=True,new_head=0,new_PnP=0,new_pose_scoring=0))
    print(json.dumps(dict(passed=True,counts=arm_counts,actual_decoder_dispatches=dict(calls),
        previously_unknown_four=[dict(id=r['id'],corner=r['corner'],error_px=r['saved_error_px'],
            reason=r['final_decoder_candidate']['reason'],selected=r['selected_decoder_corner']) for r in formerly_unknown])))


if __name__=='__main__':
    main()

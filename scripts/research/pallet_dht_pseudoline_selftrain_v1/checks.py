"""Audit receipts, GPU snapshots, and truthful close-out after a failed Stage A."""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime,timezone

import numpy as np
import torch

from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.util import read_json,sha256,canonical_sha
from .audit_mechanism import ROOT,PACKAGE,DOC,RAW,POINT,DHT,EXPORT,SOURCES,PSEUDO,save,source_inventory,summary_edges
from .contracts import paired_exposure


def correction():
    lock=read_json(DOC/'PROTOCOL_LOCK.json')
    changed={p:dict(before=d,after=sha256(ROOT/p)) for p,d in lock['source_hashes'].items() if sha256(ROOT/p)!=d}
    assert set(changed)=={str((PACKAGE/f).relative_to(ROOT)) for f in ('geometry.py','audit_mechanism.py')}
    save('IMPLEMENTATION_CORRECTION.json',dict(reason='Ignored invalid synthetic GT coordinates contain NaN. Masking AFTER norm can leak NaN through autograd; sanitize ignored targets BEFORE subtraction.',
        prior_attempt='Strict JSON serialization rejected nonfinite gradient derivatives; no final mechanism artifact was written.',
        scope='diagnostic GT gradient only; same line loss, weighting, bootstrap, population and numerical gate',
        gate_unchanged=True,student_updates_before_and_after=0,files=changed,
        unchanged_line_cache_sha256={str(p.relative_to(ROOT)):sha256(p) for p in RAW.glob('*_pseudo_lines.json')},
        adds_regression='ignored NaN GT has finite zero gradient'))


def tests():
    result=subprocess.run([sys.executable,'-m','pytest',str(PACKAGE/'tests'),'-q'],cwd=ROOT,capture_output=True,text=True)
    save('CONTRACT_TESTS.json',dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in (PACKAGE/'tests').glob('*.py')},
        student_optimizer_steps_in_experiment=0,exposure_and_optimizer_tests='static contracts; not claims that student fits were executed'))
    print(result.stdout,flush=True);assert result.returncode==0,result.stderr


def gpu():
    env=dict(os.environ);env['LD_LIBRARY_PATH']='/tmp/nvidia-580.173.02-userspace'
    query='name,driver_version,temperature.gpu,memory.used,memory.total,utilization.gpu,power.draw'
    result=subprocess.run(['nvidia-smi',f'--query-gpu={query}','--format=csv,noheader'],capture_output=True,text=True,env=env,check=True)
    processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,env=env,check=True)
    stamp=datetime.now(timezone.utc)
    save('GPU_'+stamp.strftime('%Y%m%dT%H%M%S')+'.json',dict(timestamp_utc=stamp.isoformat(),fields=query,snapshot=result.stdout.strip(),
        compute_processes=processes.stdout.strip(),other_processes_modified=False,reboot_or_driver_change=False))
    print(result.stdout,flush=True)


def baseline():
    from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.data import ObservationDataset
    result={}
    for split in SOURCES:
        data=ObservationDataset(EXPORT/f'{split}.json',targets=True)
        targets=covered=empty=0
        for item in data:
            mask=item['target_valid'][:8];targets+=int(mask.sum())
            covered+=int((mask & item['point_valid'][:8]).sum());empty+=int(not mask.any())
        result[split]=dict(N=len(data),stock_detected_frames=sum(r['stock_detected'] for r in data.records),
            stock_matched_frames=sum(r['stock_matched'] for r in data.records),target_corner_count=targets,
            covered_target_corners=covered,no_evaluable_target_frames=empty,pooled_metric_entries=targets+8*empty,
            no_target_penalty='8 diagonal penalties per no-evaluable-target frame, not 8 observed keypoints')
    save('BASELINE_DETECTION_AUDIT.json',dict(synthetic_R0=result,real_detection_not_evaluated=True))


def close():
    audit=read_json(DOC/'LINE_MECHANISM_AUDIT.json');assert audit['status']=='DHT_PSEUDOLINE_MECHANISM_FAIL'
    assert read_json(DOC/'CONTRACT_TESTS.json')['exit_code']==0
    quality=read_json(DOC/'PSEUDO_SUPERVISION_QUALITY.json');checks=audit['splits']['synth_val']['gate_checks']
    assert not all(checks.values())
    plans={str(s):paired_exposure(s,1440,273) for s in (1,2,3)}
    exposures=[]
    for seed,plan in plans.items():
        for arm in ('C0','C1','C2'):
            value=plan[arm]
            exposures.append(dict(seed=int(seed),arm=arm,planned_updates=900,actual_updates=0,
                planned_synthetic_exposures=21600,actual_synthetic_exposures=0,planned_real_exposures=value['real_exposures'],actual_real_exposures=0,
                synthetic_order_sha256=canonical_sha(value['synthetic']),real_order_sha256=None if value['real'] is None else canonical_sha(value['real']),
                augmentation_sha256=canonical_sha(value['augmentation']),status='NOT_RUN_MECHANISM_GATE_FAILED'))
    save('EXPOSURE_PARITY.json',dict(planned_contracts_verified=True,actual_training_started=False,runs=exposures))
    save('TRAINING_AUDIT.json',dict(status='NOT_RUN_MECHANISM_GATE_FAILED',student_fit_count=0,total_student_updates=0,
        C0={str(s):0 for s in (1,2,3)},C1={str(s):0 for s in (1,2,3)},C2={str(s):0 for s in (1,2,3)},
        teacher_updates=0,student_only_optimizer='static contract tested; no experiment optimizer created',actual_training_exposure_parity='all zero, no fit',
        downstream_student_results_available=False))
    save('GRADIENT_CALIBRATION.json',dict(status='NOT_RUN_MECHANISM_GATE_FAILED',lambda_line=None,g_point=None,g_line=None,ratio=None,
        planned_ratio=.25,planned_clamp=[0,1000],reason='Parameter-space lambda calibration is conditional on mechanism PASS. Output-space diagnostic norms are not substituted for parameter gradients.'))
    save('PER_SEED_RESULTS.json',dict(status='NOT_RUN_MECHANISM_GATE_FAILED',runs=exposures,C1_minus_C0=None,C2_minus_C1=None,C2_minus_R0=None))
    save('SYNTH_OR_DEV_RESULTS.json',dict(status=audit['status'],R0_synthetic_diagnostic={k:v['Point'] for k,v in quality.items()},
        C0=None,C1=None,C2=None,student_comparisons_not_measured=True,real_DEV_executed=False,PAPER_EVAL_executed=False,FINAL_opened=False,
        canonical_6D_evaluated=False,evidence_level='POSTHOC_DEVELOPMENT_ONLY',untouched_target_evaluation_used=False))
    subgroups={}
    for split,edges in audit['edge_records'].items():
        groups={field:{value:summary_edges([r for r in edges if r[field]==value]) for value in sorted({r[field] for r in edges})} for field in ('source','asset','symmetry')}
        for field in ('ambiguity','P_frame_mean_px','DHT_error_px'):
            order=np.argsort([r[field] for r in edges],kind='stable')
            groups[field+'_quartiles']=[dict(quartile=i+1,**summary_edges([edges[k] for k in ids])) for i,ids in enumerate(np.array_split(order,4))]
        # Semantic orientation: front/back horizontal, height, depth (fixed edge contract).
        from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.constants import EDGES
        groups['edge_role']={str(e)+':'+str(ends):summary_edges([r for r in edges if r['edge']==e]) for e,ends in enumerate(EDGES)}
        subgroups[split]=groups
    save('SUBGROUP_RESULTS.json',dict(teacher_mechanism=subgroups,student_subgroups=None,real_day_night_student_metrics=None,
        reason='Student/real evaluation not run after Stage A failure; synthetic source/asset/C1-C2 N reported'))
    old=read_json(DOC/'PRESERVED_SOURCE_SHA.json');assert source_inventory()==old
    protocol=read_json(DOC/'PROTOCOL_LOCK.json');assert sha256(POINT)==protocol['point_sha256'] and sha256(DHT)==protocol['DHT_sha256']
    correction=read_json(DOC/'IMPLEMENTATION_CORRECTION.json')
    assert all(sha256(ROOT/p)==h for p,h in correction['unchanged_line_cache_sha256'].items())
    for path,digest in protocol['source_hashes'].items():
        expected=correction['files'].get(path,{}).get('after',digest);assert sha256(ROOT/path)==expected
    # Direct summaries are re-derived from per-edge/frame rows without the reporting helper.
    for split,edges in audit['edge_records'].items():
        w=np.array([e['weight'] for e in edges]);p=np.array([e['P_error_px'] for e in edges]);d=np.array([e['DHT_error_px'] for e in edges]);s=audit['splits'][split]
        assert abs(np.sum(w*p)/w.sum()-s['weighted_P_mean_px'])<1e-12
        assert abs(np.sum(w*d)/w.sum()-s['weighted_DHT_mean_px'])<1e-12
        assert abs(np.mean(d<p)-s['better_edge_fraction'])<1e-12
        assert all(np.isfinite(r['gradient_cosine']) and np.isfinite(r['directional_derivative']) for r in audit['frame_records'][split])
    save('FINAL_AUDIT.json',dict(PASS=True,source_artifacts_preserved=len(old),paper_stop_lock_and_documents_unchanged=True,
        Point_before_sha256=protocol['point_sha256'],Point_after_sha256=sha256(POINT),DHT_before_sha256=protocol['DHT_sha256'],DHT_after_sha256=sha256(DHT),
        teacher_file_diff=0,teacher_optimizer_updates=0,student_optimizer_updates=0,real_GT_annotation_access_count=0,
        synthetic_GT_only_for_mechanism=True,real_manifest_metadata_only=True,FINAL_opened=False,
        initial_and_corrected_pseudo_lines_byte_identical=True,post_correction_gradient_values_finite=True,
        line_summary_independently_reaggregated=True,student_graph_contract_tested=True,no_runtime_DHT_in_stock_student=True))
    print('STOPPED AT MECHANISM GATE; ORIGINAL SOURCES PRESERVED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['correction','tests','gpu','baseline','close']);a=p.parse_args();globals()[a.phase]()

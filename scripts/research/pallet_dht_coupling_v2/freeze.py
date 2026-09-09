"""Freeze training after synthetic preflight; freeze reporting before real inference."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from .prepare import ROOT, OLD, NEW_ARMS, CONTROL_ARMS, read, write, sha, bound, check
from .driver import verify_bindings
from scripts.research.pallet_dht_joint_v1.driver import now

HERE = Path(__file__).resolve().parent


def training(root, incidence_weight):
    required = ['REUSED_CONTROLS.json', 'SOURCE_REVALIDATION.json', 'GEOMETRY_PREFLIGHT.json',
                'TRAINING_PREFLIGHT.json', 'GRADIENT_DIAGNOSIS.json']
    for name in required:
        data = read(root / name)
        check(data.get('complete') is True and data.get('PASS') is True, f'Unfinished preflight: {name}')
        verify_bindings(data, root)
    from .train import training_sources
    old = read(OLD / 'TRAIN_PROTOCOL.json')
    protocol = copy.deepcopy(old)
    protocol.update(schema='pallet_dht_coupling_protocol_v2', created_at_utc=now(),
        user_objective='Disentangle actual loss balance, PCGrad and point–line coupling; full synthetic-supervised DHT+point network',
        arms=list(NEW_ARMS), primary_arms=list(NEW_ARMS), primary_arm='balanced',
        primary_reference='hough_joint', primary_references=['hough_joint', 'point_only'],
        references={arm:dict(run_dir=str(OLD), arm=arm) for arm in CONTROL_ARMS},
        reuse_control_manifest=bound(root / 'REUSED_CONTROLS.json'),
        original_training_protocol=bound(OLD / 'TRAIN_PROTOCOL.json'),
        source_revalidation_sha256=sha(root / 'SOURCE_REVALIDATION.json'),
        source_code_sha256={str(path):sha(path) for path in training_sources()},
        preflight_required=required,
        statistics=dict(family_size=48, family_alpha=.05, session_resamples=100000,
            frame_resamples=10000, method='bonferroni_percentile', primary_references=['hough_joint','point_only'],
            uncertainty='Conditional percentile-bootstrap approximation over 13 reused sessions; not all future training-seed uncertainty'),
        diagnostics=dict(training_steps=[2,16,256,1750,3499,3500,5249,6998],
            parameters='Structural intersection of actual task autograd dependencies; task-private gradients preserved before original global clipping',
            task_a='Original combined box/class/pose/RLE loss', task_b='Weighted auxiliary structural-line loss',
            point_only_diagnostic='Separately isolated point-location+RLE vs weighted line',
            extra_training_forward_BN_or_RNG=False),
        interventions=dict(balanced='lambda_line=.1*stock.o2m/.8',
            pcgrad='Symmetric two-task PCGrad on shared dependency intersection; original partner gradients; SUM before original clipping',
            balanced_pcgrad='Both prespecified balance and PCGrad',
            incidence='Fixed .1 structural-line loss plus supervised role/instance assigned predicted-point-to-image-predicted-line mixture loss'),
        visibility_scope='Existing source contains no physical per-edge visibility label; v>0 remains amodal coordinate supervision',
        incidence=dict(coefficient=incidence_weight, geometry_contract=bound(root/'GEOMETRY_PREFLIGHT.json'),
            sigma_input_px='max(1,.01*input_diagonal)', endpoint_cost='Mean of two pseudo-Huber(delta1) signed normal residual costs',
            mixture='Negative log expectation of exp(-cost) under normalized predicted sigmoid line map on physical footprint',
            instance_scope='Verified at most one object per synthetic image; actual stock positive assignment; no GT line bin restriction',
            test_time_point_correction=False, model_forward_uses_GT=False))
    protocol['training']['incidence_weight'] = incidence_weight
    protocol['training']['line_weight_policy'] = 'Arm-specific as declared; original stock E2E decay preserved'
    protocol['line_targets']['loss'] = ('Unchanged balanced positive/background BCE normalization; '
        'coefficient .1 fixed for pcgrad/incidence, .1*stock.o2m/.8 for balanced/balanced_pcgrad')
    protocol['evaluation']['new_forward_scope'] = 'All 319 positive and 2689 negative frames for each of twelve new models; nine historical controls explicitly reused'
    protocol['evaluation']['paired_bootstrap'] = protocol['statistics']
    protocol['evaluation']['decision'] = ('Each new arm vs each of original joint/point: all6 full-population means and '
        'family-adjusted session intervals improve, with per-seed matching/pose coverage preserved. '
        'All48 registered comparisons are reported; no real-data winner selection. AP/negative FP separate.')
    protocol['evaluation']['runtime'] = 'Same26fixed DEVframes×3repeats for all21models, warmed/interleaved FP32; strict1e-4 parity failures retained'
    destination = root / 'TRAIN_PROTOCOL.json'
    check(not destination.exists(), 'Training protocol already frozen; do not overwrite')
    write(destination, protocol)
    print(f'Training protocol frozen {sha(destination)}', flush=True)


def chain(root):
    files = [HERE/name for name in ('evaluate.py','aggregate.py','report.py','audit_outputs.py','visual_qa.py','runtime.py','finalize.py')]
    files.extend(sorted((HERE.parent/'pallet_dht_joint_v1').glob('*.py')))
    files.extend(HERE.parent/'pallet_line_pose_v1'/name for name in
                 ('aggregate_results.py','report.py','paper_evaluation.py'))
    files.append(HERE.parent/'deep_hough_side_v1/dht.py')
    files.extend(Path('/home/minjae/.claude/agents/viz-expert/assets')/name for name in
                 ('palette.py','analysis.mplstyle'))
    for path in files:
        check(path.exists(), f'Missing completion implementation: {path}')
    check(not any((root/'evaluation').glob('*/PREDICTIONS.json')), 'New accuracy predictions already exist; initial reporting freeze is too late')
    result = dict(complete=True, PASS=True, source_sha256={str(path):sha(path) for path in files},
                  input_sha256={str(root/'TRAIN_PROTOCOL.json'):sha(root/'TRAIN_PROTOCOL.json'),
                                str(root/'REUSED_CONTROLS.json'):sha(root/'REUSED_CONTROLS.json')},
                  scope='Reporting/statistical sources frozen before first new real accuracy inference')
    destination = root/'COMPLETION_CHAIN_BINDING.json'
    if destination.exists():
        check(read(destination)==result, 'Reporting chain already frozen with different sources')
    else:
        write(destination,result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--phase',choices=['training','chain'],required=True)
    parser.add_argument('--incidence-weight',type=float,default=.1)
    args=parser.parse_args()
    root=args.run_dir.resolve()
    if args.phase=='training':training(root,args.incidence_weight)
    else:chain(root)

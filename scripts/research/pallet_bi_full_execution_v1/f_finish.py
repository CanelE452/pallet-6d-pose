"""Freeze the registered VAL choice, then finish both matched F systems.

Only ``choose`` reads already scored PRIMARY success counts, and only for the
one preregistered all-negative fallback bit. Ranking uses the nine own-VAL
results. ``finish`` never opens real references or chooses a configuration.
Core production must have completed before either downstream feature branch.
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

from . import s1_union as U
from .material_finish import execute

SEEDS = (20261018, 20261019, 20261020)
MATERIAL = ('S3_COMMON', 'S3_BI', 'S4_PL', 'S4_CONTOUR', 'S4_IMAGE')
PF_METHODS = ['POSEFIX_SUBPIX_PF_ANCHOR', 'POSEFIX_SUBPIX_YOLO_ANCHOR']
NAMESPACE = 'scripts.research.pallet_bi_full_execution_v1.'


def bind(path, bindings):
    path = Path(path).resolve()
    bindings[str(path)] = U.sha(path)
    return U.read(path)


def own_validation(training, prediction, bindings):
    """Read frozen own-VAL summaries, authenticating all three checkpoints."""
    training, prediction = Path(training), Path(prediction)
    manifest = bind(training/'TRAINING_MANIFEST.json', bindings)
    if not manifest.get('complete') or len(manifest['runs']) != 3:
        raise ValueError('All three own H1 fits required before configuration choice')
    protocol = manifest['protocol']
    if (protocol['seeds'] != list(SEEDS) or protocol['representative_seed'] != SEEDS[0] or
            protocol['split'] != [768, 235] or protocol['parameters'] != 16321 or
            protocol['max_epochs'] != 50 or protocol['patience'] != 5):
        raise ValueError('Registered own H1 architecture, split and training budget required')
    by_seed = {}
    for receipt in manifest['runs']:
        seed = receipt['seed']
        if seed not in SEEDS or seed in by_seed or not receipt['complete']:
            raise ValueError('Exactly the registered three completed seeds required')
        if U.sha(receipt['checkpoint']) != receipt['checkpoint_sha256']:
            raise ValueError('Own H1 checkpoint bytes differ')
        bindings[receipt['checkpoint']] = receipt['checkpoint_sha256']
        metric = receipt['best_metrics']['validation']
        if (metric['frames'] != 235 or not math.isfinite(receipt['best_validation_loss']) or
                metric['loss'] != receipt['best_validation_loss']):
            raise ValueError('All235 own validation rows and actual best-checkpoint loss required')
        by_seed[str(seed)] = dict(joint_success=metric['success_5cm5deg'],
            validation_loss=receipt['best_validation_loss'], best_epoch=receipt['best_epoch'])
    seal = bind(prediction/'PREDICTION_SEAL.json', bindings)
    if not seal.get('complete') or seal['rows'] != 238 or not seal['before_reference_decode']:
        raise ValueError('All238 predictions must have been sealed before scoring')
    if seal['training_manifest_sha256'] != bindings[str((training/'TRAINING_MANIFEST.json').resolve())]:
        raise ValueError('Prediction is not bound to the own training manifest')
    pp = prediction/'PREDICTIONS.jsonl.gz'
    if U.sha(pp) != seal['predictions_sha256']:
        raise ValueError('Sealed prediction bytes differ')
    bindings[str(pp.resolve())] = seal['predictions_sha256']
    return dict(by_seed=by_seed, validation_frames=235,
        median_joint_success=statistics.median(v['joint_success'] for v in by_seed.values()),
        median_validation_loss=statistics.median(v['validation_loss'] for v in by_seed.values()))


def choose(run, s2_core, s2_heads):
    run, s2_core, s2_heads = map(Path, (run, s2_core, s2_heads))
    destination = run/'F_SYSTEM_CHOICE.json'
    if destination.exists():
        previous = U.read(destination)
        for path, digest in previous['input_bindings'].items():
            if U.sha(path) != digest:
                raise ValueError('Completed F choice input changed')
        return previous
    bindings = {}
    plan = bind(run/'CAMPAIGN_PLAN.json', bindings)
    clarification = bind(run/'F_SELECTION_RULE_CLARIFICATION.json', bindings)
    mapping = bind(run/'F_AUXILIARY_MAPPING_IMPLEMENTATION_NOTE.json', bindings)
    bind(run/'F_OBSERVER_POLICY_LOCK_V2.json', bindings)
    bind(run/'F_CORE_PRECOMPUTE_LOCK.json', bindings)
    s2_summary = bind(s2_core/'SUMMARY.json', bindings)
    if s2_summary['status'] != 'COMPLETE_NEW' or s2_summary['missing_rows'] or s2_summary['technical_failures']:
        raise ValueError('Full completed S2 required; interrupted rows are not failures')
    all_s2 = bind(s2_heads/'ALL_REAL_PREDICTIONS_SEAL.json', bindings)
    if not all_s2['complete'] or all_s2['rows_per_arm'] != 238:
        raise ValueError('All nine S2 predictions required')
    for arm in ('C0', 'CP', 'CB'):
        if all_s2['prediction_seal_sha256'][arm] != U.sha(s2_heads/arm/'prediction/PREDICTION_SEAL.json'):
            raise ValueError('S2 aggregate prediction ancestry differs')
    material_finish = bind(run/'s3_s4_postprocess/FINISH_RECEIPT.json', bindings)
    if not material_finish['complete'] or material_finish['H1_fits'] != 15:
        raise ValueError('All S3/S4 own fits and predictions required')
    for path, digest in material_finish['sealed_artifact_paths'].items():
        if U.sha(path) != digest:
            raise ValueError('Completed S3/S4 input changed: '+path)
        bindings[path] = digest
    sources = {
        'S1': (run/'s1_training', run/'s1_predictions'),
        **{'S2_'+a: (s2_heads/a/'training', s2_heads/a/'prediction') for a in ('C0', 'CP', 'CB')},
        **{a: (run/'s3_s4_training'/a, run/'s3_s4_predictions'/a) for a in MATERIAL},
    }
    candidates = {arm: own_validation(*paths, bindings) for arm, paths in sources.items()}
    # This last tie-breaker uses the measured complete producer scope. Shared
    # S3/S4 production has one shared timer; it is never split into fake arms.
    s1_cost = bind(run/'S1_EXECUTION_SUMMARY.json', bindings)
    s1_wall = sum(s1_cost['phase_wall_seconds'][k] for k in (
        'new_synthetic_SubPix_and_F', 'real_union_feature_build', 'synthetic_union_feature_build'))
    s3_cost = bind(run/'s3_s4_production/FULL_RECEIPT.json', bindings)
    for arm, value in candidates.items():
        if arm == 'S1':
            value['production_wall_seconds'] = s1_wall
            value['production_wall_scope'] = 'S1 synthetic SubPix/F plus both-domain union phase sum'
        elif arm.startswith('S2_'):
            value['production_wall_seconds'] = s2_summary['wall_seconds']
            value['production_wall_scope'] = 'shared corrected complete S2 dispatch timer; old exact reuse costs separately reported'
        else:
            value['production_wall_seconds'] = s3_cost['invocation_wall_seconds']
            value['production_wall_scope'] = 'shared five-arm complete S3/S4 batch timer; no invented per-arm wall'
    ranked = sorted(candidates, key=lambda a: (-candidates[a]['median_joint_success'],
        candidates[a]['median_validation_loss'], candidates[a]['production_wall_seconds']))
    def rank_key(a):
        v = candidates[a]
        return (-v['median_joint_success'], v['median_validation_loss'], v['production_wall_seconds'])
    if len(ranked) > 1 and rank_key(ranked[0]) == rank_key(ranked[1]):
        raise ValueError('Frozen ranking exhausted without a unique choice; no unregistered tie-breaker')
    # All scored arms are required, but their individual scores do not rank
    # configurations. The TXT requires exactly this development fallback bit.
    representative = {}
    for folder, arms in ((run/'S1_EVALUATION', ['S1_UNION']),
                         (run/'S2_EVALUATION', ['S2_C0', 'S2_CP', 'S2_CB']),
                         (run/'S3_S4_EVALUATION', list(MATERIAL))):
        independent = {'S1_EVALUATION': run/'S1_INDEPENDENT_FINAL',
                       'S2_EVALUATION': run/'S2_INDEPENDENT',
                       'S3_S4_EVALUATION': run/'S3_S4_INDEPENDENT'}[folder.name]
        check = bind(independent/'INDEPENDENT_VERIFICATION.json', bindings)
        if check.get('PASS') is not True or check['rows'] != 238:
            raise ValueError('Independent complete score verification required')
        canonical = bind(folder/'EVALUATION_SEAL.json', bindings)
        if check['canonical_evaluation_seal_sha256'] != U.sha(folder/'EVALUATION_SEAL.json'):
            raise ValueError('Independent verification belongs to a different canonical score')
        for name, digest in canonical['artifacts'].items():
            if U.sha(folder/name) != digest:
                raise ValueError('Canonical score artifact changed: '+name)
            bindings[str((folder/name).resolve())] = digest
        summary = bind(folder/'SUMMARY.json', bindings)['PRIMARY192']
        for arm in arms:
            metric = summary[arm+'_s20261018']
            if metric['denominator'] != 192:
                raise ValueError('Full PRIMARY denominator required')
            representative[arm] = metric['successes']
    all_negative = all(n <= 120 for n in representative.values())
    # In this campaign PL has already exceeded120 before S2 completion. Do not
    # silently implement an unexecuted original120 fallback under a new name.
    if all_negative:
        raise ValueError('Mandated original120 fallback branch needs its own explicit baseline binding')
    selected = ranked[0]
    if plan['seeds'] != list(SEEDS) or len(clarification['rule']) != 6:
        raise ValueError('Registered final selection protocol differs')
    value = dict(schema='F_FROZEN_SYSTEM_CHOICE_V1', complete=True,
        time_unix=time.time(), controller_sha256=U.sha(__file__),
        selected_configuration=selected, ranked_configuration_names=ranked,
        own_validation=candidates, auxiliary_actions=mapping['root_authorized_mapping'][selected],
        all_negative_fallback=False, primary_development_fallback_bit_used=True,
        primary_configuration_ranking=False, primary_counts_used_only_for_fallback=representative,
        selected_by='highest median own VAL success; tie median VAL loss; tie measured production wall',
        active_F_observer_and_own_CAL_and_fullnormal_core_unchanged=True,
        scope='conditional native observation policy bundle with identical shared complete-estimator auxiliary proposals; downstream role information retained',
        original120_direct_parity=False, independent_real_test_n=0,
        input_bindings=bindings, before_F_optional_union_and_H1=True)
    U.write(destination, value)
    return value


def finish(run, d1, old_run, prior_run, core, output):
    run, d1, old, prior, core, out = map(Path, (run, d1, old_run, prior_run, core, output))
    choice = U.read(run/'F_SYSTEM_CHOICE.json')
    if not choice.get('complete') or not choice['before_F_optional_union_and_H1']:
        raise ValueError('Completed frozen final choice required')
    for path, digest in choice['input_bindings'].items():
        if U.sha(path) != digest:
            raise ValueError('Frozen final choice ancestry changed: '+path)
    summary = U.read(core/'SUMMARY.json'); dispatch = U.read(core/'DISPATCH_COMPLETION.json')
    if summary['status'] != 'COMPLETE_NEW' or summary['missing_rows'] or summary['technical_failures']:
        raise ValueError('Both complete F cores required')
    if not dispatch['active_rows_drained_before_exit'] or dispatch['unsubmitted_jobs'] != 0:
        raise ValueError('F production pool must exit before downstream work')
    if dispatch['summary_sha256'] != U.sha(core/'SUMMARY.json'):
        raise ValueError('F completion summary changed')
    completion = U.read(run/'F_FOUR_CPU_CONTROLLER_COMPLETION.json')
    lock = U.read(core/'PRODUCER_LOCK.json')
    if (not completion['complete'] or not completion['unconditional_core_only'] or
            completion['producer_summary_sha256'] != U.sha(core/'SUMMARY.json') or
            completion['producer_lock_sha256'] != U.sha(core/'PRODUCER_LOCK.json') or
            completion['producer_dispatch_sha256'] != U.sha(core/'DISPATCH_COMPLETION.json')):
        raise ValueError('Own F core completion ancestry differs')
    if (lock['arms'] != ['F_COMMON', 'F_BI'] or
            lock['packet_seal_sha256'] != U.sha(run/'F_CALIBRATION/PACKET_MANIFEST.json')):
        raise ValueError('Expected two own F observation/calibration cores')
    for path, digest in lock['numerical_kernel_source_binding'].items():
        if U.sha(Path(__file__).resolve().parents[3]/path) != digest:
            raise ValueError('Completed F numerical source changed')
    for arm in ('F_COMMON', 'F_BI'):
        for domain, count in (('real', 238), ('synthetic', 1003)):
            folder = core/arm/domain; seal = U.read(folder/'FEATURE_SEAL.json')
            if (not seal['complete'] or seal['rows'] != count or
                    seal['producer_lock_sha256'] != U.sha(core/'PRODUCER_LOCK.json') or
                    seal['head_features_sha256'] != U.sha(folder/'FEATURES.jsonl.gz')):
                raise ValueError('All four own F core feature seals required before downstream work')
    chosen = choice['selected_configuration']
    expected = U.read(run/'F_AUXILIARY_MAPPING_IMPLEMENTATION_NOTE.json')['root_authorized_mapping'][chosen]
    if choice['auxiliary_actions'] != expected:
        raise ValueError('Frozen auxiliary mapping changed')
    out.mkdir(parents=True, exist_ok=True)
    completed = out/'FINISH_RECEIPT.json'
    if completed.exists():
        receipt = U.read(completed)
        if (not receipt.get('complete') or receipt['controller_sha256'] != U.sha(__file__) or
                receipt['F_choice_sha256'] != U.sha(run/'F_SYSTEM_CHOICE.json')):
            raise ValueError('Completed F finalization binding differs')
        for path, digest in receipt['sealed_artifact_paths'].items():
            if U.sha(path) != digest:
                raise ValueError('Completed F finalization output changed')
        if receipt['actual_command_ledger_sha256'] != U.sha(out/'POSTPROCESS_LEDGER.jsonl'):
            raise ValueError('Completed F execution ledger changed')
        if receipt['postprocess_source_lock_sha256'] != U.sha(out/'POSTPROCESS_SOURCE_LOCK.json'):
            raise ValueError('Completed F postprocess source lock changed')
        repo = Path(__file__).resolve().parents[3]
        for path, digest in U.read(out/'POSTPROCESS_SOURCE_LOCK.json')['sources'].items():
            if U.sha(repo/path) != digest:
                raise ValueError('Completed F postprocess source changed')
        return receipt
    launch = dict(schema='F_FINALIZATION_LAUNCH_V1', argv=sys.argv,
        cwd=str(Path.cwd()), controller_sha256=U.sha(__file__),
        F_choice_sha256=U.sha(run/'F_SYSTEM_CHOICE.json'),
        core_summary_sha256=U.sha(core/'SUMMARY.json'),
        core_dispatch_sha256=U.sha(core/'DISPATCH_COMPLETION.json'),
        core_controller_completion_sha256=U.sha(run/'F_FOUR_CPU_CONTROLLER_COMPLETION.json'),
        current_real_reference_reads=0, seeds=list(SEEDS), new_H1_fits_authorized=6)
    U.write(out/'LAUNCH.json', launch)
    from . import heads as H, kernel_extra as K, material_finish as MF
    closure = dict(K.source_binding())
    closure.update(H.source_hashes())
    closure.update({str(Path(__file__).resolve()): U.sha(__file__),
                    str(Path(MF.__file__).resolve()): U.sha(MF.__file__)})
    U.write(out/'POSTPROCESS_SOURCE_LOCK.json', dict(sources=closure,
        current_real_reference_reads=0, source_scope='F controller, both child adapters and their numerical/head dependency closures'))
    def checked_execute(phase, argv):
        repo = Path(__file__).resolve().parents[3]
        for path, digest in closure.items():
            if U.sha(repo/path) != digest:
                raise ValueError('Frozen F postprocessing source changed: '+path)
        execute(phase, argv, out)
    python = Path(sys.executable)
    features = {}
    for domain, name in (('real', 'REAL238'), ('synthetic', 'SYNTHETIC1003')):
        inputs = []
        if expected:
            pf = (run/'s1_synthetic_pf/PREDICTIONS.jsonl.gz' if domain == 'synthetic' else
                  run.parent/'pallet-accuracy-closeout-20261017-v1/posefix192_v1/PREDICTIONS.jsonl.gz')
            inputs.append(dict(path=str(pf), seal=str(pf.parent/'PREDICTION_SEAL.json'), methods=PF_METHODS))
            if chosen in MATERIAL:
                inputs.append(dict(path=str(run/'s3_s4_production'/f'EXTRAS_{chosen}_{domain}.jsonl.gz'),
                    seal=str(run/'s3_s4_production/FULL_RECEIPT.json'), methods=[chosen]))
            U.write(out/(domain+'_EXTRA_INPUTS.json'), dict(extra_inputs=inputs))
        for arm in ('F_COMMON', 'F_BI'):
            folder = core/arm/domain
            feature_name = 'FEATURES.jsonl.gz'
            if inputs:
                folder = out/arm/(domain+'_union'); feature_name = 'HEAD_FEATURES.jsonl.gz'
                checked_execute(arm+'_'+domain+'_union', [python, '-B', '-m', NAMESPACE+'kernel_extra',
                    '--inputs-path', d1/'d1_upstream_v1'/f'{name}_UPSTREAM.jsonl.gz',
                    '--core-root', core, '--core-arm', arm, '--domain', domain,
                    '--extra-inputs-manifest', out/(domain+'_EXTRA_INPUTS.json'),
                    '--config-path', old/'CONFIG_FROZEN.json', '--output', folder,
                    '--run', run, '--prior-run', prior, '--root-go'])
            seal = U.read(folder/'FEATURE_SEAL.json')
            if not seal['complete'] or seal['rows'] != {'real': 238, 'synthetic': 1003}[domain]:
                raise ValueError('Both own full feature distributions required')
            if U.sha(folder/feature_name) != seal['head_features_sha256']:
                raise ValueError('Completed own F feature bytes differ')
            features[(arm, domain)] = dict(path=str(folder/feature_name), seal=str(folder/'FEATURE_SEAL.json'))
    artifacts = {}; validation = {}
    for arm in ('F_COMMON', 'F_BI'):
        synthetic, real = features[(arm, 'synthetic')], features[(arm, 'real')]
        labels, training, prediction = (out/arm/k for k in ('labels', 'training', 'prediction'))
        checked_execute(arm+'_labels', [python, '-B', '-m', NAMESPACE+'heads', 'prepare',
            '--features', synthetic['path'], '--feature-seal', synthetic['seal'],
            '--targets', old/'synthetic_v1/SYNTH_GT_LABELS.jsonl.gz',
            '--target-seal', old/'synthetic_v1/PROVISION_SEAL.json',
            '--split-lock', old/'synthetic_v1/DATA_SPLIT_LOCK.json', '--output', labels, '--run', run])
        checked_execute(arm+'_train', [python, '-B', '-m', NAMESPACE+'heads', 'train',
            '--features', synthetic['path'], '--feature-seal', synthetic['seal'],
            '--labels-dir', labels, '--output', training, '--arm', arm, '--run', run])
        checked_execute(arm+'_predict', [python, '-B', '-m', NAMESPACE+'heads', 'predict',
            '--features', real['path'], '--feature-seal', real['seal'],
            '--training-dir', training, '--output', prediction, '--arm', arm, '--run', run])
        local_bindings = {}
        validation[arm] = own_validation(training, prediction, local_bindings)
        artifacts.update(local_bindings)
        for path in (labels/'LABEL_SEAL.json', labels/'ACTION_TARGETS.jsonl.gz',
                     labels/'TRAIN_ONLY_NORMALIZER.json', labels/'NEW_CAMPAIGN_LABEL_BINDING.json'):
            artifacts[str(path)] = U.sha(path)
        for receipt in U.read(training/'TRAINING_MANIFEST.json')['runs']:
            folder = Path(receipt['checkpoint']).parent
            for name in ('RUN_RECEIPT.json', 'HISTORY.json', 'best.pt', 'last.pt', 'resume.pt'):
                artifacts[str(folder/name)] = U.sha(folder/name)
        for spec in (synthetic, real):
            for key in ('path', 'seal'):
                artifacts[spec[key]] = U.sha(spec[key])
    U.write(out/'VALIDATION_SELECTION_INPUTS.json', validation)
    artifacts[str(out/'VALIDATION_SELECTION_INPUTS.json')] = U.sha(out/'VALIDATION_SELECTION_INPUTS.json')
    for path, digest in closure.items():
        if U.sha(Path(__file__).resolve().parents[3]/path) != digest:
            raise ValueError('F source changed during the final child')
    receipt = dict(schema='F_OWN_SIX_FIT_FINISH_RECEIPT_V1', complete=True,
        arms=['F_COMMON', 'F_BI'], H1_fits=6, rows_per_arm=238, seeds=list(SEEDS),
        selected_configuration=chosen, controller_sha256=U.sha(__file__),
        F_choice_sha256=U.sha(run/'F_SYSTEM_CHOICE.json'), launch_sha256=U.sha(out/'LAUNCH.json'),
        postprocess_source_lock_sha256=U.sha(out/'POSTPROCESS_SOURCE_LOCK.json'),
        all_real_predictions_sealed_before_reference_decode=True, current_real_reference_reads=0,
        feature_inputs={a: {d: features[(a, d)] for d in ('real', 'synthetic')} for a in ('F_COMMON', 'F_BI')},
        sealed_artifact_paths=artifacts, actual_command_ledger_sha256=U.sha(out/'POSTPROCESS_LEDGER.jsonl'))
    U.write(out/'FINISH_RECEIPT.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('choose', 'finish'))
    for name in ('run', 's2-core', 's2-heads', 'd1', 'old-run', 'prior-run', 'core', 'output'):
        parser.add_argument('--'+name, type=Path)
    args = parser.parse_args()
    if args.stage == 'choose':
        value = choose(args.run, args.s2_core, args.s2_heads)
        print(U.canonical({k: value[k] for k in ('complete', 'selected_configuration', 'auxiliary_actions')}), flush=True)
    else:
        value = finish(args.run, args.d1, args.old_run, args.prior_run, args.core, args.output)
        print(U.canonical({k: value[k] for k in ('complete', 'H1_fits', 'selected_configuration')}), flush=True)


if __name__ == '__main__':
    main()

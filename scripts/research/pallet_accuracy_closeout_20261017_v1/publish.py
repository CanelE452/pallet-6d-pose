"""Copy complete small fixed results and alias local provenance paths only."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import resource
import shutil
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def run(repo, private, output, state):
    started, cpu = time.monotonic(), time.process_time()
    repo, private, out, state = map(Path, (repo, private, output, state))
    aliases = sorted([(str(p), a) for p, a in [
        (private, 'PRIVATE_RUN'), (state, 'STATE_CACHE'), (repo, 'SOURCE_ROOT'),
        (Path.home()/'Documents/github/pallet-pose', 'ASSET_ROOT'),
        (Path.home()/'Documents/pallet-research', 'RESEARCH_CACHE'),
        (Path.home()/'anaconda3', 'CONDA_ROOT')]], key=lambda x: len(x[0]), reverse=True)

    def alias(value):
        if isinstance(value, str):
            for source, replacement in aliases:
                value = value.replace(source, replacement)
            return value.replace(str(Path.home())+'/', 'LOCAL_ACCOUNT/').replace('/tmp/', 'LOCAL_TEMP/')
        if isinstance(value, list):
            return [alias(v) for v in value]
        if isinstance(value, dict):
            return {alias(k): alias(v) for k, v in value.items()}
        return value

    derivations = []

    def copied(relative, target):
        source, destination = private/relative, out/target
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            assert sha(source) == sha(destination), 'Existing public bytes differ: '+target
        else:
            shutil.copyfile(source, destination)
        derivations.append(dict(source='PRIVATE_RUN/'+relative, public=target,
            original_sha256=sha(source), public_sha256=sha(destination), derivation='exact byte copy'))

    def receipt(relative, target):
        source, destination = private/relative, out/target
        value = alias(read(source))
        if destination.exists():
            assert read(destination) == value, 'Existing aliased receipt differs: '+target
        else:
            dump(destination, value)
        derivations.append(dict(source='PRIVATE_RUN/'+relative, public=target,
            original_sha256=sha(source), public_sha256=sha(destination),
            derivation='local path aliases and JSON serialization only; scientific values unchanged'))

    # Author-produced prose/renderer receipts are retained privately before
    # replacing machine-specific example paths in the public presentation.
    originals = private/'publication_original_presentation'
    originals.mkdir(exist_ok=False)
    for name in ('REPRODUCE.md', 'POSEFIX_METHOD_AND_PARITY_KO.md'):
        source = out/name
        shutil.copyfile(source, originals/name)
        text = source.read_text().replace(str(Path.home()/'anaconda3/envs/pallet-yolo26/bin/python'), '/path/to/existing/pallet-yolo26/python')
        text = text.replace(str(state), '/path/to/existing/STATE_CACHE/score_v2')
        source.write_text(text)
        derivations.append(dict(source='PRIVATE_RUN/publication_original_presentation/'+name,
            public=name, original_sha256=sha(originals/name), public_sha256=sha(source),
            derivation='public executable/cache example path placeholders only'))
    visual = out/'gallery/VISUALIZATION_RECEIPT.json'
    shutil.copyfile(visual, originals/visual.name)
    visual.write_text(json.dumps(alias(read(visual)), indent=2, ensure_ascii=False)+'\n')
    derivations.append(dict(source='PRIVATE_RUN/publication_original_presentation/'+visual.name,
        public='gallery/VISUALIZATION_RECEIPT.json', original_sha256=sha(originals/visual.name),
        public_sha256=sha(visual), derivation='renderer source/input local paths aliased only; plots unchanged'))
    sums = out/'gallery/SHA256SUMS.txt'
    shutil.copyfile(sums, originals/sums.name)
    sums.write_text(''.join(sha(visual.parent/name)+'  '+name+'\n' for name in
        ('success_primary192.png', 'error_ecdf_primary192.png', 'index.html', 'VISUALIZATION_RECEIPT.json')))

    for name in ('POSEFIX_192_PREDICTIONS.jsonl.gz', 'POSEFIX_CAL46_PREDICTIONS.jsonl.gz'):
        copied('posefix192_v1/'+name, name)
    copied('posefix192_v1/PREDICTIONS.jsonl.gz', 'predictions/POSEFIX_ALL238.jsonl.gz')
    copied('posefix_parity95_v1/PREDICTIONS.jsonl.gz', 'parity95/PREDICTIONS.jsonl.gz')
    copied('posefix_inputs_v1/POSEFIX_INPUTS238.jsonl.gz', 'inputs/POSEFIX_INPUTS238.jsonl.gz')
    receipt('posefix_parity95_v1/POSEFIX_PARITY95.json', 'POSEFIX_PARITY95.json')
    receipt('POSEFIX_192_PROTOCOL_LOCK.json', 'POSEFIX_192_PROTOCOL_LOCK.json')
    receipt('INDEPENDENT_ARITHMETIC_V1.json', 'INDEPENDENT_ARITHMETIC.json')
    receipt('INDEPENDENT_CONTRACT_CHECK.json', 'INDEPENDENT_CONTRACT_CHECK.json')
    for name in ('START_BEFORE_FETCH.json', 'EXECUTION_START.json',
            'INTEGRATION_TESTS.json', 'INTEGRATION_TESTS_V2.json',
            'INDEPENDENT_CONTRACT_CHECK_ATTEMPT1_ROLE_LABEL.json', 'INDEPENDENT_CONTRACT_CHECK_REPAIR.json'):
        receipt(name, 'evidence/'+name)
    for folder, names in {
        'posefix_inputs_v1': ['SOURCE_AND_INPUT_BINDINGS.json', 'PREPARATION_RECEIPT.json'],
        'posefix192_v1': ['PREDICTION_SEAL.json', 'IO_BOUNDARY.json', 'RAW143_CACHE_PARITY.json', 'CAP_AND_Q_INVARIANTS.json'],
        'posefix_parity95_v1': ['PREDICTION_SEAL.json', 'IO_BOUNDARY.json', 'POSTSEAL_ACCURACY_PARITY95.json'],
        'e_contract_audit_v1': ['AUDIT_INPUT_SHA256.json'],
        'e_contract_audit_witness_v1': ['CAL_OWNER_PHASE_WITNESS.json'],
        'e_contract_audit_witness_v2': ['CAL_OWNER_PHASE_WITNESS.json'],
    }.items():
        for name in names:
            receipt(folder+'/'+name, 'evidence/'+folder+'/'+name)

    binding = dict(schema='FAIR_POSEFIX_E_CLOSEOUT_SOURCE_INPUT_BINDINGS_V1',
        execution_start=read(private/'EXECUTION_START.json'),
        original_state_result='407de865445e4987614f50f893d2ccea78d507b6',
        original_score_result='c56a4e92027a8e56f6704773f24d7c3da78082f1',
        prior_main='b843726ee2a78d7d29b79dafecdfc9d982b266fd',
        posefix=read(private/'posefix_inputs_v1/SOURCE_AND_INPUT_BINDINGS.json'),
        posefix_protocol_original_sha256=sha(private/'POSEFIX_192_PROTOCOL_LOCK.json'),
        posefix_prediction_seal_original_sha256=sha(private/'posefix192_v1/PREDICTION_SEAL.json'),
        baseline_score_seal_sha256=sha(state/'SCORE_SEAL.json'),
        fixed_reference_sha256=sha(state/'FIXED_REFERENCES.json'),
        e_audit_inputs=read(private/'e_contract_audit_v1/AUDIT_INPUT_SHA256.json'),
        new_source_sha256={str(p.relative_to(repo)): sha(p) for p in sorted((repo/'scripts/research/pallet_accuracy_closeout_20261017_v1').glob('*.py'))},
        protected_old_inputs_weights_models_defaults_papers_modified=False,
        scientific_change='NO_SUPPORTED_CHANGE', new_E192_method_evaluation=False,
        PF_outputs_are_new_F_postprocessing_of_existing_neural_cache=True,
        source_path_literal_note='posefix_replay.py contains one historical absolute provenance-key literal used to match unchanged original95 protocol keys; it is not a runtime data path')
    dump(out/'SOURCE_AND_INPUT_BINDINGS.json', alias(binding))
    dump(out/'PUBLIC_DERIVATIONS.json', dict(records=derivations, private_originals_and_failures_preserved=True,
        no_scientific_values_changed_by_publication=True, wall_seconds=time.monotonic()-started,
        CPU_seconds=time.process_time()-cpu, peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    measured = []
    for filename in ('posefix_inputs_v1/PREPARATION_RECEIPT.json', 'posefix_parity95_v1/PREDICTION_SEAL.json',
            'posefix192_v1/PREDICTION_SEAL.json', 'baseline_audit_v1/BASELINE_REPRODUCTION.json',
            'fair_evaluation_v1/FAIR_RESULTS.json', 'e_contract_audit_v1/CAL_PROVENANCE_SUMMARY.json',
            'INDEPENDENT_ARITHMETIC_V1.json', 'INDEPENDENT_CONTRACT_CHECK.json',
            'INTEGRATION_TESTS.json', 'INTEGRATION_TESTS_V2.json'):
        original = read(private/filename)
        value = original.get('timing', original)
        measured.append(dict(stage=filename, wall_seconds=value.get('wall_seconds', value.get('elapsed_seconds')),
            CPU_seconds=value.get('CPU_seconds', value.get('cpu_seconds', value.get('process_cpu_seconds'))),
            peak_RSS_KiB=value.get('peak_RSS_KiB', value.get('peak_rss_kib', value.get('max_rss_kib', value.get('maximum_resident_set_size_kib')))),
            exit_code=original.get('exit_code', 0)))
    rendering = read(visual)['resources']
    measured.append(dict(stage='CSV visualization', wall_seconds=rendering['wall_seconds_through_artifact_write'],
        CPU_seconds=rendering['cpu_seconds_through_artifact_write'], peak_RSS_KiB=1024*rendering['process_peak_rss_MiB'], exit_code=0))
    dump(out/'COST.json', dict(schema='FAIR_ACCURACY_CLOSEOUT_COST_V1', measured_stages=measured,
        known_measured_stage_wall_sum_seconds=sum(r['wall_seconds'] or 0 for r in measured),
        known_measured_CPU_sum_seconds=sum(r['CPU_seconds'] or 0 for r in measured),
        maximum_measured_peak_RSS_KiB=max(r['peak_RSS_KiB'] or 0 for r in measured),
        new_NN_forwards=0, new_training_steps=0, new_model_seed_or_data=0,
        common95_F_invocations=475, original238_fixed_PF_F_invocations=714,
        new_E_method_F_or_optimizer_invocations=0, new_E192_optin_evaluation=False,
        new_E_retained_candidate_bank_generation=0, new_covariance_fit_or_weight_search=0,
        annotation_reference_weights_defaults_papers_modified=False,
        failed_checker_and_packaging_attempts_preserved=True,
        elapsed_since_workspace_snapshot_seconds=time.time()-read(private/'START_BEFORE_FETCH.json')['start_epoch'],
        model_visual_review_seconds=None, model_visual_review_separately_not_inference_latency=True,
        notes='Cache-based classical F/SubPix and fixed-reference diagnostics, not end-to-end neural latency. Missing attempt CPU/RSS is not inferred; final publication QA timing is in VALIDATION.json.'))
    return dict(copied_records=len(derivations), new_source_files=len(binding['new_source_sha256']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('repo', 'private', 'output', 'state'):
        parser.add_argument('--'+key, required=True)
    a = parser.parse_args()
    print(run(a.repo, a.private, a.output, a.state))

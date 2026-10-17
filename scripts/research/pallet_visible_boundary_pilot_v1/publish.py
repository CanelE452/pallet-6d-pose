"""Publish small complete outputs, with explicit private/public path aliases.

Does not rerun scoring or evaluation. Original seals and failed attempts stay
in the private run; aliased receipt bytes have explicit derivation hashes.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import resource
import shutil
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else Path(path).open()) as stream:
        return [json.loads(line) for line in stream]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def run(source_root, asset_root, private_root, state_private, output):
    started, cpu = time.monotonic(), time.process_time()
    source, asset, private, state, out = map(Path, (source_root, asset_root, private_root, state_private, output))
    out.mkdir(parents=True, exist_ok=True)
    aliases = sorted([(str(path), alias) for path, alias in [
        (private, 'PRIVATE_VISIBLE'), (state, 'PRIVATE_STATE'),
        (source, 'SOURCE_ROOT'), (asset, 'ASSET_ROOT'),
        (Path.home()/'Documents/pallet-research', 'RESEARCH_CACHE'),
        (Path.home()/'anaconda3', 'CONDA_ROOT')]], key=lambda item: len(item[0]), reverse=True)

    def clean(value):
        if isinstance(value, str):
            for path, alias in aliases:
                value = value.replace(path, alias)
            value = value.replace(str(Path.home())+'/', 'LOCAL_ACCOUNT/').replace('/tmp/', 'LOCAL_TEMP/')
            if '/home/' in value or '/Users/' in value:
                raise ValueError('Unbound private path')
        elif isinstance(value, list):
            value = [clean(v) for v in value]
        elif isinstance(value, dict):
            value = {clean(k): clean(v) for k, v in value.items()}
        return value

    derived = []

    def receipt(relative, destination=None):
        path = private/relative
        if not path.is_file():
            return
        destination = destination or 'evidence/'+relative
        target = out/destination
        if path.suffix == '.jsonl':
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('x') as stream:
                for value in rows(path):
                    stream.write(json.dumps(clean(value), ensure_ascii=False, separators=(',', ':'))+'\n')
        else:
            dump(target, clean(read(path)))
        derived.append(dict(source='PRIVATE_VISIBLE/'+relative, original_sha256=sha(path),
            public=destination, public_sha256=sha(target), derivation='path aliases and JSON serialization only; coordinates/scientific values unchanged'))

    # Actual scientific outputs are copied byte-for-byte from the sealed run.
    for name in ('CANDIDATE_SCORES.jsonl.gz', 'PREDICTIONS.jsonl.gz', 'OBSERVATION_QUADRATURE.jsonl.gz', 'PREDICTION_SEAL.json'):
        target = out/name
        if target.exists():
            raise FileExistsError(target)
        shutil.copyfile(private/'actual_v2'/name, target)
    for folder in ('eval_v1', 'diagnostic_v1', 'observation_context_v1'):
        for path in sorted((private/folder).iterdir()):
            if path.suffix == '.csv' or path.name == 'SUMMARY.json':
                if (out/path.name).exists():
                    raise FileExistsError(out/path.name)
                shutil.copyfile(path, out/path.name)
            elif path.suffix == '.json':
                receipt(str(path.relative_to(private)), 'evidence/'+path.name)
    receipt('VISIBLE_INPUTS.jsonl', 'VISIBLE_INPUTS.jsonl')
    for name in ('START_BEFORE_FETCH.json', 'INPUT_REUSE_RECEIPT.json', 'PROTOCOL_LOCK.json',
            'INPUT_AUTHOR_REVIEW.json', 'KERNEL_ATTEMPT_HISTORY.json', 'KERNEL_SOURCE_READ_SCOPE.json', 'INDEPENDENT_CHECK_V1.json',
            'INDEPENDENT_MATCHED_CONTROL_CSV_CHECK.json', 'INDEPENDENT_SYNTHETIC_CHECK.json',
            'REQUIRED_DOCUMENT_READS.json', 'FINAL_VISUAL_REVIEW.json'):
        receipt(name)
    receipt('PROTOCOL_LOCK.json', 'PROTOCOL_LOCK.json')
    receipt('INDEPENDENT_CHECK_V1.json', 'INDEPENDENT_VALIDATION.json')
    receipt('observation_context_v1/OBSERVATION_CONTEXT_RECEIPT.json', 'OBSERVATION_CONTEXT_RECEIPT.json')
    for folder in ('baseline_empty_v1', 'actual_v1', 'actual_v2'):
        for name in ('PREDICTION_SEAL.json', 'ATTEMPT_STATUS.json'):
            receipt(folder+'/'+name)
    for name in ('NEUTRAL_PACKAGE.json', 'EXPORT_RECEIPT.json'):
        receipt('neutral_v1/'+name)
    snapshot = private/'INDEPENDENT_SOURCE_AT_V1.py'
    if snapshot.is_file():
        text = snapshot.read_text()
        if clean(text) != text:
            raise ValueError('Independent source snapshot needs a declared derivation')
        shutil.copyfile(snapshot, out/'evidence/INDEPENDENT_SOURCE_AT_V1.py')

    package_path = private/'neutral_v1/NEUTRAL_PACKAGE.json'
    package = read(package_path)
    frames = rows(package_path.parent/package['frames']['path'])
    candidates = rows(package_path.parent/package['candidates']['path'])
    lookup = {(c['id'], c['candidate_id']): c for c in candidates}
    predictions = rows(private/'actual_v2/PREDICTIONS.jsonl.gz')
    geometry = []
    for p in predictions:
        c = lookup[(p['id'], p['candidate_id'])]
        f = next(f for f in frames if f['id'] == p['id'])
        geometry.append(dict(id=p['id'], variant=p['variant'], stage=c['stage'], family=c['family'],
            candidate_id=c['candidate_id'], K=f['K'], native_xyz=c['native_xyz'], R=c['R'], t=c['t'],
            physical_registration_Q=c['physical_registration_Q'], cf_extents=c['cf_extents'],
            original_candidate_pose=c['pose'], same_as_sealed_output_R_cf_and_centroid=True,
            note='Read-only exact extraction; centroid equals original native_t for this centered model; no candidate generated.'))
    with gzip.open(out/'FIXED_SELECTION_GEOMETRY.jsonl.gz', 'wt') as stream:
        for value in geometry:
            stream.write(json.dumps(value, separators=(',', ':'))+'\n')
    dump(out/'REGISTRATION_CONTEXT.json', clean([dict(id=f['id'], ordinal=f['ordinal'], image=f['image'],
        image_sha256=f['image_sha256'], image_hw=f['image_hw'], K=f['K'], states=f['states'],
        original_outlines=f['outlines'], original_stage_family_metadata=f['stages']) for f in frames]))

    dependencies = [source/'scripts/research/pallet_non_target_outline_pilot_v1'/name for name in ('kernel.py', 'replay.py')]
    dependencies += [source/'scripts/research/pallet_selection_score_diagnosis_v1'/name for name in ('selection.py', 'runtime_guard.py')]
    dependencies += [source/'scripts/research/pallet_statewise_correspondence_diagnosis_v1/analysis.py']
    dependencies += list((source/'scripts/research/pallet_visible_boundary_pilot_v1').glob('*.py'))
    exporter = read(private/'neutral_v1/EXPORT_RECEIPT.json')
    assets = []
    for record in exporter['original_assets']:
        path = Path(record['path'])
        if not path.exists() or sha(path) != record['sha256']:
            raise ValueError('Direct original RGB/target mask asset changed: '+str(path))
        assets.append(dict(path=str(path), sha256=record['sha256'], kind=record['kind']))
    binding = dict(schema='VISIBLE_BOUNDARY_PILOT_BINDINGS_V1', execution_base=read(private/'INPUT_REUSE_RECEIPT.json'),
        instruction_sha256=read(private/'START_BEFORE_FETCH.json')['instruction_sha256'],
        original_state_result_commit='407de865445e4987614f50f893d2ccea78d507b6',
        original_cost_result_commit='c56a4e92027a8e56f6704773f24d7c3da78082f1',
        prior_pilot_main='f87a3d4fcfbf5c2bc638b3fbe5613a83f707bb72',
        neutral_package=package, direct_source_sha256={str(p.relative_to(source)): sha(p) for p in sorted(dependencies)},
        direct_assets=assets, export_receipt_sha256=sha(private/'neutral_v1/EXPORT_RECEIPT.json'),
        protocol_lock=read(private/'PROTOCOL_LOCK.json'), private_input_sha256=sha(private/'VISIBLE_INPUTS.jsonl'),
        public_input_sha256=sha(out/'VISIBLE_INPUTS.jsonl'),
        fixed_reference_sha256=sha(state/'score_v2/FIXED_REFERENCES.json'),
        fixed_reference_seal_sha256=sha(state/'score_v2/SCORE_SEAL.json'),
        selection_geometry_sha256=sha(out/'FIXED_SELECTION_GEOMETRY.jsonl.gz'),
        original_prior_audit_reused=True, all_779_sources_or_11603_costs_reaudited=False,
        new_model_forward=0, new_pose_solve=0, new_candidate=0, defaults_papers_weights_references_modified=False)
    dump(out/'INPUT_BINDINGS.json', clean(binding))
    dump(out/'PUBLIC_DERIVATIONS.json', dict(records=derived, original_private_attempts_preserved=True,
        immutable_scientific_outputs_copied_exactly=True,
        input_coordinates_and_scientific_values_unchanged=True,
        packaging_elapsed_seconds=time.monotonic()-started,
        packaging_process_cpu_seconds=time.process_time()-cpu,
        maximum_resident_set_size_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    return dict(candidates=len(candidates), prediction_rows=len(predictions), receipt_derivations=len(derived))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('source-root', 'asset-root', 'private-root', 'state-private', 'output'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    print(run(args.source_root, args.asset_root, args.private_root, args.state_private, args.output))

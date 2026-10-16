"""Restore exact V8 originals from direct gzip files or ordered byte parts.

Separate own freeze/run records each destination's actual initial presence.
An already correct destination is checked and left untouched. New originals
are verified in an exclusive pending file before exclusive hard-link promotion.
No production, private dependency, model, truth or numerical solver is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

REPO = Path(__file__).resolve().parents[3]
DOC = REPO / '_docs/experiments/pallet_same_observation_controls_20261010_v8'
FILES = ('RESTORE_PROTOCOL.json', 'RESTORE_STARTED.json', 'RESTORE_CHECKS.json')
ENCODING = 'ordered byte slices of original gzip; concatenate without recompression'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def reject_symlink(path):
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'symlink: ' + str(path))


def read(path):
    with Path(path).open('r', encoding='utf-8') as stream:
        return json.load(stream)


def safe(folder, name):
    relative = Path(name)
    require(isinstance(name, str) and not relative.is_absolute() and '..' not in relative.parts and
        name not in ('', '.'), 'unsafe manifest path')
    path = folder / relative
    reject_symlink(path)
    require(path.absolute().is_relative_to(folder.absolute()), 'manifest path escapes root')
    return path


def binding(path):
    path = Path(path)
    reject_symlink(path)
    require(path.is_file(), 'missing input ' + str(path))
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return dict(path=path.name, bytes=path.stat().st_size, sha256=h.hexdigest())


def same(left, right):
    return all(left[key] == right[key] for key in ('bytes', 'sha256'))


def write_new(path, value):
    reject_symlink(path)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def guard(args, stage):
    source, target, receipts = (p.absolute() for p in (args.input,args.output,args.receipts))
    for path in (source, target, receipts):
        reject_symlink(path)
    require(source.is_dir(), 'archive input directory missing')
    target.mkdir(parents=True, exist_ok=True)
    receipts.mkdir(parents=True, exist_ok=True)
    for name in FILES if stage == 'freeze' else FILES[1:]:
        require(not (receipts / name).exists(), 'preserve existing ' + name)
    return source, target, receipts


def manifest_inputs(source):
    manifest_path = source / 'ARCHIVE_MANIFEST.json'
    manifest = read(manifest_path)
    require(manifest['schema'] == 'same_observation_original_gzip_archive_manifest_v8' and
        manifest['encoding'] == ENCODING and manifest['original_files_preserved'] is True,
        'unsupported original-byte archive')
    files = {'ARCHIVE_MANIFEST.json':manifest_path, 'restore_code':Path(__file__).resolve()}
    names = []
    for item in manifest['files']:
        original = item['original']
        name = original['path']
        require(Path(name).name == name and name.endswith('.gz') and name not in names,
                'unique original gzip basename required')
        names.append(name)
        storage = item['storage']
        require(storage in ('direct_original','ordered_parts'), 'unknown archive storage')
        require(type(original['bytes']) is int and original['bytes'] >= 0, 'original byte count')
        parts = item['parts']
        if storage == 'direct_original':
            require(parts == [], 'direct original unexpectedly has parts')
            path = safe(source,name)
            require(same(binding(path), original), 'direct original differs: ' + name)
            files['original:' + name] = path
        else:
            require(parts and sum(p['bytes'] for p in parts) == original['bytes'], 'part byte total')
            for index, part in enumerate(parts):
                expected_name = 'archives/' + name + '.part%03d' % index
                require(part['path'] == expected_name and 0 < part['bytes'] <= 40 * 1024 * 1024,
                        'ordered bounded part path/size required')
                path = safe(source,part['path'])
                require(same(binding(path),part), 'part differs: ' + expected_name)
                files['part:' + expected_name] = path
    require(len(names) == manifest['original_file_count'] and len(set(names)) == len(names) and names,
            'manifest original count')
    return manifest, files


def targets(target, manifest):
    result = {}
    for item in manifest['files']:
        name = item['original']['path']
        path = safe(target,name)
        pending = safe(target,'RESTORE_PENDING_' + name)
        require(not pending.exists(), 'preserve existing restoration prefix ' + pending.name)
        exists = path.exists()
        if exists:
            require(same(binding(path),item['original']), 'existing target differs; preserve it: ' + name)
        result[name] = dict(initially_present=exists, existing_binding=binding(path) if exists else None)
    return result


def freeze(args):
    source, target, receipts = guard(args,'freeze')
    manifest, files = manifest_inputs(source)
    state = targets(target,manifest)
    own = dict(schema='supplemental_same_observation_restore_protocol_v8',
        inputs={name:binding(path) for name,path in files.items()}, source_folder=str(source),
        target_folder=str(target), initial_targets=state, frozen_before_own_restoration=True,
        recompression_calls=0, already_correct_targets_left_unchanged=True,
        exclusive_verified_pending_promotion=True, no_automatic_retry=True,
        model_PnP_training_RGB_GT_metric_arithmetic_calls=0)
    write_new(receipts / FILES[0],own)
    print(json.dumps(dict(frozen=True, protocol=binding(receipts / FILES[0]),
        initially_absent=sum(not x['initially_present'] for x in state.values()))), flush=True)


def run(args):
    source, target, receipts = guard(args,'run')
    own = read(receipts / FILES[0])
    require(own['schema'] == 'supplemental_same_observation_restore_protocol_v8' and
        own['source_folder'] == str(source) and own['target_folder'] == str(target), 'own restore roots differ')
    manifest, files = manifest_inputs(source)
    current = {name:binding(path) for name,path in files.items()}
    require(current == own['inputs'], 'frozen archive/checker bytes changed')
    state = targets(target,manifest)
    require(state == own['initial_targets'], 'destination presence changed since own freeze')
    write_new(receipts / FILES[1],dict(protocol=binding(receipts / FILES[0]),inputs=current,
        initial_targets=state, actual_restoration_runs=1,
        model_PnP_training_RGB_GT_metric_arithmetic_calls=0))
    start = time.perf_counter()
    result = dict(schema='supplemental_same_observation_restore_checks_v8',complete=False,passed=False,
        protocol=binding(receipts / FILES[0]),inputs=current,files=[],failure_count=0,failures=[],
        initial_targets=state,recompression_calls=0,no_automatic_retry=True,
        model_PnP_training_RGB_GT_metric_arithmetic_calls=0)
    try:
        for item in manifest['files']:
            original = item['original']
            name = original['path']
            path = safe(target,name)
            existing = state[name]['initially_present']
            if not existing:
                pending = safe(target,'RESTORE_PENDING_' + name)
                sources = ([safe(source,name)] if item['storage'] == 'direct_original' else
                    [safe(source,p['path']) for p in item['parts']])
                with pending.open('xb') as stream:
                    for part in sources:
                        with part.open('rb') as original_stream:
                            for block in iter(lambda: original_stream.read(8 * 1024 * 1024), b''):
                                stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
                require(same(binding(pending),original), 'pending restored gzip differs: ' + name)
                os.link(pending,path)  # Fails atomically if another writer created the destination.
                pending.unlink()
            require(same(binding(path),original), 'restored gzip differs: ' + name)
            result['files'].append(dict(path=name,bytes=original['bytes'],sha256=original['sha256'],
                verified=True,initially_absent=not existing,existing_unchanged=existing,
                actually_created=not existing,storage=item['storage']))
        require({name:binding(path) for name,path in files.items()} == current, 'archive inputs changed during restore')
        result.update(complete=True,passed=True,original_files=len(result['files']),
            actual_created_files=sum(row['actually_created'] for row in result['files']),
            existing_unchanged_files=sum(row['existing_unchanged'] for row in result['files']),
            actual_created_bytes=sum(row['bytes'] for row in result['files'] if row['actually_created']),
            all_targets_byte_identical=True)
    except Exception as error:
        result.update(failure_count=1,failures=[dict(type=type(error).__name__,message=str(error))],
            preserved_pending_files=sorted(p.name for p in target.glob('RESTORE_PENDING_*.gz')))
    result['elapsed_seconds'] = time.perf_counter()-start
    write_new(receipts / FILES[2],result)
    print(json.dumps({key:result.get(key) for key in ('complete','passed','original_files',
        'actual_created_files','existing_unchanged_files','actual_created_bytes','failure_count')}), flush=True)
    if not result['passed']:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','run'))
    parser.add_argument('--input',type=Path,default=DOC,help='published archive directory')
    parser.add_argument('--output',type=Path,default=DOC,help='restore original files here')
    parser.add_argument('--receipts',type=Path,required=True,help='new own protocol/receipt directory')
    args = parser.parse_args()
    freeze(args) if args.stage == 'freeze' else run(args)


if __name__ == '__main__':
    main()

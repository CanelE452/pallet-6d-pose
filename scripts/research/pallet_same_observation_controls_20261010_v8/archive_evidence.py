"""Freeze, then package unchanged V8 gzip evidence as bounded byte slices.

Only originals at least 50 MiB are split into 40 MiB parts. Smaller originals
remain directly published; a zero-split manifest is a valid explicit result.
This standalone standard-library supplement never imports an experiment.
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
FILES = ('ARCHIVE_PROTOCOL.json', 'ARCHIVE_STARTED.json', 'ARCHIVE_CHECKS.json')
PART_BYTES = 40 * 1024 * 1024
SPLIT_MIN_BYTES = 50 * 1024 * 1024
ENCODING = 'ordered byte slices of original gzip; concatenate without recompression'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def reject_symlink(path):
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'symlink: ' + str(path))


def read(path):
    with Path(path).open('r', encoding='utf-8') as stream:
        return json.load(stream)


def binding(path):
    path = Path(path)
    reject_symlink(path)
    require(path.is_file(), 'missing input ' + str(path))
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return dict(path=path.name, bytes=path.stat().st_size, sha256=h.hexdigest())


def write_new(path, value):
    reject_symlink(path)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def guard(args, stage):
    folder, output = args.input.absolute(), args.output.absolute()
    for path in (folder, output):
        reject_symlink(path)
    require(folder.is_dir(), 'input evidence directory missing')
    output.mkdir(parents=True, exist_ok=True)
    for name in FILES if stage == 'freeze' else FILES[1:]:
        require(not (output / name).exists(), 'preserve existing ' + name)
    for name in ('ARCHIVE_MANIFEST.json', 'archives', '.gitignore'):
        require(not (folder / name).exists() and not (folder / name).is_symlink(),
                'preserve existing archive artifact ' + name)
    return folder, output


def inputs(folder):
    originals = sorted(folder.glob('*.gz'), key=lambda p: p.name)
    require(originals, 'no original gzip evidence')
    files = {p.name: p for p in originals}
    files.update({'PROTOCOL.json': folder / 'PROTOCOL.json',
                  'archive_code': Path(__file__).resolve(),
                  'restore_code': Path(__file__).resolve().with_name('restore_archives.py')})
    return files


def freeze(args):
    folder, output = guard(args, 'freeze')
    files = inputs(folder)
    own = dict(schema='supplemental_same_observation_archive_protocol_v8',
        inputs={name: binding(path) for name, path in files.items()},
        original_names=sorted(name for name in files if name.endswith('.gz')),
        original_folder=str(folder), encoding=ENCODING, part_size_bytes=PART_BYTES,
        split_minimum_bytes=SPLIT_MIN_BYTES, smaller_files_remain_direct_originals=True,
        zero_split_manifest_is_valid=True, original_gzip_recompression=False,
        frozen_before_own_packaging=True, no_automatic_retry=True,
        model_PnP_training_RGB_GT_metric_arithmetic_calls=0)
    write_new(output / FILES[0], own)
    print(json.dumps(dict(frozen=True, protocol=binding(output / FILES[0]))), flush=True)


def run(args):
    folder, output = guard(args, 'run')
    own = read(output / FILES[0])
    require(own['schema'] == 'supplemental_same_observation_archive_protocol_v8' and
            own['original_folder'] == str(folder) and own['part_size_bytes'] == PART_BYTES and
            own['split_minimum_bytes'] == SPLIT_MIN_BYTES and own['encoding'] == ENCODING,
            'own frozen archive policy differs')
    files = inputs(folder)
    current = {name: binding(path) for name, path in files.items()}
    require(current == own['inputs'], 'frozen original/helper bytes changed')
    write_new(output / FILES[1], dict(protocol=binding(output / FILES[0]), inputs=current,
        actual_packaging_runs=1, model_PnP_training_RGB_GT_metric_arithmetic_calls=0))
    start = time.perf_counter()
    result = dict(schema='supplemental_same_observation_archive_checks_v8', complete=False,
        passed=False, protocol=binding(output / FILES[0]), inputs=current,
        original_files_preserved=True, original_seal_and_receipts_unchanged=True,
        recompression_calls=0, model_PnP_training_RGB_GT_metric_arithmetic_calls=0,
        no_automatic_retry=True, files=[], parts_created=0, failure_count=0, failures=[])
    try:
        eligible = [name for name in own['original_names'] if current[name]['bytes'] >= SPLIT_MIN_BYTES]
        if eligible:
            (folder / 'archives').mkdir(exist_ok=False)
        for name in own['original_names']:
            path = files[name]
            parts = []
            if name in eligible:
                with path.open('rb') as source:
                    index = 0
                    while True:
                        block = source.read(PART_BYTES)
                        if not block:
                            break
                        relative = 'archives/' + name + '.part%03d' % index
                        target = folder / relative
                        with target.open('xb') as stream:
                            stream.write(block)
                            stream.flush()
                            os.fsync(stream.fileno())
                        part = binding(target)
                        require(part['bytes'] == len(block) and
                            part['sha256'] == hashlib.sha256(block).hexdigest(), 'written part differs')
                        part['path'] = relative
                        parts.append(part)
                        index += 1
                        result['parts_created'] += 1
                require(sum(p['bytes'] for p in parts) == current[name]['bytes'], 'part byte total differs')
                reconstructed = hashlib.sha256()
                for part in parts:
                    with (folder / part['path']).open('rb') as stream:
                        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                            reconstructed.update(block)
                require(reconstructed.hexdigest() == current[name]['sha256'], 'ordered parts differ from original')
            result['files'].append(dict(original=current[name], storage='ordered_parts' if parts else
                'direct_original', parts=parts))
        require({name: binding(path) for name, path in inputs(folder).items()} == current,
                'original/helper bytes changed during packaging')
        manifest = dict(schema='same_observation_original_gzip_archive_manifest_v8',
            encoding=ENCODING, purpose='byte-identical public evidence; no gzip rewriting',
            protocol=binding(output / FILES[0]), archive_code=current['archive_code'],
            restore_code=current['restore_code'], part_size_bytes=PART_BYTES,
            split_minimum_bytes=SPLIT_MIN_BYTES, files=result['files'],
            original_file_count=len(own['original_names']), split_file_count=len(eligible),
            direct_original_file_count=len(own['original_names'])-len(eligible),
            part_count=result['parts_created'], no_split_required=not eligible,
            original_files_preserved=True, original_seal_and_receipts_unchanged=True,
            model_PnP_training_RGB_GT_metric_arithmetic_calls=0)
        write_new(folder / 'ARCHIVE_MANIFEST.json', manifest)
        if eligible:
            with (folder / '.gitignore').open('x', encoding='utf-8') as stream:
                stream.write('# Original bytes are preserved locally and published as verified ordered parts.\n')
                for name in eligible:
                    stream.write('/' + name + '\n')
                stream.flush()
                os.fsync(stream.fileno())
        result.update(complete=True, passed=True, manifest=binding(folder / 'ARCHIVE_MANIFEST.json'),
            original_file_count=len(own['original_names']), split_file_count=len(eligible),
            direct_original_file_count=len(own['original_names'])-len(eligible),
            no_split_required=not eligible, original_bytes=sum(current[n]['bytes'] for n in own['original_names']))
    except Exception as error:
        result.update(failure_count=1, failures=[dict(type=type(error).__name__, message=str(error))],
            preserved_created_parts=sorted(str(p.relative_to(folder)) for p in
                (folder / 'archives').glob('*')) if (folder / 'archives').is_dir() else [])
    result['elapsed_seconds'] = time.perf_counter()-start
    write_new(output / FILES[2], result)
    print(json.dumps({key: result.get(key) for key in ('complete','passed','original_file_count',
        'split_file_count','direct_original_file_count','parts_created','no_split_required','failure_count')}), flush=True)
    if not result['passed']:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'run'))
    parser.add_argument('--input', type=Path, default=DOC)
    parser.add_argument('--output', type=Path, default=DOC, help='new supplemental protocol/receipt directory')
    args = parser.parse_args()
    freeze(args) if args.stage == 'freeze' else run(args)


if __name__ == '__main__':
    main()

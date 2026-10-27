"""Run the unchanged e83 diagnostic kernel with one explicit metadata adapter.

The existing WOOD80 allowlist records manifest_sha256 rather than sha256.
During Inputs.metadata() only, a deepcopy of its guarded JSON return is
normalized using the already byte-verified snapshot and prepared MANIFEST.
Original files, opaque expected hashes, scientific functions and SOURCE remain
unchanged. The adapter and pure normalizer are explicit extra GO dependencies.
Import is stdlib-only, performs no IO, and never imports a neural runtime.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
from pathlib import Path
import re
import sys
import types

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parents[1]
WOOD = OUT.parent / 'dope-wood640-eval80-20261027-v1'
SOURCE = Path(__file__).resolve()
CORE = WOOD / 'code/merged_filter_diagnostics.py'
CORE_SHA = 'e83f23bfe3f1e1429654ed7e2036993abd68849498ff0cb0b30a1f9693f6832a'
NORMALIZER = OUT / 'code/prepare_ratio_evaluation_v2.py'
NORMALIZER_SHA = '11278e1b623c251c98276305390de32cb3cd030cd740c25c52aeeb19da959322'
ALLOWLIST = WOOD / 'ALLOWLIST.json'
SNAPSHOT = WOOD / 'preflight/SNAPSHOT_BEFORE.json'
MANIFEST = WOOD / 'MANIFEST.json'
AUTHORIZATION = WOOD / 'AUTHORIZATION.json'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def source_sha(path):
    """Only the three exact source files can be opened by this SHA helper."""
    p = Path(path)
    require(p in (SOURCE, CORE, NORMALIZER) and p.resolve() == p
            and not p.is_symlink() and p.is_file(), 'Unlisted/unsafe adapter source')
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_source(path, expected, name):
    require(source_sha(path) == expected, 'Frozen metadata adapter dependency changed')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    require(source_sha(path) == expected, 'Source changed during guarded import')
    return module


def metadata_inputs_class(core, normalizer):
    """The sole behavior override is this non-scientific metadata() method."""
    base = core.Inputs

    class MetadataInputs(base):
        def metadata(self):
            # All reads go through the unchanged original guard, recording the
            # original byte SHA. No RGB/annotation path named here is followed.
            original_read = base.read.__get__(self, type(self))
            before = original_read(SNAPSHOT)
            raw = original_read(ALLOWLIST)
            prepared = original_read(MANIFEST)
            entries = raw['input_files']
            baseline = before['files']
            frames = prepared['records']
            require(before['allowlist_sha256'] == self.used[str(ALLOWLIST)]
                    and raw['authorization_sha256'] == self.go['wood80_authorization_sha256']
                    and self.go['wood80_manifest_sha256'] == self.used[str(MANIFEST)]
                    and len(raw['records']) == len(frames) == 80
                    and len(entries) == len(baseline) == 160
                    and set(entries) == set(baseline),
                    'Exact authorized80 snapshot/allowlist/manifest contract differs')
            # The original metadata() checks actual AUTHORIZATION bytes through
            # its own guard, plus merged identity, geometry and exact GO scope.
            normalized = copy.deepcopy(raw)
            paths = set()
            for frame in frames:
                path = frame['image_path']
                require(path not in paths, 'Duplicate authorized80 image identity')
                paths.add(path)
                normalized['input_files'][path] = normalizer(
                    entries[path], baseline[path], frame)
            require(len(paths) == 80 and paths == {p for p, item in entries.items()
                    if item['category'] == 'authorized_WOOD80_RGB'},
                    'Only exact authorized80 RGB entries may be normalized')
            require({p for p in entries if p not in paths} ==
                    {p for p in normalized['input_files'] if p not in paths}
                    and all(normalized['input_files'][p] == entries[p]
                            for p in entries if p not in paths),
                    'Non-RGB authorization metadata changed')
            require('read' not in self.__dict__, 'Unexpected existing read override')

            def metadata_read(path):
                value = original_read(path)
                if Path(path) == ALLOWLIST:
                    require(value == raw, 'Original allowlist changed within metadata')
                    return copy.deepcopy(normalized)
                return value

            self.read = metadata_read
            try:
                result = super().metadata()
            finally:
                del self.read
            require(not self.opened_reference_cache,
                    'Metadata adapter must not open reference-coordinate caches')
            self.metadata_adapter_evidence = dict(
                original_allowlist_sha256=self.used[str(ALLOWLIST)],
                byte_verified_snapshot_sha256=self.used[str(SNAPSHOT)],
                prepared_manifest_sha256=self.used[str(MANIFEST)],
                authorization_sha256=self.used[str(AUTHORIZATION)],
                normalized_RGB_entries=80, untouched_non_RGB_entries=80,
                normalization_scope='deepcopy return of guarded ALLOWLIST read during metadata() only',
                original_input_files_modified=False, expected_input_SHA_changed=False,
                reference_coordinate_cache_reads=False, raw_RGB_or_annotation_reads=0)
            return result

    return MetadataInputs


def function_identities(core):
    return {name: value for name, value in vars(core).items()
            if isinstance(value, types.FunctionType)}


def run(stage, tag, go):
    own_sha = source_sha(SOURCE)
    core = load_source(CORE, CORE_SHA, '_frozen_e83_diagnostic_kernel_metadata_v2')
    helper = load_source(NORMALIZER, NORMALIZER_SHA, '_frozen_wood80_metadata_normalizer_v2')
    require(core.SOURCE == CORE and len(core.DEPS) == 12,
            'Original scientific kernel SOURCE/dependency identity differs')
    scientific_functions = function_identities(core)
    original_inputs = core.Inputs
    original_input_methods = dict(vars(original_inputs))
    original_deps = dict(core.DEPS)
    extension = {str(SOURCE): own_sha, str(NORMALIZER): NORMALIZER_SHA}
    require(not set(extension) & set(original_deps), 'Adapter dependency collision')
    # Only provenance dependency identities are extended in memory. SOURCE and
    # scientific function objects stay attached to the byte-identical e83 file.
    core.DEPS = {**original_deps, **extension}
    expected_deps = dict(core.DEPS)

    def science_unchanged():
        require(source_sha(CORE) == CORE_SHA and source_sha(NORMALIZER) == NORMALIZER_SHA
                and source_sha(SOURCE) == own_sha, 'Frozen execution source changed')
        require(core.SOURCE == CORE and core.Inputs is original_inputs
                and dict(vars(original_inputs)) == original_input_methods
                and function_identities(core) == scientific_functions
                and core.DEPS == expected_deps,
                'Original scientific function/class identity or provenance changed')

    science_unchanged()
    io = metadata_inputs_class(core, helper.normalized_authorized_rgb)(go, stage, tag)
    require(io.source == {str(CORE): CORE_SHA, **expected_deps}
            and len(io.source) == 15 and not io.opened_reference_cache,
            'GO must explicitly bind unchanged e83 plus original12 and adapter2')
    require('read' not in io.__dict__ and type(io).read is original_inputs.read,
            'Temporary metadata read override escaped metadata()')
    io.unchanged()
    science_unchanged()
    io.folder.mkdir(parents=True, exist_ok=True)
    preflight = io.folder / ('METADATA_ADAPTER_' + stage + '.json')
    core.write(preflight, dict(complete=True, passed=True, stage=stage, tag=tag,
        role='metadata_adapter_preflight_not_stage_completion',
        scientific_kernel=dict(path=str(CORE), sha256=CORE_SHA, SOURCE_unchanged=True,
            original_function_objects_unchanged=True, original_Inputs_unchanged=True),
        metadata_adapter=dict(path=str(SOURCE), sha256=own_sha),
        pure_normalizer=dict(path=str(NORMALIZER), sha256=NORMALIZER_SHA),
        explicit_GO_source_sha256=io.source,
        evidence=io.metadata_adapter_evidence, input_sha256=dict(io.used)))
    io.hash(preflight)
    # Invoke the original function object directly, with the original code's
    # GPU lock/runtime/filter/parity/reference-read/output guards intact.
    scientific_functions[stage](io)
    science_unchanged()
    io.unchanged()
    original_receipt = io.folder / stage / ('EQUIVALENCE.json' if stage == 'select' else 'COMPLETE.json')
    receipt = io.read(original_receipt)
    require(receipt.get('complete') is True and receipt.get('passed') is True,
            'Unchanged scientific stage did not complete successfully')
    binding_key = 'source_input_sha256' if stage == 'select' else 'input_sha256'
    for path, digest in {str(CORE): CORE_SHA, **extension,
            str(preflight): io.used[str(preflight)]}.items():
        require(receipt[binding_key].get(path) == digest,
                'Original stage receipt omitted actual adapter/kernel identity')
    core.write(io.folder / ('METADATA_ADAPTER_' + stage + '_COMPLETE.json'),
        dict(complete=True, passed=True, stage=stage, tag=tag,
            scientific_kernel_path=str(CORE), scientific_kernel_sha256=CORE_SHA,
            original_function_objects_unchanged=True, original_SOURCE_unchanged=True,
            metadata_adapter_path=str(SOURCE), metadata_adapter_sha256=own_sha,
            pure_normalizer_path=str(NORMALIZER), pure_normalizer_sha256=NORMALIZER_SHA,
            original_stage_receipt_path=str(original_receipt),
            original_stage_receipt_sha256=io.used[str(original_receipt)],
            metadata_preflight_path=str(preflight), metadata_preflight_sha256=io.used[str(preflight)],
            source_sha256=io.source, metadata_only_override=True,
            original_files_modified=False, scientific_functions_replaced=False))
    io.unchanged()
    science_unchanged()
    print('METADATA_ADAPTER_STAGE_COMPLETE', stage, tag, CORE_SHA, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('infer', 'select', 'score'))
    parser.add_argument('--tag', required=True)
    parser.add_argument('--go', default=str(WOOD / 'preflight/FILTER_DIAGNOSTIC_GO.json'))
    args = parser.parse_args()
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.tag), 'Unsafe diagnostic tag')
    run(args.stage, args.tag, args.go)


if __name__ == '__main__':
    main()

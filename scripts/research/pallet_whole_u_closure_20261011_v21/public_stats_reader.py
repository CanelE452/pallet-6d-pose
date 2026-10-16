"""Run the unchanged V21 saved-statistics checker in explicit copied public roots.

The exact V19 c127 resolver is reused from a SHA-bound public sibling source.
Manifest schema observed_frame_public_relocation_manifest_v19 is deliberately
reused for V21: every historical file origin must have its explicit byte mapping.
No producer JSON is edited, no /tmp origin is consulted, and no missing-file
search or fallback is allowed. Only paths and the private output/source roots
are relocated; the full frozen checker AST and arithmetic remain unchanged.
A portable PASS requires ROOT's actual separate freeze/run on copied files.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

CHECKER_SHA = '9ea0be750a7f921bb572801b48953b9530f6a34799a368dc0507c0e34f9cf0d2'
RESOLVER_SHA = 'c12701e12dcd3e983c69033652958f9c4e26671c3e12670bd17c6ccf9c8411ad'
HISTORICAL_STATS_SHA = '4cad27af50af1711ac1817e3c24cfddf73386b535f1cf8c30950950ef2ab1b0c'
SCHEMA = 'observed_frame_public_relocation_manifest_v19'
RESOLVER_REL = 'scripts/research/pallet_finite_support_orthogonal_audit_20261011_v19/public_stats_reader.py'

def require(value, message):
    if not value: raise ValueError(message)

def copied_path(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'symlink: ' + str(path))
    return path

def source_sha(path):
    path = copied_path(path)
    require(path.is_file(), 'required source missing: ' + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()

def resolver_namespace(repo):
    path = copied_path(repo / RESOLVER_REL)
    require(source_sha(path) == RESOLVER_SHA, 'exact unchanged V19 public resolver source required')
    tree = ast.parse(path.read_bytes(), filename=str(path))
    before = ast.dump(tree, include_attributes=False)
    namespace = dict(__name__='_v21_explicit_v19_public_resolver', __file__=str(path), __package__=None)
    exec(compile(tree, str(path), 'exec'), namespace)
    require(ast.dump(tree, include_attributes=False) == before, 'whole resolver AST changed')
    require(namespace['SCHEMA'] == SCHEMA, 'versioned explicit relocation schema differs')
    return path, namespace

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'run'))
    for name in ('repo', 'artifact-root', 'manifest', 'input', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--checker', type=Path)
    args = parser.parse_args()
    require(all(getattr(args, name).is_absolute() for name in ('repo', 'artifact_root', 'manifest', 'input', 'output')),
        'all copied roots, manifest, input and output must be explicit absolute paths')
    require(args.checker is None or args.checker.is_absolute(), 'explicit checker override must be absolute')
    repo, artifact, folder, out = [copied_path(p) for p in (args.repo, args.artifact_root, args.input, args.output)]
    require(repo.is_dir() and artifact.is_dir() and folder.is_dir(), 'existing copied public roots and evaluation input')
    require(folder.is_relative_to(artifact) and out.is_relative_to(artifact) and out != artifact
        and out != folder and not out.is_relative_to(folder) and not folder.is_relative_to(out),
        'input and separate new output must be inside the copied artifact root')
    resolver_source, reused = resolver_namespace(repo)
    relocation = reused['Relocations'](repo, artifact, args.manifest)
    reader = relocation.resolve(__file__)
    require(reader == copied_path(__file__), 'executing reader must be in copied public roots')
    require(relocation.resolve(resolver_source) == resolver_source, 'resolver location must be declared copied public source')
    checker = relocation.resolve(args.checker or Path(__file__).with_name('stats_check.py'))
    namespace = reused['load_checker'](checker, CHECKER_SHA)
    original_historical_source = Path(namespace['ORIGINAL'])
    historical_source = relocation.resolve(original_historical_source)
    require(source_sha(historical_source) == HISTORICAL_STATS_SHA, 'exact immutable public V19 statistics source')
    # These are explicit location globals only. ORIGINAL is the same SHA source;
    # the V21 whole checker AST and every formula/threshold remain unchanged.
    namespace['PRIVATE'] = artifact
    namespace['HISTORY'] = historical_source.parent
    namespace['ORIGINAL'] = historical_source
    original_paths = namespace['paths']
    def paths(a):
        result = {key: relocation.resolve(path) for key, path in original_paths(a).items()}
        result.update(public_reader=reader, relocation_manifest=relocation.manifest,
            original_checker=checker, unchanged_resolver=resolver_source,
            relocated_immutable_historical_statistics=historical_source)
        return result
    namespace['paths'] = paths
    reused['dispatch'](namespace, [str(checker), args.stage, '--repo', str(repo),
        '--input', str(folder), '--output', str(out)])

if __name__ == '__main__': main()

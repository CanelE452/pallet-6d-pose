"""Publish frozen public evidence only; no model, metric, GT or timing calls.

Executed by Root in a separate publication checkout. Existing committed bytes
must match; this program never overwrites a previously committed path.
"""
import hashlib, json, pathlib, re, subprocess, sys, datetime
SOURCE = pathlib.Path('/tmp/pallet-kp-publication-20261010-v6')
DEST = pathlib.Path('/tmp/pallet-observation-publication-stage-20261011')
PRIVATE = pathlib.Path('/tmp/pallet-consensus-decision-repair-private-20261011-v18')
STATUS = '_docs/experiments/pallet_observation_refiner_status_20261011'
BRANCH = 'research/observation-refiner-robust-pnp-20261009'
PARENT = '691a4ece4e719ddfda2aa69a6afb6578f8923511'
FAMILIES = [
 'pallet_factor_subset_consensus_20261011_v15',
 'pallet_hybrid_factor_subset_consensus_20261011_v16',
 'pallet_hybrid_standard_controls_20261011_v17',
 'pallet_consensus_decision_repair_20261011_v18',
]
DOC18 = '_docs/experiments/' + FAMILIES[-1]
MANIFEST_PATH = STATUS + '/PUBLICATION_MANIFEST_V16_V18.json'
PLAN_PATH = PRIVATE / 'PUBLICATION_PREPARED_V16_V18.json'
PATHSPEC_PATH = PRIVATE / 'PUBLICATION_NEW_PATHS_V16_V18.nul'
ALLOWED = {'.py', '.md', '.json', '.gz', '.csv', '.png', '.txt'}
EXCLUDED_DRAFT = '_docs/experiments/' + FAMILIES[1] + '/REPORT_DRAFT_KO.md'

def git(root, *args, data=None):
 p = subprocess.run(['git', '-C', str(root), *args], input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
 return p.stdout

def load(path):
 return json.loads((SOURCE / path).read_text())

def digest(path):
 h = hashlib.sha256()
 with path.open('rb') as f:
  while b := f.read(1024 * 1024): h.update(b)
 return h.hexdigest()

def valid(path):
 p = pathlib.PurePosixPath(path)
 if p.is_absolute() or '..' in p.parts or not (path.startswith('scripts/research/') or path.startswith('_docs/experiments/')):
  raise RuntimeError('unexpected public path: ' + path)
 if '__pycache__' in p.parts or not (p.suffix in ALLOWED or p.name == '.gitignore' or re.fullmatch(r'.+\.part[0-9]{3}', p.name)):
  raise RuntimeError('unexpected public file type: ' + path)
 q = SOURCE / path
 if q.is_symlink() or not q.is_file(): raise RuntimeError('public regular file missing: ' + path)
 for parent in q.parents:
  if parent == SOURCE: break
  if parent.is_symlink(): raise RuntimeError('public parent symlink: ' + path)
 return q

def same_binding(path, binding):
 q = valid(path)
 if q.stat().st_size != binding['bytes'] or digest(q) != binding['sha256']:
  raise RuntimeError('frozen public byte mismatch: ' + path)

if git(DEST, 'rev-parse', 'HEAD').decode().strip() != PARENT or git(DEST, 'status', '--porcelain=v1', '-z'):
 raise RuntimeError('publication checkout must remain clean at the published parent')
if (SOURCE / MANIFEST_PATH).exists() or PLAN_PATH.exists() or PATHSPEC_PATH.exists():
 raise RuntimeError('new publication outputs must initially be absent')
reader = load(DOC18 + '/PUBLIC_READER_CHECKS.json')
if not reader.get('complete') or not reader.get('passed') or len(reader.get('proofs', {})) != 4:
 raise RuntimeError('four completed copied public readers are required')
bundle = load(DOC18 + '/PUBLIC_BUNDLE_PROTOCOL.json')['snapshot']['files']
selected, excluded, archived = set(), [], {}
print('Collect frozen direct files and ordered parts', flush=True)
for path, record in bundle.items():
 if record['source_kind'] == 'direct_public_file':
  same_binding(path, record['binding']); selected.add(path)
 elif record['source_kind'] == 'ordered_archive_parts':
  parts = record['ordered_parts']
  if not parts: raise RuntimeError('empty ordered archive: ' + path)
  for binding in parts:
   same_binding(binding['path'], binding); selected.add(binding['path'])
  archived[path] = {'original': record['binding'], 'ordered_parts': parts}
 else: raise RuntimeError('unknown frozen file kind')
for family in FAMILIES:
 doc = '_docs/experiments/' + family
 manifest = load(doc + '/ARCHIVE_MANIFEST.json')
 for record in manifest['files']:
  original = record['original']; path = doc + '/' + original['path']
  if record['storage'] == 'ordered_parts':
   parts = [{**p, 'path': doc + '/' + p['path']} for p in record['parts']]
   for p in parts: same_binding(p['path'], p); selected.add(p['path'])
   archive = {'original': {**original, 'path': path}, 'ordered_parts': parts}
   if path in archived:
    if [(p['path'], p['bytes'], p['sha256']) for p in archived[path]['ordered_parts']] != [(p['path'], p['bytes'], p['sha256']) for p in parts]:
     raise RuntimeError('archive ordering metadata mismatch: ' + path)
   else: archived[path] = archive
  elif record['storage'] != 'direct_original': raise RuntimeError('unknown family archive storage')
 for tree in ['scripts/research/' + family, doc]:
  for q in sorted((SOURCE / tree).rglob('*')):
   if q.is_symlink(): raise RuntimeError('symlink in public tree: ' + str(q))
   if not q.is_file(): continue
   path = str(q.relative_to(SOURCE))
   if '__pycache__' in q.parts or q.suffix == '.pyc': continue
   if path == EXCLUDED_DRAFT:
    excluded.append({'path': path, 'reason': 'unfinished historical draft; preserved at source'}); continue
   valid(path)
   if q.stat().st_size > 50 * 1024 * 1024:
    if path not in archived: raise RuntimeError('large original lacks strict ordered parts metadata: ' + path)
    excluded.append({'path': path, 'reason': 'large gzip preserved at source; publish exact ordered parts'}); continue
   selected.add(path)
for path in [STATUS + '/LATEST_KO.md', STATUS + '/PRE_PUBLICATION_STATE_V16_V18.json', STATUS + '/PUSH_V10_V15_RECEIPT.json', STATUS + '/PUBLICATION_COLLECTOR.py',
 'scripts/research/pallet_three_head_observation_20261010_v7/restore_archives.py',
 'scripts/research/pallet_same_observation_controls_20261010_v8/restore_archives.py',
 'scripts/research/pallet_sparse_local_line_20261010_v9/restore_archives.py',
 'scripts/research/pallet_finite_edge_consensus_repair_20261011_v13/restore_archives.py']:
 valid(path); selected.add(path)
paths = sorted(selected)
print('Hash selected public files and compare committed parent bytes', len(paths), flush=True)
old = {}
for record in git(DEST, 'ls-tree', '-r', '-z', 'HEAD').split(b'\0'):
 if not record: continue
 meta, name = record.split(b'\t', 1); mode, typ, oid = meta.split()
 old[name.decode()] = (mode.decode(), typ.decode(), oid.decode())
ids = git(SOURCE, 'hash-object', '--stdin-paths', data=b''.join((str(SOURCE / p) + '\n').encode() for p in paths)).decode().splitlines()
if len(ids) != len(paths): raise RuntimeError('incomplete Git object identity list')
files, new, unchanged, conflicts = [], [], [], []
for path, oid in zip(paths, ids):
 q = valid(path)
 if q.stat().st_size > 50 * 1024 * 1024: raise RuntimeError('oversize selected Git blob: ' + path)
 entry = {'path': path, 'bytes': q.stat().st_size, 'sha256': digest(q), 'git_blob': oid}
 files.append(entry)
 if path in old:
  if old[path] != ('100644', 'blob', oid): conflicts.append({'path': path, 'parent': old[path], 'source_git_blob': oid})
  else: unchanged.append(path)
 else: new.append(entry)
if conflicts:
 print(json.dumps({'conflicts': conflicts}, ensure_ascii=False, indent=2), flush=True)
 raise RuntimeError('existing published bytes are protected; collector will not overwrite')
print('Zero conflicts; copy new public bytes only', len(new), flush=True)
for e in new:
 dest = DEST / e['path']; dest.parent.mkdir(parents=True, exist_ok=True)
 if dest.exists() or dest.is_symlink(): raise RuntimeError('new destination must be absent: ' + e['path'])
 with (SOURCE / e['path']).open('rb') as src, dest.open('xb') as out:
  while b := src.read(1024 * 1024): out.write(b)
 if dest.stat().st_size != e['bytes'] or digest(dest) != e['sha256']: raise RuntimeError('copy verification failed: ' + e['path'])
manifest = {
 'schema': 'public_evidence_publication_manifest_v16_v18', 'branch': BRANCH, 'published_parent': PARENT,
 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'files': files,
 'file_count': len(files), 'public_bytes': sum(e['bytes'] for e in files),
 'new_file_count': len(new), 'new_public_bytes': sum(e['bytes'] for e in new),
 'existing_committed_byte_identical_count': len(unchanged), 'existing_committed_paths_not_overwritten': True,
 'ordered_archives': list(archived.values()), 'excluded': excluded,
 'frozen_bundle_file_count': len(bundle), 'frozen_direct_file_count': sum(v['source_kind'] == 'direct_public_file' for v in bundle.values()),
 'four_actual_copied_readers_complete_and_PASS': True,
 'late_outputs_are_additional_to_original_frozen_byte_bundle': True,
 'standalone_archive_restore_helpers_explicitly_included': True,
 'manifest_does_not_include_itself': True,
 'accuracy_goal_solved': False, 'full_V16_scalar_PASS_claim': False,
 'evaluation_scope': 'known DEV GEOMETRIC_PROXY Clean153 Moderate92; 245 frames; Severe74 excluded',
 'new_training_RGB_GT_model_numeric_timing_calls': 0,
 'main_mutation_force_push_or_merge': False,
}
raw = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
for root in [SOURCE, DEST]:
 q = root / MANIFEST_PATH; q.parent.mkdir(parents=True, exist_ok=True)
 with q.open('xb') as out: out.write(raw)
manifest_binding = {'path': MANIFEST_PATH, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'git_blob': git(SOURCE, 'hash-object', str(SOURCE / MANIFEST_PATH)).decode().strip()}
with PATHSPEC_PATH.open('xb') as out: out.write(b''.join((e['path'] + '\0').encode() for e in new + [manifest_binding]))
plan = {'schema': 'prepared_publication_v16_v18', 'complete': True, 'parent': PARENT,
 'new_files': new + [manifest_binding], 'unchanged_paths': unchanged, 'manifest': manifest_binding,
 'pathspec_file': str(PATHSPEC_PATH), 'no_index_commit_push_calls': True}
with PLAN_PATH.open('x') as out: json.dump(plan, out, ensure_ascii=False, indent=2); out.write('\n')
print(json.dumps({'complete': True, 'selected_files': len(files), 'new_files': len(new), 'new_bytes': sum(e['bytes'] for e in new), 'unchanged_files': len(unchanged), 'ordered_archives': len(archived), 'excluded': len(excluded)}, ensure_ascii=False), flush=True)

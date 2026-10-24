#!/usr/bin/env python3
"""Recalculate published V184 primary metrics without models or third-party packages.

Examples (run from any directory after cloning the publication):
  python3 verify_published_results.py --self-test
  python3 verify_published_results.py --root _docs/experiments/pallet-keypoint-visibility
  python3 verify_published_results.py --root PATH --manifest PUBLICATION_MANIFEST.json

--ledger/--results are publication-relative logical paths. A path can name a
plain file, .gz, or a .parts/.parts.json descriptor; no source absolute path is
followed. Optional manifest files/entries map logical paths to storage specs.
"""
from pathlib import Path, PurePosixPath
from collections import Counter
import argparse
import gzip
import hashlib
import json
import math
import sys
import tempfile

PREFIX = 'archive/task-2/pallet-keypoint-visibility/query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/'
DEFAULT_LEDGER = PREFIX+'saved_real_head_evaluation_v1/current1784/ALL1784_CURRENT_NATIVE_LEDGER.jsonl'
DEFAULT_RESULTS = PREFIX+'ROOT_RESULTS_KO.json'
MODELS = ('v61', 'legacy', 'C52', 'C53')
GROUPS = ('ALL', 'WOOD_PRIMARY105', 'SQUARE118')
D, E = 'DIRECT_VISIBLE', 'EXTERNAL_OCCLUDED'
FLOAT_KEYS = ('D_precision', 'D_recall', 'E_precision', 'E_recall', 'U_rate')
COUNT_KEYS = ('N', 'D', 'E', 'E_wrongly_passed_as_visible', 'E_predicted_U')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def boolean(value):
    if value is True or value == 'True':
        return True
    if value is False or value == 'False':
        return False
    raise ValueError('Original boolean flag must be bool or True/False text')


def relative_path(value):
    require(isinstance(value, str) and bool(value), 'Nonempty publication-relative path required')
    p = PurePosixPath(value)
    require(not p.is_absolute() and '..' not in p.parts and '\\' not in value, 'Path must stay inside publication root')
    require('v112' not in value.lower(), 'V112 access prohibited')
    return p


def manifest_records(manifest):
    """Small explicit interchange schema; no recursive guesses or directory scan."""
    records = manifest.get('files', manifest.get('entries'))
    require(isinstance(records, (list, dict)), 'Manifest requires files or entries list/mapping')
    if isinstance(records, dict):
        return [dict(value, logical_path=logical) if isinstance(value, dict)
                else dict(logical_path=logical, published_path=value) for logical, value in records.items()]
    require(all(isinstance(r, dict) for r in records), 'Manifest entries must be objects')
    return records


def record_storage(record):
    """Publication manifest v2 plus the smaller generic files/entries schema."""
    if record.get('gzipchunks') is not None:
        spec = record['gzipchunks']
        require(isinstance(spec, dict), 'gzipchunks must be explicit descriptor object')
        require(record.get('source_sha256', spec.get('content_sha256')) == spec.get('content_sha256'),
                'Original bytes SHA and reassembled content SHA must match')
        return spec
    if record.get('published_relative') is not None:
        return dict(path=record['published_relative'], sha256=record.get('published_sha256'),
                    stored_size_bytes=record.get('size_bytes'))
    return record.get('storage', record)


class PublicationReader:
    """Only this adapter knows storage placement; metric code consumes bytes."""
    def __init__(self, root, manifest=None):
        self.root = Path(root).resolve()
        self.mapping = {}
        self.read_log = []
        if manifest is not None:
            for r in manifest_records(manifest):
                logical = r.get('published_relative', r.get('logical_path', r.get('original_path')))
                if logical is not None:
                    relative_path(logical)
                    require(logical not in self.mapping, 'Duplicate logical publication path')
                    self.mapping[logical] = record_storage(r)

    def stored_path(self, path):
        relative_path(path)
        p = (self.root/path).resolve()
        require(p.is_relative_to(self.root), 'Publication symlink must remain inside root')
        return p

    def stored_bytes(self, path, expected_sha=None):
        data = self.stored_path(path).read_bytes()
        digest = sha256(data)
        if expected_sha is not None:
            require(digest == expected_sha, 'Published stored-byte SHA256 mismatch: '+path)
        self.read_log.append(dict(path=path, bytes=len(data), sha256=digest))
        return data

    def hash_stored_file(self, path, expected_sha, expected_size=None):
        """Streaming stored-byte integrity only; never deserialize dataset arrays."""
        digest = hashlib.sha256(); size = 0
        with self.stored_path(path).open('rb') as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b''):
                size += len(chunk); digest.update(chunk)
        require(digest.hexdigest() == expected_sha, 'Published stored-byte SHA256 mismatch: '+path)
        if expected_size is not None:
            require(size == expected_size, 'Published stored-byte size mismatch: '+path)
        result = dict(path=path, bytes=size, published_sha256=digest.hexdigest())
        self.read_log.append(dict(result, operation='STREAM_SHA_ONLY'))
        return result

    def read_bytes(self, path_or_spec, resolve_mapping=True):
        if isinstance(path_or_spec, str):
            relative_path(path_or_spec)
            spec = self.mapping.get(path_or_spec, path_or_spec) if resolve_mapping else path_or_spec
            if isinstance(spec, str):
                data = self.stored_bytes(spec)
                if spec.endswith(('.parts', '.parts.json')):
                    return self.read_bytes(json.loads(data.decode('utf-8')), False)
                return gzip.decompress(data) if spec.endswith('.gz') else data
            return self.read_bytes(spec, False)
        spec = path_or_spec
        require(isinstance(spec, dict), 'Storage spec must be path string or object')
        if 'parts' in spec:
            require(spec.get('format') == 'concatenated-bytes-v1', 'Explicit parts format required')
            require(isinstance(spec['parts'], list) and len(spec['parts']) > 0, 'Ordered nonempty parts required')
            data = b''.join(self.read_bytes(part, False) for part in spec['parts'])
        else:
            path = spec.get('published_path', spec.get('path'))
            expected = spec.get('publishedSHA', spec.get('published_sha256', spec.get('sha256')))
            data = self.stored_bytes(path, expected)
            if spec.get('stored_size_bytes') is not None:
                require(len(data) == spec['stored_size_bytes'], 'Stored part size mismatch')
            compression = spec.get('compression', 'gzip' if path.endswith('.gz') else 'none')
            require(compression in ('none', 'gzip'), 'Supported compression: none/gzip')
            if path.endswith(('.parts', '.parts.json')):
                require(compression == 'none', 'Parts descriptor itself must be plain JSON')
                data = self.read_bytes(json.loads(data.decode('utf-8')), False)
            elif compression == 'gzip':
                data = gzip.decompress(data)
            if spec.get('decoded_size_bytes') is not None:
                require(len(data) == spec['decoded_size_bytes'], 'Decoded part size mismatch')
        if spec.get('content_sha256') is not None:
            require(sha256(data) == spec['content_sha256'], 'Decompressed/assembled content SHA256 mismatch')
        if spec.get('content_size_bytes') is not None:
            require(len(data) == spec['content_size_bytes'], 'Assembled content size mismatch')
        return data


def verify_manifest(reader, manifest):
    """Verify stored published bytes, independently from metric reconstruction."""
    checked = []
    skipped = 0
    for record in manifest_records(manifest):
        status = record.get('status', '')
        require(not status.startswith('CANDIDATE_'), 'Candidate inventory is not a completed publication manifest')
        if (record.get('published_relative') is None and 'published_relative' in record
                and record.get('gzipchunks') is None):
            require(not status.startswith('PUBLISHED_'), 'Published record lacks storage path')
            skipped += 1
            continue
        spec = record_storage(record)
        if 'parts' in spec:
            require(spec.get('format') == 'concatenated-bytes-v1', 'Explicit parts manifest format')
            parts = spec['parts']
        else:
            parts = [spec]
        for part in parts:
            require(isinstance(part, dict), 'Each manifest stored file requires object/SHA')
            path = part.get('published_path', part.get('path'))
            expected = part.get('publishedSHA', part.get('published_sha256', part.get('sha256')))
            require(isinstance(expected, str) and len(expected) == 64, 'Each published file requires SHA256')
            size = part.get('stored_size_bytes')
            checked.append(reader.hash_stored_file(path, expected, size))
    return dict(PASS=True, stored_files_checked=len(checked), unpublished_records_skipped=skipped, files=checked)


def metric(rows, model, group):
    """Independent scalar 2x3 kernel. U remains in the recall denominator."""
    cm = [[0, 0, 0], [0, 0, 0]]
    for row in rows:
        if group != 'ALL' and row['group'] != group:
            continue
        if not boolean(row['paired_same_query_RGB_D_vs_E']) or row['human_state'] not in (D, E):
            continue
        response = row[model+'_D_E_U']
        require(response in ('D', 'E', 'U'), 'Saved D/E/U enum: '+model)
        truth = 0 if row['human_state'] == D else 1
        predicted = ('D', 'E', 'U').index(response)
        cm[truth][predicted] += 1
    n_d, n_e = sum(cm[0]), sum(cm[1])
    ratio = lambda a, b: a/b if b else None
    return dict(model=model, group=group, N=n_d+n_e, D=n_d, E=n_e,
        CM_rows_D_E_cols_D_E_U=cm,
        D_precision=ratio(cm[0][0], cm[0][0]+cm[1][0]), D_recall=ratio(cm[0][0], n_d),
        E_precision=ratio(cm[1][1], cm[0][1]+cm[1][1]), E_recall=ratio(cm[1][1], n_e),
        U_rate=ratio(cm[0][2]+cm[1][2], n_d+n_e),
        E_wrongly_passed_as_visible=cm[1][0], E_predicted_U=cm[1][2])


def compare_metric(actual, reported):
    for key in ('model', 'group', *COUNT_KEYS, 'CM_rows_D_E_cols_D_E_U'):
        require(actual[key] == reported[key], 'Published report/count disagreement '+str((actual['model'], actual['group'], key)))
    for key in FLOAT_KEYS:
        a, b = actual[key], reported[key]
        require((a is None and b is None) or (a is not None and b is not None and math.isfinite(b)
                and math.isclose(a, b, rel_tol=0, abs_tol=1e-12)), 'Published report ratio disagreement '+key)


def verify_results(ledger_bytes, result_bytes):
    rows = [json.loads(line) for line in ledger_bytes.decode('utf-8').splitlines() if line.strip()]
    report = json.loads(result_bytes.decode('utf-8'))
    require(len(rows) == 1784 and [r['global_index'] for r in rows] == list(range(1784)), 'All original1784 ordered IDs')
    require(len({(r['id'], r['native_corner']) for r in rows}) == 1784, 'Unique original id/native pairs')
    require(all(type(r['native_corner']) is int and 0 <= r['native_corner'] < 8 for r in rows), 'Original native0..7')
    require({r['group'] for r in rows} == set(GROUPS[1:]), 'Both original WOOD/SQUARE groups retained')
    available = sum(boolean(r['query_available']) for r in rows)
    require(available == 1739, 'Original1739 available/45 inputU')
    truth = Counter(r['human_state'] for r in rows)
    if 'human_state_inventory' in report:
        require(dict(truth) == report['human_state_inventory'], 'Original human-state inventory')
    require(report['original_current_slots'] == 1784 and report['original_current_available'] == available
            and report['original_current_inputU'] == 45, 'Published original coverage')
    primary = report['primary_all']; material = report['primary_by_material']
    expected = {(m, g) for m in MODELS for g in GROUPS}
    reported = {(r['model'], r['group']): r for r in primary+material}
    require(len(primary) == 4 and len(material) == 8 and len(reported) == 12 and set(reported) == expected, 'Exactly4models×3groups')
    calculated = []
    for group in GROUPS:
        for model in MODELS:
            actual = metric(rows, model, group)
            if group == 'ALL':
                require((actual['N'], actual['D'], actual['E']) == (1237, 1147, 90), 'Original primary1237 D1147/E90')
            compare_metric(actual, reported[model, group]); calculated.append(actual)
    return dict(completed=True, PASS=True, original_slots=1784, original_available=available, original_input_U=45,
        primary_known=1237, primary_D=1147, primary_E=90, model_group_rows_verified=12,
        human_state_inventory=dict(truth), independent_scalar_metrics=calculated,
        source_ledger_content_sha256=sha256(ledger_bytes), source_report_content_sha256=sha256(result_bytes),
        scope='Saved publication arithmetic only. Existing human semantic/cohort authority and historical phase/review bindings remain unchanged. Current data are evaluation-only, not training inputs. No new native physical gold, model, threshold, label, selection, or training claim.')


def self_tests():
    tests = []
    def row(i, truth, pred='D', primary=True, available=True):
        return dict(global_index=i, id='toy'+str(i//8), native_corner=i%8,
            group=GROUPS[1+i%2], human_state=truth, paired_same_query_RGB_D_vs_E=primary,
            query_available=available, **{m+'_D_E_U':pred for m in MODELS})
    toy = [row(0,D,'D'),row(1,D,'E'),row(2,D,'U'),row(3,E,'E'),row(4,E,'D'),row(5,E,'U'),
           row(6,'SELF_OCCLUDED','D'),row(7,D,'D',False)]
    x=metric(toy,'C52','ALL')
    require(x['CM_rows_D_E_cols_D_E_U']==[[1,1,1],[1,1,1]] and x['D_precision']==.5 and x['E_precision']==.5
            and x['D_recall']==1/3 and x['E_recall']==1/3 and x['U_rate']==1/3,'Scalar U denominators')
    tests.append('Independent2x3_counts_precision_recall_U_and_original_flag')
    z=metric([row(0,D,'U'),row(1,E,'U')],'v61','ALL')
    require(z['D_precision'] is None and z['E_precision'] is None and z['U_rate']==1,'No-call precision null/allU retained')
    tests.append('AllU_zero_call_denominator_null')
    try: boolean('unknown')
    except ValueError: tests.append('Malformed_original_flag_rejected')
    else: raise AssertionError('Malformed flag accepted')
    allrows=[row(i,D if i<1147 else E if i<1237 else 'SELF_OCCLUDED',('D','E','U')[i%3],i<1237,i>=45) for i in range(1784)]
    primary=[metric(allrows,m,'ALL') for m in MODELS]
    material=[metric(allrows,m,g) for g in GROUPS[1:] for m in MODELS]
    report=dict(primary_all=primary,primary_by_material=material,human_state_inventory=dict(Counter(r['human_state'] for r in allrows)),
                original_current_slots=1784,original_current_available=1739,original_current_inputU=45)
    ledger=('\n'.join(json.dumps(r) for r in allrows)+'\n').encode(); result=json.dumps(report).encode()
    require(verify_results(ledger,result)['model_group_rows_verified']==12,'Full invented schema')
    tests.append('All1784_full_schema_12_metric_rows_and_material_partition')
    bad=json.loads(json.dumps(report));bad['primary_all'][0]['CM_rows_D_E_cols_D_E_U'][0][0]+=1
    try: verify_results(ledger,json.dumps(bad).encode())
    except ValueError: tests.append('Changed_report_count_rejected')
    else: raise AssertionError('Changed report accepted')
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder); (root/'toy.jsonl').write_bytes(ledger); (root/'toy.jsonl.gz').write_bytes(gzip.compress(ledger,mtime=0))
        a,b=ledger[:12345],ledger[12345:];(root/'p0').write_bytes(a);(root/'p1.gz').write_bytes(gzip.compress(b,mtime=0))
        parts=dict(format='concatenated-bytes-v1',parts=[dict(path='p0',sha256=sha256(a)),dict(path='p1.gz',compression='gzip',sha256=sha256((root/'p1.gz').read_bytes()))],content_sha256=sha256(ledger))
        (root/'toy.parts').write_text(json.dumps(parts))
        reader=PublicationReader(root)
        require(reader.read_bytes('toy.jsonl')==reader.read_bytes('toy.jsonl.gz')==reader.read_bytes('toy.parts')==ledger,'Byte exact raw/gzip/parts')
        tests.append('Plain_gzip_ordered_mixedcompressed_parts_byteexact')
        manifest=dict(files=[dict(logical_path='logical/ledger',storage=parts)])
        reader=PublicationReader(root,manifest)
        require(reader.read_bytes('logical/ledger')==ledger and verify_manifest(reader,manifest)['stored_files_checked']==2,'Logical map + stored SHA')
        tests.append('Manifest_logical_mapping_and_all_published_SHA')
        v2=dict(entries=[dict(published_relative='archive/raw.jsonl',published_sha256=sha256(ledger),
                             source_sha256=sha256(ledger),size_bytes=len(ledger),status='PUBLISHED_RAW',gzipchunks=None),
                         dict(published_relative='archive/chunked.jsonl',published_sha256=None,
                             source_sha256=sha256(ledger),status='PUBLISHED_GZIP_CHUNKS',gzipchunks=parts),
                         dict(published_relative=None,status='EXCLUDED',gzipchunks=None)])
        (root/'archive').mkdir();(root/'archive/raw.jsonl').write_bytes(ledger)
        reader=PublicationReader(root,v2)
        require(reader.read_bytes('archive/chunked.jsonl')==reader.read_bytes('archive/raw.jsonl')==ledger
                and verify_manifest(reader,v2)['unpublished_records_skipped']==1,'Publicationv2 physical/logical split')
        tests.append('Publication_v2_raw_and_embedded_gzipchunks_exclusions')
        corrupt=dict(parts);corrupt['content_sha256']='0'*64
        try: reader.read_bytes(corrupt)
        except ValueError: tests.append('Corrupt_parts_content_SHA_rejected')
        else: raise AssertionError('Bad content SHA accepted')
        for p in ('/absolute','../outside','some/V112/file'):
            try: reader.read_bytes(p)
            except ValueError: pass
            else: raise AssertionError('Unsafe publication path accepted')
        tests.append('Absolute_traversal_V112_paths_rejected')
    return dict(completed=True,PASS=True,invented_only=True,tests=tests,count=len(tests),actual_published_package_reads=0,
                runtime_dependencies='Python standard library only')


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--self-test',action='store_true');parser.add_argument('--root',type=Path)
    parser.add_argument('--ledger',default=DEFAULT_LEDGER);parser.add_argument('--results',default=DEFAULT_RESULTS)
    parser.add_argument('--manifest',help='Publication-relative JSON files/entries manifest; verify all publishedSHA fields')
    parser.add_argument('--report',type=Path,help='Optional NEW output JSON file; otherwise stdout only')
    args=parser.parse_args()
    if args.self_test:
        require(args.root is None and args.manifest is None,'Self-test must not read publication package')
        answer=self_tests()
    else:
        require(args.root is not None,'--root is required for actual publication verification')
        reader=PublicationReader(args.root)
        manifest=json.loads(reader.read_bytes(args.manifest,False)) if args.manifest else None
        if manifest is not None: reader=PublicationReader(args.root,manifest)
        manifest_review=verify_manifest(reader,manifest) if manifest is not None else dict(status='NOT_REQUESTED')
        answer=verify_results(reader.read_bytes(args.ledger),reader.read_bytes(args.results))
        answer['publication_root']=str(reader.root);answer['publication_manifest_verification']=manifest_review;answer['published_file_reads']=reader.read_log
    text=json.dumps(answer,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if args.report:
        with args.report.open('x',encoding='utf-8') as handle:handle.write(text)
    print(text,end='')


if __name__=='__main__':
    try: main()
    except (ValueError,KeyError,TypeError,OSError,json.JSONDecodeError) as error:
        print(json.dumps(dict(completed=False,PASS=False,error_type=type(error).__name__,error=str(error)),ensure_ascii=False),file=sys.stderr)
        sys.exit(1)

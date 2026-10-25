"""Pinned public metadata collector; future exact-allowlist opaque downloader."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import stat
import urllib.error
import urllib.request
import urllib.parse

API_SHA = 'df02a2d129392f76d65a1130f741e2de29d114fd3a608ef431229b313ddce3dd'
README_SHA = '06344d868baf10389964124e263e4d7e9333bd0f7b7a36f4b03ced68cc08efd3'
PILOTS = {'0.55m_hall/0000', '0.55m_hall/0011'}
ROLE_RULES = {'rgb': ('RGB', '.png'), 'depth': ('DEPTH', '.png'),
              'annotated_poses': ('POSE', '.txt'), 'masks': ('SAM', '_sam1_mask.png')}
REQUIRED = ('RGB', 'DEPTH', 'POSE', 'SAM')
SPLIT_SALT = 'V260_PUBLIC_CAPTURE_GROUP_HASH_V1:'


def need(ok, message):
    if not bool(ok):
        raise ValueError(message)


def safe(path, exists=True):
    p = Path(path)
    need(p.is_absolute() and '..' not in p.parts, 'Exact absolute path')
    need(not any(x in str(p).casefold() for x in
                 ('v112', 'cache', '.pyc', 'current223', 'publicheld', 'holdout', 'oldval')), 'Forbidden before FS')
    for a in [*reversed(p.parents), p]:
        try:
            mode = os.lstat(a).st_mode
        except FileNotFoundError:
            need(not exists, 'Exact input absent')
            continue
        need(not stat.S_ISLNK(mode), 'No ancestor/leaf symlink')
    return p


def bound_read(b):
    fd = os.open(safe(b['path']), os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as f:
        v = f.read()
    need(len(v) == b['size_bytes'] and hashlib.sha256(v).hexdigest() == b['sha256'], 'Bound input SHA/size')
    return v


def binding(p):
    v = safe(p).read_bytes()
    return {'path': str(p), 'sha256': hashlib.sha256(v).hexdigest(), 'size_bytes': len(v)}


def dump(p, v):
    with safe(p, False).open('x', encoding='utf-8') as f:
        json.dump(v, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    return binding(p)


def public_file(entry):
    f = entry['dataFile']
    directory = entry.get('directoryLabel', '')
    path = (directory+'/' if directory else '')+entry['label']
    need(not Path(path).is_absolute() and '..' not in Path(path).parts and
         not any(t in path.casefold() for t in ('v112','cache','.pyc')), 'Publisher lexical path')
    checksum = f['checksum']
    need(checksum['type'] == 'MD5' and len(checksum['value']) == 32, 'Publisher MD5 available')
    need(f['filesize'] > 0 and entry['restricted'] is False, 'Public positive-size file metadata')
    return {'file_id': f['id'], 'publisher_relative_path': path, 'size_bytes': f['filesize'],
            'publisher_MD5': checksum['value'].lower(), 'restricted': False,
            'content_type': f['contentType'], 'original_filename': f['filename'],
            'payload_read': False, 'local_payload_sha256': None}


def collect(api_binding, readme_binding, out):
    need(api_binding['sha256'] == API_SHA and api_binding['size_bytes'] == 725973, 'Pinned official API')
    need(readme_binding['sha256'] == README_SHA and readme_binding['size_bytes'] == 4874, 'Pinned official README')
    api = json.loads(bound_read(api_binding))
    bound_read(readme_binding)  # Documentation only, no serialized source path follow.
    need(api['status'] == 'OK', 'Official metadata API status')
    version = api['data']['latestVersion']
    need(version['datasetPersistentId'] == 'doi:10.48804/DJLDFS' and version['versionState'] == 'RELEASED', 'Pinned public dataset')
    bundles, calibrations = {}, {}
    for f in version['files']:
        directory = f.get('directoryLabel', '')
        parts = directory.split('/') if directory else []
        if len(parts) == 1 and f['label'] == 'cam_K.txt':
            need(parts[0] not in calibrations, 'Unique K per collection')
            calibrations[parts[0]] = public_file(f)
        elif len(parts) == 2 and parts[1] in ROLE_RULES:
            role, suffix = ROLE_RULES[parts[1]]
            need(f['label'].endswith(suffix), 'Publisher expected component suffix')
            frame = f['label'][:-len(suffix)]
            need(frame.isdigit(), 'Literal numeric publisher frame stem')
            key = parts[0]+'/'+frame
            row = bundles.setdefault(key, {'capture_directory': parts[0], 'frame_stem': frame, 'files': {}})
            need(role not in row['files'], 'Unique component per publisher bundle')
            row['files'][role] = public_file(f)
    groups = sorted({r['capture_directory'] for r in bundles.values()})
    need(set(groups) == set(calibrations), 'Each directory has official K')
    group_hashes = {g: hashlib.sha256((SPLIT_SALT+g).encode()).hexdigest() for g in groups}
    ranked = sorted(groups, key=lambda g: (group_hashes[g], g))
    # Prospective coarse-group proposal only. Exactly two of six groups; never frame/label/model based.
    dev_groups = set(ranked[:2])
    proposal = []
    for g in groups:
        proposal.append({'group_id': g, 'group_resolution': 'Official height_environment directory; conservative coarse capture collection, not proven acquisition-session identity',
                         'hash_sha256': group_hashes[g], 'hash_rank': ranked.index(g),
                         'proposed_role': 'PUBLIC_SOURCE_DEV_PROPOSAL' if g in dev_groups else 'PUBLIC_SOURCE_TRAIN_PROPOSAL',
                         'actual_role_adopted': None, 'known_diagnostic_exposed_group': any(x.startswith(g+'/') for x in PILOTS),
                         'fresh_clean_blind_claim': False, 'training_admission': False})
    by_group = {g['group_id']: g for g in proposal}
    complete, missing = [], []
    for key in sorted(bundles):
        row = bundles[key]
        group = row['capture_directory']
        absent = [role for role in REQUIRED if role not in row['files']]
        row.update({'bundle_id': 'Pallox_DOI10.48804_DJLDFS_V1.0/'+key,
                    'publisher_frame_path': key, 'coarse_capture_group_id': group,
                    'K': calibrations[group], 'missing_roles': absent, 'four_way_complete': not absent,
                    'known_diagnostic_exposed_frame': key in PILOTS,
                    'known_diagnostic_exposed_group': by_group[group]['known_diagnostic_exposed_group'],
                    'other_historical_exposure_not_exhaustively_audited': True,
                    'proposed_role': by_group[group]['proposed_role'], 'actual_role_adopted': None,
                    'native_mapping_or_visibility_labels': None, 'training_admission': False})
        if 'POSE' in row['files']:
            row['pose_file_MD5_alias_identity_only'] = group+':'+row['files']['POSE']['publisher_MD5']
        (missing if absent else complete).append(row)
    need(len(complete) == 284 and len(groups) == 6, 'Publisher four-way metadata completeness; do not silently filter')
    census = []
    for g in groups:
        rr = [x for x in [*complete, *missing] if x['capture_directory'] == g]
        roles = Counter(role for x in rr for role in x['files'])
        pose_alias = Counter(x['files']['POSE']['publisher_MD5'] for x in rr if 'POSE' in x['files'])
        census.append({'capture_directory': g, 'union_frame_stems': len(rr),
                       'component_counts': dict(sorted(roles.items())),
                       'complete_four_way': sum(x['four_way_complete'] for x in rr),
                       'incomplete': sum(not x['four_way_complete'] for x in rr),
                       'missing_role_counts': dict(Counter(z for x in rr for z in x['missing_roles'])),
                       'distinct_pose_file_MD5s_metadata_only': len(pose_alias),
                       'pose_MD5_is_not_physical_capture_equivalence': True,
                       'known_exposed_pilot_frames': sorted(x['publisher_frame_path'] for x in rr if x['known_diagnostic_exposed_frame']),
                       'proposed_role': by_group[g]['proposed_role']})
    file_map = {}
    for row in complete:
        for f in [*row['files'].values(), row['K']]:
            if f['file_id'] in file_map:
                need(file_map[f['file_id']] == f, 'Same public file ID metadata')
            file_map[f['file_id']] = f
    exact_files = sorted(file_map.values(), key=lambda f: f['publisher_relative_path'])
    result = {'completed': True, 'metadata_only': True,
              'dataset': {'DOI': '10.48804/DJLDFS', 'version': '1.0', 'release_time': version['releaseTime'], 'license': version['license']},
              'source_API': api_binding, 'source_README': readme_binding,
              'complete_bundles': complete, 'complete_bundle_count': len(complete),
              'exact_required_public_files': exact_files, 'unique_required_file_count': len(exact_files),
              'publisher_bytes_required': sum(f['size_bytes'] for f in exact_files),
              'split_proposal': {'adopted': False, 'salt': SPLIT_SALT,
                                 'rule': 'Sort six official coarse collection IDs by SHA256(salt+group); first2 DEV proposal, other4 TRAIN proposal. No pixel/label/model/occlusion input.',
                                 'groups': proposal},
              'scope_limits': ['All284complete included; incomplete entries kept separately. No occlusion, score or model filtering.',
                               'Six directory groups are coarser than unknown acquisition-session/capture identities; grouping is not proof of independent frames.',
                               'Known exposed pilots0000/0011 and their entire0.55m_hall collection are disclosed. No samecapture clean held-out claim.',
                               'Current223 human-confirmed unrelated Pallox does not establish MR6D45 capture disjointness. Encoded-byte no-hit does not exclude reencoding/capture overlap.',
                               'Public source preparation does not require unavailable global capture proof, but later training/adoption and clean performance claims need separate Root scope; never use protected benchmark to claim clean performance.'],
              'downloads_decodes_or_new_labels': 0, 'training_admission': False}
    out = safe(out, False);need(not out.exists(), 'Fresh metadata output');out.mkdir()
    manifest = dump(out/'ALL284_FOUR_WAY_METADATA_MANIFEST.json', result)
    missing_b = dump(out/'INCOMPLETE_BUNDLES_RETAINED.json', {'completed': True, 'all_incomplete_retained': missing,
                     'count': len(missing), 'not_dropped_by_occlusion_model_or_visibility': True})
    census_b = dump(out/'GROUP_CENSUS_AND_SPLIT_PROPOSAL.json', {'completed': True, 'census': census, 'proposal': result['split_proposal'],
                     'total_union_frames': len(bundles), 'complete': len(complete), 'incomplete': len(missing),
                     'all_complete_file_count_with_K': len(exact_files), 'publisher_total_bytes': result['publisher_bytes_required'],
                     'not_a_training_or_blind_performance_admission': True})
    receipt = dump(out/'COMPLETE.json', {'completed': True, 'PASS': True, 'manifest': manifest,
                       'missing_bundles': missing_b, 'census': census_b, 'code': binding(Path(__file__).absolute()),
                       'metadata_inputs': [api_binding, readme_binding], 'payload_FS_access_or_network_or_decode': 0,
                       'training_admission': False})
    print(json.dumps({'completed': True, 'PASS': True, 'COMPLETE': receipt, 'group_census': census,
                      'unique_public_files': len(exact_files), 'publisher_bytes': result['publisher_bytes_required']}, sort_keys=True))
    return receipt


STORAGE_SCOPE = {'scheme': 'https', 'host': 'rdmo.icts.kuleuven.be', 'port': 443,
                 'path_prefix': '/dataverse/10.48804/DJLDFS/', 'userinfo': None}


def storage_url_in_scope(url):
    try:
        p = urllib.parse.urlsplit(url)
        return (p.scheme == 'https' and p.hostname == STORAGE_SCOPE['host'] and
                p.port in (None, 443) and p.username is None and p.password is None and
                not p.fragment and p.path.startswith(STORAGE_SCOPE['path_prefix']) and
                '%' not in p.path and '\\' not in p.path and
                not any(x in ('.', '..') for x in p.path.split('/')))
    except (ValueError, TypeError):
        return False


def redacted_url_metadata(url):
    # No signed query values or full final URL are returned, logged or persisted.
    p = urllib.parse.urlsplit(url)
    return {'scheme': p.scheme, 'host': p.hostname, 'port': p.port or 443,
            'path': p.path, 'query_sha256': hashlib.sha256(p.query.encode()).hexdigest(),
            'query_keys': sorted({key for key, _ in urllib.parse.parse_qsl(p.query, keep_blank_values=True)})}


class DatasetScopedRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, initial_url):
        super().__init__()
        p = urllib.parse.urlsplit(initial_url)
        need(p.scheme == 'https' and p.hostname == 'rdr.kuleuven.be' and
             p.port in (None, 443) and p.username is None and p.password is None and
             not p.query and not p.fragment and
             p.path.startswith('/api/access/datafile/') and p.path.rsplit('/', 1)[-1].isdigit(),
             'Only an official exact file-ID initial URL')
        self.initial_url = initial_url
        self.events = []

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        need(req.full_url == self.initial_url and not self.events,
             'Only one official initial-to-storage redirect; no chained redirect')
        need(code in (301, 302, 303, 307, 308) and storage_url_in_scope(newurl),
             'Redirect outside the exact official HTTPS dataset storage scope')
        self.events.append({'initial_public_url': self.initial_url, 'HTTP_redirect_status': code,
                            'storage': redacted_url_metadata(newurl)})
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def validated_reuse_plan(e, previous_entry, previous_failure, needed):
    need(e.get('explicit_user_authorized_resumption') is True,
         'Root explicit user-authorized resumption required; never automatic retry')
    need(previous_entry['code']['sha256'] == '23c59ba4e769a0c988993441bf2eee541060cdd262257edae4e8a2afa4f0d629' and
         previous_entry['metadata_manifest'] == e['metadata_manifest'] and
         previous_entry['bundle_ids'] == e['bundle_ids'] and
         previous_entry['exact_download_allowlist'] == e['exact_download_allowlist'] and
         previous_entry['redirect_scope'] == e['redirect_scope'], 'Same sealed V2 official allowlist/scope')
    need(previous_failure.get('completed') is False and previous_failure.get('PASS') is False and
         previous_failure.get('automatic_retry') is False, 'Preserved failed/interrupted V2 receipt')
    completed = previous_failure['completed_downloads']
    old_by_id = {r['file_id']: r for r in completed}
    need(len(old_by_id) == len(completed), 'Unique previous completed IDs')
    reusable = set(old_by_id) & set(needed)
    planned = e['reuse_files']
    need(len(planned) == len(reusable) == e['expected_reused_completed_files'] and
         {r['publisherfile_id'] for r in planned} == reusable, 'Exact completed-file reuse subset; no partial records')
    result = {}
    for r in planned:
        file_id = r['publisherfile_id']; before = old_by_id[file_id]; expected = needed[file_id]
        source = r['source']
        need(source == {k: before[k] for k in ('path', 'sha256', 'size_bytes')} and
             r['original_MD5'] == before['publisher_MD5_verified'] == expected['publisher_MD5'] and
             source['size_bytes'] == expected['size_bytes'] and
             before['publisher_relative_path'] == expected['publisher_relative_path'],
             'Reuse exact completed binding/metadata MD5/size/path')
        need(not str(source['path']).casefold().endswith('.part'), 'Partial files never reused')
        need(len(source['sha256']) == 64, 'Completed SHA256 required')
        result[file_id] = r
    return result


def copy_verified_reuse(source, destination, expected):
    # Source is an explicitly authorized completed-file binding, never a directory or .part.
    need(not str(source['path']).casefold().endswith('.part'), 'Partial files never copied')
    fd = os.open(safe(source['path']), os.O_RDONLY | os.O_NOFOLLOW)
    part = safe(destination.with_name(destination.name+'.part'), False)
    count, sha, md5 = 0, hashlib.sha256(), hashlib.md5()
    with os.fdopen(fd, 'rb') as src, part.open('xb') as dst:
        while True:
            chunk = src.read(1024*1024)
            if not chunk:break
            count += len(chunk);need(count <= source['size_bytes'], 'Reuse source exceeds exact bound size')
            sha.update(chunk);md5.update(chunk);dst.write(chunk)
    need(count == source['size_bytes'] == expected['size_bytes'] and
         sha.hexdigest() == source['sha256'] and md5.hexdigest() == expected['publisher_MD5'],
         'Reuse source exact SHA256/size/publisherMD5; no GET fallback')
    part.rename(destination)
    return {'sha256':sha.hexdigest(), 'size_bytes':count, 'publisher_MD5_verified':md5.hexdigest()}


def download(entry_path):
    """Future only: opaque MD5/size-checked files, no decoder/labels/models."""
    e = json.loads(safe(entry_path).read_bytes())
    need(e.get('approved_actual_public_allowlist_download') is True and e.get('opaque_only') is True,
         'Root future exact public download entry required')
    need(e.get('explicit_user_authorized_resumption') is True,
         'Root explicit user-authorized resumption required before metadata access')
    need(e.get('training_admission') is False and e.get('automatic_retry') is False,
         'No automatic retry or training permission')
    need(e.get('Rootapproved_same_dataset_official_redirect') is True and
         e.get('redirect_scope') == STORAGE_SCOPE, 'Root explicit same-dataset redirect scope required')
    need(e['code'] == binding(Path(__file__).absolute()), 'Pinned downloader code')
    manifest = json.loads(bound_read(e['metadata_manifest']))
    need(manifest['source_API']['sha256'] == API_SHA and manifest['source_README']['sha256'] == README_SHA,
         'Same sealed official metadata')
    all_bundles = {r['bundle_id']: r for r in manifest['complete_bundles']}
    need(len(e['bundle_ids']) == len(set(e['bundle_ids'])) > 0 and set(e['bundle_ids']) == set(all_bundles),
         'Root explicit full284 complete bundle set, no new selection or discovery')
    needed = {}
    for name in e['bundle_ids']:
        row = all_bundles[name]
        for f in [*row['files'].values(), row['K']]:needed[f['file_id']] = f
    allowed = e['exact_download_allowlist']
    need(len(allowed) == len(needed) and {x['file_id'] for x in allowed} == set(needed),
         'Exact all components plus K for admitted bundle set')
    for x in allowed:
        f = needed[x['file_id']]
        need(x['publisher_relative_path'] == f['publisher_relative_path'] and x['size_bytes'] == f['size_bytes'] and
             x['publisher_MD5'] == f['publisher_MD5'], 'Entry matches public metadata')
        need(x['url'] == 'https://rdr.kuleuven.be/api/access/datafile/'+str(x['file_id']), 'Exact official HTTPS endpoint')
    previous_entry = json.loads(bound_read(e['previous_download_entry']))
    previous_failure = json.loads(bound_read(e['previous_failure']))
    reuse = validated_reuse_plan(e, previous_entry, previous_failure, needed)
    out = safe(e['output_directory'], False);need(not out.exists(), 'Fresh V3 opaque output; never in-place resume');out.mkdir()
    completed, stage = [], 'before_first_request'
    try:
        for x in sorted(allowed, key=lambda a: a['publisher_relative_path']):
            stage = x['publisher_relative_path']
            relative = Path(stage)
            need(not relative.is_absolute() and '..' not in relative.parts and
                 not any(t in stage.casefold() for t in ('v112','cache','.pyc')), 'Safe declared destination')
            p = safe(out/relative, False);p.parent.mkdir(parents=True, exist_ok=True)
            if x['file_id'] in reuse:
                source = reuse[x['file_id']]['source']
                copied = copy_verified_reuse(source, p, needed[x['file_id']])
                completed.append({'file_id':x['file_id'], 'publisher_relative_path':stage, 'url':x['url'],
                                  'path':str(p), **copied, 'operation':'REUSE_VERIFIED_COPY', 'source':source})
                continue
            part = safe(p.with_name(p.name+'.part'), False)
            count, md5, sha = 0, hashlib.md5(), hashlib.sha256()
            request = urllib.request.Request(x['url'], headers={'User-Agent':'PublicMetadataBoundOpaqueCollector/1.0'})
            redirect = DatasetScopedRedirect(x['url'])
            opener = urllib.request.build_opener(redirect)
            with opener.open(request, timeout=90) as response, part.open('xb') as f:
                final_url = response.geturl()
                need(response.status == 200 and (final_url == x['url'] or storage_url_in_scope(final_url)),
                     'Final response outside exact public initial or scoped dataset storage')
                final_origin = redacted_url_metadata(final_url)
                while True:
                    chunk = response.read(1024*1024)
                    if not chunk:break
                    count += len(chunk);need(count <= x['size_bytes'], 'Response exceeds exact publisher size')
                    md5.update(chunk);sha.update(chunk);f.write(chunk)
            need(count == x['size_bytes'] and md5.hexdigest() == x['publisher_MD5'], 'Opaque size/originalMD5')
            part.rename(p)
            completed.append({'file_id':x['file_id'], 'publisher_relative_path':stage, 'url':x['url'],
                              'path':str(p), 'sha256':sha.hexdigest(), 'size_bytes':count, 'publisher_MD5_verified':md5.hexdigest(),
                              'official_redirect_events':redirect.events, 'final_origin_redacted':final_origin,
                              'operation':'GET_OFFICIAL_SCOPE'})
            if len(completed) % 25 == 0:
                print(json.dumps({'stage':'opaque_public_files', 'completed_files':len(completed),
                                  'required_files':len(needed), 'reused_files':len(reuse)}), flush=True)
        receipt = dump(out/'OPAQUE_DOWNLOAD_COMPLETE.json', {'completed':True,'PASS':True,'entry':binding(entry_path),
                       'metadata_manifest':e['metadata_manifest'],'downloads':completed,'decoded_files':0,
                       'previous_failure':e['previous_failure'], 'previous_download_entry':e['previous_download_entry'],
                       'explicit_user_authorized_resumption':True, 'reused_verified_files':len(reuse),
                       'new_GET_files':len(completed)-len(reuse), 'required_files':len(needed),
                       'labels_models_training_or_evaluation':0,'training_admission':False})
        print(json.dumps({'completed':True,'PASS':True,'receipt':receipt},sort_keys=True))
    except BaseException as exc:
        dump(out/'FAILURE.json', {'completed':False,'PASS':False,'stage':stage,'error_type':type(exc).__name__,
                               'HTTP_status':getattr(exc, 'code', None) if isinstance(exc, urllib.error.HTTPError) else None,
                               'error_summary':'Opaque public download failed; URL query values suppressed.',
                               'completed_downloads':completed,'partial_bytes_preserved':True,'automatic_retry':False})
        raise RuntimeError('Opaque public download failed; signed URL query values suppressed.') from None


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--collect-entry')
    ap.add_argument('--download-entry')
    args = ap.parse_args()
    need(bool(args.collect_entry) != bool(args.download_entry), 'One operation')
    if args.collect_entry:
        e = json.loads(safe(args.collect_entry).read_bytes())
        need(e['approved_metadata_only_collection'] is True and e['code'] == binding(Path(__file__).absolute()), 'Pinned metadata-only entry')
        collect(e['official_API'],e['official_README'],e['output_directory'])
    else:
        download(args.download_entry)

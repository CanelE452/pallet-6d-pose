"""Seal exact physical publication files after completed independent checks."""
from pathlib import Path
import argparse, hashlib, json, subprocess


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repository', required=True, type=Path)
    ap.add_argument('--admin', required=True, type=Path)
    ap.add_argument('--initial-main', required=True)
    a = ap.parse_args()
    prefix = '_docs/experiments/pallet-keypoint-visibility/'
    root = a.repository/prefix
    manifest_file = root/'PUBLICATION_MANIFEST.json'
    content = manifest_file.read_bytes(); manifest = json.loads(content)
    reports = {}
    for key, name in [('metrics', 'ACTUAL_FINAL_SAVED_METRIC_AND_STORED_SHA_REVIEW.json'),
                      ('original_bytes', 'ACTUAL_ALL_ORIGINAL_BYTE_ROUNDTRIP_REVIEW.json'),
                      ('portable_and_credentials', 'ACTUAL_CREDENTIAL_SIZE_AND_PORTABLE_LINK_QA.json')]:
        f = root/'publication'/name
        reports[key] = json.loads(f.read_text())
        assert reports[key]['completed'] is True and reports[key]['PASS'] is True
    assert reports['original_bytes']['manifest_stored_sha256'] == sha(content)
    assert reports['portable_and_credentials']['manifest_stored_sha256'] == sha(content)
    summary = dict(completed=True, PASS=True, main_publication_status='Verified payload ready for bounded main pushes',
                   manifest_sha256=sha(content), original_files_verified=reports['original_bytes']['published_originals_verified'],
                   saved_primary_points=reports['metrics']['primary_known'], model_material_rows_verified=12,
                   credential_or_portable_link_issues=reports['portable_and_credentials']['issues'],
                   excluded_source_files=sum(e['status']=='EXCLUDED' for e in manifest['entries']),
                   raw_dataset_banks_foundation_weights_environment_and_V112='excluded with provenance; V112 not accessed',
                   model_adoption=False, full_visibility_goal='active_unsolved',
                   scope='Publication arithmetic and artifact-byte/link checks, not model success or new training. Verification outputs sit outside the manifest they verify; Git commit binds them.')
    for key, name in [('metrics', 'ACTUAL_FINAL_SAVED_METRIC_AND_STORED_SHA_REVIEW.json'),
                      ('original_bytes', 'ACTUAL_ALL_ORIGINAL_BYTE_ROUNDTRIP_REVIEW.json'),
                      ('portable_and_credentials', 'ACTUAL_CREDENTIAL_SIZE_AND_PORTABLE_LINK_QA.json')]:
        f=root/'publication'/name
        summary[key] = dict(path='publication/'+name, sha256=sha(f.read_bytes()), bytes=f.stat().st_size)
    f=root/'publication/PUBLICATION_VERIFICATION_SUMMARY.json'
    with f.open('x') as stream: json.dump(summary, stream, ensure_ascii=False, indent=2); stream.write('\n')
    seal=a.admin/'FINAL_MANIFEST_BEFORE_ACTUAL_VERIFICATION.json'
    (root/'publication'/seal.name).write_bytes(seal.read_bytes())
    files={}
    def add(path, expected=None, size=None):
        assert 'v112' not in path.lower() and '..' not in Path(path).parts and not Path(path).is_absolute()
        f=root/path
        assert f.is_file() and not f.is_symlink()
        actual_size=f.stat().st_size
        assert actual_size < 100*1024**2
        if size is not None: assert size==actual_size
        h=expected or sha(f.read_bytes())
        r=dict(path=prefix+path, sha256=h, size_bytes=actual_size)
        if path in files: assert files[path]==r
        else:files[path]=r
    for e in manifest['entries']:
        if e['status']=='EXCLUDED':continue
        if e['gzipchunks'] is None:add(e['published_relative'], e['published_sha256'], e['size_bytes'])
        else:
            for part in e['gzipchunks']['parts']:add(part['path'], part['sha256'], part['stored_size_bytes'])
    for name in ['PUBLICATION_MANIFEST.json','publication/ACTUAL_FINAL_SAVED_METRIC_AND_STORED_SHA_REVIEW.json',
                 'publication/ACTUAL_ALL_ORIGINAL_BYTE_ROUNDTRIP_REVIEW.json',
                 'publication/ACTUAL_CREDENTIAL_SIZE_AND_PORTABLE_LINK_QA.json',
                 'publication/PUBLICATION_VERIFICATION_SUMMARY.json',
                 'publication/FINAL_MANIFEST_BEFORE_ACTUAL_VERIFICATION.json']:
        add(name)
    # Preserve the previous global README byte-for-byte after the new entry.
    old=subprocess.check_output(['git','show',a.initial_main+':readme.md'],cwd=a.repository)
    readme=(a.repository/'readme.md').read_bytes();assert readme.endswith(old)
    listing=list(files.values())
    archive=[r for r in listing if r['path'].startswith(prefix+'archive/')]
    final=[r for r in listing if not r['path'].startswith(prefix+'archive/')]
    final.append(dict(path='readme.md',sha256=sha(readme),size_bytes=len(readme)))
    limit=480*1024**2;batches=[]
    for group in (archive,final):
        current=[];size=0
        for row in sorted(group,key=lambda r:r['path']):
            if current and size+row['size_bytes']>limit:
                batches.append(dict(files=current));current=[];size=0
            current.append(row);size+=row['size_bytes']
        if current:batches.append(dict(files=current))
    for i,b in enumerate(batches):
        b['message']=(f'Archive keypoint visibility experiments ({i+1}/{len(batches)})'
                      if i+1<len(batches) else 'Publish visibility experiment index, galleries and independent verification')
    plan=dict(authorized_main_push=True,repository=str(a.repository),
              remote='https://github.com/CanelE452/pallet-6d-pose.git',branch='main',
              initial_remote_main=a.initial_main,manifest_sha256=sha(content),batch_max_bytes=limit,batches=batches)
    out=a.admin/'ROOT_SEALED_MAIN_PUSH_PLAN.json'
    with out.open('x') as stream:json.dump(plan,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(dict(completed=True, files=sum(len(b['files']) for b in batches), batches=len(batches),
                          stored_bytes=sum(r['size_bytes'] for b in batches for r in b['files']), plan=str(out))))


if __name__=='__main__':main()

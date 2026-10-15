"""Opaque-byte inventory and lossless post-snapshot release delta; no git writes.

Each source file is hashed once for inventory. Exact duplicate content may be
stored as tar hardlinks; all original names are restored. Existing snapshot,
source files, results, manifests, checkout and index remain unchanged.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import gzip,hashlib,json,os,stat,subprocess,tarfile,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
D=ROOT/'_docs/experiments/pallet_visibility_curriculum_20261013_v1'
B=D/'nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
PUB=D/'publication_followup_20261015_v11'
PREVIOUS=D/'publication_20261015_v1'
WT=Path('/tmp/pallet-visibility-main-publication-20261015-v1')
EXTERNAL=Path('/tmp/pallet-visibility-followup-assets-20261015-v11')
ROOTS=(D,ROOT/'scripts/research/pallet_visibility_curriculum_20261013_v1',
 ROOT/'scripts/research/pallet_foundation_mask_20261013_v1',ROOT/'scripts/research/pallet_nonsevere_segmentation_20261013_v1')
PROTOCOLS=(B/'clean_fresh_BI_pose_eval_v4_32_r2/PROTOCOL.json',
 B/'initial_only_observed_ray_observations_v4_32/FINAL_PROTOCOL.json',
 B/'largest_component_global_outline_pose_eval_v4_33/PROTOCOL.json',
 B/'largest_component_global_outline_observations_v4_33/FINAL_PROTOCOL.json',
 B/'new_gain10_paired_BI_v2/PROTOCOL.json',
 B/'raw_yolo_N3_subpix_gain10_process_v1/ANALYSIS_PLAN.json',
 B/'new_gain10_paired_BI_process_v1/PROTOCOL.json')
BLOCK=8*1024*1024
PART=512*1024*1024
PREFIX='pallet-visibility-followup-20261015-v11.tar.zst'

def now():return datetime.now(timezone.utc).isoformat()
def rel(path):return Path(path).relative_to(ROOT).as_posix()
def write(path,value):
    if path.exists():raise FileExistsError('Preserve existing inventory/packaging records: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(BLOCK),b''):h.update(block)
    return h.hexdigest()
def excluded(path):
    parts=path.relative_to(ROOT).parts;lower=[p.lower() for p in parts];name=path.name.lower()
    if path.is_relative_to(PUB):return 'current_publication_recursion'
    if any(p in ('.git','__pycache__','.pytest_cache','.mypy_cache','.ruff_cache') for p in parts):return 'git_or_cache'
    if path.suffix in ('.pyc','.pyo'):return 'interpreter_cache'
    if path.is_symlink():return 'symlink'
    if any(p in ('.ssh','.aws','.azure','.config','.kube','.gnupg') for p in lower):return 'machine_configuration'
    if name=='.env' or name.startswith('.env.') or name.startswith('credential') or name in ('id_rsa','id_dsa','id_ecdsa','id_ed25519','authorized_keys','known_hosts') or name.endswith(('.key','.p12','.pfx')):return 'machine_secret_name'
    return None
def group(path):
    p=ROOT/path
    if p.is_relative_to(B):return p.relative_to(B).parts[0]
    if p.is_relative_to(ROOTS[1]):return 'SOURCES'
    if p.is_relative_to(D):return p.relative_to(D).parts[0]
    return 'RUNTIME_OUTSIDE_MAIN_EXPERIMENT_ROOT'
def dumpgz(path,rows):
    if path.exists():raise FileExistsError(path)
    with path.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as stream:
        for row in rows:stream.write((json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode())
def rows(path):
    with gzip.open(path,'rt',encoding='utf8') as stream:
        for line in stream:yield json.loads(line)


def inventory():
    if (PUB/'repro_inventory_summary_v2.json').exists():raise FileExistsError('Inventory already closed')
    start=time.monotonic();previous={r['relpath']:r for r in rows(PREVIOUS/'FILE_MANIFEST.jsonl.gz')}
    git={}
    for item in subprocess.check_output(['git','-C',str(WT),'ls-tree','-r','-z','HEAD']).split(b'\0'):
        if not item:continue
        meta,path=item.split(b'\t',1);mode,kind,blob=meta.decode().split()
        if kind=='blob':git[path.decode()]=blob
    head=subprocess.check_output(['git','-C',str(WT),'rev-parse','HEAD']).decode().strip()
    if head!='5f5f619ef9af3c38313a5619cc554109d116a090':raise ValueError('Explicit publication cutoff changed')
    paths=set();exclusions=Counter()
    for base in ROOTS:
        for directory,dirs,files in os.walk(base,followlinks=False):
            current=Path(directory);kept=[]
            for name in dirs:
                p=current/name;reason=excluded(p)
                if reason:exclusions[reason]+=1
                else:kept.append(name)
            dirs[:]=sorted(kept)
            for name in files:
                p=current/name;reason=excluded(p)
                if reason:exclusions[reason]+=1;continue
                if not stat.S_ISREG(p.lstat().st_mode):exclusions['nonregular']+=1;continue
                paths.add(rel(p))
    expected={};protocols=[];conflicts=[]
    for protocol in PROTOCOLS:
        if not protocol.is_file():
            protocols.append(dict(path=rel(protocol),absent_optional=True));continue
        data=json.loads(protocol.read_text());protocols.append(dict(path=rel(protocol),sha256=sha(protocol)))
        for field in ('source_sha256','input_sha256','protected_user_sha256'):
            for key,digest in data.get(field,{}).items():
                p=Path(key) if Path(key).is_absolute() else ROOT/key
                if not p.is_relative_to(ROOT) or '..' in p.relative_to(ROOT).parts or not p.resolve().is_relative_to(ROOT):raise ValueError('Runtime input outside repository: '+key)
                reason=excluded(p)
                if reason:raise ValueError('Required runtime binding excluded: '+key+' '+reason)
                r=rel(p)
                if r in expected and expected[r]['sha256']!=digest:conflicts.append(r)
                item=expected.setdefault(r,dict(sha256=digest,roles=[]));item['roles'].append(dict(protocol=rel(protocol),field=field))
                paths.add(r)
    if conflicts:raise ValueError('Conflicting current runtime maps: '+str(sorted(set(conflicts))))
    result=[];delta=[];missing=[];hash_groups=defaultdict(list);sizes=Counter();last=start;runtime_fail=[]
    for index,r in enumerate(sorted(paths),1):
        p=ROOT/r
        if not p.is_file():missing.append(r);continue
        before=p.stat();h=hashlib.sha256();blob=hashlib.sha1() if r in git else None
        if blob:blob.update(('blob '+str(before.st_size)+'\0').encode())
        with p.open('rb') as stream:
            for block in iter(lambda:stream.read(BLOCK),b''):
                h.update(block)
                if blob:blob.update(block)
        after=p.stat()
        if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):raise ValueError('File changed during opaque hash: '+r)
        digest=h.hexdigest();old=previous.get(r)
        old_exact=old is not None and old['size']==after.st_size and old['sha256']==digest
        main_exact=blob is not None and blob.hexdigest()==git[r]
        record=dict(relpath=r,size=after.st_size,sha256=digest,group=group(r),mtime_ns=after.st_mtime_ns,
            prior_release_exact=old_exact,prior_release_path_present=old is not None,
            main_path_present=r in git,main_blob_exact=main_exact)
        if blob:record['current_git_blob_sha1']=blob.hexdigest();record['published_git_blob_sha1']=git[r]
        if r in expected:
            record['runtime_roles']=expected[r]['roles']
            if expected[r]['sha256']!=digest:runtime_fail.append(r)
        result.append(record)
        if not old_exact:delta.append(record)
        hash_groups[(after.st_size,digest)].append(r)
        tick=time.monotonic()
        if tick-last>=10:
            print(json.dumps(dict(stage='opaque_inventory_hash_once',files=index,total=len(paths),bytes=sum(x['size'] for x in result),seconds=tick-start)),flush=True);last=tick
    if missing or runtime_fail:raise ValueError('Missing/runtime byte mismatch: '+str(dict(missing=missing,mismatch=runtime_fail)))
    # Exact duplicate payloads are aliases only in tar storage, never omitted names.
    canon={}
    for row in delta:
        key=(row['size'],row['sha256']);first=canon.setdefault(key,row['relpath'])
        row['archive_member_kind']='file' if first==row['relpath'] else 'hardlink'
        if first!=row['relpath']:row['hardlink_to']=first
    dumpgz(PUB/'repro_inventory_current_files_v2.jsonl.gz',result)
    dumpgz(PUB/'repro_inventory_delta_manifest_v1.jsonl.gz',delta)
    (PUB/'repro_inventory_delta_paths_v1.null').write_bytes(b''.join(x['relpath'].encode()+b'\0' for x in delta))
    groups=defaultdict(Counter)
    for r in result:
        c=groups[r['group']];c['files']+=1;c['bytes']+=r['size']
        if not r['main_blob_exact']:c['not_main_exact_files']+=1;c['not_main_exact_bytes']+=r['size']
        if not r['prior_release_exact']:c['delta_files']+=1;c['delta_bytes']+=r['size']
        if 'spool' in r['relpath']:c['spool_files']+=1;c['spool_bytes']+=r['size']
        if 'code_snapshot' in r['relpath'] or 'source_snapshot' in r['relpath']:c['snapshot_files']+=1;c['snapshot_bytes']+=r['size']
    dup=[dict(size=k[0],sha256=k[1],paths=v) for k,v in hash_groups.items() if len(v)>1]
    write(PUB/'repro_inventory_duplicate_content_v1.json',dict(groups=dup,duplicate_names=sum(len(x['paths'])-1 for x in dup),bytes_savable_if_all_groups_deduplicated=sum(x['size']*(len(x['paths'])-1) for x in dup)))
    summary=dict(schema='EXACT_POST_ORIGINAL_RELEASE_DELTA_INVENTORY_V11_V2',completed_utc=now(),main_commit=head,
        previous_release_tag='visibility-curriculum-snapshot-20261015-v1',previous_manifest_sha256=sha(PREVIOUS/'FILE_MANIFEST.jsonl.gz'),previous_archive_sha256=json.loads((PREVIOUS/'ARCHIVE_FILES.json').read_text())['concatenated_archive_sha256'],
        current_files=len(result),current_bytes=sum(x['size'] for x in result),
        main_exact_files=sum(x['main_blob_exact'] for x in result),main_exact_bytes=sum(x['size'] for x in result if x['main_blob_exact']),
        original_release_exact_files=sum(x['prior_release_exact'] for x in result),original_release_exact_bytes=sum(x['size'] for x in result if x['prior_release_exact']),
        delta_files=len(delta),delta_logical_bytes=sum(x['size'] for x in delta),delta_unique_payload_bytes=sum(x['size'] for x in delta if x['archive_member_kind']=='file'),delta_hardlink_members=sum(x['archive_member_kind']=='hardlink' for x in delta),
        runtime_unique_files=len(expected),runtime_binding_failures=0,protocols=protocols,excluded=dict(exclusions),
        groups=[dict(group=k,**v) for k,v in sorted(groups.items())],large_files=[r for r in result if r['size']>=100_000_000],
        source_sha256={rel(Path(__file__)):sha(__file__)},source_bytes_hashed_once_per_current_file=True,
        numeric_journal_or_reference_decodes=0,git_writes=0,source_mutations_or_deletions=0,
        restoration='original full release archive, then this delta over it; verify every path with current manifest; current publication manifest/assets supplied separately',
        current_manifest_sha256=sha(PUB/'repro_inventory_current_files_v2.jsonl.gz'),delta_manifest_sha256=sha(PUB/'repro_inventory_delta_manifest_v1.jsonl.gz'),elapsed_seconds=time.monotonic()-start)
    write(PUB/'repro_inventory_summary_v2.json',summary)
    print(json.dumps({k:summary[k] for k in ('current_files','current_bytes','delta_files','delta_logical_bytes','delta_unique_payload_bytes','delta_hardlink_members','runtime_unique_files','elapsed_seconds')}),flush=True)


def archive():
    if EXTERNAL.exists() and any(EXTERNAL.iterdir()):raise FileExistsError('Preserve previous delta archive attempt')
    manifest=list(rows(PUB/'repro_inventory_delta_manifest_v1.jsonl.gz'));summary=json.loads((PUB/'repro_inventory_summary_v2.json').read_text())
    if sha(PUB/'repro_inventory_delta_manifest_v1.jsonl.gz')!=summary['delta_manifest_sha256']:raise ValueError('Closed delta manifest changed')
    EXTERNAL.mkdir(parents=True,exist_ok=True);started=time.monotonic();parts=[];whole=hashlib.sha256();total=0
    # A file-backed zstd stdout is split without retaining the tar or full archive.
    process=subprocess.Popen(['zstd','-T2','-3','-c'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=(EXTERNAL/'zstd.stderr.log').open('wb'))
    import threading
    errors=[]
    def produce():
        try:
            with tarfile.open(fileobj=process.stdin,mode='w|',format=tarfile.PAX_FORMAT) as tar:
                for index,row in enumerate(manifest,1):
                    p=ROOT/row['relpath'];st=p.stat()
                    if st.st_size!=row['size'] or st.st_mtime_ns!=row['mtime_ns']:raise ValueError('Closed source changed before archive: '+row['relpath'])
                    info=tarfile.TarInfo(row['relpath']);info.mode=stat.S_IMODE(st.st_mode);info.uid=info.gid=0;info.mtime=int(st.st_mtime)
                    if row['archive_member_kind']=='hardlink':
                        info.type=tarfile.LNKTYPE;info.linkname=row['hardlink_to'];info.size=0;tar.addfile(info)
                    else:
                        info.size=row['size']
                        with p.open('rb') as source:tar.addfile(info,source)
                    if index%2000==0:print('tar members',index,'/',len(manifest),flush=True)
        except BaseException as error:errors.append(repr(error))
        finally:process.stdin.close()
    producer=threading.Thread(target=produce);producer.start();out=None;part_sha=None;part_size=0
    while True:
        block=process.stdout.read(BLOCK)
        if not block:break
        whole.update(block);total+=len(block)
        offset=0
        while offset<len(block):
            if out is None:
                path=EXTERNAL/f'{PREFIX}.part-{len(parts):03d}';out=path.open('wb');part_sha=hashlib.sha256();part_size=0
            piece=block[offset:offset+min(PART-part_size,len(block)-offset)];out.write(piece);part_sha.update(piece);part_size+=len(piece);offset+=len(piece)
            if part_size==PART:
                out.close();parts.append(dict(name=path.name,size=part_size,sha256=part_sha.hexdigest()));out=None
                print('compressed part',len(parts),'bytes',total,flush=True)
    if out is not None:out.close();parts.append(dict(name=path.name,size=part_size,sha256=part_sha.hexdigest()))
    producer.join();code=process.wait()
    if errors or code:raise RuntimeError(dict(tar_errors=errors,zstd_exit=code))
    result=dict(schema='LOSSLESS_FOLLOWUP_DELTA_ARCHIVE_V11',archive_member_files=len(manifest),hardlink_members=summary['delta_hardlink_members'],logical_source_bytes=summary['delta_logical_bytes'],
        unique_regular_payload_bytes=summary['delta_unique_payload_bytes'],format='POSIX pax tar with exact-content hardlink aliases; zstd -T2 -3; sorted original repo paths',
        concatenated_archive_sha256=whole.hexdigest(),total_compressed_bytes=total,part_size_bytes=PART,parts=parts,
        source_manifest_sha256=summary['delta_manifest_sha256'],elapsed_seconds=time.monotonic()-started,root=str(ROOT))
    write(EXTERNAL/'ARCHIVE_FILES.json',result);write(PUB/'repro_inventory_archive_files_v1.json',result)
    text=''.join(x['sha256']+'  '+x['name']+'\n' for x in parts)
    (EXTERNAL/'ARCHIVE_SHA256SUMS.txt').write_text(text);(PUB/'repro_inventory_archive_SHA256SUMS_v1.txt').write_text(text)
    print(json.dumps(dict(compressed_bytes=total,parts=len(parts),sha256=whole.hexdigest())),flush=True)


def verify():
    manifest={r['relpath']:r for r in rows(PUB/'repro_inventory_delta_manifest_v1.jsonl.gz')}
    info=json.loads((EXTERNAL/'ARCHIVE_FILES.json').read_text());parts=[EXTERNAL/p['name'] for p in info['parts']]
    # cat/zstd stream bytes are opaque. No JSON journal, RGB, mask or GT decode.
    cat=subprocess.Popen(['cat',*[str(p) for p in parts]],stdout=subprocess.PIPE)
    unpack=subprocess.Popen(['zstd','-d','-c'],stdin=cat.stdout,stdout=subprocess.PIPE,stderr=(EXTERNAL/'verify-zstd.stderr.log').open('wb'));cat.stdout.close()
    checked={};logical=0;fail=[];started=time.monotonic()
    with tarfile.open(fileobj=unpack.stdout,mode='r|') as tar:
        for member in tar:
            r=manifest.get(member.name)
            if r is None or member.name in checked:raise ValueError('Unexpected/duplicate tar member')
            if member.islnk():
                if member.linkname!=r.get('hardlink_to') or member.linkname not in checked:raise ValueError('Hardlink target not a previously verified byte identity')
                size,digest=checked[member.linkname]
            elif member.isfile():
                h=hashlib.sha256();size=0;source=tar.extractfile(member)
                for block in iter(lambda:source.read(BLOCK),b''):h.update(block);size+=len(block)
                digest=h.hexdigest()
            else:raise ValueError('Non-file tar member')
            if size!=r['size'] or digest!=r['sha256']:fail.append(member.name)
            checked[member.name]=(size,digest);logical+=size
    catcode=cat.wait();zcode=unpack.wait()
    if fail or len(checked)!=len(manifest) or catcode or zcode:raise ValueError(dict(fail=fail,verified=len(checked),expected=len(manifest),cat=catcode,zstd=zcode))
    write(PUB/'repro_inventory_archive_stream_verification_v1.json',dict(schema='EXACT_ALL_DELTA_MEMBER_STREAM_VERIFICATION_V11',completed_utc=now(),verified_files=len(checked),verified_logical_bytes=logical,
        hardlink_identities_validated=True,all_original_paths_retained=True,all_regular_member_bytes_SHA256_match=True,
        archive_manifest_sha256=sha(EXTERNAL/'ARCHIVE_FILES.json'),delta_manifest_sha256=sha(PUB/'repro_inventory_delta_manifest_v1.jsonl.gz'),
        failure_count=0,numeric_decodes=0,files_extracted=0,source_mutations=0,elapsed_seconds=time.monotonic()-started))
    print('verified',len(checked),'logical_bytes',logical,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('inventory','archive','verify'));a=parser.parse_args()
    {'inventory':inventory,'archive':archive,'verify':verify}[a.stage]()

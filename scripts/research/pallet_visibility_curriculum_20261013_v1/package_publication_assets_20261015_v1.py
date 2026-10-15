"""Read-only complete experiment snapshot, split zstd archive and byte audit.

No source data are deleted/edited, no git/upload is performed, and reference
documents are only hashed as bytes. The publication directory itself is
excluded from the snapshot to prevent recursive archive/manifest inclusion.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tarfile
import threading
import time

ROOT=Path(__file__).resolve().parents[3]
EXPERIMENT=ROOT/'_docs/experiments/pallet_visibility_curriculum_20261013_v1'
PUBLICATION=EXPERIMENT/'publication_20261015_v1'
BOUNDARY=EXPERIMENT/'nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
PROTOCOLS=(BOUNDARY/'prepartial_RGB_preservation_eval_v4_26/PROTOCOL.json',
           BOUNDARY/'geometry_guided_edge_observations_v4_29b/PROTOCOL.json')
BASE_ROOTS=(EXPERIMENT,ROOT/'scripts/research/pallet_visibility_curriculum_20261013_v1',
            ROOT/'scripts/research/pallet_foundation_mask_20261013_v1',
            ROOT/'scripts/research/pallet_nonsevere_segmentation_20261013_v1')
EXTERNAL=Path('/tmp/pallet-visibility-publication-assets-20261015-v1')
CHUNK=8*1024*1024
PART_SIZE=512*1024*1024
CACHE_NAMES={'.git','__pycache__','.pytest_cache','.mypy_cache','.ruff_cache'}


def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb')as stream:
        for block in iter(lambda:stream.read(CHUNK),b''):h.update(block)
    return h.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
def relative(path):return path.relative_to(ROOT).as_posix()


def exclusion(path):
    parts=path.relative_to(ROOT).parts
    if 'publication_20261015_v1'in parts:return'publication_recursion'
    if any(p in CACHE_NAMES for p in parts):return'git_or_interpreter_cache'
    if path.suffix.lower()in{'.pyc','.pyo'}:return'interpreter_cache'
    if path.is_symlink():return'symlink'
    # This scope is experiment data and explicit verified inputs only. Never
    # include machine credentials even if a future broad root contains one.
    lowered=[p.lower()for p in parts];name=path.name.lower()
    if any(p in{'.ssh','.aws','.azure','.config','.kube','.gnupg'}for p in lowered):return'machine_configuration'
    if(name=='.env'or name.startswith('.env.')or name in{'id_rsa','id_dsa','id_ecdsa','id_ed25519','authorized_keys',
        'known_hosts','credentials','credentials.json','credentials.yaml','credentials.yml'}
        or name.startswith('credential')or name.endswith(('.key','.p12','.pfx'))):return'machine_secret_name'
    return None


def protocol_bindings():
    bindings={};protocol_info=[]
    for protocol in PROTOCOLS:
        document=json.loads(protocol.read_text())
        protocol_info.append(dict(path=relative(protocol),sha256=sha(protocol)))
        for kind in('source_sha256','input_sha256','protected_user_sha256'):
            for key,digest in document[kind].items():
                lexical=Path(key)
                if lexical.is_absolute():
                    try:lexical=lexical.relative_to(ROOT)
                    except ValueError:raise ValueError('Protocol path lies outside repository: '+str(key))
                if'..'in lexical.parts:raise ValueError('Parent traversal in protocol path')
                path=ROOT/lexical
                if not path.resolve().is_relative_to(ROOT):raise ValueError('Resolved protocol path outside repository')
                reason=exclusion(path)
                if reason:raise ValueError('Required runtime binding excluded by safe scope: '+str(key)+' ('+reason+')')
                if not path.is_file():raise FileNotFoundError('Required runtime file missing: '+str(key))
                rel=relative(path)
                if rel in bindings and bindings[rel]['sha256']!=digest:
                    raise ValueError('Conflicting immutable protocol hashes: '+rel)
                item=bindings.setdefault(rel,dict(sha256=digest,roles=[]))
                item['roles'].append(dict(protocol=relative(protocol),kind=kind))
    return bindings,protocol_info


def inventory(bindings):
    paths=set();excluded=Counter()
    for base in BASE_ROOTS:
        if not base.is_dir():raise FileNotFoundError('Requested publication root missing: '+str(base))
        for directory,dirs,files in os.walk(base,followlinks=False):
            current=Path(directory)
            keep=[]
            for name in dirs:
                path=current/name;reason=exclusion(path)
                if reason:excluded[reason]+=1
                else:keep.append(name)
            dirs[:]=sorted(keep)
            for name in files:
                path=current/name;reason=exclusion(path)
                if reason:excluded[reason]+=1;continue
                if not stat.S_ISREG(path.lstat().st_mode):excluded['non_regular']+=1;continue
                paths.add(relative(path))
    for rel in bindings:paths.add(rel)
    return sorted(paths),dict(excluded)


def manifest(paths,bindings,protocol_info):
    target=PUBLICATION/'FILE_MANIFEST.jsonl.gz'
    if target.exists():raise FileExistsError('Preserve existing publication manifest')
    started=time.monotonic();last=started;total=0;entries={};verified=[]
    with target.open('wb')as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0)as stream:
        for index,rel in enumerate(paths,1):
            path=ROOT/rel;before=path.stat();digest=sha(path);after=path.stat()
            if(before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                raise ValueError('Source file changed during manifest hashing: '+rel)
            item=dict(relpath=rel,size=after.st_size,sha256=digest);entries[rel]=item;total+=after.st_size
            if rel in bindings:
                if digest!=bindings[rel]['sha256']:raise ValueError('Runtime source/input/protected bytes changed: '+rel)
                verified.append(dict(path=rel,size=after.st_size,sha256=digest,roles=bindings[rel]['roles']))
            stream.write((json.dumps(item,ensure_ascii=False,sort_keys=True)+'\n').encode())
            tick=time.monotonic()
            if tick-last>=10:
                progress=dict(stage='source_manifest',completed_files=index,total_files=len(paths),
                    hashed_bytes=total,elapsed_seconds=tick-started)
                write(PUBLICATION/'PACKAGING_LIVE_PROGRESS.json',progress);print(json.dumps(progress),flush=True);last=tick
    write(PUBLICATION/'RUNTIME_BINDING_VERIFICATION.json',dict(protocols=protocol_info,
        unique_runtime_files=len(bindings),verified_files=len(verified),failure_count=0,files=verified,
        source_and_input_bytes_verified_before_archive=True,reference_numeric_decode=False))
    (PUBLICATION/'ARCHIVE_PATHS.null').write_bytes(b''.join(rel.encode()+b'\0'for rel in paths))
    return entries,total


def compress_archive(entries):
    if EXTERNAL.exists()and any(EXTERNAL.iterdir()):raise FileExistsError('Preserve existing external archive generation')
    EXTERNAL.mkdir(parents=True,exist_ok=True)
    tar_error=EXTERNAL/'tar.stderr.log';zstd_error=EXTERNAL/'zstd.stderr.log'
    parts=[];whole=hashlib.sha256();compressed=0;started=time.monotonic();last=started
    with tar_error.open('wb')as te,zstd_error.open('wb')as ze:
        tar=subprocess.Popen(['tar','-C',str(ROOT),'--format=posix','--pax-option=delete=atime,delete=ctime',
            '--numeric-owner','--no-recursion','--null','--verbatim-files-from','-c','-f','-',
            '-T',str(PUBLICATION/'ARCHIVE_PATHS.null')],stdout=subprocess.PIPE,stderr=te)
        zstd=subprocess.Popen(['zstd','-T2','-3','-c'],stdin=tar.stdout,stdout=subprocess.PIPE,stderr=ze)
        tar.stdout.close();part=None;part_hash=None;part_bytes=0
        try:
            while True:
                block=zstd.stdout.read(CHUNK)
                if not block:break
                whole.update(block);compressed+=len(block);offset=0
                while offset<len(block):
                    if part is None:
                        part_path=EXTERNAL/('pallet-visibility-publication-20261015-v1.tar.zst.part-%03d'%len(parts))
                        part=part_path.open('xb');part_hash=hashlib.sha256();part_bytes=0
                    take=min(PART_SIZE-part_bytes,len(block)-offset);piece=block[offset:offset+take]
                    part.write(piece);part_hash.update(piece);part_bytes+=take;offset+=take
                    if part_bytes==PART_SIZE:
                        part.close();parts.append(dict(name=part_path.name,size=part_bytes,sha256=part_hash.hexdigest()));part=None
                tick=time.monotonic()
                if tick-last>=10:
                    progress=dict(stage='tar_zstd_split',compressed_bytes=compressed,closed_parts=len(parts),
                        elapsed_seconds=tick-started)
                    write(PUBLICATION/'PACKAGING_LIVE_PROGRESS.json',progress);print(json.dumps(progress),flush=True);last=tick
            if part is not None:
                part.close();parts.append(dict(name=part_path.name,size=part_bytes,sha256=part_hash.hexdigest()));part=None
        finally:
            if part is not None:part.close()
        zcode=zstd.wait();tcode=tar.wait()
    if zcode or tcode:raise RuntimeError('Archive subprocess failed: tar=%s zstd=%s'%(tcode,zcode))
    result=dict(format='POSIX tar stream compressed by zstd -T2 -3',part_size_bytes=PART_SIZE,
        total_compressed_bytes=compressed,concatenated_archive_sha256=whole.hexdigest(),parts=parts,
        sorted_paths_null_sha256=sha(PUBLICATION/'ARCHIVE_PATHS.null'),
        source_manifest_sha256=sha(PUBLICATION/'FILE_MANIFEST.jsonl.gz'),archive_member_files=len(entries))
    write(EXTERNAL/'ARCHIVE_FILES.json',result);write(PUBLICATION/'ARCHIVE_FILES.json',result)
    return result


def verify_tar_stream(entries,archive):
    started=time.monotonic();last=started;seen=set();total=0;errors=[];feeder_errors=[]
    log=EXTERNAL/'verify-zstd.stderr.log'
    with log.open('wb')as stderr:
        process=subprocess.Popen(['zstd','-d','-c'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr)
        def feed():
            whole=hashlib.sha256()
            try:
                for item in archive['parts']:
                    path=EXTERNAL/item['name'];h=hashlib.sha256();size=0
                    with path.open('rb')as stream:
                        for block in iter(lambda:stream.read(CHUNK),b''):
                            h.update(block);whole.update(block);size+=len(block);process.stdin.write(block)
                    if size!=item['size']or h.hexdigest()!=item['sha256']:
                        raise ValueError('Compressed archive part hash mismatch: '+item['name'])
                if whole.hexdigest()!=archive['concatenated_archive_sha256']:raise ValueError('Whole compressed SHA mismatch')
            except Exception as error:feeder_errors.append(str(error))
            finally:process.stdin.close()
        feeder=threading.Thread(target=feed,daemon=True);feeder.start()
        try:
            with tarfile.open(fileobj=process.stdout,mode='r|')as stream:
                for member in stream:
                    name=member.name
                    if not member.isfile()or name not in entries or name in seen:
                        raise ValueError('Unexpected/duplicate/nonregular tar member: '+name)
                    expected=entries[name];h=hashlib.sha256();count=0
                    data=stream.extractfile(member)
                    for block in iter(lambda:data.read(CHUNK),b''):h.update(block);count+=len(block)
                    if count!=expected['size']or member.size!=expected['size']or h.hexdigest()!=expected['sha256']:
                        raise ValueError('Extracted member bytes differ from original manifest: '+name)
                    seen.add(name);total+=count;tick=time.monotonic()
                    if tick-last>=10:
                        progress=dict(stage='streamed_tar_member_SHA_verification',verified_files=len(seen),
                            total_files=len(entries),verified_uncompressed_bytes=total,elapsed_seconds=tick-started)
                        write(PUBLICATION/'PACKAGING_LIVE_PROGRESS.json',progress);print(json.dumps(progress),flush=True);last=tick
            # Drain tar padding/trailing output so a valid feeder never blocks.
            for _ in iter(lambda:process.stdout.read(CHUNK),b''):pass
            feeder.join();code=process.wait()
            if code or feeder_errors:raise ValueError('Archive decompression failed: '+str((code,feeder_errors)))
            if seen!=set(entries):raise ValueError('Archive dropped manifest members')
        except Exception:
            process.terminate();process.wait();raise
    receipt=dict(completed_utc=now(),verified_files=len(seen),verified_uncompressed_bytes=total,
        expected_files=len(entries),failure_count=0,all_tar_member_bytes_SHA256_match=True,
        compressed_part_and_concatenated_SHA256_match=True,
        verification='streamed decompression plus SHA256 of every member; no files extracted',
        elapsed_seconds=time.monotonic()-started,archive_files_manifest_sha256=sha(PUBLICATION/'ARCHIVE_FILES.json'))
    write(PUBLICATION/'ARCHIVE_STREAM_VERIFICATION.json',receipt)
    return receipt


def main():
    PUBLICATION.mkdir(parents=True,exist_ok=True)
    if(PUBLICATION/'SUMMARY.json').exists():raise FileExistsError('Preserve completed publication generation')
    started=now();bindings,protocol_info=protocol_bindings();paths,excluded=inventory(bindings)
    print(json.dumps(dict(stage='inventory',files=len(paths),runtime_binding_files=len(bindings),excluded=excluded)),flush=True)
    entries,total=manifest(paths,bindings,protocol_info)
    print(json.dumps(dict(stage='manifest_complete',files=len(entries),uncompressed_bytes=total)),flush=True)
    archive=compress_archive(entries);verification=verify_tar_stream(entries,archive)
    summary=dict(schema='FULL_VISIBILITY_EXPERIMENT_PUBLICATION_SNAPSHOT_V1',started_utc=started,completed_utc=now(),
        snapshot_roots=[relative(p)for p in BASE_ROOTS],excluded=excluded,files=len(entries),uncompressed_bytes=total,
        runtime_binding_files=len(bindings),runtime_binding_failures=0,protocols=protocol_info,
        additional_runtime_files_outside_snapshot_roots=sum(not any((ROOT/rel).is_relative_to(base)for base in BASE_ROOTS)for rel in bindings),
        source_manifest=dict(path='FILE_MANIFEST.jsonl.gz',sha256=sha(PUBLICATION/'FILE_MANIFEST.jsonl.gz')),
        archive_manifest=dict(path='ARCHIVE_FILES.json',sha256=sha(PUBLICATION/'ARCHIVE_FILES.json')),
        archive_parts=len(archive['parts']),compressed_bytes=archive['total_compressed_bytes'],
        archive_sha256=archive['concatenated_archive_sha256'],verification=verification,
        packaging_source=dict(path=relative(Path(__file__).resolve()),sha256=sha(__file__)),
        source_or_data_deletions_or_edits=0,git_commands=0,uploads=0,reference_numeric_decodes=0,
        draft_unvalidated_optimization_sources_are_archived_without_adoption=True)
    write(PUBLICATION/'SUMMARY.json',summary)
    assets=['FILE_MANIFEST.jsonl.gz','RUNTIME_BINDING_VERIFICATION.json','ARCHIVE_PATHS.null',
        'ARCHIVE_FILES.json','ARCHIVE_STREAM_VERIFICATION.json','SUMMARY.json']
    write(PUBLICATION/'PUBLICATION_ASSET_SEAL.json',dict(completed_utc=now(),
        assets={name:sha(PUBLICATION/name)for name in assets},archive_sha256=archive['concatenated_archive_sha256']))
    write(PUBLICATION/'PACKAGING_LIVE_PROGRESS.json',dict(stage='complete',files=len(entries),
        archive_parts=len(archive['parts']),compressed_bytes=archive['total_compressed_bytes'],failure_count=0))
    print(json.dumps(dict(stage='complete',files=len(entries),uncompressed_bytes=total,
        parts=len(archive['parts']),compressed_bytes=archive['total_compressed_bytes'],verification_failures=0)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args();main()

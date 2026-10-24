#!/usr/bin/env python3
"""Verify or restore original artifact bytes from a publication manifest.

Standard library only; POSIX O_NOFOLLOW/dir_fd support is required. No artifact
is parsed (code/JSON/image/array payloads remain bytes). Only manifest JSON is
parsed; source_absolute fields are never followed.

  python3 tools/restore_original_artifacts.py --self-test
  python3 tools/restore_original_artifacts.py --root PUBLICATION_ROOT --manifest MANIFEST.json --verify-all
  python3 tools/restore_original_artifacts.py --root PUBLICATION_ROOT --manifest MANIFEST.json --logical archive/path.jsonl --output NEW_PATH

Without --output the action is verification only. Output parents must already
exist; symlinks and overwriting existing output names are rejected.
"""
from pathlib import Path, PurePosixPath
import argparse
import gzip
import hashlib
import json
import os
import stat
import sys
import tempfile
import uuid


def require(value, message):
    if not value: raise ValueError(message)


def components(value, absolute=False):
    require(isinstance(value,str) and bool(value) and 'v112' not in value.lower(), 'Forbidden/empty path name precheck')
    require('\\' not in value, 'Backslash path prohibited')
    path=PurePosixPath(value)
    require(path.is_absolute()==absolute and '..' not in path.parts, 'Path must have the declared safe root')
    return [part for part in path.parts if part not in ('/','.','')]


def directory_fd(path):
    """Open every absolute ancestor without following a symbolic link."""
    absolute=str(Path(path).absolute())  # no filesystem resolution
    parts=components(absolute,True)
    require(os.name=='posix' and hasattr(os,'O_NOFOLLOW'), 'POSIX O_NOFOLLOW support required')
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in parts:
            new=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd);fd=new
        return fd
    except BaseException:
        os.close(fd);raise


def open_relative(root_fd, name):
    parts=components(name);require(bool(parts),'File path required')
    parent=os.dup(root_fd)
    try:
        for part in parts[:-1]:
            new=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
            os.close(parent);parent=new
        fd=os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW,dir_fd=parent)
        try:require(stat.S_ISREG(os.fstat(fd).st_mode),'Stored artifact must be a regular file')
        except BaseException:os.close(fd);raise
        return os.fdopen(fd,'rb')
    finally:os.close(parent)


def sha(data): return hashlib.sha256(data).hexdigest()


def parse_manifest(data, path):
    if path.endswith('.jsonl'):
        return dict(entries=[json.loads(line) for line in data.splitlines() if line.strip()])
    return json.loads(data)


def manifest_entries(manifest):
    entries=manifest.get('entries',manifest.get('files'))
    require(isinstance(entries,list) and all(isinstance(e,dict) for e in entries),'Final manifest entries list required')
    result={}
    for entry in entries:
        status=entry.get('status','')
        require(not status.startswith('CANDIDATE_'),'Candidate inventory is not a final publication')
        if status not in ('PUBLISHED_RAW','PUBLISHED_GZIP_CHUNKS'):
            require(entry.get('published_relative') is None,'Unrecognized published record status')
            continue
        logical=entry.get('published_relative');components(logical)
        require(logical not in result,'Duplicate logical publication path')
        require(isinstance(entry.get('source_sha256'),str) and len(entry['source_sha256'])==64,'Original source SHA256 required')
        require(type(entry.get('size_bytes')) is int and entry['size_bytes']>=0,'Original source byte size required')
        result[logical]=entry
    return result


class HashingReader:
    def __init__(self, handle):self.handle=handle;self.digest=hashlib.sha256();self.size=0
    def read(self,n=-1):
        value=self.handle.read(n);self.digest.update(value);self.size+=len(value);return value
    def seekable(self):return False
    def readable(self):return True


def stream_original(root_fd, entry, sink=None):
    logical=entry['published_relative'];components(logical)
    original_hash=hashlib.sha256();original_size=0;part_results=[]
    def accept(data):
        nonlocal original_size
        original_hash.update(data);original_size+=len(data)
        if sink is not None:sink.write(data)
    if entry['status']=='PUBLISHED_RAW':
        require(entry.get('gzipchunks') is None,'Raw record cannot also contain chunks')
        require(entry.get('published_sha256')==entry['source_sha256'],'Raw published SHA must equal original SHA')
        with open_relative(root_fd,logical) as handle:
            for block in iter(lambda:handle.read(1<<20),b''):accept(block)
        part_results.append(dict(path=logical,stored_sha256=original_hash.hexdigest(),stored_size_bytes=original_size))
    else:
        descriptor=entry.get('gzipchunks')
        require(isinstance(descriptor,dict) and descriptor.get('format')=='concatenated-bytes-v1','Explicit gzip chunk descriptor required')
        require(entry.get('published_sha256') is None,'Chunked logical file has no physical original published SHA')
        require(descriptor.get('content_sha256')==entry['source_sha256'] and descriptor.get('content_size_bytes')==entry['size_bytes'],'Descriptor/original SHA and size agreement')
        descriptor_review=None
        if entry.get('gzip_descriptor_path') is not None:
            name=entry['gzip_descriptor_path'];components(name)
            descriptor_hash=hashlib.sha256();descriptor_size=0
            with open_relative(root_fd,name) as handle:
                for block in iter(lambda:handle.read(1<<20),b''):
                    descriptor_hash.update(block);descriptor_size+=len(block)
            require(descriptor_hash.hexdigest()==entry.get('gzip_descriptor_sha256')
                    and descriptor_size==entry.get('gzip_descriptor_size_bytes'),'Physical descriptor stored SHA/size mismatch')
            descriptor_review=dict(path=name,sha256=descriptor_hash.hexdigest(),size_bytes=descriptor_size)
        parts=descriptor.get('parts');require(isinstance(parts,list) and bool(parts),'Nonempty ordered gzip parts required')
        for part in parts:
            require(isinstance(part,dict) and part.get('compression')=='gzip','Each chunk is gzip')
            name=part.get('path');components(name)
            require(isinstance(part.get('sha256'),str) and len(part['sha256'])==64,'Stored chunk SHA required')
            require(type(part.get('stored_size_bytes')) is int and type(part.get('decoded_size_bytes')) is int,'Stored/decoded chunk byte sizes required')
            with open_relative(root_fd,name) as handle:
                raw=HashingReader(handle);decoded_size=0
                with gzip.GzipFile(fileobj=raw,mode='rb') as decoder:
                    for block in iter(lambda:decoder.read(1<<20),b''):
                        decoded_size+=len(block);accept(block)
                # Include any remaining stored bytes in the compressed-byte hash.
                while raw.read(1<<20):pass
                require(raw.digest.hexdigest()==part['sha256'],'Stored chunk SHA256 mismatch')
                require(raw.size==part['stored_size_bytes'] and decoded_size==part['decoded_size_bytes'],'Stored/decoded chunk byte size mismatch')
                part_results.append(dict(path=name,stored_sha256=raw.digest.hexdigest(),stored_size_bytes=raw.size,decoded_size_bytes=decoded_size))
    require(original_hash.hexdigest()==entry['source_sha256'],'Restored original SHA256 mismatch')
    require(original_size==entry['size_bytes'],'Restored original byte size mismatch')
    return dict(logical_path=logical,status=entry['status'],PASS=True,original_sha256=original_hash.hexdigest(),original_size_bytes=original_size,
                stored_parts=part_results,physical_descriptor=(descriptor_review if entry['status']=='PUBLISHED_GZIP_CHUNKS' else None),
                artifact_payloads_parsed=0,original_source_paths_accessed=0)


def restore_one(root_fd, entry, output):
    absolute=str(Path(output).absolute());parts=components(absolute,True)
    require(bool(parts),'Output filename required')
    parent=directory_fd(str(PurePosixPath(absolute).parent));name=parts[-1]
    temporary='.restore-'+uuid.uuid4().hex+'.partial'
    fd=None
    try:
        # This stat is no-follow; a pre-existing file, directory or symlink blocks restoration.
        try:os.stat(name,dir_fd=parent,follow_symlinks=False)
        except FileNotFoundError:pass
        else:raise FileExistsError('Requested output already exists; refusing overwrite')
        fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=parent)
        with os.fdopen(fd,'wb') as sink:
            fd=None;review=stream_original(root_fd,entry,sink);sink.flush();os.fsync(sink.fileno())
        # link() creates only a NEW final name; concurrent existing names cannot be overwritten.
        os.link(temporary,name,src_dir_fd=parent,dst_dir_fd=parent,follow_symlinks=False)
        os.unlink(temporary,dir_fd=parent)
        review.update(restored_output=absolute,output_no_overwrite=True)
        return review
    except BaseException:
        if fd is not None:os.close(fd)
        # A failed write/verification deliberately retains any exclusive partial evidence.
        raise
    finally:os.close(parent)


def verify(root, manifest_path, logical=None, output=None):
    components(manifest_path)
    root_fd=directory_fd(root)
    try:
        with open_relative(root_fd,manifest_path) as handle:manifest_bytes=handle.read()
        entries=manifest_entries(parse_manifest(manifest_bytes,manifest_path))
        if logical is not None:
            components(logical);require(logical in entries,'Requested logical path is not published')
            chosen=[entries[logical]]
        else:
            require(output is None,'Output requires one explicit logical path');chosen=list(entries.values())
        results=[]
        for entry in chosen:
            results.append(restore_one(root_fd,entry,output) if output is not None else stream_original(root_fd,entry))
        return dict(completed=True,PASS=True,manifest_stored_sha256=sha(manifest_bytes),published_originals_verified=len(results),
            restored_files=int(output is not None),results=results,artifact_payloads_parsed=0,original_source_paths_accessed=0,
            symlinks_followed=0,verification_only=output is None)
    finally:os.close(root_fd)


def write_new_report(path, payload):
    absolute=str(Path(path).absolute());parts=components(absolute,True)
    require(bool(parts),'New report filename required')
    parent=directory_fd(str(PurePosixPath(absolute).parent))
    try:
        fd=os.open(parts[-1],os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=parent)
        with os.fdopen(fd,'wb') as handle:handle.write(payload)
    finally:os.close(parent)


def self_tests():
    tests=[]
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);raw=b'opaque\x00\xff\n'+b'{this is not parsed JSON}\n'*101
        (root/'raw.bin').write_bytes(raw)
        def raw_entry():return dict(published_relative='raw.bin',status='PUBLISHED_RAW',source_absolute='/never/access/original',
                    source_sha256=sha(raw),published_sha256=sha(raw),size_bytes=len(raw),gzipchunks=None)
        parts=[]
        for index,data in enumerate((raw[:83],raw[83:])):
            compressed=gzip.compress(data,mtime=0);name='chunk'+str(index)+'.gz';(root/name).write_bytes(compressed)
            parts.append(dict(path=name,sha256=sha(compressed),compression='gzip',stored_size_bytes=len(compressed),decoded_size_bytes=len(data)))
        chunk_entry=dict(published_relative='logical.bin',status='PUBLISHED_GZIP_CHUNKS',source_absolute='/never/access/original',
            source_sha256=sha(raw),published_sha256=None,size_bytes=len(raw),gzipchunks=dict(format='concatenated-bytes-v1',parts=parts,content_sha256=sha(raw),content_size_bytes=len(raw)))
        descriptor_bytes=json.dumps(chunk_entry['gzipchunks']).encode();(root/'logical.bin.parts.json').write_bytes(descriptor_bytes)
        chunk_entry.update(gzip_descriptor_path='logical.bin.parts.json',gzip_descriptor_sha256=sha(descriptor_bytes),gzip_descriptor_size_bytes=len(descriptor_bytes))
        manifest=dict(entries=[raw_entry(),chunk_entry]);(root/'manifest.json').write_text(json.dumps(manifest))
        result=verify(root,'manifest.json');require(result['published_originals_verified']==2 and result['artifact_payloads_parsed']==0,'RAW_MULTIPART_BYTE_VERIFICATION')
        tests.append('Raw_and_ordered_multipart_gzip_original_SHA_size_bytes_only')
        (root/'manifest.jsonl').write_bytes(b'\n'.join(json.dumps(e).encode() for e in manifest['entries'])+b'\n')
        require(verify(root,'manifest.jsonl')['published_originals_verified']==2,'JSONL_MANIFEST')
        tests.append('Canonical_JSONL_manifest_and_physical_descriptor_SHA_size')
        one=dict(chunk_entry);one['gzipchunks']=dict(chunk_entry['gzipchunks']);allgzip=gzip.compress(raw,mtime=0);(root/'single.gz').write_bytes(allgzip)
        one['gzipchunks']['parts']=[dict(path='single.gz',sha256=sha(allgzip),compression='gzip',stored_size_bytes=len(allgzip),decoded_size_bytes=len(raw))]
        (root/'single.json').write_text(json.dumps(dict(entries=[one])));require(verify(root,'single.json')['PASS'],'SINGLE_GZIP')
        tests.append('Single_gzip_original_roundtrip')
        output=root/'restored.bin';require(verify(root,'manifest.json','logical.bin',output)['restored_files']==1 and output.read_bytes()==raw,'RESTORE_NEW_BYTES')
        tests.append('Selected_one_logical_file_exclusive_byteexact_restore')
        try:verify(root,'manifest.json','logical.bin',output)
        except FileExistsError:pass
        else:raise AssertionError('Output overwrite accepted')
        require(output.read_bytes()==raw,'EXISTING_OUTPUT_UNCHANGED');tests.append('Existing_output_never_overwritten')
        bad=json.loads(json.dumps(manifest));bad['entries'][1]['gzipchunks']['parts'][0]['sha256']='0'*64
        (root/'bad.json').write_text(json.dumps(bad))
        try:verify(root,'bad.json')
        except ValueError:tests.append('Corrupt_stored_chunk_SHA_rejected')
        else:raise AssertionError('Corruption accepted')
        bad=json.loads(json.dumps(chunk_entry));bad['source_sha256']='0'*64;bad['gzipchunks']['content_sha256']='0'*64
        (root/'badcontent.json').write_text(json.dumps(dict(entries=[bad])))
        try:verify(root,'badcontent.json')
        except ValueError:tests.append('Decoded_original_SHA_independently_recomputed')
        else:raise AssertionError('Wrong original content SHA accepted')
        bad=raw_entry();bad['size_bytes']+=1;(root/'badsize.json').write_text(json.dumps(dict(entries=[bad])))
        try:verify(root,'badsize.json')
        except ValueError:tests.append('Original_size_mismatch_rejected')
        else:raise AssertionError('Wrong size accepted')
        bad=raw_entry();bad['published_sha256']='0'*64;(root/'badraw.json').write_text(json.dumps(dict(entries=[bad])))
        try:verify(root,'badraw.json')
        except ValueError:tests.append('Raw_published_original_SHA_disagreement_rejected')
        else:raise AssertionError('Wrong raw SHA accepted')
        (root/'symlink.bin').symlink_to(root/'raw.bin');bad=raw_entry();bad['published_relative']='symlink.bin'
        (root/'link.json').write_text(json.dumps(dict(entries=[bad])))
        try:verify(root,'link.json')
        except OSError:tests.append('Stored_file_symlink_not_followed')
        else:raise AssertionError('Symlink followed')
        (root/'real-dir').mkdir();(root/'real-dir/raw.bin').write_bytes(raw);(root/'dirlink').symlink_to(root/'real-dir',target_is_directory=True)
        bad=raw_entry();bad['published_relative']='dirlink/raw.bin';(root/'dirlink.json').write_text(json.dumps(dict(entries=[bad])))
        try:verify(root,'dirlink.json')
        except OSError:tests.append('Ancestor_directory_symlink_not_followed')
        else:raise AssertionError('Ancestor symlink followed')
        bad=raw_entry();bad['published_relative']='V112/does-not-exist';(root/'forbidden.json').write_text(json.dumps(dict(entries=[bad])))
        try:verify(root,'forbidden.json')
        except ValueError:tests.append('Forbidden_name_precheck_before_stored_file_access')
        else:raise AssertionError('Forbidden path accepted')
        require(all(r['original_source_paths_accessed']==0 for r in result['results']),'SOURCE_ABSOLUTE_IGNORED')
        tests.append('Original_source_absolute_is_metadata_never_followed')
    return dict(completed=True,PASS=True,invented_only=True,checks=tests,count=len(tests),actual_staged_package_reads=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--self-test',action='store_true');parser.add_argument('--root',type=Path);parser.add_argument('--manifest')
    group=parser.add_mutually_exclusive_group();group.add_argument('--verify-all',action='store_true');group.add_argument('--logical')
    parser.add_argument('--output',type=Path);parser.add_argument('--report',type=Path);args=parser.parse_args()
    if args.self_test:
        require(args.root is None and args.manifest is None and args.output is None,'Self-test cannot read staged artifacts');result=self_tests()
    else:
        require(args.root is not None and args.manifest is not None,'Staged root and manifest required')
        require(args.output is None or args.logical is not None,'--output requires exactly one --logical file')
        result=verify(args.root,args.manifest,args.logical,args.output)
    text=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if args.report:write_new_report(args.report,text.encode('utf-8'))
    print(text,end='')


if __name__=='__main__':
    try:main()
    except (ValueError,OSError,KeyError,TypeError,EOFError,gzip.BadGzipFile) as error:
        print(json.dumps(dict(completed=False,PASS=False,error_type=type(error).__name__,error=str(error)),ensure_ascii=False),file=sys.stderr)
        sys.exit(1)

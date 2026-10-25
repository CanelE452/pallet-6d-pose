#!/usr/bin/env python3
"""Verify an exact portable-package manifest; stdlib, read-only, no tree walk."""
import argparse
import hashlib
import json
import os
from pathlib import PurePosixPath
import stat
import sys


class VerificationError(ValueError):
    pass


def lexical_guard(value):
    if not isinstance(value,str) or not value or '\x00' in value:
        raise VerificationError('Nonempty path string required')
    folded=value.casefold()
    if 'v112' in folded:
        raise VerificationError('Forbidden v112 path before filesystem access')
    if any(part=='__pycache__' or part.endswith('.pyc') for part in folded.replace('\\','/').split('/')):
        raise VerificationError('Compiled Python cache path rejected before filesystem access')


def relative_parts(value):
    lexical_guard(value)
    path=PurePosixPath(value)
    parts=value.split('/')
    if path.is_absolute() or '\\' in value or any(part in ('','.','..') for part in parts):
        raise VerificationError('Canonical package-relative POSIX path required: '+value)
    return tuple(parts)


def directory_fd(package):
    lexical_guard(package)
    absolute=os.path.abspath(package)
    lexical_guard(absolute)
    flags=os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW
    fd=os.open('/',flags)
    try:
        for part in absolute.split('/')[1:]:
            if not part:continue
            following=os.open(part,flags,dir_fd=fd)
            os.close(fd);fd=following
        return fd
    except BaseException:
        os.close(fd)
        raise


def file_fd(root_fd,relative):
    parts=relative_parts(relative)  # lexical/schema rejection BEFORE any open/stat
    fd=os.dup(root_fd)
    try:
        for part in parts[:-1]:
            following=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd);fd=following
        leaf=os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
        if not stat.S_ISREG(os.fstat(leaf).st_mode):
            os.close(leaf)
            raise VerificationError('Regular file required: '+relative)
        return leaf
    finally:
        os.close(fd)


def verify_package(package,manifest):
    relative_parts(manifest)
    root=directory_fd(package)
    try:
        with os.fdopen(file_fd(root,manifest),'rb') as source:
            manifest_bytes=source.read()
        document=json.loads(manifest_bytes)
        if not isinstance(document,dict) or not isinstance(document.get('files'),list):
            raise VerificationError('Manifest must contain files[]')
        planned=[];seen=set()
        # Validate every declared path before accessing any declared file body.
        for item in document['files']:
            if not isinstance(item,dict):raise VerificationError('Manifest file record must be an object')
            path=item.get('path');relative_parts(path)
            if path==manifest:raise VerificationError('Manifest must exclude its own file')
            if path in seen:raise VerificationError('Duplicate manifest path: '+path)
            seen.add(path)
            expected_sha=item.get('sha256');expected_size=item.get('bytes')
            if not isinstance(expected_sha,str) or len(expected_sha)!=64 or any(c not in '0123456789abcdefABCDEF' for c in expected_sha):
                raise VerificationError('SHA256 hex required: '+path)
            if type(expected_size) is not int or expected_size<0:
                raise VerificationError('Nonnegative integer bytes required: '+path)
            planned.append((path,expected_sha.casefold(),expected_size))
        total=0
        for path,expected_sha,expected_size in planned:
            digest=hashlib.sha256();actual_size=0
            with os.fdopen(file_fd(root,path),'rb') as source:
                for block in iter(lambda:source.read(1024*1024),b''):
                    digest.update(block);actual_size+=len(block)
            if actual_size!=expected_size:raise VerificationError('Byte-size mismatch: '+path)
            if digest.hexdigest()!=expected_sha:raise VerificationError('SHA256 mismatch: '+path)
            total+=actual_size
        return {'completed':True,'PASS':True,'files_verified':len(planned),'bytes_verified':total,
                'manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),
                'scope':'Manifest allowlist only; unlisted paths are not enumerated or certified.',
                'manifest_entry_files_only':True,'pointer_targets_followed':False}
    finally:
        os.close(root)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',required=True,help='Existing package directory')
    parser.add_argument('--manifest',required=True,help='Package-relative POSIX manifest path')
    args=parser.parse_args(argv)
    try:
        print(json.dumps(verify_package(args.package,args.manifest),ensure_ascii=False))
        return 0
    except (VerificationError,OSError,ValueError,TypeError) as error:
        print(json.dumps({'completed':False,'PASS':False,'error':str(error)},ensure_ascii=False),file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())

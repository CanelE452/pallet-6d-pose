"""Static ZIP/pickle-opcode metadata only; no unpickle, tensor decode or model import."""
from pathlib import Path
import hashlib
import json
import pickletools
import re
import zipfile


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inspect(path, coco_names):
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        member = next(name for name in archive.namelist() if name.endswith('/data.pkl'))
        data = archive.read(member)
    operations = list(pickletools.genops(data))
    memo = {}
    scalar_ops = {'BINUNICODE', 'SHORT_BINUNICODE', 'UNICODE', 'BININT', 'BININT1', 'BININT2', 'INT'}
    memo_ops = {'BINPUT', 'LONG_BINPUT', 'PUT', 'MEMOIZE'}
    for i, (op, argument, _) in enumerate(operations[:-1]):
        following, index, _ = operations[i + 1]
        if op.name in scalar_ops and following.name in memo_ops - {'MEMOIZE'}:
            memo[index] = argument

    def after(key):
        for i, (_, argument, _) in enumerate(operations):
            if argument == key:
                for op, value, _ in operations[i + 1:i + 8]:
                    if op.name not in memo_ops:
                        return value
        return None

    names = {}
    for i, (_, argument, _) in enumerate(operations):
        if argument != 'names':
            continue
        index = None
        for op, value, _ in operations[i + 1:]:
            if op.name in {'SETITEM', 'SETITEMS'}:
                break
            if op.name in {'BININT', 'BININT1', 'BININT2', 'INT'}:
                index = value
            elif op.name in {'BINUNICODE', 'SHORT_BINUNICODE', 'UNICODE'} and index is not None:
                names[index] = value
                index = None
            elif op.name in {'BINGET', 'LONG_BINGET', 'GET'} and index is not None and isinstance(memo.get(value), str):
                names[index] = memo[value]
                index = None
        break
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path),
        metadata_member=member, metadata_bytes=len(data),
        metadata_sha256=hashlib.sha256(data).hexdigest(),
        nc=after('nc'), task=after('task'), checkpoint_declared_version=after('version'),
        epoch=after('epoch'), names=names, names_count=len(names),
        names_equal_local_COCO80=(names == coco_names),
        model_and_head_globals=[arg for op, arg, _ in operations if op.name == 'GLOBAL'
            and ('Model' in arg or '.head ' in arg)],
        static_opcode_interpretation_only=True, torch_or_pickle_loads=0,
        model_imports=0, tensor_storage_members_extracted=0,
        entire_checkpoint_bytes_streamed_only_for_opaque_sha256=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coco-yaml', type=Path, required=True)
    parser.add_argument('checkpoints', nargs='+', type=Path)
    args = parser.parse_args()
    names = {int(key): value for key, value in re.findall(
        r'^  (\d+): ([^\n]+)$', args.coco_yaml.read_text(), re.M)}
    if len(names) != 80:
        raise ValueError('Pinned COCO80 class-name metadata required')
    print(json.dumps(dict(static_checkpoint_metadata=[inspect(p, names) for p in args.checkpoints],
        coco_yaml=str(args.coco_yaml), coco_yaml_sha256=sha(args.coco_yaml)),
        ensure_ascii=False, indent=2))

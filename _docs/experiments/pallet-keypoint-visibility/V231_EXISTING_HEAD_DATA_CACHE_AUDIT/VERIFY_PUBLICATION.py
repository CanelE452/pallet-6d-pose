"""Verify this documentation package using only package-relative declared files."""
import hashlib
import json
from pathlib import Path, PurePosixPath


def main():
    root = Path(__file__).absolute().parent
    if root.is_symlink():
        raise ValueError('Package root must not be a symlink')

    def read_declared(rel):
        path = PurePosixPath(rel)
        if (path.is_absolute() or '..' in path.parts or not path.parts
                or 'v112' in rel.lower() or '__pycache__' in path.parts):
            raise ValueError('Invalid declared path: ' + rel)
        current = root
        for part in path.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError('Symlink rejected: ' + rel)
        return current.read_bytes()

    manifest = json.loads(read_declared('PUBLICATION_MANIFEST.json'))
    checked = set()
    total = 0
    for item in manifest['entries']:
        rel = item['relative_path']
        if rel in checked:
            raise ValueError('Duplicate declared path: ' + rel)
        checked.add(rel)
        data = read_declared(rel)
        if len(data) != item['size_bytes']:
            raise ValueError('Size mismatch: ' + rel)
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('SHA256 mismatch: ' + rel)
        total += len(data)
    print(json.dumps({'PASS': True, 'files_verified': len(checked),
                      'bytes_verified': total,
                      'scope': 'Published documentation bytes only; no model/data replay'}))


if __name__ == '__main__':
    main()

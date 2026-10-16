"""Package unchanged large raw evidence into auditable 40 MiB byte parts once."""
import hashlib
import json
from pathlib import Path
from .restore_archives import DOC, digest


def main():
    manifest = DOC / 'ARCHIVE_MANIFEST.json'
    ignore = DOC / '.gitignore'
    assert not manifest.exists() and not ignore.exists(), 'preserve completed archive'
    folder = DOC / 'archives'
    folder.mkdir(exist_ok=False)
    files = []
    size = 40 * 1024 * 1024
    for path in sorted(DOC.glob('*.gz')):
        if path.stat().st_size < 50 * 1024 * 1024:
            continue
        parts = []
        total_hash = hashlib.sha256()
        with path.open('rb') as stream:
            index = 0
            while True:
                block = stream.read(size)
                if not block:
                    break
                chunk = folder / (path.name + '.part%03d' % index)
                with chunk.open('xb') as out:
                    out.write(block)
                parts.append(dict(path=str(chunk.relative_to(DOC)), bytes=len(block), sha256=hashlib.sha256(block).hexdigest()))
                total_hash.update(block)
                index += 1
        assert digest(path) == total_hash.hexdigest()
        files.append(dict(original=dict(path=path.name, bytes=path.stat().st_size, sha256=total_hash.hexdigest()), parts=parts))
    assert files, 'no large evidence files'
    data = dict(encoding='ordered byte slices of original gzip; concatenate without recompression',
                purpose='complete public raw evidence with individual Git blobs below the service file limit',
                original_files_preserved=True, original_seal_and_receipts_unchanged=True,
                part_size_bytes=size, files=files,
                restore_command='python3 -B -m scripts.research.pallet_partial_line_heldout_20261010_v6.restore_archives',
                model_PnP_training_RGB_calls=0)
    with manifest.open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')
    with ignore.open('x') as stream:
        stream.write('# Exact original large evidence files; byte-identical public parts are tracked.\n')
        for item in files:
            stream.write('/' + item['original']['path'] + '\n')
    print(json.dumps(dict(files=len(files), parts=sum(len(item['parts']) for item in files), original_bytes=sum(item['original']['bytes'] for item in files))))


if __name__ == '__main__':
    main()

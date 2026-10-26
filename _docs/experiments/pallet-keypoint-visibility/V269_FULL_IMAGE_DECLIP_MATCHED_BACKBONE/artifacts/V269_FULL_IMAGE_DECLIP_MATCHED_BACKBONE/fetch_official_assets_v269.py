"""Download two authorized generic checkpoints and pinned official source archives.

No Hugging Face default storage, imports of downloaded code, or model forwards.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import tarfile
import time
import urllib.request

HERE = Path(__file__).absolute().parent


def safe(p):
    p = Path(p)
    assert p.is_absolute() and not any(t in str(p).lower() for t in ('v112', 'cache', '.pyc', '..')), str(p)
    for a in (p, *p.parents):
        assert not a.is_symlink(), str(a)
    return p


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Codex-V269-official-provenance'}), timeout=90)


def write(path, obj):
    with safe(path).open('x') as f:
        json.dump(obj, f, indent=2, sort_keys=True); f.write('\n')


def download_weight(repo, filename, record, output):
    with get('https://huggingface.co/api/models/' + repo + '/revision/main') as f:
        revision = json.load(f)['sha']
    url = 'https://huggingface.co/' + repo + '/resolve/' + revision + '/' + filename + '?download=true'
    expected, size = record['lfs']['oid'], record['lfs']['size']
    out = safe(HERE / 'pretrained' / output); assert not out.exists()
    partial = safe(Path(str(out) + '.part')); assert not partial.exists()
    h = hashlib.sha256(); n = 0; next_report = 128 * 1024 * 1024; started = time.monotonic()
    with get(url) as response, partial.open('xb') as f:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk: break
            f.write(chunk); h.update(chunk); n += len(chunk)
            if n >= next_report:
                print(json.dumps(dict(file=output, downloaded_bytes=n, total_bytes=size)), flush=True)
                next_report += 128 * 1024 * 1024
    assert n == size and h.hexdigest() == expected, (output, n, h.hexdigest())
    partial.rename(out)
    result = dict(repository=repo, revision=revision, file=filename, download_url=url,
                  path=str(out), sha256=expected, size_bytes=size, elapsed_seconds=time.monotonic()-started,
                  official_LFS_SHA256_and_size_verified=True, no_default_HF_storage=True)
    write(HERE / 'pretrained' / (output + '.DOWNLOAD.json'), result)
    print(json.dumps(dict(download_complete=output, sha256=expected, size_bytes=n)), flush=True)
    return result


def source_archive(branch, sha):
    url = 'https://codeload.github.com/xiaomoguhz/DeCLIP/tar.gz/' + sha
    with get(url) as response:
        raw = response.read()
    out = safe(HERE / 'official_source' / branch); assert not out.exists(); out.mkdir()
    files = []; skipped = []
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
        for member in archive.getmembers():
            parts = Path(member.name).parts[1:]
            if not parts: continue
            relative = Path(*parts)
            if any(t in str(relative).lower() for t in ('v112', 'cache', '.pyc', '..')):
                skipped.append(str(relative)); continue
            assert not member.issym() and not member.islnk() and not relative.is_absolute(), member.name
            target = safe(out / relative)
            if member.isdir(): target.mkdir(parents=True, exist_ok=True); continue
            assert member.isfile(), member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = archive.extractfile(member).read()
            with target.open('xb') as f: f.write(payload)
            files.append(dict(relative_path=relative.as_posix(), sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload)))
    result = dict(branch=branch, commit=sha, official_repository='https://github.com/xiaomoguhz/DeCLIP',
                  source_archive_url=url, archive_sha256=hashlib.sha256(raw).hexdigest(), archive_size_bytes=len(raw),
                  path=str(out), files=files, forbidden_names_skipped_without_FS_read=skipped, nested_git_directory=False)
    write(HERE / 'official_source' / (branch + '_PROVENANCE.json'), result)
    print(json.dumps(dict(source_ready=branch, commit=sha, files=len(files))), flush=True)
    return result


def main():
    meta = json.loads(safe(HERE / 'OFFICIAL_REMOTE_METADATA.json').read_bytes())
    safe(HERE / 'pretrained').mkdir(exist_ok=True)
    safe(HERE / 'official_source').mkdir(exist_ok=True)
    sources = [source_archive(branch, meta[branch]['sha']) for branch in ('main', 'ZSSS')]
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(download_weight, 'QuanSun/EVA-CLIP', 'EVA02_CLIP_B_psz16_s8B.pt', meta['eva'][0], 'EVA02_CLIP_B_psz16_s8B.pt')
        b = pool.submit(download_weight, 'xiaomoguhzz/DeCLIP_evab_dinov2B_csa_560_0.25_seg',
                        'checkpoints/evab_dinov2B_csa_560_0.25_seg.pt', meta['declip'][0], 'evab_dinov2B_csa_560_0.25_seg.pt')
        weights = [a.result(), b.result()]
    write(HERE / 'OFFICIAL_ASSETS_COMPLETE.json', dict(completed=True, PASS=True, source_branches=sources,
          checkpoints=weights, original_dataset_or_current_GT_reads=0, model_forwards=0,
          provenance_caveat='Released DeCLIP has public COCO dense-adaptation history; only subsequent pallet maskhead fits are SYN-only.'))


if __name__ == '__main__':
    main()

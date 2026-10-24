"""Fetch official COCO metadata only after ZIP directory confirms no annotations."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,time,urllib.request
OUT=Path(__file__).resolve().parent
def main():
    directory=OUT/'annotations';directory.mkdir(exist_ok=False)
    pre=json.loads((OUT/'REMOTE_CENTRAL_DIRECTORY_PREFLIGHT.json').read_text());assert not pre['annotation_like_entries']
    tree=json.loads((OUT/'OFFICIAL_REPO_TREE.json').read_text())
    files=[r for r in tree['tree'] if r['path'].startswith('rgb/') and r['path'].endswith('.json')]
    assert len(files)==6
    planned=769055500+sum(r['size'] for r in files)+pre['range_download_bytes']+100000
    assert planned<1200000000
    lock=OUT/'ANNOTATION_FETCH_LOCK.json';assert not lock.exists()
    lock.write_text(json.dumps(dict(schema='V73_OFFICIAL_COCO_METADATA_FETCH_LOCK',files=files,annotations_absent_from_archive_directory=True,
      planned_total_archive_annotation_metadata_range_bytes_upper_bound=planned,budget_bytes=1200000000,no_RGB_payloads_decoded=True),indent=2)+'\n')
    start=time.monotonic()
    def download(r):
        url='https://raw.githubusercontent.com/tum-fml/loco/main/'+r['path'];dest=directory/Path(r['path']).name
        sha=hashlib.sha256();git=hashlib.sha1(f"blob {r['size']}\0".encode());count=0
        with urllib.request.urlopen(url,timeout=60) as response,dest.with_suffix('.json.part').open('wb') as f:
            status=response.status
            for block in iter(lambda:response.read(1048576),b''):
                f.write(block);sha.update(block);git.update(block);count+=len(block)
        assert count==r['size'] and git.hexdigest()==r['sha'],r['path']
        dest.with_suffix('.json.part').rename(dest)
        a=dict(path=str(dest),official_url=url,status=status,bytes=count,sha256=sha.hexdigest(),official_tree_git_blob_sha1=r['sha'],git_blob_identity_match=True)
        print(json.dumps(a),flush=True);return a
    with ThreadPoolExecutor(2) as pool:receipts=list(pool.map(download,files))
    p=OUT/'ANNOTATION_DOWNLOAD_RECEIPT.json';assert not p.exists()
    p.write_text(json.dumps(dict(completed=True,files=receipts,download_bytes=sum(r['bytes'] for r in receipts),seconds=time.monotonic()-start,archive_annotations_redundant_copy_downloaded=False),indent=2)+'\n')
if __name__=='__main__':main()

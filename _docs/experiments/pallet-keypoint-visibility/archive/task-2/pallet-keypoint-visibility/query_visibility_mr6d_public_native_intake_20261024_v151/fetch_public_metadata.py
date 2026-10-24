"""Fetch only pinned public dataset annotation metadata, never image payloads."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,json,urllib.request,datetime,traceback
V=Path(__file__).resolve().parent;M=V/'metadata';OUT=V/'provided_metadata'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if __name__=='__main__':
    assert sha(V/'ROOT_PUBLIC_INTAKE_POLICY_BEFORE_REMOTE_SAMPLE_BODIES.json')=='3b23bd1ef0c48819b8c84f07a8f0a53c372620457ca972878e47cfad21ebdfba'
    info=json.loads((M/'HF_DATASET_INFO.json').read_text());commit=info['sha'];assert commit=='3c4ed8453312655499c3d3a3cf38b4929fae77ee' and info.get('private')is False and info['cardData']['license']=='cc-by-4.0'
    files={x['rfilename']for x in info['siblings']}
    paths=sorted(f for f in files if f in ['README.md','models/models_info.json','models_names.json']or f.endswith(('/scene_gt.json','/scene_gt_info.json'))or f.endswith('/scene_camera.json'))
    # Camera metadata of unlabeled official test subsets is retained as metadata;
    # it does not become object GT or authorize test mask-based truth.
    assert len([f for f in paths if f.endswith('/scene_gt.json')])==16
    OUT.mkdir(exist_ok=False)
    def fetch(path):
        url=f'https://huggingface.co/datasets/anas-gouda/mr6d/resolve/{commit}/{path}'
        record=dict(path=path,url=url,requested_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'pallet-query-visibility-source-audit/1.0'})
            with urllib.request.urlopen(req,timeout=30)as r:
                data=r.read(20000001);assert len(data)<=20000000
                record.update(status=r.status,final_url=r.url,headers={k:r.headers[k]for k in ['Content-Type','ETag','X-Repo-Commit']if k in r.headers})
            if path.endswith('.json'):json.loads(data)
            p=OUT/path;p.parent.mkdir(parents=True,exist_ok=True)
            with p.open('xb')as f:f.write(data)
            record.update(PASS=True,local_path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        except BaseException:
            record.update(PASS=False,error=traceback.format_exc())
        return record
    records=[]
    with ThreadPoolExecutor(max_workers=6)as pool:
        tasks={pool.submit(fetch,path):path for path in paths}
        for future in as_completed(tasks):
            r=future.result();records.append(r)
            if len(records)%20==0 or not r['PASS']:print(json.dumps(dict(done=len(records),total=len(paths),last=r['path'],PASS=r['PASS'])),flush=True)
    records.sort(key=lambda r:r['path']);total=sum(r.get('bytes',0)for r in records)
    result=dict(completed=all(r['PASS']for r in records),PASS=all(r['PASS']for r in records),commit=commit,policy_sha=sha(V/'ROOT_PUBLIC_INTAKE_POLICY_BEFORE_REMOTE_SAMPLE_BODIES.json'),code_sha=sha(Path(__file__)),files=records,total_body_bytes=total,image_depth_mask_mesh_bodies_downloaded=0,new_model_calls=0)
    with (M/'PROVIDED_METADATA_DOWNLOAD_RECEIPT.json').open('x')as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    assert total+ (M/'HF_DATASET_INFO.json').stat().st_size<100000000
    print(json.dumps(dict(completed=result['completed'],files=len(records),bytes=total,failed=[r['path']for r in records if not r['PASS']])),flush=True)
    if not result['PASS']:raise SystemExit(1)

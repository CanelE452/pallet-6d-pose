"""Resumable public NGC download with streaming publisher MD5 and SHA256."""
from pathlib import Path
import hashlib,json,os,shutil,time
import requests

ROOT=Path(__file__).resolve().parent
URL='https://api.ngc.nvidia.com/v2/resources/nvidia/isaac/r2bdataset2023/versions/3/files/r2b_storage/r2b_storage_0.db3'
SIZE=3087908864
MD5='5e8f11201fe10dbac7307a8628553e94'
DEST=ROOT/'r2b_storage_0.db3'
PART=DEST.with_suffix('.db3.part')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()
def write(p,x):
    tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def hashes(p):
    s=hashlib.sha256();m=hashlib.md5();n=0
    if p.exists():
        with p.open('rb')as f:
            while b:=f.read(8*1024*1024):s.update(b);m.update(b);n+=len(b)
    return n,s,m

def main():
    assert not DEST.exists(),'Existing verified DB3 must be preserved.'
    bindings=[dict(path=str(ROOT/f),sha256=sha(ROOT/f))for f in ['metadata.yaml','OFFICIAL_API_FILES.json','OFFICIAL_CARD.html','LICENSE_CC_BY_4_0.html']]
    protocol=ROOT/'INTAKE_PROTOCOL.json'
    definition=dict(schema='R2B_STORAGE_V3_PUBLIC_RGB_FIXED_INTAKE_V1',source_url=URL,version='3',expected_size=SIZE,publisher_md5=MD5,
        publisher_checksum_source='Official NGC card and recipe.description; multipart ETag is NOT MD5.',
        official_card='https://catalog.ngc.nvidia.com/orgs/nvidia/isaac/resources/r2bdataset2023/3',license='CC-BY-4.0',
        selection='For each observed official d455_1_rgb_image and hawk_0_left_rgb_image topic, sort by(timestamp,id), select np.linspace(0,N-1,min(16,N)).astype(int). Maximum32RGB images. No content/score selection.',
        allowed_reads='Public DB3, metadata/license, and EVAL_SHA256_DENYLIST only; no current RGB/GT/model/feature/weights.',
        annotation='RGB-only intake, no independent pallet/visible/full/occlusion GT.',training_admission='HOLD pending source-family, labels and TRAIN/VAL separation review.',
        bindings=bindings,downloader_sha256=sha(__file__))
    if protocol.exists():assert json.loads(protocol.read_text())==definition
    else:write(protocol,definition)
    n,s,m=hashes(PART);assert 0<=n<=SIZE
    free=shutil.disk_usage(ROOT).free
    assert free>SIZE-n+1024**3,('Insufficient free space',free)
    sess=requests.Session();sess.headers['Accept-Encoding']='identity'
    begin=time.monotonic();initial=n;last=begin;failures=[];headers=[]
    for attempt in range(1,9):
        if n==SIZE:break
        try:
            req={'Range':f'bytes={n}-'}if n else{}
            with sess.get(URL,headers=req,stream=True,timeout=(30,60))as response:
                if response.status_code in [429,500,502,503,504]:
                    delay=float(response.headers.get('Retry-After',min(60,2**attempt)))
                    failures.append(dict(attempt=attempt,status=response.status_code,retry_after=delay,bytes=n));time.sleep(delay);continue
                response.raise_for_status()
                assert response.headers.get('Content-Encoding','identity')in ['identity',''],response.headers.get('Content-Encoding')
                if n:
                    assert response.status_code==206,'Range ignored; partial preserved rather than overwritten.'
                    assert response.headers.get('Content-Range')==f'bytes {n}-{SIZE-1}/{SIZE}'
                else:assert response.status_code==200
                relevant={k:v for k,v in response.headers.items()if k.lower()in ['content-length','content-range','accept-ranges','content-type','etag','last-modified']}
                headers.append(dict(attempt=attempt,resume_offset=n,status=response.status_code,headers=relevant))
                with PART.open('ab')as f:
                    for block in response.iter_content(chunk_size=4*1024*1024):
                        if not block:continue
                        assert n+len(block)<=SIZE
                        f.write(block);s.update(block);m.update(block);n+=len(block)
                        now=time.monotonic()
                        if now-last>=10:
                            rate=(n-initial)/(now-begin)
                            state=dict(bytes=n,expected_bytes=SIZE,percent=100*n/SIZE,MB_per_second=rate/1e6,ETA_seconds=(SIZE-n)/rate if rate else None,attempt=attempt)
                            write(ROOT/'DOWNLOAD_PROGRESS.json',state);print(json.dumps(state),flush=True);last=now
                    f.flush();os.fsync(f.fileno())
        except(requests.RequestException,OSError)as e:
            failures.append(dict(attempt=attempt,error=type(e).__name__,message=str(e).split('?')[0],bytes=n))
            time.sleep(min(60,2**attempt))
    assert n==SIZE,('Incomplete download; .part preserved',n,SIZE)
    assert m.hexdigest()==MD5,('Publisher MD5 mismatch; .part preserved',m.hexdigest())
    with PART.open('rb')as f:assert f.read(16)==b'SQLite format 3\x00'
    final=dict(PASS=True,source_url=URL,expected_size=SIZE,actual_size=n,publisher_md5=MD5,actual_md5=m.hexdigest(),sha256=s.hexdigest(),
        sqlite_header_valid=True,streaming_hashes=True,resume_start_bytes=initial,HTTP_transactions=headers,retry_history=failures,
        multipart_ETag_not_used_as_MD5=True,seconds=time.monotonic()-begin,free_disk_bytes_at_start=free,protocol_sha256=sha(protocol))
    PART.replace(DEST);write(ROOT/'DOWNLOAD_VERIFICATION.json',final)
    print('OFFICIAL_DB3_VERIFIED',json.dumps(final),flush=True)

if __name__=='__main__':main()

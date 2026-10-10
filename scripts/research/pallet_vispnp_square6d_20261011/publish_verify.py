"""Verify published artifacts against an immutable commit and normal remote refs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.request import Request,urlopen

REPOSITORY='CanelE452/pallet-6d-pose'
BRANCH='research/vispnp-square6d-20261011'
DOC='_docs/experiments/pallet_vispnp_square6d_20261011'
CODE='scripts/research/pallet_vispnp_square6d_20261011'

def git(*args):return subprocess.check_output(['git',*args])
def get(url):
    with urlopen(Request(url,headers={'User-Agent':'pallet-vis-artifact-verifier'}),timeout=45) as response:
        assert response.status==200
        return response.read()

def main():
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert len(a.commit)==40 and git('rev-parse',a.commit+'^{commit}').decode().strip()==a.commit
    assert not a.output.exists()
    refs={ref:sha for sha,ref in (x.split() for x in git('ls-remote','origin','refs/heads/main','refs/heads/'+BRANCH).decode().splitlines())}
    assert refs=={'refs/heads/main':a.commit,'refs/heads/'+BRANCH:a.commit},refs
    files=git('ls-tree','-r','--name-only',a.commit,'--',DOC,CODE).decode().splitlines()
    assert DOC+'/REPORT_KO.md' in files and any(f.endswith('.png') for f in files)
    def check(path):
        content=git('show',a.commit+':'+path);url='https://raw.githubusercontent.com/'+REPOSITORY+'/'+a.commit+'/'+path
        observed=get(url);assert hashlib.sha256(content).digest()==hashlib.sha256(observed).digest(),path
        return dict(path=path,sha256=hashlib.sha256(observed).hexdigest(),bytes=len(observed),status='HTTP_200_SHA256_MATCH',url=url)
    with ThreadPoolExecutor(max_workers=4) as pool:artifacts=list(pool.map(check,files))
    links=[]
    for path in files:
        if path.endswith('.md') or path.endswith('.png'):
            url='https://github.com/'+REPOSITORY+'/blob/'+a.commit+'/'+path;get(url)
            links.append(dict(path=path,status='HTTP_200',url=url))
    result=dict(status='PASS',artifact_commit_sha=a.commit,remote_refs=refs,artifacts=artifacts,
        report_and_image_links=links,verified_at=datetime.now(timezone.utc).isoformat(),force_push=False)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print('REMOTE_VERIFICATION_PASS',a.commit,len(files),len(links),flush=True)

if __name__=='__main__':main()

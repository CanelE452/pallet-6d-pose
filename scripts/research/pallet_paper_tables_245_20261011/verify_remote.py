"""Read-only GitHub publication proof: SHA, raw bytes, rendered report/images."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT=Path(__file__).resolve().parents[3]
PREFIX='_docs/experiments/pallet_paper_tables_245_20261011/'
REPOSITORY='CanelE452/pallet-6d-pose'
FILES=['REPORT_KO.md','REPRODUCE.md','README.md','TABLE_1.md','TABLE_2.md','TABLE_3.md',
       'METHOD_LOCK.json','LATENCY.json','PAPER_NUMBERS.json','VERIFICATION.json',
       'STATISTICS_VERIFICATION.json','REPLAY_VERIFICATION.json','PAPER_NUMBERS_VERIFICATION.json',
       'NUMERICAL_AMENDMENT.json','figures/FIGURE_A_errors.png','figures/FIGURE_B_wd_elevation.png',
       'figures/FIGURE_C_coordinates.png','figures/FIGURE_D_coordinates.png']

def remote_sha():
    text=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=ROOT,text=True)
    value=text.split()[0];assert len(value)==40;return value

def fetch(url,head=False):
    request=urllib.request.Request(url,headers={'User-Agent':'pallet-paper-artifact-verification'},method='HEAD' if head else 'GET')
    with urllib.request.urlopen(request,timeout=45) as response:
        assert response.status==200,(url,response.status)
        return response.read() if not head else response.status

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--science-sha',required=True);parser.add_argument('--receipt',required=True)
    args=parser.parse_args();receipt=Path(args.receipt);assert not receipt.exists()
    observed=remote_sha();assert observed==args.science_sha,('Main differs from requested verification SHA',observed,args.science_sha)
    def check(name):
        path=PREFIX+name;local=(ROOT/path).read_bytes();expected=hashlib.sha256(local).hexdigest()
        raw=f'https://raw.githubusercontent.com/{REPOSITORY}/{args.science_sha}/{path}'
        data=fetch(raw);actual=hashlib.sha256(data).hexdigest();assert actual==expected,path
        record=dict(path=path,raw_commit_url=raw,status=200,sha256=actual,bytes=len(data))
        if name=='REPORT_KO.md' or name.startswith('figures/'):
            main=f'https://github.com/{REPOSITORY}/blob/main/{path}'
            record['main_blob_url']=main;record['main_blob_status']=fetch(main,head=True)
            current=fetch(f'https://raw.githubusercontent.com/{REPOSITORY}/main/{path}')
            assert hashlib.sha256(current).hexdigest()==expected,path
            record['main_raw_sha256_equal']=True
        return record
    with ThreadPoolExecutor(max_workers=4) as executor:records=list(executor.map(check,FILES))
    assert remote_sha()==observed,'Main changed during verification'
    result=dict(status='PASS',repository=REPOSITORY,branch='main',science_commit_sha=observed,
        force_push=False,artifacts=records,artifacts_verified=len(records),rendered_report_and_public_figures_status200=5,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with receipt.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print('GITHUB_REMOTE_SHA_AND18_ARTIFACT_HASHES_PASS',observed,flush=True)

if __name__=='__main__':main()

"""Publish an explicit progress snapshot or completed experimental draft.

Only the listed new namespaces and small reproducibility dependencies are copied.
The working repository's git index and unrelated edits are never touched.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
NAME=Path(__file__).parent.name
RAW=ROOT/'data/pallet/results'/NAME
DOC=ROOT/'_docs/experiments'/NAME
CHECKOUT=Path('/tmp/pallet-pose-github-review-20260930')
REMOTE='https://github.com/CanelE452/pallet-6d-pose.git'
NAMES=['pallet_dope_refiner_20261001_v1','pallet_resnet18_refiner_20261001_v1',NAME]
PAPER='_docs/paper/sensors_dope_extension_20261001_v1'
DEPENDENCIES=[
 'data/pallet/eval_results/stage16_truncation_addon/capturecad_b2_eval/eval_capturecad_b2.py',
 'data/pallet/results/pallet_line_pose_v1/TRAIN_PROTOCOL.json',
 *['data/pallet/results/pallet_sensors_refinement_closeout_v1/external/integral-human-pose/'+p for p in (
     'LICENSE','pytorch_projects/common_pytorch/base_modules/resnet.py','pytorch_projects/common_pytorch/base_modules/deconv_head.py')]]

def read(p):return json.loads(Path(p).read_text())
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.pending')
    q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def git(*args):
    return subprocess.check_output(['git',*args],cwd=CHECKOUT,text=True,
        env=dict(os.environ,GIT_TERMINAL_PROMPT='0')).strip()
def verify(b):assert sha(ROOT/b['path'])==b['sha256'],b
def verify_tree(value):
    if isinstance(value,dict):
        if 'path' in value and 'sha256' in value:verify(value)
        for child in value.values():verify_tree(child)
    elif isinstance(value,list):
        for child in value:verify_tree(child)

def wait_for_postprocess():
    while not (DOC/'POSTPROCESS_COMPLETE.json').exists():
        p=RAW/'POSTPROCESS_STATUS.json'
        if p.exists():
            v=read(p)
            assert not str(v.get('state','')).startswith(('STOPPED','FAILED')),v
        q=read(RAW/'QUEUE_STATUS.json')
        assert q['state']!='STOPPED_ON_ERROR',q
        time.sleep(30)

def main(mode,wait):
    assert mode in ('progress','results')
    if wait:
        assert mode=='results';wait_for_postprocess()
    if mode=='results':
        p=read(DOC/'POSTPROCESS_COMPLETE.json');assert p['complete']
        verify_tree(p)
        for name in NAMES:
            result=read(ROOT/'_docs/experiments'/name/'REPORT_COMPLETE.json')
            assert result['complete'];verify(result['code']);verify(result['report'])
    build=read(ROOT/PAPER/'DRAFT_BUILD.json');assert build['complete']
    verify_tree(build)
    assert git('remote','get-url','origin')==REMOTE
    assert not git('status','--porcelain'),'Publication checkout must be clean; preserve its edits'
    git('fetch','origin','main');git('merge','--ff-only','origin/main')
    before=git('rev-parse','HEAD')
    if mode=='results':
        (DOC/'README_KO.md').write_text('# IEEE Sensors: 세 기반 추정기 결과\n\n'
            'YOLO·DOPE(VGG)·SimpleBaseline-derived ResNet-18의 보정 전후 실험과 실제 측정 보고서가 생성되었습니다. '
            '원고는 측정값이 삽입된 검토용 초안이며 논문 투고·채택이나 독립 TEST 확인을 뜻하지 않습니다.\n\n'
            '[세 모델 통합 결과·이미지·치수](REPORT_KO.md) · [전체 지표 CSV](RESULTS.csv) · '
            '[측정값을 넣은 원고](../../paper/sensors_dope_extension_20261001_v1/manuscript.pdf) · '
            '[보충자료](../../paper/sensors_dope_extension_20261001_v1/supplementary.pdf) · '
            '[게시 시점 기록](PUBLICATION_SNAPSHOT.json)\n\n'
            '수치는 각 기반 내 동일 지원점의 전후 비교이며 실패·결측을 전체 GT PCK와 coverage에 포함합니다. '
            '원래 실사 T/R 안정적 공동 개선 목표는 아직 입증되지 않았습니다. '
            '데이터·가중치·큰 SOURCE_MANIFEST·로컬 Tectonic/cache는 새로 게시하지 않았으며 코드 단독 재실행 패키지가 아닙니다.\n')
        for name in NAMES[:2]:
            (ROOT/'_docs/experiments'/name/'README_KO.md').write_text(
                f'# {name}\n\n실제 측정 실행을 완료했습니다. [이미지·치수·표·한계를 포함한 보고서](REPORT_KO.md), '
                '[전체 지표](DEV_RESULTS.json), [paired 통계](DEV_PAIRED_RESULTS.json), [측정 속도](RUNTIME.json)를 확인할 수 있습니다. '
                '추가 실험 완료가 모든 성능 개선이나 원래 T/R 안정성 목표 달성을 의미하지 않습니다.\n')
    snapshot=dict(time=now(),mode=mode,queue=read(RAW/'QUEUE_STATUS.json'),
        original_stable_TR_goal_achieved=False,independent_TEST=False,
        raw_weights_images_and_large_source_manifest_not_published=True,
        GitHub_snapshot_not_a_live_monitor=True,not_submitted=True,
        PDF_visual_review='See current PDF_VISUAL_REVIEW.json; automated final build may require author layout review',
        code_sha256=sha(__file__))
    for label,folder in [('dope','pallet_dope_refiner_20261001_v1'),('resnet18','pallet_resnet18_refiner_20261001_v1')]:
        for filename in ('CACHE_PROGRESS.json','TRAIN_PROGRESS.json','BASELINE_PROGRESS.json'):
            path=ROOT/'data/pallet/results'/folder/filename
            if path.exists():snapshot[label+'_'+filename]=read(path)
    write(DOC/'PUBLICATION_SNAPSHOT.json',snapshot)
    roots=[*[f'scripts/research/{n}' for n in NAMES],*[f'_docs/experiments/{n}' for n in NAMES],PAPER]
    files=[]
    excluded_parts={'.tex_cache','pdf_render','__pycache__','.git'}
    for relative in roots:
        for path in sorted((ROOT/relative).rglob('*')):
            if not path.is_file() or excluded_parts.intersection(path.parts):continue
            if path.suffix in ('.pyc','.aux','.log','.out','.bbl','.blg') or path.name.endswith('.pending'):continue
            files.append(path)
    files += [ROOT/p for p in DEPENDENCIES]
    files.append(ROOT/'_docs/paper/IEEE_SENSORS_COMPLETION_BRIEF_20261001_KO.md')
    manifest=[]
    for source in sorted(set(files)):
        assert source.is_file() and not source.is_symlink(),source
        assert source.stat().st_size<40*1024*1024,('Unexpected large publication file',source)
        rel=source.relative_to(ROOT);target=CHECKOUT/rel
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        digest=sha(source);assert sha(target)==digest
        manifest.append(dict(path=str(rel),sha256=digest,bytes=source.stat().st_size))
    # Force only explicitly inspected allowlisted paths, including ignored PDF assets.
    for first in range(0,len(manifest),80):git('add','-f','--',*[r['path'] for r in manifest[first:first+80]])
    changed=git('diff','--cached','--name-only').splitlines()
    assert changed and set(changed).issubset({r['path'] for r in manifest})
    git('diff','--cached','--check')
    title=('Prepare three-estimator Sensors experiments with measured progress'
           if mode=='progress' else 'Report three-estimator refinement measurements and manuscript draft')
    git('commit','-m',title)
    commit=git('rev-parse','HEAD')
    for row in manifest:
        # Check repository blob bytes as well as the copied working files.
        blob=subprocess.check_output(['git','show',commit+':'+row['path']],cwd=CHECKOUT)
        assert hashlib.sha256(blob).hexdigest()==row['sha256'],row['path']
    git('push','origin','HEAD:main')
    remote=git('ls-remote','origin','refs/heads/main').split()[0]
    assert remote==commit,('Remote HEAD mismatch',commit,remote)
    write(RAW/f'GITHUB_{mode.upper()}_PUSH_RECEIPT.json',dict(complete=True,verified=True,time=now(),
        mode=mode,before=before,commit=commit,remote_head=remote,remote=REMOTE,
        files=manifest,changed_paths=changed,source_git_index_untouched=True))
    print(json.dumps(dict(mode=mode,commit=commit,verified=True,files=len(manifest))),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['progress','results'])
    parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    try:main(args.mode,args.wait)
    except BaseException as exc:
        write(RAW/f'GITHUB_{args.mode.upper()}_PUBLISH_FAILED.json',dict(time=now(),error_type=type(exc).__name__,error=str(exc)))
        raise

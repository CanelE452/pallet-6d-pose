"""Verify complete model comparisons, open the real report, notify Discord."""
from __future__ import annotations
import argparse
import re
from pathlib import Path
import shutil
import subprocess
import time
from scripts.research.pallet_dht_joint_v1.driver import audit_training, read, write, sha, now

TITLE='Pallet DHT Joint · 점과 선 공동 학습'


def show_page(page):
    launcher=next((shutil.which(x) for x in ['google-chrome','google-chrome-stable','chromium','chromium-browser','firefox'] if shutil.which(x)),None)
    receipt=dict(html=str(page),html_sha256=sha(page),window_visibility_confirmed=False)
    if launcher:
        with (page.parent/'GALLERY_OPEN.log').open('a') as log:
            child=subprocess.Popen([launcher,'-new-window' if Path(launcher).name=='firefox' else '--new-window',page.as_uri()],
                                   stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        receipt.update(status='launch_requested',pid=child.pid,launcher=launcher)
        for _ in range(12):
            try:
                windows=subprocess.run(['xprop','-root','_NET_CLIENT_LIST_STACKING'],capture_output=True,text=True,timeout=5).stdout
                for window in re.findall(r'0x[0-9a-fA-F]+',windows):
                    title=subprocess.run(['xprop','-id',window,'_NET_WM_NAME'],capture_output=True,text=True,timeout=5).stdout
                    if TITLE not in title:continue
                    info=subprocess.run(['xwininfo','-id',window],capture_output=True,text=True,timeout=5).stdout
                    state=subprocess.run(['xprop','-id',window,'_NET_WM_STATE'],capture_output=True,text=True,timeout=5).stdout
                    if 'Map State: IsViewable' in info and '_NET_WM_STATE_HIDDEN' not in state:
                        receipt.update(status='window_confirmed',window_visibility_confirmed=True,window_id=window,title=TITLE)
                        break
            except (OSError,subprocess.TimeoutExpired):break
            if receipt['window_visibility_confirmed']:break
            time.sleep(.5)
    else:receipt['status']='browser_not_found'
    write(page.parent/'GALLERY_OPEN.json',receipt)
    return receipt


def marker(path, allow_recorded_runtime_failure=False):
    value=read(path)
    recorded_failure=(allow_recorded_runtime_failure and Path(path).name=='RUNTIME.json'
        and value.get('PASS') is False and value.get('parity_PASS') is False
        and value.get('timing_collection_complete') is True
        and value.get('status')=='COMPLETE_WITH_STRICT_PARITY_FAILURE'
        and len(value.get('strict_failures',[]))>0
        and value.get('parity_policy',{}).get('atol')==1e-4
        and value.get('parity_policy',{}).get('rtol')==0
        and value.get('parity_policy',{}).get('criterion_changed') is False)
    if not(value.get('complete') is True and (value.get('PASS',True) is True or recorded_failure)):
        raise ValueError(f'Incomplete artifact {path}')
    for key in ['source_sha256','input_sha256','output_sha256']:
        for file,digest in value.get(key,{}).items():
            p=Path(file);p=p if p.is_absolute() else Path(path).parent/p
            if sha(p)!=digest:raise ValueError(f'Changed bound artifact {p}')
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);args=p.parse_args()
    root=args.run_dir.resolve();protocol=read(root/'TRAIN_PROTOCOL.json')
    audit_training(root,protocol)
    names=['COMPLETION_CHAIN_BINDING.json','SOURCE_REVALIDATION.json','ARCHITECTURE_AUDIT.json','SMOKE_CHECKS.json','TRAINING_AUDIT.json','RUNTIME.json',
           'SUMMARY.json','VERDICT.json','AGGREGATE_COMPLETE.json','REPORT_RENDER.json',
           'INDEPENDENT_FINAL_AUDIT.json','ACTUAL_VISUAL_QA.json']
    for name in names:marker(root/name,allow_recorded_runtime_failure=name=='RUNTIME.json')
    runtime=read(root/'RUNTIME.json')
    runtime_audit=read(root/'INDEPENDENT_FINAL_AUDIT.json')['runtime_audit']
    if not(runtime_audit['timing_collection_complete'] and runtime_audit['observed_timing_count']==702
           and runtime_audit['strict_parity_PASS'] is runtime['parity_PASS']
           and runtime_audit['strict_failure_count']==len(runtime['strict_failures'])
           and runtime_audit['criterion_changed'] is False):
        raise ValueError('Runtime failure reporting lacks independent verification')
    evaluations=[]
    for seed in protocol['training']['seeds']:
        for arm in protocol['arms']:
            rel=f'evaluation/{arm}_seed{seed}/COMPLETION.json';c=marker(root/rel)
            if c['actual_positive_forwards']!=319 or c['actual_negative_forwards']!=2689 or c['baseline_candidate_copying']:
                raise ValueError('New-model actual inference denominator differs')
            train=read(root/'runs'/f'{arm}_seed{seed}'/'COMPLETION.json')
            if c['checkpoint_sha256']!=train['checkpoint_sha256']:raise ValueError('Wrong evaluation checkpoint')
            evaluations.append(dict(arm=arm,seed=seed,checkpoint_sha256=c['checkpoint_sha256']))
            names.append(rel)
    render=read(root/'REPORT_RENDER.json');page=root/'index.html'
    if not(render.get('experiment_complete') and render.get('n_completed_evaluations')==len(evaluations)
           and render['html_sha256']==sha(page) and TITLE in page.read_text()):
        raise ValueError('Report is not the complete bound comparison')
    qa=marker(root/'VISUAL_QA.json');names.append('VISUAL_QA.json')
    if qa.get('html_sha256')!=sha(page) or qa.get('broken_local_images',0)!=0:
        raise ValueError('HTML visual QA changed or failed')
    opened=show_page(page);verdict=read(root/'VERDICT.json')
    from scripts.research.discord_notify import notify
    message='\n'.join(['팔레트 Deep Hough Transform + 점 공동 학습 아키텍처의 구현·학습·비교 검증이 완료됐습니다.',
        verdict['headline_ko'],*verdict.get('discord_lines_ko',[]),
        'Self-training 없이 합성55,980장, 3개 구조×3개 seed를 같은 예산으로 학습했습니다.',
        '각 모델은 실사319장과 negative2689장을 모두 새로 추론했습니다. 반복 사용한 DEV 평가입니다.',
        ('속도 측정의 원래 예측 일치 검사(atol1e-4)는 통과했습니다.' if runtime['parity_PASS'] else
         f"속도는 참고 실측값입니다. 원래 예측 일치 검사(atol1e-4)는 {len(runtime['strict_failures'])}/702회 실패했고 이를 그대로 보고했습니다."),
        f'시각화 HTML(작업 PC): {page}',
        '브라우저 보고서 창 표시를 확인했습니다.' if opened['window_visibility_confirmed'] else '브라우저 창 표시 여부는 확인되지 않았습니다.'])
    delivered=notify(message,root,evidence=root/'VERDICT.json')
    done=dict(complete=True,PASS=True,completed_at_utc=now(),runs=evaluations,
              scope='Research execution and honest result reporting completed; PASS is neither an accuracy gain nor a claim that every validation check passed.',
              all_checks_passed=runtime['parity_PASS'], runtime_strict_parity_PASS=runtime['parity_PASS'],
              runtime_strict_failure_count=len(runtime['strict_failures']),
              headline_ko=verdict['headline_ko'],overall_accuracy_improved=verdict['overall_accuracy_improved'],
              report=str(page),html_sha256=sha(page),browser_window_confirmed=opened['window_visibility_confirmed'],
              discord_status=delivered['status'],discord_http_status=delivered.get('http_status'),
              artifact_sha256={name:sha(root/name) for name in names})
    write(root/'COMPLETION.json',done)
    if delivered['status']!='sent':raise RuntimeError('Work completed; Discord delivery not confirmed')
    print(verdict['headline_ko'],flush=True)


if __name__=='__main__':main()

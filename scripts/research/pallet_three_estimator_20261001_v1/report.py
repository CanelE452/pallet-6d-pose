"""Combine completed estimator-specific evidence; never fill pending experiments."""
from pathlib import Path
from datetime import datetime,timezone
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
DOC=ROOT/'_docs/experiments'/Path(__file__).parent.name
EXPERIMENTS={'DOPE':'pallet_dope_refiner_20261001_v1','ResNet-18':'pallet_resnet18_refiner_20261001_v1'}
HISTORICAL=ROOT/'_docs/experiments/pallet_sensors_submission_v1'
def read(p):return json.loads(Path(p).read_text())
def bound(p):
    p=Path(p);return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def verify(b):assert bound(ROOT/b['path'])['sha256']==b['sha256'],b
def fmt(v):return 'NA' if v is None else f'{v:.3f}'
def avg(values):return None if any(v is None for v in values) else float(np.mean(values))
def mean(methods,arm):
    rows=[methods[f'{arm}{s}'] for s in (1,2,3)]
    return dict(**{k:avg([v[k] for v in rows]) for k in ('median_px','p90_px','matched_frames')},
        ALL_GT_PCK={'10':avg([v['ALL_GT_PCK']['10'] for v in rows])},
        pose={k:avg([v['pose'].get(k) for v in rows]) for k in ('translation_median_cm','rotation_median_deg','translation_p90_cm','rotation_p90_deg','coverage')})

def main():
    evidence=[];results={};runtime={};galleries={};contrasts={}
    for label,name in EXPERIMENTS.items():
        folder=ROOT/'_docs/experiments'/name
        completed=read(folder/'REPORT_COMPLETE.json');assert completed['complete']
        for b in [completed['report'],completed['code'],*completed['sources'],*completed['figures'],*completed['data_files']]:verify(b)
        result=read(folder/'DEV_RESULTS.json');assert result['complete'] and result['role']=='REUSED_DEV'
        verify(result['historical_yolo_results']['source'])
        assert result['historical_yolo_results']['source']['sha256']==bound(HISTORICAL/'UNIFIED_DEV_RESULTS.json')['sha256']
        assert (result['positive_frames'],result['sessions'],result['gt_denominator'])==(319,13,2818)
        assert result['same_conditional_support']
        results[label]=result;runtime[label]=read(folder/'RUNTIME.json')
        galleries[label]=read(folder/'GALLERY_MANIFEST.json')
        base='DOPE' if label=='DOPE' else 'RESNET18'
        pair=read(folder/'DEV_PAIRED_RESULTS.json')
        contrasts[label]=pair['results']['P_minus_'+base]['conditional_keypoint_median']['session']
        evidence.append(bound(folder/'REPORT_COMPLETE.json'))
    historical=read(HISTORICAL/'UNIFIED_DEV_RESULTS.json');assert historical['complete']
    evidence.extend([bound(HISTORICAL/'UNIFIED_DEV_RESULTS.json'),bound(HISTORICAL/'FINAL_STATUS.json')])
    contrasts['YOLO']=dict(status='COMPLETE',**read(HISTORICAL/'FINAL_STATUS.json')['EVIDENCE']['P_minus_R0']['session'])
    rows=[]
    for label,methods,base in [('YOLO',historical['methods'],'R0'),('DOPE',results['DOPE']['methods'],'DOPE'),('ResNet-18',results['ResNet-18']['methods'],'RESNET18')]:
        for arm in ('baseline','D','P'):
            r=methods[base] if arm=='baseline' else mean(methods,arm)
            rows.append(dict(estimator=label,arm=arm,median_px=r['median_px'],p90_px=r['p90_px'],
                matched_frames=r['matched_frames'],PCK10_percent=100*r['ALL_GT_PCK']['10'],
                T_median_cm=r['pose']['translation_median_cm'],R_median_deg=r['pose']['rotation_median_deg'],
                T_p90_cm=r['pose'].get('translation_p90_cm'),R_p90_deg=r['pose'].get('rotation_p90_deg'),
                pose_coverage_percent=100*r['pose']['coverage']))
    DOC.mkdir(parents=True,exist_ok=True)
    with (DOC/'RESULTS.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axs=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for ax,metric,title in zip(axs,('median_px','p90_px','PCK10_percent'),('Conditional median (px)','Conditional P90 (px)','Full-GT PCK10 (%)')):
        vals=[np.nan if r[metric] is None else r[metric] for r in rows]
        ax.bar(range(9),vals,color=['#52687a','#d6a24a','#258878']*3)
        ax.set_xticks(range(9),['Y base','Y+D','Y+P','Do base','Do+D','Do+P','R base','R+D','R+P'],rotation=45)
        ax.set_title(title);ax.grid(axis='y',alpha=.2)
        for i,v in enumerate(vals):
            ax.text(i,v if np.isfinite(v) else 0,f'{v:.2f}' if np.isfinite(v) else 'NA',ha='center',va='bottom',fontsize=7)
    fig.suptitle('YOLO (Y), DOPE (Do), ResNet-18 (R): within-estimator refinement\nReused DEV319 / 2,818 GT points; D/P = mean of three seed statistics')
    fig.savefig(DOC/'three_estimators.png',dpi=170);fig.savefig(DOC/'three_estimators.pdf');plt.close(fig)
    text=['# IEEE Sensors 원고: 세 기반 추정기 실험 결과',
        '\nYOLO·DOPE(VGG)·SimpleBaseline-derived ResNet-18의 각 기반 안에서 P/D 보정 전후를 비교했다. 이 보고서는 세 모델의 실제 결과가 모두 있을 때만 생성된다. 실험 완료와 원고 최종 검토·GitHub 게시·논문 투고는 별도 단계다.',
        '\n## 정확도와 실패 분모\n\n| 기반 | 보정 | 2D 중앙값 px | 2D P90 px | 전체GT PCK10 % | 매칭/319 | T 중앙값 cm | R 중앙값 deg | Pose coverage % |\n|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:text.append('| '+' | '.join([r['estimator'],r['arm'],*[fmt(r[k]) for k in ('median_px','p90_px','PCK10_percent','matched_frames','T_median_cm','R_median_deg','pose_coverage_percent')]])+' |')
    text+=['\nD/P는3seed 통계의 평균이다. 원래 box·score·중심·결측 mask를 유지하며 각 기반의 조건부 매칭 분모는 서로 다를 수 있다. 전체 GT PCK는 제외·결측을 실패로 포함한다. T/R은 기존 주석·카메라·치수로 만든 geometry 참조 기준이며 독립 물리 측정 정확도를 의미하지 않는다. [정밀도 CSV 및 T/R P90](RESULTS.csv). 역사적 YOLO pose P90 미기록은 NA로 남겼다.',
        '\n![세 기반의 실제 측정 결과](three_estimators.png)',
        '\n## P−기반 추정기 차이: 세션 bootstrap\n\n| 기반 | 2D 중앙값 차이 px | 95% 구간 | 상태 |\n|---|---:|---|---|']
    supported=[]
    for label in ('YOLO','DOPE','ResNet-18'):
        c=contrasts[label]
        text.append(f"| {label} | {fmt(c.get('delta'))} | [{fmt(c.get('low'))}, {fmt(c.get('high'))}] | {c.get('status')} |")
        if c.get('status')=='COMPLETE' and c['high']<0:supported.append(label)
    text += [f'\n현재 재사용 DEV 세션 구간이 2D 중앙값 감소 방향을 지지한 기반: {", ".join(supported) or "없음"}. 13세션·10,000 bootstrap의 탐색적 비교이며 모든 지표 개선이나 임의 backbone 일반화를 뜻하지 않는다.',
        '\n## 추가 실험의 실제 속도\n\n| 별도 측정 패널 | 방법 | 2D 평균 ms | Pose 평균 ms | 전체 평균 ms | 전체 P90 ms |\n|---|---|---:|---:|---:|---:|']
    for label in ('DOPE','ResNet-18'):
        rr=runtime[label];assert rr['complete'] and rr['measured_calls']==390 and rr['all_cache_replays_PASS']
        for arm,r in rr['results'].items():text.append('| '+' | '.join([label,arm,fmt(r['keypoints_ms']['mean']),fmt(r['pose_ms']['mean']),fmt(r['full_ms']['mean']),fmt(r['full_ms']['p90'])])+' |')
    text += ['\n26장·13세션·5반복, batch1, 각 방법20회 warmup, 대표 seed1. 파일 읽기·모델 초기화는 제외한다. 역사적 YOLO 속도와 다른 시점·thread 설정이므로 세 기반의 동시 속도 순위로 해석하지 않는다. 각 패널 안의 보정 추가 비용을 비교한다.',
        '\n## 이미지, 치수, 개별 실패\n\n각 보고서에 실제 RGB 전후 좌표, 원영상 크기, long×short×height 치수(mm), 카메라 K와 이미지 해시가 있다. 사례는 사전 명시한 결과 양끝·결측 규칙으로 선택했고 전체 성능을 대표한다고 주장하지 않는다.']
    for label,name in EXPERIMENTS.items():
        text.append(f'\n### {label}\n\n[전체 실험·학습·치수·이미지 보고서](../{name}/REPORT_KO.md)')
        for c in galleries[label]['cases']:text.append(f"\n![{label} {c['selection']}](../{name}/{c['figure']})")
    text+=['\n## 학습 범위와 주장 제한\n\nYOLO와 DOPE는 기존 기반 가중치이고, 세 번째 ResNet-18만 ImageNet 초기값에서 같은 합성 TRAIN55,980장으로60epoch 새로 학습했다. 각 모델 초기화와 전체 학습 예산이 같다고 주장하지 않는다. 새 실사 학습은0장이다. 기본 추정기를 고정한 후의 P/D 예산은 각3seed×6,000updates×16, 같은 source 분할·선택 그리드이며, 실제 유효 학습 행 수·adapter 채널 수는 기반마다 공개한다.',
        '\nResNet-18 모델은 full-image pallet9에 맞춘 SimpleBaseline adaptation이다. DOPE affinity 다중 객체 grouping은 사용하지 않았다. 이번 비교는 각 기반에 보정 원리를 적용하고 head를 각각 학습한 것이다. 하나의 YOLO head를 다른 모델에 무학습 전이한 실험이 아니다. 원래 실사 T/R 안정적 공동 개선 목표는 아직 입증되지 않았으며 독립 TEST도 없다.',
        '\n[기존 YOLO/PoseFix-derived 큰 보정기 비교](../pallet_sensors_submission_v1/FINAL_REPORT_KO.md)에서는 더 큰 보정기가 더 정확하고 P가 측정 지연시간이 작았다. 이 결과를 새 모델의 측정치와 섞어 성능을 부풀리지 않는다.']
    (DOC/'REPORT_KO.md').write_text('\n'.join(text)+'\n')
    value=dict(complete=True,time=datetime.now(timezone.utc).isoformat(),report=bound(DOC/'REPORT_KO.md'),
        code=bound(__file__),evidence=evidence,outputs=[bound(DOC/n) for n in ('RESULTS.csv','three_estimators.png','three_estimators.pdf')],
        models=3,original_stable_TR_goal_achieved=False,independent_TEST=False,publication_verified=False)
    (DOC/'REPORT_COMPLETE.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()

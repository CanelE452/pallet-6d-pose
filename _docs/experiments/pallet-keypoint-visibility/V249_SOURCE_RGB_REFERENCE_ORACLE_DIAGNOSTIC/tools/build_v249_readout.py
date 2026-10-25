"""Plot all predeclared V249 SOURCE reference scores; no model or source-input I/O."""
from pathlib import Path
import csv, json, html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).absolute().parent
H=R/'V249_ROOT_TARGET_RGB_DIAGNOSTIC'
D=H/'oracle_rgb_reference_diagnostic_v1'
VIEWS=('full_CLS','fixed_centre_Q','centre_CLS')

def safe(p):
    p=Path(p)
    if any(t in str(p).lower() for t in ('v112','__pycache__','.pyc')) or any(a.is_symlink() for a in [p,*p.parents]):raise ValueError(str(p))
    return p

def run():
    summary=json.loads(safe(D/'SUMMARY.json').read_bytes())
    tables=json.loads(safe(D/'DESCRIPTIVE_TABLES.json').read_bytes())['tables']
    assert summary['PASS'] and summary['reference_identity_is_oracle']
    with safe(D/'DEV397_DESCRIPTOR_SCORES.csv').open(newline='') as f:scores=list(csv.DictReader(f))
    rows=[]
    for stratum in ('all','<=3px'):
        for view in VIEWS:
            t=next(t for t in tables if t['view']==view and t['q_r_stratum']==stratum and t['object_id'] is None)
            rows.append({'view':view,'q_r_stratum':stratum,'D':t['D'],'E':t['E'],
                         'target_cosine_AUROC':t['E_AUROC_negative_nominal_cosine']['auc'],
                         'margin_AUROC':t['E_AUROC_negative_margin']['auc'],
                         'target_D_supported':t['E_AUROC_negative_nominal_cosine']['D_supported'],
                         'target_E_supported':t['E_AUROC_negative_nominal_cosine']['E_supported'],
                         'D_nominal_gt_other_fraction':t['D_nominal_gt_other']['fraction']})
    with safe(H/'AUC_COMPARISON.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for ax,stratum in zip(axs,('all','<=3px')):
        take=[r for r in rows if r['q_r_stratum']==stratum];x=np.arange(3)
        for offset,key,label,color in [(-.18,'target_cosine_AUROC','E=-target cosine','#285c9a'),(.18,'margin_AUROC','E=-target/other margin','#c87834')]:
            vals=[r[key] for r in take]
            ax.bar(x+offset,[0 if v is None else v for v in vals],.34,label=label,color=color)
            for xx,v in zip(x+offset,vals):ax.text(xx,.015 if v is None else v+.018,'NA' if v is None else f'{v:.3f}',ha='center',fontsize=9)
        ax.axhline(.5,color='#777',ls='--',lw=1);ax.set_ylim(0,1.08)
        ax.set_xticks(x,['Full CLS','Centre Q','Centre CLS']);ax.set_ylabel('AUROC, E positive (fixed direction)')
        ax.set_title(f"SOURCE DEV {stratum}: D{take[0]['D']} / E{take[0]['E']}");ax.legend(fontsize=8,loc='lower left')
    fig.suptitle('Teacher-identity RGB-reference oracle; AUROC is not pallet accuracy')
    fig.savefig(safe(H/'ORACLE_REFERENCE_AUC.png'),dpi=155);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(11,4),constrained_layout=True)
    for ax,view in zip(axs,VIEWS):
        groups=[[float(r['cosine_nominal_target']) for r in scores if r['view']==view and r['associated_r_state']==state and r['cosine_nominal_target']!=''] for state in ('D','E')]
        ax.boxplot(groups,tick_labels=['D','E'],whis=(5,95));ax.set_ylim(-1.02,1.02);ax.set_title(view);ax.set_ylabel('Cosine to nominal target FIT-D prototype')
    fig.suptitle('SOURCE DEV distributions, 397 unique query rows across three descriptor views')
    fig.savefig(safe(H/'REFERENCE_SCORE_DISTRIBUTIONS.png'),dpi=155);plt.close(fig)
    fig,axs=plt.subplots(1,3,sharey=True,figsize=(11,4),constrained_layout=True)
    for ax,view in zip(axs,VIEWS):
        groups=[[float(r['nominal_minus_other_margin']) for r in scores if r['view']==view and r['associated_r_state']==state and r['nominal_minus_other_margin']!=''] for state in ('D','E')]
        ax.boxplot(groups,tick_labels=['D','E'],whis=(5,95));ax.axhline(0,color='#777',ls='--',lw=1);ax.set_title(view);ax.set_ylabel('Nominal target cosine - fixed other cosine')
    fig.suptitle('SOURCE DEV reference-margin distributions; no D/E threshold chosen')
    fig.savefig(safe(H/'REFERENCE_MARGIN_DISTRIBUTIONS.png'),dpi=155);plt.close(fig)
    def fmt(v):return 'NA' if v is None else f'{v:.4f}'
    table=['|특징|q–r 범위|D / E|E: −참조 cosine AUROC|E: −참조 차이 AUROC|D에서 자기 참조>다른 참조|','|---|---|---:|---:|---:|---:|']
    table += [f"|{r['view']}|{r['q_r_stratum']}|{r['D']} / {r['E']}|{fmt(r['target_cosine_AUROC'])}|{fmt(r['margin_AUROC'])}|{fmt(r['D_nominal_gt_other_fraction'])}|" for r in rows]
    text=['# V249 — 정답 물체 RGB 참조를 준 SOURCE 특징 진단','','**이 결과는 teacher-identity oracle 진단이다. 배포 모델·팔레트 정확도·새 v61 비교가 아니다.**','',
          'V248의 픽셀 보조학습 성공이 기존 판정으로 전이되지 않아, 이미 저장된 특징에 대상 RGB 참조를 연결했을 때의 구분 신호를 확인했다. 기존 q8/context와 same-image context cosine 경로는 이미 시험한 것이며, 이를 새 해결책으로 다시 제안하지 않는다.','',
          '## 고정한 계산','',
          '원 YCB known 1,521행의 좌표·crop·라벨·분할을 유지했다. SOURCE FIT D 841행만 물체별 참조 계산에 사용했다. 각 특징을 원 sample 순서로 FP64 평균하고 L2 정규화했다. SOURCE DEV 397행(D340/E57)은 참조 계산에 쓰지 않았다. 숫자순으로 다음 물체를 다른 참조로 정했으며, 유사도나 DEV 결과로 비교 물체를 고르지 않았다.','',
          '세 특징(전체 crop CLS, 고정 중심 Q, 중심 crop CLS)을 모두 보고한다. 정답 물체 ID는 SOURCE 교사 메타데이터에서 참조를 선택하는 oracle이며 배포 입력이 아니다. 보임 참조도 r 상태가 D인 q crop 특징의 평균이므로, 대상만 잘라낸 깨끗한 template이나 물리 부위 대응을 보장하지 않는다.','',
          'SOURCE FIT 참조 통계는 실제로 적합했다. 새 optimizer update·encoder/head forward·원 RGB/depth/mesh/annotation archive·가중치 읽기·PnP는 없다. 사용한 SOURCE 파생 정답 메타데이터(r·대상 ID·D/E)는 숨기지 않는다. 현재·보호 평가 자료는 사용하지 않았다.','',
          '## 모든 사전 지정 특징의 결과','']+table+['',
          '![전체 및 3px 이하 AUROC](ORACLE_REFERENCE_AUC.png)','',
          '**AUROC는 정확도가 아니다.** 0.5는 이 점수 방향의 순위 구분이 우연 수준이라는 뜻이다. E의 점수 방향을 −참조 cosine 및 −참조 차이로 미리 정했고, 결과를 보고 반전·문턱·best-view를 선택하지 않았다. 비교 점수가 없는 행과 클래스가 없는 stratum은 UNKNOWN/NA로 남겼다.','',
          '![참조 유사도 분포](REFERENCE_SCORE_DISTRIBUTIONS.png)','',
          'YCB의 정답은 q 픽셀 소유자가 아니라 연결된 표면점 r의 상태다. q는 r 주변 반경≤4px의 인공 jitter로 만들어졌으므로 실제 초기 검출 오차를 재현했다고 부르지 않는다. <=1px, >1–3px, <=3px, >3px와 물체별 전체 결과는 원 JSON에 모두 보존했다. 1,191개의 점수 행은 397개 query를 세 특징으로 본 것으로, 독립 표본 1,191개가 아니다.','',
          '## 해석의 한계','',
          '이 단순 참조 점수에서 구분 신호가 있어도 배포 가능한 참조 선택·분류기가 완성된 것은 아니다. 반대로 구분이 약해도 모든 RGB 기반 가림 판정이 불가능하다거나 다른 참조·특징 구조도 실패한다고 증명하지 않는다. SOURCE DEV는 반복 노출된 개발 자료이며, 물체별 지원과 장면·외관의 상관관계를 함께 봐야 한다. physical native 대응과 SELF 의미는 이 진단에서 만들지 않았다. 새 모델 채택과 현재 평가 진입은 없다.','',
          '## 확인 자료','',
          '- [수치 표 CSV](AUC_COMPARISON.csv)',
          '- [모든 오류 범위·물체별 분포 및 지원](artifacts/V249_ROOT_TARGET_RGB_DIAGNOSTIC/oracle_rgb_reference_diagnostic_v1/DESCRIPTIVE_TABLES.json)',
          '- [397 query × 3 특징의 원 점수](artifacts/V249_ROOT_TARGET_RGB_DIAGNOSTIC/oracle_rgb_reference_diagnostic_v1/DEV397_DESCRIPTOR_SCORES.csv)',
          '- [참조 지원 표](artifacts/V249_ROOT_TARGET_RGB_DIAGNOSTIC/oracle_rgb_reference_diagnostic_v1/PROTOTYPE_SUPPORT.csv)',
          '- [데이터 읽기 전 명세](artifacts/V249_ROOT_TARGET_RGB_DIAGNOSTIC/PROTOCOL_BEFORE_SOURCE_REFERENCE_PAYLOADS.json)',
          '- [기존 입력·시도와 차이의 CODE 근거](artifacts/V249_TARGET_IDENTITY_CODE_A/KO.md)',
          '- [공개된 SOURCE 입력 세 개의 SHA 매핑](artifacts/V249_PUBLISHED_DEPENDENCIES_A/dependency_map.json)','']
    safe(H/'PUBLIC_README_KO.md').write_text('\n'.join(text)+'\n')
    tr=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [r['view'],r['q_r_stratum'],f"{r['D']} / {r['E']}",fmt(r['target_cosine_AUROC']),fmt(r['margin_AUROC'])])+'</tr>' for r in rows)
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>V249 SOURCE 참조 진단</title><style>body{font:17px/1.7 sans-serif;max-width:1080px;margin:35px auto;padding:20px;background:#f6f8fb;color:#192335}img{max-width:100%}table{width:100%;border-collapse:collapse}td,th{border:1px solid #ccd4df;padding:9px}.card{background:white;padding:22px;margin:20px 0;border-radius:14px}</style><h1>정답 물체 RGB 참조를 준 특징 진단</h1><div class="card"><b>교사 물체 ID를 사용하는 oracle 진단입니다. 배포 모델·팔레트 정확도가 아닙니다.</b><p>SOURCE FIT 보임 841개로 참조를 계산하고 DEV397개를 평가했습니다. 새 학습 update·문턱 선택·현재 평가·PnP는 없습니다.</p></div><div class="card"><table><tr><th>특징</th><th>오차 범위</th><th>D / E</th><th>−참조 cosine AUROC</th><th>−참조 차이 AUROC</th></tr>'''+tr+'''</table></div><div class="card"><img src="ORACLE_REFERENCE_AUC.png" alt="고정 AUROC 비교"><img src="REFERENCE_SCORE_DISTRIBUTIONS.png" alt="참조 점수 분포"></div><div class="card"><a href="README_KO.md">전체 해석·한계·코드·지원 표</a> · <a href="AUC_COMPARISON.csv">표 CSV</a><p>AUROC는 정확도가 아닙니다. 세 특징을 모두 보고하며 유리한 특징이나 문턱을 선택하지 않았습니다. SOURCE 파생 정답 메타데이터를 참조 선택에 사용했고, 물리 native 대응·SELF·배포 참조 선택은 미확인입니다.</p></div></html>'''
    safe(H/'ROOT_READOUT_KO.html').write_text(page)
    print(json.dumps({'completed':True,'PASS':True,'rows':rows,'unique_DEV_queries':397,'new_optimizer_or_model_forward':0}))

if __name__=='__main__':run()

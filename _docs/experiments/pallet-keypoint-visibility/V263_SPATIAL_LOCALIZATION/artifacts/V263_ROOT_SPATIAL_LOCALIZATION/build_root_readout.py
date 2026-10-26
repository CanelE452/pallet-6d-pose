"""Build an auditable scientific readout from sealed saved results only."""
from pathlib import Path
import csv
import datetime
import hashlib
import html
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
OUT = R/'V263_ROOT_SPATIAL_LOCALIZATION'
A = R/'V263_CURRENT_EVALUATION_A/actual_evaluation_v1'
B = R/'V263_CURRENT_UTILITY_B/current223_utility_v1'
C = R/'V263_SPATIAL_HEAD_C'

def safe(p):
    if '..' in p.parts or any(t in str(p).lower() for t in ['v112','cache','.pyc']):
        raise ValueError(p)
    for a in reversed([p,*p.parents]):
        if a.is_symlink(): raise ValueError(a)
    return p

def load(p): return json.loads(safe(p).read_bytes())
def rows(p):
    with safe(p).open() as f: return list(csv.DictReader(f))
def bind(p):
    b = safe(p).read_bytes()
    return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}
def percent(v): return '—' if v in ('',None) else f'{float(v)*100:.2f}%'
def table(headers, data):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in data)+'</tbody></table>'

def main():
    cls = [r for r in rows(A/'CLASSIFICATION.csv') if r['cohort']=='PRIMARY' and not r['status']]
    loc = [r for r in rows(A/'LOCATION.csv') if r['cohort']=='MANUAL_CLICK_D_ALL']
    utility = rows(B/'REFERENCE_RELATIVE_SUMMARY.csv')
    transitions = rows(B/'REFERENCE_RELATIVE_PAIRED_TRANSITIONS.csv')
    syn = load(C/'FOUR_FIT_COMPLETE.json')
    replay = load(OUT/'ROOT_SAVED_METRIC_REPLAY.json')
    assert replay['PASS']
    def lp(g,m): return next(r for r in loc if r['group']==g and r['model']==m)
    def up(g,m): return next(r for r in utility if r['group']==g and r['arm']==m)
    def cp(g,m): return next(r for r in cls if r['group']==g and r['model']==m)
    groups=['WOOD_PRIMARY105','SQUARE118']; group_labels=['Wood (323 points)','Square (531 points)']
    colors=['#526579','#277da8','#d67c24']; models=['ORIGINAL_QUERY','LOC_ONLY_INITIAL_GEOMETRY','JOINT_SELECTED']; labels=['Original','Location only','Joint']
    fig,axs=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    x=np.arange(2); w=.23
    for i,(m,label,col) in enumerate(zip(models,labels,colors)):
        v=[float(lp(g,m)['all_scorable_accuracy_LE3'])*100 for g in groups]
        bar=axs[0,0].bar(x+(i-1)*w,v,w,label=label,color=col)
        axs[0,0].bar_label(bar,fmt='%.1f',padding=3,fontsize=10)
    axs[0,0].set(xticks=x,xticklabels=group_labels,ylim=(0,100),ylabel='Points within 3 px (%)',title='Real manual-visible points: partial location gain')
    axs[0,0].legend(fontsize=9)
    cms=['CAL_LEGACY','V261_CONTROL','VIS_ONLY_SELECTED','JOINT_SELECTED']; xx=np.arange(len(cms))
    for i,(key,label,col) in enumerate([('D_recall','Visible recall','#526579'),('E_recall','External recall','#277da8'),('E_precision','External precision','#d67c24')]):
        bars=axs[0,1].bar(xx+(i-1)*w,[float(cp('ALL',m)[key] or 0)*100 for m in cms],w,label=label,color=col)
        axs[0,1].bar_label(bars,fmt='%.1f',padding=3,fontsize=8)
    axs[0,1].set(xticks=xx,xticklabels=['Legacy real-CAL','V261 control','New VIS','New joint'],ylim=(0,112),ylabel='Percent',title='Same real 1,237 queries: visibility remains unsolved')
    axs[0,1].legend(fontsize=8)
    for i,(m,label,col) in enumerate(zip(['ORIGINAL','LOC_ONLY_INITIAL_GEOMETRY','JOINT_SELECTED'],labels,colors)):
        v=[int(up(g,m)['success']) for g in groups]
        bars=axs[1,0].bar(x+(i-1)*w,v,w,label=label,color=col)
        axs[1,0].bar_label(bars,padding=3,fontsize=10)
    axs[1,0].set(xticks=x,xticklabels=['Wood (105 frames)','Square (118 frames)'],ylim=(0,118),ylabel='Successful frames',title='Unchanged fixed solver: reference-relative T<5 cm, R<5 deg')
    axs[1,0].legend(fontsize=9)
    idx=np.arange(4); loc_cells=[('WOOD_PRIMARY105','LOC_ONLY_INITIAL_GEOMETRY'),('WOOD_PRIMARY105','JOINT_SELECTED'),('SQUARE118','LOC_ONLY_INITIAL_GEOMETRY'),('SQUARE118','JOINT_SELECTED')]
    rec=[int(lp(g,m)['recovery_at3px']) for g,m in loc_cells]; dam=[-int(lp(g,m)['damage_at3px']) for g,m in loc_cells]
    b1=axs[1,1].bar(idx,rec,color='#25836c',label='Recovered: >3 to <=3 px');b2=axs[1,1].bar(idx,dam,color='#bb4050',label='Damaged: <=3 to >3 px')
    axs[1,1].bar_label(b1,padding=3);axs[1,1].bar_label(b2,padding=3);axs[1,1].axhline(0,color='#555',lw=1)
    axs[1,1].set(xticks=idx,xticklabels=['Wood LOC','Wood joint','Square LOC','Square joint'],ylim=(-60,145),ylabel='Manual-visible point count',title='Keep recovery and damage separate')
    axs[1,1].legend(fontsize=8)
    for ax in axs.flat:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('V263 bounded comparison — existing 223 images are development evaluation',fontsize=16)
    fig.savefig(OUT/'RESULT_OVERVIEW.png',dpi=160);plt.close(fig)
    summary={'completed':True,'created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'scientific_status':'Visibility unresolved; synthetic localization learned; square location transfer improves, wood damage persists; no model adopted.',
             'executed_learned_settings':4,'optimizer_updates_each':1000,'seed':17,'new_real_fit_or_threshold_selection':0,
             'independent_real_validation':False,'current_role':'Repeated development evaluation',
             'source_DEV':syn['SYN_DEV_reports'],'source_rule_selection':syn['JOINT_rule_selection'],
             'source_oracle':load(R/'V263_ORACLE_DIAGNOSTIC_B/CONSTRUCTIVE_FINAL_RESULT.json'),
             'current_primary_classification':cls,'current_manual_visible_location':loc,
             'current_reference_relative_utility':utility,'current_pose_paired_transitions':transitions,
             'additional_real_inventory':{'path':'ADDITIONAL_REAL_DATA_INVENTORY.json','raw_candidates':1505,'source_videos':20,'current_or_green851_encoded_overlap':0,'native_visibility_gold_available':False},
             'visual_observations':['A visible wood corner with initial error<=3px is classified E while its heatmap remains near the corner (CELL01).',
                                    'An externally hidden wood point produces a peak on the foreground/background edge (CELL03).',
                                    'A square visible corner is corrected toward the manual point (CELL10); another already-correct corner moves away (CELL11).'],
             'limitations':['Selected crop examples do not establish population-wide causal frequencies.',
                            'Geometric two-direction witness labels are not independently measured photometric gold.',
                            'Pose success uses the unchanged historical C2/C4 reference-relative contract, not new independent 6D ground truth.',
                            'Additional 1505 real frames have no pointwise D/E gold; exact encoded SHA non-overlap does not certify session independence.'],
             'source_results':[bind(C/'FOUR_FIT_COMPLETE.json'),bind(A/'COMPLETE.json'),bind(B/'COMPLETE.json'),bind(OUT/'ROOT_SAVED_METRIC_REPLAY.json')]}
    (OUT/'FINAL_RESULT_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    cls_data=[]
    for r in cls:
        cls_data.append([r['group'],r['model'],r['truth_D']+'/'+r['truth_E'],f"[{r['D_D']},{r['D_E']},{r['D_U']}] / [{r['E_D']},{r['E_E']},{r['E_U']}]",percent(r['D_recall']),percent(r['E_precision']),percent(r['E_recall']),percent(r['U_rate'])])
    loc_data=[]
    for r in loc:
        e=json.loads(r['final_error_px']);loc_data.append([r['group'],r['model'],r['scorable_N'],percent(r['all_scorable_accuracy_LE3']),r['recovery_at3px'],r['damage_at3px'],percent(r['accepted_accuracy_LE3']),f"{e['median']:.3f} / {e['p90']:.3f}"])
    pose_data=[[r['group'],r['arm'],r['success']+'/'+r['n'],r['fallback'],f"{float(r['T_cm_mean']):.2f} / {float(r['T_cm_P95']):.2f}",f"{float(r['R_deg_mean']):.2f} / {float(r['R_deg_P95']):.2f}",r['canonical_near90']] for r in utility]
    body='''<h1>V263 공간 위치 보정과 가림 비교 결과</h1>
<p><strong>가림 판별 개선은 확인하지 못했습니다. 위치 보정은 정사각형에서 개선됐지만 목재 손상이 남습니다. 네 설정의 제한된 비교를 완료했으며 새 모델은 채택하지 않았습니다.</strong></p>
<p>합성 7,040 query·원 FIT86/DEV24 base 분할, frozen DINOv2 중간층 공간 token, native8 condition, 21,989 parameter head를 사용했습니다. seed17·각1,000 AdamW 업데이트로 LOC 두 crop, 선택 crop의 VIS와 JOINT를 학습했습니다. 현재223장으로 학습·문턱·crop·모델을 고르지 않았습니다. 모든 checkpoint와 설정을 고정한 뒤 현재 전체 예측을 저장하고 정답과 결합했습니다.</p>
<h2>고정 후단 합성 oracle 진단</h2><p>880그룹의 같은 후단에서 ORIGINAL / 가시성 정답만 / 직접 관측 위치 정답만 / 둘 다를 비교했습니다. post의 고정 W/D 정규화 기준 성공은625 / 621 / 850 / 861장이지만, 물리 canonical C1 기준은242 / 243 / 313 / 313장입니다. 위치 개선 여지와 전역 방향·대응 문제를 구분해야 합니다. symmetry UNKNOWN에 임의 C2/C4 최소값을 적용하지 않았습니다. 기존 가상 cuboid 합성1,003장은 원 baseline만 전수 재현했고, 직접 관측 GT가 없는 세 oracle을 대체 정답으로 만들지 않았습니다. <a href="../V263_ORACLE_DIAGNOSTIC_B/constructive_source110_v1/CONSTRUCTIVE_REPORT_KO.md">oracle 전체 통계와 복구·손상</a></p>
<p>합성 DEV의 위치 감독 유효1,284점에서 3px 이내는 771→1,263점, JOINT 중앙오차는2.436→0.503px, 복구492·손상0입니다. 합성의 직접 관측 마스크는 물리적 native anchor와 두 방향의 visible witness 근거이며 독립적인 photometric 정답은 아닙니다.</p>
<img class="full" src="RESULT_OVERVIEW.png" alt="위치 가림 고정후단 복구 손상 비교">
<h2>실사 가림 혼동행렬과 precision recall</h2><p>PRIMARY 총1,237 query(D1,147·E90). 행은 정답 D/E, 열은 예측 D/E/U입니다. recall 분모에 U를 포함합니다. SELF는 외부가림에 합치지 않았으며 별도 CSV에 보존했습니다.</p>'''
    body+=table(['집단','모델','정답 D/E','CM D행 / E행','D recall','E precision','E recall','보류율'],cls_data)
    body+='''<h2>수동 클릭 보임 점의 위치</h2><p>총857점 중 원입력U3점을 제외한854점입니다. 위치 보류는 원q 유지입니다. accepted 정확도는 이 수동 클릭 보임 집단의3px 기준이며 전체 가림 판별 precision과 다릅니다. JOINT 전체3px 이내376→457점, 복구140·손상59입니다. 목재는153→140/323점으로 감소했고 정사각형은223→317/531점으로 증가했습니다. 원오차≤3px인376점에서도 JOINT가102점을 외부가림으로 오판했습니다.</p>'''
    body+=table(['집단','구성','측정가능 N','전체≤3px','복구','손상','accepted≤3px','최종 median / p90 px'],loc_data)
    body+='''<h2>실제 crop에서 확인한 문제</h2><p>미리 정한12개 결과 cell에서 전체 cell 수를 표시하고 최소 row index를 택했습니다. 성공·실패를 설명하는 선택 사례이며 원인별 전체 비율은 아닙니다. 보이는 목재 모서리를 E로 부르는 사례, 가려진 점 대신 주변의 강한 경계를 고르는 사례, 정확하던 점을 옮겨 손상시키는 사례가 있습니다. 합성에서 고른 관측 적합성 규칙의 실사 전이가 충분하지 않습니다.</p>
<p><a href="../V263_CURRENT_EVALUATION_A/current_fixed12_gallery_v1/INDEX_KO.html">12사례 전체와 셀별 census</a> · <a href="../V263_LOCALIZATION_DATA_A/qualitative_fixed16_v1/INDEX_KO.html">합성16사례</a></p>
<div class="cards"><img src="../V263_CURRENT_EVALUATION_A/current_fixed12_gallery_v1/CELL01_CARD.png"><img src="../V263_CURRENT_EVALUATION_A/current_fixed12_gallery_v1/CELL03_CARD.png"><img src="../V263_CURRENT_EVALUATION_A/current_fixed12_gallery_v1/CELL10_CARD.png"><img src="../V263_CURRENT_EVALUATION_A/current_fixed12_gallery_v1/CELL11_CARD.png"></div>
<h2>동일 고정 후단에서의 영향</h2><p>기존 solver·문턱·참조 C2/C4 대칭 계약을 그대로 재생했습니다. 아래는 기존 참조에 상대적인 T&lt;5cm·R&lt;5도 성공이며 독립6D 정답 검증이 아닙니다. ORIGINAL223은 기존 초기 pose와 최대8.81e−10 차이입니다. LOC는 정사각형30→72/118, 목재57→51/105입니다. 목재 LOC의 P95 이동오차84.25→119.58cm와 전체 평균 회전오차7.83→8.30도 악화도 남습니다. JOINT는120/223장에서 원결과 fallback을 사용했습니다.</p>'''
    body+=table(['집단','구성','성공','fallback','T mean/P95 cm','R mean/P95 deg','canonical near90'],pose_data)
    body+='''<h2>추가 실사 자료</h2><p>9월6일20개 영상에서 추출한1,505장 ZIP을 찾았습니다. 현재223장·기존학습851장과 encoded SHA 중복은0입니다. 점별 D/E 정답이 없고 이전 실험 노출은 확인되지 않아 이번 학습·평가에 편입하지 않았습니다. 기존 추가주석215장과194/290/51장 ZIP의 중복·과거 사용 이력은 <a href="ADDITIONAL_REAL_DATA_READOUT_KO.html">조사 표</a>에 있습니다.</p>
<h2>검토와 재현 경로</h2><p><a href="FINAL_RESULT_SUMMARY.json">전체 수치 JSON</a> · <a href="ROOT_SAVED_METRIC_REPLAY.json">독립 산술 재검산12CM·6위치표</a> · <a href="../V263_CURRENT_EVALUATION_A/PORTABLE_CURRENT_REPORT_KO.md">가림·위치 상세 보고</a> · <a href="../V263_SPATIAL_HEAD_C/RUN_GUIDE_KO.md">학습·특징·고정추론 재현 안내</a> · <a href="../V263_CURRENT_EVALUATION_A/actual_evaluation_v1/HUMAN_SELF.csv">SELF 별도표</a></p>
<p>기존223장은 반복 개발 평가입니다. 이번 제한된 비교는 위치 학습의 가능성과 목재·가림·실사 acceptance의 미해결 범위를 보여줍니다. 수치 개선만으로 실제 배포의 신뢰도90%를 확인했다고 해석하지 않습니다.</p>'''
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>V263 공간 위치 보정과 가림 비교 결과</title><style>body{font:16px sans-serif;max-width:1350px;margin:32px auto;padding:0 18px;line-height:1.65;color:#17232b}h1,h2{line-height:1.3}table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}th,td{border:1px solid #bfc9cf;padding:8px;text-align:left}th{background:#edf2f5}tr:nth-child(even){background:#f7f9fa}.full{width:100%}.cards{display:grid;grid-template-columns:1fr 1fr;gap:12px}.cards img{width:100%}a{color:#13659a}</style>'+body+'</html>'
    (OUT/'ROOT_READOUT_KO.html').write_text(page)
    print(json.dumps({'summary':str(OUT/'FINAL_RESULT_SUMMARY.json'),'readout':str(OUT/'ROOT_READOUT_KO.html'),'figure':str(OUT/'RESULT_OVERVIEW.png')}))

if __name__=='__main__': main()

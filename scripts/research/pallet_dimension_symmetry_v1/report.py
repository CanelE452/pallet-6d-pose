"""Readable Korean report, paired comparisons and real image overlays."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-dimsym-mpl')
import html
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from core import *

NOTE=ROOT/'_docs/notes/pose_geometry/dimension-symmetry-2x2-results.md'
TITLE='팔레트 치수 × 대칭 로스 — 네 군 비교'

def main():
    s=read(OUT/'SUMMARY.json');audit=read(OUT/'CACHE_AUDIT.json');train=read(OUT/'TRAINING_COMPLETE.json')
    records=read(OUT/'val_records.json');per=read(OUT/'PER_FRAME.json');pred=np.load(OUT/'predictions.npz')
    font=FontProperties(fname='/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
    plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    names=['stock',*ARMS];values=[];lines=[]
    lines+=['| 방법 | 평균 px | P90 px | 주 지표 | 통과 |','|---|---:|---:|---:|---|']
    cells=[]
    for a in names:
        rr=[s['results'][a]] if a=='stock' else s['results'][a]
        vals=np.array([r['mean_px'] for r in rr]);values.append(vals)
        primary=float(np.mean([r['primary'] for r in rr]));p90=float(np.mean([r['p90_px'] for r in rr]))
        status='기준' if a=='stock' else ('학습 대조' if a=='A_indexed' else ('통과' if s['gates'][a]['PASS'] else '미통과'))
        cells.append([LABELS[a],f'{vals.mean():.4f}',f'{p90:.4f}',f'{primary:.6f}',status])
        lines.append('| '+' | '.join(cells[-1])+' |')
    fig,ax=plt.subplots(figsize=(10,4))
    ax.bar(np.arange(5),[v.mean() for v in values],color=['#8494a4','#4c87bd','#9270b5','#37a495','#e39b47'])
    for i,v in enumerate(values):ax.scatter(np.repeat(i,len(v)),v,c='black',s=18,zorder=4)
    ax.set_xticks(np.arange(5),[LABELS[a] for a in names]);ax.set_ylabel('전체 프레임 평균 오차 px')
    ax.set_title('검출 실패를 포함한 같은 512장 · 점은 각 시드의 값')
    fig.tight_layout();fig.savefig(figures/'arm_comparison.png',dpi=160);plt.close(fig)
    keys=['dims_without_sym','dims_with_sym','sym_without_dims','sym_with_dims','interaction']
    labels=['치수 효과 C-A','치수 효과 D-B','대칭 효과 B-A','대칭 효과 D-C','상호작용 D-C-B+A']
    fig,ax=plt.subplots(figsize=(10,4))
    for i,k in enumerate(keys):
        b=s['contrasts'][k]['bootstrap'];ax.errorbar(b['mean'],i,xerr=[[b['mean']-b['low']],[b['high']-b['mean']]],fmt='o',capsize=4)
    ax.axvline(0,color='gray',linestyle='--');ax.set_yticks(range(5),labels);ax.invert_yaxis()
    ax.set_xlabel('정규화 오차 변화 · 음수가 개선 · 95% 프레임 그룹 bootstrap 구간')
    fig.tight_layout();fig.savefig(figures/'factor_effects.png',dpi=160);plt.close(fig)
    a=np.mean([[r['mean_px'] for r in per[f'A_indexed_seed{i}']] for i in [1,2,3]],axis=0)
    d=np.mean([[r['mean_px'] for r in per[f'D_both_seed{i}']] for i in [1,2,3]],axis=0)
    matched=np.array([r['matched'] for r in per['stock']])
    eligible=np.flatnonzero(matched)
    ranked=eligible[np.argsort((d-a)[eligible])]
    chosen=[('가장 개선',int(ranked[0])),('중간 사례',int(ranked[len(ranked)//2])),('가장 악화',int(ranked[-1]))]
    case_lines=[]
    for ci,(label,i) in enumerate(chosen):
        r=records[i];img=cv2.imread(str(ROOT/r['image']));raw=img[100:-100,100:-100,::-1]
        fig,axs=plt.subplots(1,3,figsize=(15,5))
        for ax,tag,title in zip(axs,['stock','A_indexed_seed1','D_both_seed1'],['기존 YOLO','A 기존 손실 · seed1','D 치수+대칭 · seed1']):
            row=per[tag][i];gt=np.asarray(r['gt_raw']);valid=np.asarray(r['valid'])
            if row['chosen_symmetry']:
                gt=gt[r['perm']];valid=valid[r['perm']]
            points=pred[tag][i];ax.imshow(raw)
            ax.scatter(gt[valid,0],gt[valid,1],c='#46ea79',marker='x',s=48,label='GT')
            ax.scatter(points[:8,0],points[:8,1],edgecolors='#ff55cc',facecolors='none',s=48,label='예측 점')
            for j,(x,y) in enumerate(points[:8]):ax.annotate(str(j),(x,y),color='#ff55cc',fontsize=8)
            ax.set_xlim(0,raw.shape[1]);ax.set_ylim(raw.shape[0],0);ax.axis('off')
            ax.set_title(f"{title}\n평균 {row['mean_px']:.2f}px");ax.legend(loc='lower right',fontsize=8)
        fig.suptitle(f"{label} · {r['id']} · 3시드 평균 D-A {d[i]-a[i]:+.3f}px")
        fig.tight_layout();fig.savefig(figures/f'case_{ci}.png',dpi=140);plt.close(fig)
        case_lines.append(f"![{label}](../../../data/pallet/results/pallet_dimension_symmetry_v1/figures/case_{ci}.png)")
    activation=s['initial_symmetry_activation'];gates=s['gates'];passed=[LABELS[a] for a in ARMS if gates[a]['PASS']]
    verdict=' / '.join(passed)+'에서 제한된 출력층 시험의 개선 기준을 통과했다.' if passed else '이번 출력층 학습 범위에서 안정적 개선 기준을 통과한 군은 없다.'
    descriptions=[]
    check_labels={'all_seed_gain_at_least_1pct':'모든 시드에서 A 대비1% 개선',
                  'all_seed_mean_better_than_stock':'모든 시드에서 기존 YOLO 평균 초과',
                  'all_seed_p90_nonworse_than_A':'모든 시드의 P90 비악화',
                  'paired_bootstrap_upper_below_zero':'개선 구간이0을 포함하지 않음',
                  'no_new_bad_frames':'정상 프레임의 새 큰 오류 없음'}
    for arm in ['B_symmetry','C_dimensions','D_both']:
        g=gates[arm]
        descriptions.append(f"- {LABELS[arm]}: A 대비 시드별 개선율 "+' / '.join(f'{100*x:+.2f}%' for x in g['gain_over_A_per_seed'])+
                            f". 미충족 기준: {', '.join(check_labels[k] for k,v in g['checks'].items() if not v) or '없음'}.")
    text='\n'.join([
        '# '+TITLE,'','2026-09-11. **'+verdict+'**','',
        '**재부팅 없이 GPU를 복구한 뒤 실제 학습을 완료했다.** 4개 군 × 3시드 × 1,000업데이트 = 12,000업데이트. 합성 학습2,048장·검증512장. 실사나 전체 네트워크의 최종 성능 검증은 아니다.','',
        '## 무엇을 비교했나','',
        'A: 기존 손실, B: 대칭만, C: 치수만, D: 둘 다. 기존 YOLO의 마지막 점·sigma 출력층과 동일 초기 FiLM을 학습했다. 백본·검출기·공간 점 특징·BN·flow는 고정했다. 치수 없는 군도 동일 모듈에0벡터를 넣어 파라미터 수를 맞췄다. 모든 군은 최초에 기존 점 출력을 재현한다.','',
        '손실은 한 객체에 대해 기존 OKS·가시성·RLE 항을12/1/1로 사용한다. C2일 때 세 항이 같은 배치 하나를 선택한다. C1은 원래 배치만 허용한다. 이 첫 비교에는 선이나 자세 손실을 넣지 않았다. 배치 전체 RLE가 아닌 객체별 RLE reduction이며, 객체 하나에서 기존 구현과 값·gradient를 대조했다.','',
        '## 동일 평가 결과','',*lines,'',
        '표의 학습군 값은3시드 평균이다. 주 지표는 코너 평균 오차/원본 이미지 대각선이며1에서 제한한다. 최고 confidence 검출을 GT 없이 선택하고 평가에서 IoU≥0.5를 확인한다. 검출 실패에는 이미지 대각선 벌점을 주어 모든512장을 분모에 유지했다. 모든 군이 같은 C1/C2 평가를 사용한다.','',
        f"동일 검출 성공: {s['results']['stock']['matched']}/512장. 검출기는 고정돼 있으므로 검출률 개선 실험은 아니다.",'',
        '![군별 비교](../../../data/pallet/results/pallet_dimension_symmetry_v1/figures/arm_comparison.png)','',
        '## 치수와 대칭의 효과 분리','',
        '![요인별 효과](../../../data/pallet/results/pallet_dimension_symmetry_v1/figures/factor_effects.png)','',
        '**치수 입력의 작은 추가 효과는 관측됐다.** C-A의 주 지표 개선은3시드 모두0.29~0.58%였고, seed 평균의 paired bootstrap 구간도 개선 쪽이었다. 신경망에 넣는 치수만 섞으면 C와 D의 모든 시드가 악화했다. 이때 PnP에는 계속 올바른 치수를 넣었으므로, 이 차이를 PnP의 잘못된 스케일 탓으로 설명할 수 없다. 다만 C/D의 seed1은 학습 전 YOLO 평균보다 나빠, 안정적인 기존 모델 개선으로 채택할 수 없다.','',
        '**대칭을 치수에 더한 추가 효과는 거의 없었다.** D-C의 bootstrap 구간은0을 포함한다. 이 출력층 구조에서 두 요소의 큰 상승효과를 확인한 결과가 아니다.','',
        f"초기 학습 데이터의 C2 객체 {activation['initial_C2_objects']}개 중 공동 손실이 대체 배치를 선택한 것은 {activation['initial_alternate_targets']}개였다. 평균 손실 감소는 {activation['initial_mean_loss_reduction']:.8f}다. 선택 빈도가 낮다면 이미 일관된 점 번호를 갖춘 모델에는 대칭 로스가 바꿀 감독 자체가 적다는 해석이 가능하다. 이는 대칭 감독이 필요 없는 모든 상황을 증명하지 않는다.",'',
        *descriptions,'',
        '## 실제 이미지 사례','',
        '같은 검증 집합에서3시드 평균 D−A를 기준으로 개선·중간·악화 사례를 선택했다. 그림은 seed1이다. 초록×는 해당 방법에 선택된 유효 대칭의 GT, 분홍○는 예측 코너다. **Hough 선은 이번 시험에 없다.**','',*case_lines,'',
        '## GT·실행 무결성','',
        f"- 원본3D 기하의 GT 재투영 최대 오차: 학습 {audit['splits']['train']['reprojection_max_px']:.6f}px, 검증 {audit['splits']['val']['reprojection_max_px']:.6f}px.",
        f"- 기존 출력과 캐시의 최대 차이: {max(audit['splits'][x]['cache_parity_max_px'] for x in ['train','val']):.8f}px.",
        '- 객체의 물리 Y축180도 회전을 현재 점 번호로 변환했고 중심점·모서리 연결 보존을 검사했다. 학습·검증의 선택된 pair_group은 중복되지 않는다.',
        '- 같은 시드에서 네 군의 초기 state 및 모든 배치 순서 해시가 동일하다. 마지막1,000번째 업데이트를 평가했으며 검증으로 checkpoint를 선택하지 않았다.',
        '- 비교·판정 조건은 학습 전에 기록했다. 동일 CAD 계열 및 기존 검증 데이터 재사용이므로 새로운 팔레트 종류·새 실사 일반화로 주장하지 않는다. C4는 없다.',
        '- PnP에는 모든 군에서 올바른 치수를 사용했다. GT-free 두 앞면 축 가설의 translation 보조 진단이며 물리적 면 방향/rotation 식별을 해결했다는 지표가 아니다.',
        '- 사용자 수작업 실사 GT를 이번에 검사한 결과가 아니다. 기존 annotation과 체크포인트는 수정하지 않았다.','',
        '## 판단 범위','',
        verdict+' 이 결과는 학습된 공간 특징을 고정한 마지막 출력층 시험이다. 전체 YOLO의 특징까지 치수와 대칭으로 다시 학습한 결과로 확대하지 않는다. 치수 효과의 존재와 새 논문 아키텍처의 기여도 별개다. 다음 실험은 이 비교의 요인별 결과를 근거로 선택해야 하며, Hough를 추가했다는 이유만으로 개선을 기대하지 않는다.','',
        '## 실행 파일','',
        '- [프로토콜](../../../data/pallet/results/pallet_dimension_symmetry_v1/PROTOCOL.json)',
        '- [수치·비교·판정 전체](../../../data/pallet/results/pallet_dimension_symmetry_v1/SUMMARY.json)',
        '- [원문과 과거 실행 대조](shared-dimension-symmetry-review.md)',
        '- [재부팅 없는 GPU 복구 기록](../../handoff/2026-09-11_gpu_driver_recovery.md)',''])
    NOTE.write_text(text)
    local=text.replace('../../../data/pallet/results/pallet_dimension_symmetry_v1/','')
    local=local.replace('(shared-dimension-symmetry-review.md)','(../../../../_docs/notes/pose_geometry/shared-dimension-symmetry-review.md)')
    local=local.replace('(../../handoff/2026-09-11_gpu_driver_recovery.md)','(../../../../_docs/handoff/2026-09-11_gpu_driver_recovery.md)')
    (OUT/'REPORT_KO.md').write_text(local)
    table='<table><tr>'+''.join('<th>'+x+'</th>' for x in ['방법','평균 px','P90 px','주 지표','통과'])+'</tr>'
    table+=''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in cells)+'</table>'
    body=f'<h1>{TITLE}</h1><p><b>{html.escape(verdict)}</b></p><p>재부팅 없는 GPU 복구 완료 · 실제12모델 / 12,000업데이트 · 합성2,048장 학습 / 512장 평가</p><p>기존 YOLO의 마지막 출력층 비교입니다. 실사·전체 네트워크·Hough 성능 검증은 아닙니다.</p>'+table
    body+='<h2>세 시드와 요인별 비교</h2><img src="figures/arm_comparison.png"><img src="figures/factor_effects.png">'
    body+=f'<p>초기 C2 객체 {activation["initial_C2_objects"]}개 중 대체 배치 선택 {activation["initial_alternate_targets"]}개.</p>'
    body+='<h2>실제 이미지 사례</h2><p>초록× GT · 분홍○ 예측 점 · seed1 표시. Hough 선은 이번 시험에 없습니다.</p>'
    body+=''.join(f'<img src="figures/case_{i}.png">' for i in range(3))
    body+='<h2>해석 및 검사</h2><pre>'+html.escape(text)+'</pre>'
    (OUT/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>'+TITLE+'</title><style>body{font-family:sans-serif;max-width:1200px;margin:30px auto;padding:20px;background:#111b27;color:#e9eef5;line-height:1.7}table{border-collapse:collapse;width:100%}th,td{padding:12px;border-bottom:1px solid #41546b;text-align:right}td:first-child,th:first-child{text-align:left}img{width:100%;background:white;border-radius:8px;margin:12px 0}pre{white-space:pre-wrap;font-size:14px}h1,h2{color:#84d5e1}</style>'+body+'</html>')
    write(OUT/'REPORT_COMPLETE.json',dict(PASS=True,note=str(NOTE.relative_to(ROOT)),table_cells=[x for row in cells for x in row],
        summary_sha256=sha(OUT/'SUMMARY.json'),figures=5,title=TITLE))

if __name__=='__main__':main()

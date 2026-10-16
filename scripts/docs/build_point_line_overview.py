"""Documentation figures from frozen experiment JSON; no inference or training."""
from pathlib import Path
import os,json,hashlib,shutil
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-docs-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'_docs/assets/point_line_fusion'
def read(path):return json.loads((ROOT/path).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True);base='data/pallet/results/';inputs=[base+'pallet_identifiability_v24/SUMMARY.json',base+'pallet_vector_v18/SUMMARY.json',base+'pallet_identifiability_v24/CLASSIFICATION.json',base+'pallet_identifiability_v24/TINY_COMPLETE.json',base+'pallet_quality_v23/MATCHED_DIAGNOSIS.json'];s,v,cl,t,match=[read(p) for p in inputs]
    plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False,'font.size':11});conds=[('corner20','hidden','코너 가림'),('crop20','outside','잘림20%'),('crop40','outside','잘림40%')]
    methods=[('stock','기존 YOLO',s),('base','V12 점 기준',s),('point_vector','V18 점 보정',v),('line_vector','V18 점+선 보정',s),('unit_bce','V22 적용률',s),('unit_mass','V23 균형 학습',s),('unit_sign','V24 방향 점수',s)];rows=[]
    for arm,label,data in methods:
        values=[]
        for c,k,_ in conds:
            r=data['results']['real/'+c];rr=[r['stock']] if arm=='stock' else [r[f'{arm}_seed{i}'] for i in (1,2,3)];values.append(float(np.mean([x['strata'][k]['mean_px'] for x in rr])))
        rows.append(dict(method=arm,label=label,error_px=values))
    values=np.array([r['error_px'] for r in rows]);delta=values-values[1];fig,ax=plt.subplots(figsize=(11,5.5));lim=np.max(np.abs(delta));im=ax.imshow(delta,cmap='RdYlGn_r',vmin=-lim,vmax=lim,aspect='auto');ax.set_xticks(range(3),[x[2] for x in conds]);ax.set_yticks(range(len(rows)),[r['label'] for r in rows])
    for i in range(len(rows)):
        for j in range(3):ax.text(j,i,f'{values[i,j]:.2f}px\n({delta[i,j]:+.2f})',ha='center',va='center',fontsize=11,color='white' if abs(delta[i,j])>.75*lim else 'black')
    ax.set_title('어려운 조건의 개선과 약한 잘림의 악화가 함께 나타남\n셀: 실제 오차 · 괄호: V12 점 기준 대비 차이',pad=16);fig.colorbar(im,ax=ax,label='기준 대비 오차 차이px · 음수가 개선');fig.tight_layout();fig.savefig(OUT/'performance.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.8));rr=[x for x in t['rows'] if x['arm']=='unit_tiny'];steps=[x['step'] for x in rr[0]['curve']]
    for split,color,label in [('train','#2563eb','학습한 원본12장'),('check','#d97706','tiny 학습에서 제외한48장')]:
        vals=np.array([[p[split]['all']['regret_px'] for p in r['curve']] for r in rr]);axes[0].plot(steps,vals.mean(0),color=color,label=label);axes[0].fill_between(steps,vals.min(0),vals.max(0),color=color,alpha=.15)
    axes[0].set_yscale('log');axes[0].set_yticks([.001,.01,.1,1],['0.001','0.01','0.1','1']);axes[0].set_ylim(.001,1.5);axes[0].set_title('작은 집합은 맞추지만, 다른 원본으로 이전은 제한');axes[0].set_xlabel('학습 업데이트');axes[0].set_ylabel('GT 최적 적용률 대비 추가 오차px');axes[0].legend(fontsize=9)
    auc_rows=[]
    for j,(c,k,label) in enumerate(conds):
        for z,(split,color) in enumerate([('training','#2563eb'),('real','#d97706')]):
            auc=np.array([cl['results'][split+'/'+c]['models'][f'unit_sign_seed{i}'][k]['auc'] for i in (1,2,3)]);x=j+(z-.5)*.24;axes[1].bar(x,auc.mean(),width=.22,color=color,alpha=.75,label=('합성 학습' if z==0 else '실사 평가') if j==0 else None);axes[1].scatter(x+np.array([-.04,0,.04]),auc,color='black',s=16,zorder=3);axes[1].text(x,auc.max()+.025,f'{auc.mean():.2f}',ha='center',fontsize=10);auc_rows.append(dict(condition=c,split=split,per_seed=auc.tolist()))
    axes[1].axhline(.5,color='gray',ls='--');axes[1].set_xticks(range(3),[x[2] for x in conds]);axes[1].set_ylim(0,1);axes[1].set_ylabel('방향 판별 AUROC · 클수록 좋음');axes[1].set_title('약한 잘림의 방향 판단이 실사로 이전되지 않음');axes[1].legend(fontsize=9)
    for ax in axes:ax.grid(axis='y',alpha=.15)
    fig.text(.5,.01,'왼쪽 음영: 3시드 범위 · 오른쪽 검은 점: 개별 시드 · 신뢰구간은 아님',ha='center',fontsize=9);fig.tight_layout(rect=(0,.04,1,1));fig.savefig(OUT/'learnability_and_transfer.png',dpi=170);plt.close(fig)
    source=ROOT/(base+'pallet_quality_v23/figures/case_4.png');shutil.copyfile(source,OUT/'wrong_direction_example.png')
    data=dict(performance=rows,conditions=[x[0] for x in conds],auc=auc_rows,matched5px=[r for r in match['rows'] if r['matching']=='point_id_and_5px_bins'],sources={p:sha(ROOT/p) for p in inputs},image_source=dict(path=str(source.relative_to(ROOT)),sha256=sha(source)),scope='Documentation-only aggregation of existing3seed results. No new training, inference, selection or statistical test. Tiny band is min/max; AUROC dots are individual seeds. Real297/9sessions reusedDEV.')
    (OUT/'FIGURE_DATA.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');(OUT/'FIGURES.json').write_text(json.dumps(dict(source_script=str(Path(__file__).relative_to(ROOT)),source_sha256=sha(Path(__file__)),figures={p.name:sha(p) for p in OUT.glob('*.png')},input_binding='FIGURE_DATA.json'),ensure_ascii=False,indent=2)+'\n');print('FIGURES COMPLETE',len(rows),'methods')
if __name__=='__main__':main()

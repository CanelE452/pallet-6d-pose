"""V173 saved-result scientific readout only; no source RGB/mask/depth reread."""
from pathlib import Path
import hashlib,html,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

V=Path(__file__).resolve().parent.parent;OUT=V/'scientific_readout_v1'
COMPLETE=V/'literal_raster_census_v1/COMPLETE.json';EXPECTED='025ea05d354404046bcaf92cef0af29653c77defd7e0a4c9faa7d9115fd29a96'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def checked(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def summary(a,mask):
    x=a[mask];known=np.isfinite(x).all(1);mass=x[known].sum(0);missing=int((~known).sum());N=int(mask.sum())
    assert np.isclose(mass.sum()+missing,N,atol=2e-9,rtol=0)
    return dict(N_all=N,N_measured=int(known.sum()),N_missing=missing,weighted_mass=mass.tolist(),fraction_of_all=(mass/N).tolist() if N else None,missing_fraction=missing/N if N else None)
def main():
    assert sha(COMPLETE)==EXPECTED;c=read(COMPLETE);assert c['PASS'] and c['counts']['rows']==6144
    packets={}
    for split,b in c['sidecars'].items():
        with np.load(checked(b),allow_pickle=False) as z:packets[split]={k:z[k].copy() for k in z.files}
    allpacket={k:np.concatenate([packets['TRAIN'][k],packets['VALIDATION'][k]]) for k in ('scene_owner_point','BEFORE_owner_point','joint_point','scene_owner_central4','joint_central4','scene_owner_central32','joint_central32','original_semantic_state','query_available','query_xy_missing','L_missing','selected_index_missing','original_global_index')}
    assert np.array_equal(allpacket['original_global_index'],np.arange(6144))
    owner=c['owner_class_order'];joint=c['joint_class_order'];cohorts={};cross={}
    for split,p in [('ALL',allpacket),('TRAIN',packets['TRAIN']),('VALIDATION',packets['VALIDATION'])]:
        mask=np.ones(len(p['original_semantic_state']),bool);N=len(mask)
        cohorts[split]=dict(N_all=N,N_input_available=int(p['query_available'].sum()),N_inputU=int((~p['query_available']).sum()),N_null_q=int(p['query_xy_missing'].sum()),N_null_L=int(p['L_missing'].sum()),N_null_selected=int(p['selected_index_missing'].sum()),semantic_counts={s:int((p['original_semantic_state']==s).sum()) for s in 'DESU'},
            measures={k:summary(p[k],mask) for k in ('scene_owner_point','BEFORE_owner_point','joint_point','scene_owner_central4','joint_central4','scene_owner_central32','joint_central32')})
        cross[split]={s:summary(p['scene_owner_point'],p['original_semantic_state']==s) for s in 'DESU'}
    assert cohorts['ALL']['N_inputU']==88 and cohorts['ALL']['N_null_q']==56
    numeric=OUT/'SOFT_MASS_READOUT_COUNTS.json';write(numeric,dict(completed=True,owner_class_order=owner,joint_class_order=joint,denominator='All original rows. Missing adds one missing unit; inputU is an overlapping original flag, never added as a second mass.',cohorts=cohorts,semantic_cross_owner_weighted_mass=cross,hard_label_cutoff_used=False))
    plt.rcParams.update({'font.size':12,'axes.titlesize':16,'axes.labelsize':12})
    fig=plt.figure(figsize=(23,14.8));grid=fig.add_gridspec(2,2,height_ratios=[.95,1.15],hspace=.57,wspace=.22)
    owner_labels=['Incident target\n(declared CF)','Other target','Foreign','Background','Ambiguous','Invalid support','Missing q']
    joint_labels=['Visible target','External over\nbefore-target','Foreign without\nbefore-target','Background both','Ambiguous either','Inconsistent\nchange','Invalid support','Missing q']
    owner_colors=['#3ba272','#5470c6','#ee6666','#c9cdd3','#9a60b4','#333333','#f4c95d'];joint_colors=['#3ba272','#d84a5b','#f39c4a','#c9cdd3','#9a60b4','#8b5a3c','#333333','#f4c95d']
    folds=['ALL','TRAIN','VALIDATION'];ylabels=[]
    for split in folds:
        r=cohorts[split];tag={'ALL':'ALL','TRAIN':'TRAIN source / apparent','VALIDATION':'VALIDATION source'}[split]
        ylabels.append(f"{tag}\nN all={r['N_all']}; available={r['N_input_available']}\ninputU={r['N_inputU']}; null q={r['N_null_q']}")
    for col,key,labels,colors,title in [(0,'scene_owner_point',owner_labels,owner_colors,'A  Original-q owner: soft mass / all original rows'),(1,'joint_point',joint_labels,joint_colors,'B  Same-pixel causal conjunction: soft mass / all rows')]:
        ax=fig.add_subplot(grid[0,col]);values=[]
        for split in folds:
            r=cohorts[split]['measures'][key];values.append(np.r_[r['weighted_mass'],r['N_missing']]/r['N_all']*100)
        values=np.array(values);left=np.zeros(3)
        for j,(label,color) in enumerate(zip(labels,colors)):
            ax.barh(np.arange(3),values[:,j],left=left,color=color,label=label,edgecolor='white',linewidth=.6,height=.62,hatch='///' if j==len(labels)-1 else None)
            for i,w in enumerate(values[:,j]):
                if w>=5:ax.text(left[i]+w/2,i,f'{w:.1f}%',ha='center',va='center',fontsize=12,color='white' if color in ('#5470c6','#d84a5b','#9a60b4','#333333','#8b5a3c') else '#1b2637')
            left+=values[:,j]
        ax.set_yticks(np.arange(3),ylabels,fontsize=11);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Weighted proportion of N all (%)');ax.set_title(title,loc='left',pad=15);ax.spines[['top','right']].set_visible(False)
        ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.49,-.13),frameon=False,fontsize=10,handlelength=1.3,columnspacing=1.1)
    for col,split,letter in [(0,'ALL','C'),(1,'VALIDATION','D')]:
        ax=fig.add_subplot(grid[1,col]);masses=[];percents=[];yl=[]
        for s in 'DESU':
            r=cross[split][s];m=np.r_[r['weighted_mass'],r['N_missing']];masses.append(m);percents.append(m/r['N_all']*100 if r['N_all'] else np.zeros(7));yl.append(f"Original semantic {s}\nN all={r['N_all']}; measured={r['N_measured']}")
        masses=np.array(masses);percents=np.array(percents);im=ax.imshow(percents,vmin=0,vmax=100,cmap='Blues',aspect='auto')
        for i in range(4):
            for j in range(7):ax.text(j,i,f'{masses[i,j]:.1f}\n({percents[i,j]:.1f}%)',ha='center',va='center',fontsize=11,color='white' if percents[i,j]>60 else '#172334')
        ax.set_xticks(np.arange(7),owner_labels,fontsize=10);ax.set_yticks(np.arange(4),yl,fontsize=11);ax.set_title(f'{letter}  {split}: semantic × q-owner weighted mass',loc='left',pad=16)
        ax.set_xlabel('Cell = fractional row-mass (percentage within original semantic row)');fig.colorbar(im,ax=ax,fraction=.035,pad=.025,label='Within-semantic weighted mass (%)')
    fig.suptitle('V173 | Complete OLD6144 literal-raster source census',fontsize=23,x=.035,ha='left',y=.985)
    fig.text(.035,.945,'No model scores or hard owner cutoffs. Native semantic D/E/S/U and literal query ownership are separate observations; these are not accuracy/confusion matrices.',fontsize=13)
    fig.text(.035,.045,'All 6144 rows retained: TRAIN4096 + VALIDATION2048; original inputU88 and null q56 remain in the denominators. InputU overlaps measured/invalid/missing categories.',fontsize=12)
    fig.text(.035,.022,'INCIDENT uses declared CF registration; initializer physical native identity is uncertified. No relabel, q/L correction, training admission, source qualification change or model-success claim.',fontsize=12)
    fig.subplots_adjust(left=.135,right=.97,top=.90,bottom=.135)
    png=OUT/'FULL6144_SOFT_OWNER_CAUSAL_SEMANTIC_CENSUS.png';fig.savefig(png,dpi=140,facecolor='white');plt.close(fig)
    results=dict(completed=True,phase='V173',publication_ready=False,pending_final_bindings=['independent actual census verification by A','root actual scientific PNG visual QA'],
        conclusion_KO='원 V129 합성 source6144 전체에서 원 semantic 역할과 literal q 소속·같은위치 외부가림을 별도 수치로 확인했다. 이는 라벨 오류·전체 실패 원인·모델 성공 판정이 아니며 기존 source qualification과 전체 목표 미해결 상태를 바꾸지 않는다.',
        counts=c['counts'],soft_owner_class_order=owner,joint_causal_class_order=joint,cohorts=cohorts,original_semantic_cross_q_owner_weighted_mass=cross,
        interpretation_KO=['수치는 hard owner label 수가 아니라 float64 bilinear onehot weighted mass다. 원6144 전체분모와 측정가능분모를 모두 표시하고 U88/null56을 보존했다.','SCENE foreign mass 중 BEFORE target이 있던 같은 소스픽셀만 EXTERNAL_OVER_BEFORE_TARGET이다. BEFORE BG 위 foreign은 FOREIGN_WITHOUT_BEFORE_TARGET으로 분리했다. marginal 확률의 곱을 쓰지 않았다.','TRAIN은 기존 학습 source의 apparent 진단, VALIDATION은 원source 고정분할 진단이다. 이 단계에서 새 모델학습·성능평가·문턱탐색은 없다.','INCIDENT는 선언 CF-to-canonical 등록 부위에 상대적인 표현이며 초기query의 물리 native identity를 인증하지 않는다. correspondence_boundary는 CF/V4 순서 불일치로 pixelboundary와 다르다.','전수범위는 원 합성 V129 corpus이다. 실제 데이터 일반화·human physical native visibility gold·라벨오류 또는 모든 실패 원인을 증명하지 않는다.'],
        bindings=dict(actual_census_complete=bind(COMPLETE),numeric_counts=bind(numeric),figure=bind(png),original_full6144_rows=c['rows'],fixedsplit_sidecars=c['sidecars'],all768_scene_diagnostics=c['scene_diagnostics'],original_group_aggregates=c['aggregates'],report_code=bind(__file__)),
        source_qualification_changed=False,original_labels_qL_flags_and_splits_unchanged=True,training_admission=False,new_model_fit_forward_threshold_or_label_selection=0,new_original_scene_RGB_mask_depth_or_forbidden_bodies_read_for_report=0,selected_model=None,adopted=False,fully_solved=False,goal_complete=False,HTML_browser_rendered=False)
    rp=V/'ROOT_RESULTS_KO.json';write(rp,results)
    rowshtml=[]
    for split in folds:
        r=cohorts[split];om=r['measures']['scene_owner_point']['weighted_mass'];jm=r['measures']['joint_point']['weighted_mass']
        rowshtml.append('<tr>'+''.join('<td>'+html.escape(str(z))+'</td>' for z in [split,r['N_all'],r['N_input_available'],r['N_inputU'],r['N_null_q'],f'{om[2]:.3f}',f'{jm[1]:.3f}',f'{jm[2]:.3f}'])+'</tr>')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V173 전체 source query census</title><style>body{font:16px/1.65 sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;color:#18283b}h1{font-size:27px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccd4df;padding:8px}th{background:#eef2f8}img{width:100%;height:auto}.note{background:#edf3fa;padding:14px}small{color:#46556a}</style><h1>V173 · 원 합성 source6144 전수 query 진단</h1>
<p class="note">192base ×4variant ×8native를 전부 보존했습니다. 원 semantic D/E/S/U, literal q 소속, 같은 위치의 target 외부가림을 분리했습니다. <strong>새 학습·라벨/좌표 교정·학습 편입은 없으며 전체 목표는 미해결입니다.</strong></p>
<p>아래 숫자는 개수로 hard분류한 값이 아니라 onehot bilinear <strong>weighted mass</strong>입니다. 소스픽셀별 BEFORE/SCENE conjunction을 먼저 만들었고 marginal 확률을 곱하지 않았습니다. foreign이 원target을 덮은 경우와 원background 위에 놓인 경우를 나눴습니다. TRAIN은 기존 학습 source의 apparent 진단이고 VALIDATION은 원 고정source 분할이며, 모델 성공률이 아닙니다.</p>
<table><tr><th>분할</th><th>N 전체</th><th>N 입력가능</th><th>원 inputU</th><th>null q</th><th>q FOREIGN mass</th><th>실제 BEFORE target 위 외부 mass</th><th>BEFORE BG 위 FOREIGN mass</th></tr>'''+''.join(rowshtml)+'''</table>
<p>U88은 measured/invalid/missing 범주와 겹치는 기존 flag로서 별도로 유지했습니다. null56은 JSONnull/NPZ NaN+명시flag이며 가짜0이나background로 바꾸지 않았습니다. 그림의 분모는 전체행이고 각 교차표에 측정가능 N도 함께 표시했습니다. 원S/U semantic도 삭제하지 않았습니다.</p>
<a href="scientific_readout_v1/FULL6144_SOFT_OWNER_CAUSAL_SEMANTIC_CENSUS.png"><img src="scientific_readout_v1/FULL6144_SOFT_OWNER_CAUSAL_SEMANTIC_CENSUS.png" alt="전체 및 분할별 q-owner causal7 soft 분포와 원semantic 교차표"></a>
<p>INCIDENT는 선언된 CF 등록 부위 기준이며 initializer의 물리 native identity 인증이 아닙니다. CF/V4 correspondence_boundary는 순서 불일치로 pixelboundary가 아닙니다. 전수진단 범위는 원합성 corpus이며 human physical native gold, 라벨 오류, 전체 실패 원인 또는 실제 일반화 성공을 증명하지 않습니다. 기존 source qualification·모델·문턱은 바뀌지 않았습니다.</p>
<p><a href="ROOT_RESULTS_KO.json">결과 JSON</a> · <a href="scientific_readout_v1/SOFT_MASS_READOUT_COUNTS.json">그림의 정확한 수치</a> · <a href="literal_raster_census_v1/COMPLETE.json">실제 census 완료 기록</a> · <a href="literal_raster_census_v1/ALL6144_LITERAL_RASTER_SOURCE_DIAGNOSTICS.jsonl">원6144행 전체</a></p>
<small>독립 실제검산과 root의 PNG 실제 시각 QA 최종 바인딩은 대기 중입니다. 이 보고서 작성은 저장결과만 사용했으며 원scene/RGB/mask/depth·모델·private/current/V169/V112를 새로 읽지 않았습니다. HTML 브라우저 렌더링은 수행하지 않았습니다.</small></html>'''
    hp=V/'ROOT_READOUT_KO.html'
    with hp.open('x') as f:f.write(page+'\n')
    phase=dict(completed=True,publication_ready=False,phase='V173',actual_complete=bind(COMPLETE),results=bind(rp),readout=bind(hp),figure=bind(png),numeric_counts=bind(numeric),pending_final_bindings=results['pending_final_bindings'],source_qualification_changed=False,new_training_or_admission=False,labels_qL_splits_inputU_preserved=True,selected_model=None,fully_solved=False,goal_complete=False,HTML_browser_rendered=False)
    pp=V/'ROOT_PHASE_RESULT.json';write(pp,phase)
    print(json.dumps(dict(results=bind(rp),readout=bind(hp),phase=bind(pp),figure=bind(png))),flush=True)
if __name__=='__main__':main()

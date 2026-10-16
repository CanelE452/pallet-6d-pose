"""Display actual line-only correction and independent point-hint bypass."""
from probe28 import *
import os,cv2,markdown
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-v28-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(KIT.parent/'pallet_direction_utility_v27_cases'))
from show_cases import clipline
from features25 import actual_image
NOTE=ROOT/'_docs/notes/point_line_fusion/04_failure_analysis/pallet_line_only_v28.md'
NAMES={'base':'기존 점','actual_v27':'기존 V27 점+선','line_mix':'선 후보만·학습 이동량','line_keep':'유지+선 후보','line_mix_full_amount':'선 후보만·이동량100%','top_line_full':'상위1선·이동량100%','line_oracle_GT':'GT 최적 선 후보·이동량'}

def main():
    s=read(RUN/'SUMMARY.json');dh=read(RUN/'DHT_ONLY.json');c=s['case'];row=c['row'];im=actual_image(row);raw=PREP.raw_image(row);gt=np.array(c['gt']);h,w=im.shape[:2];figdir=RUN/'figures';figdir.mkdir(exist_ok=True)
    def avg(arm,condition,key='mean_px'):return float(np.mean([r[key] for r in s['rows'] if r['condition']==condition and r['method']==arm]))
    fig,axes=plt.subplots(1,3,figsize=(14,3.8),layout='constrained');box=np.array(row['box']);top_ids={}
    for ax,mode in zip(axes,('guided','zero_point_maps','gate_bypass')):
        ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));v=dh['variants'][mode];ll=np.array(v['lines']);mass=np.array(v['mass']);ids=np.argsort(-mass,kind='stable')[:6];top_ids[mode]=ids.tolist();rect=(0,max(0,box[1]-40),w,min(h,box[3]+40))
        for k,line in enumerate(ll):
            seg=clipline(line,rect)
            if seg is not None:ax.plot(*np.array(seg).T,c='#eeeeee',alpha=.18,lw=.6)
        for rank,k in enumerate(ids):
            seg=clipline(ll[k],rect)
            if seg is not None:ax.plot(*np.array(seg).T,c=plt.cm.tab10(rank),lw=1.5,label=f'line{k} / mass{mass[k]:.2f}')
        ax.set_xlim(rect[0],rect[2]);ax.set_ylim(rect[3],rect[1]);ax.set_title({'guided':'Existing point-guided DHT','zero_point_maps':'Point maps zeroed; gate retained','gate_bypass':'Explicit point-hint gate bypassed'}[mode],fontsize=10);ax.legend(fontsize=7,loc='upper right');ax.set_xlabel('No point decoding shown')
    fig.suptitle('Same cropped input and DHT weights / all36lines faint; top6mass colored\nRight panel generates lines without point coordinates; not a newly trained line-only detector',fontsize=11);fig.savefig(figdir/'dht_only.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained');ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));v=dh['variants']['gate_bypass'];ll=np.array(v['lines']);mass=np.array(v['mass']);ids=top_ids['gate_bypass'];rect=(0,max(0,box[1]-40),w,min(h,box[3]+40))
    for k,line in enumerate(ll):
        seg=clipline(line,rect)
        if seg:ax.plot(*np.array(seg).T,c='#eeeeee',alpha=.25,lw=.65)
    for rank,k in enumerate(ids):
        seg=clipline(ll[k],rect)
        if seg:ax.plot(*np.array(seg).T,c=plt.cm.tab10(rank),lw=1.8,label=f'DHT line{k}')
    ax.set_xlim(rect[0],rect[2]);ax.set_ylim(rect[3],rect[1]);ax.legend(fontsize=8);ax.set_title('DHT lines only / explicit point-hint gate bypassed\nSame cropped input and weights; no point decoding or new training',fontsize=11);fig.savefig(figdir/'dht_bypass_only.png',dpi=170);plt.close(fig)
    points={k:np.array(v) for k,v in c['points'].items()};shown=('base','actual_v27','line_mix','line_mix_full_amount','line_oracle_GT');colors=['#168bff','#ed3030','#ff9800','#7251b5','#d100bb'];markers=['o','s','D','v','*'];lines=np.array(c['lines']);allp=np.stack([gt]+[points[k] for k in shown]);low=allp.min(0)-15;high=allp.max(0)+15;center=(low+high)/2;span=max((high-low).max(),60);rect=(center[0]-span/2,center[1]-span/2,center[0]+span/2,center[1]+span/2)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,limits in zip(axes,[(min(-180,low[0]),max(0,box[1]-30),w,h),rect]):
        ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));ax.set_facecolor('#e0e3e7')
        for j,line in enumerate(lines):
            seg=clipline(line,limits)
            if seg:ax.plot(*np.array(seg).T,c='#607d8b',lw=1,ls='--',alpha=.7)
            seg=clipline(line,(0,0,w,h))
            if seg:ax.plot(*np.array(seg).T,c='#607d8b',lw=1,alpha=.7)
        ax.scatter(*gt,c='#00ae3d',marker='x',s=90,lw=2,label='GT',zorder=7)
        for key,color,marker in zip(shown,colors,markers):ax.scatter(*points[key],s=90 if key=='line_mix' else 60,c=color,marker=marker,edgecolors='white',lw=.7,label=key+f' {c["errors"][key]:.2f}px',zorder=6)
        ax.set_xlim(limits[0],limits[2]);ax.set_ylim(limits[3],limits[1]);ax.axvline(0,color='black',ls=':');ax.set_aspect('equal');ax.set_xlabel('Original image px; grey = cropped-out area')
    axes[0].set_title('Existing guided DHT candidates; point correction removed',fontsize=10);axes[1].set_title('Line-only amounts still predicted by the existing V27 head',fontsize=10);handles,labels=axes[1].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8,bbox_to_anchor=(.5,-.08));fig.suptitle('Case3 / frame000546 / point0: actual V27 78.80px -> line candidates only12.08px\nThis correction still uses the base point and its line association; it is not point-free pose decoding',fontsize=11);fig.savefig(figdir/'case_correction.png',dpi=170,bbox_inches='tight');plt.close(fig)
    table=[dict(method=a,values=[avg(a,cond) for cond in ('clean','corner20','crop10','crop20','crop30','crop40')]) for a in METHODS];move=avg('line_mix','crop20','mean_move_px');still=avg('line_mix','crop20','near_zero_move_fraction')
    text=['# V28: 3번 사례에 선만 적용한 결과','', '**3번 사례는 선 후보만 사용하면 실제 오차가78.80→12.08px로 줄었다. 기존 점14.19px보다도 작다. 다만 선만 사용하는 방식이 전체에서 안정적으로 개선된 것은 아니다.**','',
    '## 현재 점이 선에 힌트를 주는가','',
    '**그렇다. 두 단계에서 직접 사용한다.** 원래 YOLO 점 주변을3개 크기의 거리 맵으로 만들고 이미지 특징과 결합해 DHT 앞의 특징 가중치를 조절한다. 선을 만든 뒤에도 점 주변 특징·점과 선의 거리로 대응 점수를 만들며, V27에서는 점 위치/문맥/보정벡터와 선 특징을 함께 사용한다. 따라서 현재 점 경로와 선 경로는 독립 전문가가 아니다.','',
    '코드 근거: `pallet_point_attention_v9/model.py`의 `point_maps`·`GuidedDHT.forward`, `pallet_optional_lines_v11/optional_model.py`의 `predict_lines`, `pallet_point_tokens_v10/refiner.py`의 point query/line association, `pallet_direction_utility_v27/model27.py`의 후보 효용 헤드. 점을 서로 연결해서 선을 만드는 구조는 아니다.','',
    '## 먼저 실행한 것: 점 보정 후보를 빼고 선 후보만','',
    '기존 V27의 유지·점 벡터·x/y 후보를 제거하고, 선 법선/비평행 선 쌍 후보의 기존 가중치만 재정규화했다. 이동량은 학습된 값을 사용했다. **기준 점 위치와 점에 따른 선 선택은 남아 있다.** 이는 선 경로만 쓰는 추론 개입이며, 독립적인 선 전용 모델의 새 학습이 아니다. GT는 실제 보정에 사용하지 않았다.','',
    '| 3번 점 ID0의 방법 | 오차px |','|---|---:|']
    for a in METHODS:text.append(f'| {NAMES[a]} | {c["errors"][a]:.3f} |')
    text+=['','![선 후보만으로 보정한 실제 결과](figures/case_correction.png)','',
    '**선까지 무조건 이동하는 것은 해결책이 아니었다.** 선 후보의 이동량을100%로 바꾸면22.19px, 가장 높은 association의 선까지100% 이동하면16.50px였다. GT로 고른0.40px는 선에 완전히 붙인 결과가 아니라 후보 이동량까지 정답으로 조절한 oracle다. 좋은 방향 단서와 정확한 선 위치/이동량은 구별해야 한다.','',
    '## 별도로 실행한 것: 점 힌트를 끄고 DHT 선만 추론','',
    '같은 이미지 특징·DHT 체크포인트에서 (1)기존 점 가이드, (2)점 거리맵만0, (3)점 가이드 gate 자체를 우회한 선 출력을 비교했다. 세 번째는 점 좌표를1000px 바꿔도 선 logits가 동일함을 검사했다. YOLO box/이미지 특징/K/치수는 남아 있으며 기존에 학습된 DHT 파라미터를 쓴다. 새로 학습한 점 독립 선 검출기와는 다르다.','',
    '![점 힌트 유무에 따른 DHT 선만 보기](figures/dht_only.png)','',
    '희미한 선은36개 전체 후보, 색선은 GT를 보지 않고 mass가 높은6개다. mass는 실제 파렛트 선의 정확도나 역할 신뢰도로 검증된 확률이 아니다. **이 그림의 gate 우회 선으로 위12.08px를 만든 것은 아니다.** 12.08px는 기존 guided DHT 후보만 남긴 결과이며, 점 힌트 우회는 별도 선 생성 개입이다.','',
    '## 같은 개입을 전체에도 적용하면','',
    '실사297장·9세션 재사용DEV·3시드 평균px. clean은 검출 영상 평균, 코너 가림은 가려진 점, 나머지는 화면 밖 점의 오차다. 검출 결과는 기존과 같고 요청/검출 점 수는 JSON에 보존했다.','',
    '| 방법 | clean | 코너 가림 | 잘림10 | 잘림20 | 잘림30 | 잘림40 |','|---|---:|---:|---:|---:|---:|---:|']
    for r0 in table:text.append('| '+NAMES[r0['method']]+' | '+' | '.join(f'{v:.3f}' for v in r0['values'])+' |')
    text+=['',f'선 후보만 남긴 경우 잘림20에서 평균 이동량은 **{move:.4f}px**, 이동0.1px 미만은 **{still:.2%}**였다. 기존 혼합 모델에서 선의 이동량이 작게 학습된 상태를 그대로 사용하므로, 점 후보를 제거하는 것만으로 선 전문가가 충분한 복원을 수행하게 되지는 않았다. 가림/큰 잘림의 기존 개선도 거의 사라졌다.','',
    '## 다음 설계에 대한 판단','',
    '3번 사례는 **점 보정을 그대로 믿으면 악화되고 선 방향의 작은 보정은 도움 되는 경우**라는 해석을 지지한다. 그러나 “선만으로 정확한 점을 바로 복원할 수 있다”는 증거는 아니다.','',
    '다음 설계는 점 위치를 필수 입력으로 쓰지 않는 이미지/선 역할 분기를 따로 학습하고, 관측된 면의 서로 다른 방향 경계나 끝점으로 점 후보를 만든 뒤 점 분기와 비교하는 쪽이 타당하다. 선 분기를 먼저 단독 손실로 학습해 복원 능력을 확인한 후 결합해야, 현재처럼 점 분기가 학습을 대부분 담당하는 상황을 분리할 수 있다. 잘린 점에는 보이는 다른 경계와 물체 형상 제약이 필요하며, 선 하나의 법선만으로 접선 위치까지 결정하지는 못한다. **이 새 학습은 이번에 실행하지 않았다.**','',
    '이번 완료 범위는 요청한 사례의 선 후보 전용 보정, 전체6조건의 같은 개입 확인, 점 힌트가 없는 선 생성 추론 비교다. GT없는 실제 결과와 GT oracle를 분리했고 기존 완료 실험 파일은 변경하지 않았다.','',
    '[전체 수치](SUMMARY.json) · [DHT 선 계수·점 힌트 제거 검사](DHT_ONLY.json) · [실험 범위](PROTOCOL.json)','']
    report='\n'.join(text);(RUN/'REPORT_KO.md').write_text(report);(RUN/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>V28 선만 적용한 결과</title><style>body{font:17px/1.75 sans-serif;max-width:1350px;margin:35px auto;padding:0 24px;color:#183048}img{max-width:100%}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:9px;border-bottom:1px solid #ccd5de}th{background:#eaf0f6}h2{margin-top:2.5em}a{color:#096ba9}</style>'+markdown.markdown(report,extensions=['tables'])+'</html>')
    import re
    prefix='../../../../data/pallet/results/pallet_line_only_v28/';NOTE.write_text(re.sub(r'\]\(([^)]+)\)',lambda m:']('+prefix+m.group(1)+')',report));write(RUN/'FIGURE_DATA.json',dict(case=c,table=table,top_mass_line_ids=top_ids,source_hashes={n:sha(RUN/n) for n in ('SUMMARY.json','DHT_ONLY.json')}))
    checked=0
    for prior in (PRIOR,ROOT/'data/pallet/results/pallet_direction_utility_v27_cases'):
        for path,hsh in read(prior/'COMPLETION.json')['files'].items():assert sha(ROOT/path)==hsh,path;checked+=1
    write(RUN/'AUDIT.json',dict(status='PASS',prior_files_unchanged=checked,guided_line_cache_replay_max_px=dh['guided_cache_replay_max'],bypassed_branch_point_coordinate_invariance=dh['bypass_point_coordinate_invariance'],new_training=False));print('REPORT28',move,still,flush=True)

if __name__=='__main__':main()

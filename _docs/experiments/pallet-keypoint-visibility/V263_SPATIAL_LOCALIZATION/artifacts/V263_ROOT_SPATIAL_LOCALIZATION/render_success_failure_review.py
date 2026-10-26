"""Visual rearrangement of frozen V263 records. No inference or image alteration.

RGB is shown unchanged with display-only point markers. All errors are copied
from saved POINT_DETAILS, not estimated by looking at crops.
"""
from pathlib import Path
import csv
import hashlib
import html
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
G = R/'V263_CURRENT_EVALUATION_A/current_fixed12_gallery_v1'
A = R/'V263_CURRENT_EVALUATION_A/actual_evaluation_v1'
OUT = R/'V263_ROOT_SPATIAL_LOCALIZATION/visual_success_failure_v1'
FONT = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')

def safe(p):
    p=Path(p)
    if '..' in p.parts or any(t in str(p).lower() for t in ['v112','cache','.pyc']):
        raise ValueError(p)
    for a in reversed([p,*p.parents]):
        if a.is_symlink(): raise ValueError(a)
    return p

def binding(p):
    b=safe(p).read_bytes()
    return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}

def load_bound(b):
    p=safe(b['path']);data=p.read_bytes()
    assert hashlib.sha256(data).hexdigest()==b['sha256']
    if 'size_bytes' in b: assert len(data)==b['size_bytes']
    return p

def records(p):
    with safe(p).open() as f:return list(csv.DictReader(f))

def text(fig,x,y,value,size=14,**kwargs):
    return fig.text(x,y,value,fontproperties=FONT,fontsize=size,**kwargs)

def show(ax,rgb):
    ax.imshow(rgb,interpolation='nearest',extent=(-.5,95.5,95.5,-.5))
    ax.set(xlim=(-.5,95.5),ylim=(95.5,-.5));ax.set_axis_off()

def marker(ax,xy,kind):
    if xy is None:return
    x,y=xy
    if kind=='q':
        ax.plot(x,y,'o',ms=9,mfc='none',mec='black',mew=3,zorder=6)
        ax.plot(x,y,'o',ms=9,mfc='none',mec='#00e7ff',mew=1.8,zorder=7)
    elif kind=='r':
        ax.plot(x,y,'+',ms=15,mew=4,color='black',zorder=8)
        ax.plot(x,y,'+',ms=15,mew=2,color='#ff5ce1',zorder=9)
    elif kind=='f':
        ax.plot(x,y,'x',ms=10,mew=4,color='black',zorder=6)
        ax.plot(x,y,'x',ms=10,mew=2,color='#a7ff32',zorder=7)

def main():
    safe(OUT).mkdir(exist_ok=True)
    data=json.loads(safe(G/'CASE_DATA.json').read_bytes())
    cases={int(c['cell']['selected_global_index']):c for c in data['cases']}
    point={(int(r['global_index']),r['model']):r for r in records(A/'POINT_DETAILS.csv')}
    joined={int(r['global_index']):r for r in records(A/'ALL1784_CLASSIFICATION_LOCATION_JOIN.csv')}
    complete_path=R/'V263_SPATIAL_HEAD_C/current223_frozen_inference_v1/COMPLETE.json'
    complete=json.loads(safe(complete_path).read_bytes())
    hm_binding=complete['heatmaps']['JOINT_SELECTED']['binding']
    with np.load(load_bound(hm_binding),allow_pickle=False) as z:hm=np.array(z['probability'])
    selected=[848,981,201,200,908,996,216,844,256]
    descriptions={
        848:('정사각형 · 위치 복구','success','실제 모서리 쪽으로 이동해 3px 안으로 복구'),
        981:('정사각형 · 보정으로 손상','failure','원래 맞던 점을 옮겨 3px 밖으로 악화'),
        201:('목재 · 위치 복구','success','초기 오차를 줄이고 3px 안으로 복구'),
        200:('목재 · 보정으로 손상','failure','보임 판단은 맞지만 적용한 좌표가 악화'),
        908:('외부 가림 · 올바른 판단','success','콘이 가린 점을 외부 가림으로 판별'),
        996:('외부 가림 · 놓친 사례','failure','가린 점을 보임으로 판단하고 보정도 적용'),
        216:('목재 보임 · 잘못 제외','failure','초기 오차 0.73px인 보이는 모서리를 외부 가림으로 판단'),
        844:('정사각형 보임 · 잘못 제외','failure','수동 보임 점을 외부 가림으로 판단'),
        256:('목재 외부 가림 · 주변 경계로 이동','failure','콘·바닥 crop에서 보임으로 판단하고 바닥 방향으로 보정')}
    summary=[]
    for idx in selected:
        c=cases[idx];j=c['details']['JOINT_SELECTED'];rgb=np.array(Image.open(load_bound(j['clean'])).convert('RGB'))
        assert hashlib.sha256(rgb.tobytes()).hexdigest()==j['RGB96_HWC_sha256']
        p=point.get((idx,'JOINT_SELECTED'))
        summary.append({'global_index':idx,'row_id':c['row_id'],'title':descriptions[idx][0],
                        'status':descriptions[idx][1],'observation':descriptions[idx][2],
                        'truth':c['human_state'],'prediction':j['semantic_VIS_prediction'],
                        'location_accepted':str(j['accepted']).lower()=='true',
                        'initial_error_px':float(p['initial_error_px']) if p else None,
                        'final_error_px':float(p['final_error_px']) if p else None,
                        'cell_census':c['cell']['census_N'],'cell':c['cell']['cell'],
                        'clean_binding':j['clean'],'existing_card':c['card'],
                        'q_crop':j['q_crop'],'reference_crop':j['reference_crop'],'final_crop':j['final_crop']})
    byid={c['global_index']:c for c in summary}
    # Two clear paired comparisons per row. Position errors use original-image pixels.
    for mode,ids,filename in [
        ('location',[848,981,201,200],'LOCATION_SUCCESS_FAILURE_KO.png'),
        ('visibility',[908,996,216,844],'VISIBILITY_SUCCESS_FAILURE_KO.png')]:
        fig,axs=plt.subplots(2,4,figsize=(15.5,10),dpi=140)
        fig.subplots_adjust(top=.80,bottom=.06,left=.025,right=.975,wspace=.035,hspace=.44)
        heading='위치 보정: 되는 사례와 손상되는 사례' if mode=='location' else '가림 판단: 맞는 사례와 틀린 사례'
        text(fig,.5,.966,heading,24,ha='center',weight='bold')
        text(fig,.5,.926,'V263 공동 모델 · 지정 키포인트 기준 · 같은 RGB crop · 저장 출력 재사용',12,ha='center',color='#52636f')
        handles=[Line2D([],[],marker='o',mfc='none',mec='#00c5df',mew=2,ls='',ms=9,label='초기 추론'),
                 Line2D([],[],marker='+',color='#d92fbc',mew=2,ls='',ms=12,label='수동 주석 위치'),
                 Line2D([],[],marker='x',color='#76b915',mew=2,ls='',ms=10,label='적용한 위치')]
        fig.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(.5,.91),prop=FONT,frameon=False)
        for tile,idx in enumerate(ids):
            row=tile//2;col=(tile%2)*2;c=byid[idx];j=cases[idx]['details']['JOINT_SELECTED']
            rgb=np.array(Image.open(safe(j['clean']['path'])).convert('RGB'))
            a,b=axs[row,col],axs[row,col+1]
            show(a,rgb);show(b,rgb)
            marker(a,j['q_crop'],'q');marker(a,j['reference_crop'],'r')
            if mode=='location':
                marker(b,j['q_crop'],'q');marker(b,j['final_crop'],'f');marker(b,j['reference_crop'],'r')
                q=np.array(j['q_crop']);f=np.array(j['final_crop'])
                b.annotate('',xy=f,xytext=q,arrowprops={'arrowstyle':'->','color':'#a7ff32','lw':1.6})
                detail=f"오차 {c['initial_error_px']:.2f} → {c['final_error_px']:.2f} px · 같은 유형 {c['cell_census']}점"
                a.text(.5,-.055,'보정 전',transform=a.transAxes,ha='center',fontproperties=FONT,fontsize=12)
                b.text(.5,-.055,'보정 후',transform=b.transAxes,ha='center',fontproperties=FONT,fontsize=12)
            else:
                h=hm[idx];h=(h-h.min())/max(float(h.max()-h.min()),1e-12)
                b.imshow(h,cmap='inferno',alpha=.62,interpolation='bilinear',extent=(-.5,95.5,95.5,-.5),vmin=0,vmax=1)
                marker(b,j['q_crop'],'q')
                if c['location_accepted']:marker(b,j['final_crop'],'f')
                marker(b,j['reference_crop'],'r')
                truth='보임' if c['truth']=='DIRECT_VISIBLE' else '외부 가림'
                predicted='보임' if c['prediction']=='D' else '외부 가림'
                detail=f"사람 정답: {truth}  /  모델 판단: {predicted} · 같은 유형 {c['cell_census']}점"
                if c['truth']=='DIRECT_VISIBLE':detail+=f"\n초기 오차 {c['initial_error_px']:.2f}px · 위치 적용 {'예' if c['location_accepted'] else '보류'}"
                a.text(.5,-.055,'실제 crop + 주석',transform=a.transAxes,ha='center',fontproperties=FONT,fontsize=12)
                b.text(.5,-.055,'모델 heatmap + 적용 위치',transform=b.transAxes,ha='center',fontproperties=FONT,fontsize=12)
            pos=a.get_position();pos2=b.get_position();center=(pos.x0+pos2.x1)/2
            text(fig,center,pos.y1+.063,c['title'],17,ha='center',color='#14785e' if c['status']=='success' else '#b43747',weight='bold')
            text(fig,center,pos.y1+.015,detail,11,ha='center',color='#263640')
        footer='오차는 원본 영상 좌표 기준. 분류 수는 이 공동 모델의 해당 셀 전체 수이며 그림은 그중 고정된 첫 사례입니다.' if mode=='location' else '밝은 heatmap은 상대적인 peak 표시이며 신뢰 확률이 아닙니다. 가린 점의 주석 위치는 직접 관측 위치 정답이 아닙니다.'
        text(fig,.5,.016,footer,10,ha='center',color='#52636f')
        fig.savefig(OUT/filename,facecolor='white');plt.close(fig)
    outdata={'new_fit_or_inference_or_PnP':0,'principal_model':'JOINT_SELECTED',
             'source_cases_unchanged':True,'selection':'Rearranged original fixed12 cell first-row cases; no new visual search.',
             'heatmap_display':'Per-case min/max relative display, not calibrated probability',
             'cases':summary,'input_bindings':[binding(G/'CASE_DATA.json'),binding(A/'POINT_DETAILS.csv'),binding(A/'ALL1784_CLASSIFICATION_LOCATION_JOIN.csv'),hm_binding]}
    (OUT/'CASE_SUMMARY.json').write_text(json.dumps(outdata,ensure_ascii=False,indent=2)+'\n')
    cards=[]
    for c in summary:
        oldcard='../../V263_CURRENT_EVALUATION_A/current_fixed12_gallery_v1/'+Path(c['existing_card']['path']).name
        error='' if c['initial_error_px'] is None else f" · 위치 오차 {c['initial_error_px']:.2f}→{c['final_error_px']:.2f}px"
        cards.append(f'<article class="{c["status"]}"><h3>{html.escape(c["title"])}</h3><p>{html.escape(c["observation"]+error)}</p><p>셀 전체 {c["cell_census"]}점 · {html.escape(c["row_id"])}</p><a href="{oldcard}" target="_blank"><img src="{oldcard}" alt="{html.escape(c["title"])}"></a></article>')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>되는 사례와 실패 사례 V263</title><style>body{font:16px system-ui,sans-serif;line-height:1.7;color:#1d303b;max-width:1280px;margin:30px auto;padding:0 18px}h1{line-height:1.3}img{max-width:100%}.board{display:block;margin:26px 0}button{font:inherit;padding:8px 18px;margin-right:8px;cursor:pointer}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}article{border:1px solid #cdd7dd;border-top:5px solid #b43747;border-radius:8px;padding:14px}.success{border-top-color:#14785e}article p{margin:8px 0}a{color:#17659a}.note{padding:14px 20px;background:#eef3f5}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style>
<h1>되는 사례와 실패 사례</h1><p>V263 공동 모델의 저장된 실사 crop을 재배치했습니다. 새 학습·추론·문턱 변경은 없습니다.</p>
<p class="note"><b>보이는 모서리의 위치를 잘 고치는 사례는 있습니다.</b> 동시에 보이는 점을 외부 가림으로 부르거나, 정확하던 점을 잘못 옮기거나, 가린 점 대신 주변 경계를 고르는 사례도 남습니다. 여기서 가림은 외부 가림이며 자기 가림과 합치지 않았습니다.</p>
<a class="board" href="LOCATION_SUCCESS_FAILURE_KO.png" target="_blank"><img src="LOCATION_SUCCESS_FAILURE_KO.png" alt="위치 보정 성공과 손상 비교"></a>
<a class="board" href="VISIBILITY_SUCCESS_FAILURE_KO.png" target="_blank"><img src="VISIBILITY_SUCCESS_FAILURE_KO.png" alt="가림 판별 성공과 실패 비교"></a>
<h2>사례별 확대</h2><p>그림을 누르면 확대할 수 있습니다. 기존 고정12셀의 선택 규칙을 유지했고, 표시한 수는 각 셀의 전체 사례 수입니다.</p>
<p><button onclick="filterCards('all')">전체</button><button onclick="filterCards('success')">되는 사례</button><button onclick="filterCards('failure')">실패 사례</button></p><div class="grid">'''+''.join(cards)+'''</div>
<h2>시각적으로 확인되는 공통 문제</h2><p>정사각형 복구 사례는 제안점이 사람이 찍은 모서리 쪽으로 움직입니다. 실패 사례에서는 주변 모서리나 경계에 peak가 있어도 해당 native 점의 직접 보임을 입증하지 못합니다. 목재 사례는 초기 오차가0.73px인데도 외부 가림으로 판별되어, 초기 좌표 오차만으로는 이 오류를 설명할 수 없습니다.</p>
<p>이는 선택 사례의 관찰입니다. 합성→실사 전이, 점 ID 대응, 공동 손실 중 어느 원인이 얼마나 기여했는지는 이 그림만으로 확정하지 않습니다. Heatmap 색은 보정된 신뢰 확률이 아닙니다.</p><p><a href="CASE_SUMMARY.json">사례별 수치와 출처</a> · <a href="../ROOT_READOUT_KO.html">전체 평가표</a></p>
<script>function filterCards(x){document.querySelectorAll('article').forEach(e=>e.hidden=x!=='all'&&!e.classList.contains(x));}</script></html>'''
    (OUT/'INDEX_KO.html').write_text(page)
    manifest={'completed':True,'PASS':True,'new_model_operations':0,'files':[binding(OUT/n) for n in ['LOCATION_SUCCESS_FAILURE_KO.png','VISIBILITY_SUCCESS_FAILURE_KO.png','CASE_SUMMARY.json','INDEX_KO.html']], 'code':binding(Path(__file__))}
    (OUT/'COMPLETE.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(OUT),'case_count':len(summary),'new_model_operations':0},ensure_ascii=False))

if __name__=='__main__':main()

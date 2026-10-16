"""Two per-image error scatter plots from sealed scores, without inference."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import time

import numpy as np

ARMS=('POSEFIX_RAW','R0','N0','N1','E0')
NAMES={'POSEFIX_RAW':'PoseFix','R0':'RAW YOLO + F','N0':'N3→SubPix + F','N1':'N3→SubPix + 최소 B/I','E0':'기존 전체 방법 e'}
COLORS={'POSEFIX_RAW':'#b73594','R0':'#697c90','N0':'#df940e','N1':'#dc4a2c','E0':'#146ba4'}
MARKERS={'POSEFIX_RAW':'D','R0':'x','N0':'o','N1':'^','E0':'s'}
PLOTLY_MARKERS={'POSEFIX_RAW':'diamond','R0':'x','N0':'circle','N1':'triangle-up','E0':'square'}
COHORTS={'FULL192':'전체192장 · 제외 없음','RETAINED121':'아웃라이어 공통 제외 후121장','EXCLUDED71':'공통 제외71장만'}
METRICS={'translation_cm':('위치 오차','cm','position'),'rotation_deg':('회전 오차','°','rotation')}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,obj):
    with Path(p).open('x')as f:f.write(obj if isinstance(obj,str)else json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def build(summary,outliers,output):
    start=time.monotonic();summary,outliers,output=map(Path,(summary,outliers,output))
    if output.exists():raise FileExistsError('Keep prior scatter plots')
    original_seal=read(summary.parent/'SEAL.json')
    if sha(summary)!=original_seal['artifacts'][summary.name]:raise ValueError('Saved score CSV seal mismatch')
    analysis=read(outliers);rule=analysis['cohorts']['ALL5_IQR_T_R'];excluded=set(rule['excluded_ids']);kept=set(rule['retained_ids'])
    points=[]
    with summary.open(newline='')as f:
        for row in csv.DictReader(f):
            if row['partition']=='CAL':continue
            metrics={a:{m:float(row[a+'__'+m])for m in METRICS}for a in ARMS}
            if any(not np.isfinite(v)for a in metrics.values()for v in a.values()):raise ValueError('Do not silently omit unavailable errors')
            points.append(dict(index=len(points)+1,id=row['id'],session=row['session'],partition=row['partition'],
                excluded=row['id']in excluded,metrics=metrics))
    if len(points)!=192 or len({r['id']for r in points})!=192 or len(excluded)!=71 or len(kept)!=121:raise ValueError('Fixed192/121/71 required')
    if set(r['id']for r in points)!=kept|excluded or kept&excluded:raise ValueError('Same shared image populations required')
    populations={'FULL192':points,'RETAINED121':[r for r in points if not r['excluded']],'EXCLUDED71':[r for r in points if r['excluded']]}
    groups=[]
    for p in points:
        if not groups or groups[-1]['session']!=p['session']:groups.append(dict(session=p['session'],first=p['index'],last=p['index']))
        else:groups[-1]['last']=p['index']
    stats={key:{a:{m:dict(n=len(rs),mean=float(np.mean([r['metrics'][a][m]for r in rs])))for m in METRICS}for a in ARMS}for key,rs in populations.items()}
    output.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt,font_manager
    font=Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf');font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False})
    for cohort in ('FULL192','RETAINED121'):
        selected=populations[cohort]
        for metric,(label,unit,stem)in METRICS.items():
            fig,ax=plt.subplots(figsize=(17,6.4),layout='constrained')
            for i,g in enumerate(groups):
                if i%2:ax.axvspan(g['first']-.5,g['last']+.5,color='#eaf0f5',alpha=.7,zorder=0)
                if i:ax.axvline(g['first']-.5,c='#acbbc8',lw=.8,zorder=0)
            for arm in ARMS:
                x=[r['index']for r in selected];y=[r['metrics'][arm][metric]for r in selected];mean=stats[cohort][arm][metric]['mean']
                ax.scatter(x,y,c=COLORS[arm],marker=MARKERS[arm],s=29,alpha=.72,label=f'{NAMES[arm]} · 평균 {mean:.2f}{unit}',linewidths=.7,zorder=3)
                ax.axhline(mean,c=COLORS[arm],ls='--',lw=1.3,alpha=.8,zorder=2)
            ax.set(xlim=(.5,192.5),ylim=(0,None),xlabel='이미지 순번 (원래192장 순서 유지)',ylabel=f'이미지별 {label} ({unit})',
                   title=f'{label} 산점도 · {COHORTS[cohort]}\n점 하나 = 이미지 한 장의 개별 오차 / 색 = 방법 / 점선 = 해당 방법의 평균')
            ax.grid(axis='y',alpha=.22);ax.legend(loc='upper left',ncols=2,fontsize=10,framealpha=.96)
            fig.savefig(output/f'{stem}_{cohort.lower()}.png',dpi=150)
            if cohort=='FULL192':fig.savefig(output/f'{stem}_{cohort.lower()}.svg')
            plt.close(fig)
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.11,subplot_titles=('위치 오차 · 이미지별 cm','회전 오차 · 이미지별 °'))
    order=[]
    for metric,(label,unit,stem)in METRICS.items():
        panel=1 if metric=='translation_cm'else 2
        for arm in ARMS:
            custom=[[r['id'],r['session'],'제외71장'if r['excluded']else'유지121장',NAMES[arm]]for r in points]
            fig.add_trace(go.Scatter(x=[r['index']for r in points],y=[r['metrics'][arm][metric]for r in points],customdata=custom,
                mode='markers',marker=dict(color=COLORS[arm],symbol=PLOTLY_MARKERS[arm],size=7,opacity=.75),name=NAMES[arm],legendgroup=arm,showlegend=panel==1,
                hovertemplate=f'%{{customdata[3]}}<br>이미지 #%{{x}}: %{{customdata[0]}}<br>세션: %{{customdata[1]}}<br>{label}: %{{y:.3f}}{unit}<br>%{{customdata[2]}}<extra></extra>'),row=panel,col=1)
            order.append((metric,arm,'points'))
            mean=stats['FULL192'][arm][metric]['mean']
            fig.add_trace(go.Scatter(x=[.5,192.5],y=[mean,mean],mode='lines',line=dict(color=COLORS[arm],dash='dash',width=1.5),
                name=NAMES[arm]+' 평균',legendgroup=arm,showlegend=False,hovertemplate=f'{NAMES[arm]} 평균: %{{y:.3f}}{unit}<extra></extra>'),row=panel,col=1)
            order.append((metric,arm,'mean'))
        for g in groups[1:]:fig.add_vline(x=g['first']-.5,line_width=1,line_color='#c4ced8',row=panel,col=1)
        fig.update_yaxes(title_text=f'{label} ({unit})',rangemode='tozero',row=panel,col=1)
        fig.update_xaxes(title_text='이미지 순번 · 원래192장 내 순서',range=[.5,192.5],row=panel,col=1)
    buttons=[]
    for cohort,selected in populations.items():
        xs=[];ys=[];custom=[]
        for metric,arm,kind in order:
            if kind=='points':
                xs.append([r['index']for r in selected]);ys.append([r['metrics'][arm][metric]for r in selected]);custom.append([[r['id'],r['session'],'제외71장'if r['excluded']else'유지121장',NAMES[arm]]for r in selected])
            else:
                xs.append([.5,192.5]);ys.append([stats[cohort][arm][metric]['mean']]*2);custom.append(None)
        buttons.append(dict(label=COHORTS[cohort],method='update',args=[dict(x=xs,y=ys,customdata=custom),
            {'title.text':f'이미지별 오차 산점도 · {COHORTS[cohort]}','yaxis.autorange':True,'yaxis2.autorange':True,'xaxis.range':[.5,192.5],'xaxis2.range':[.5,192.5]}]))
    fig.update_layout(title='이미지별 오차 산점도 · 전체192장 · 제외 없음',height=1050,template='plotly_white',
        hovermode='closest',legend=dict(orientation='h',y=1.11,x=0,groupclick='togglegroup'),
        updatemenus=[dict(type='dropdown',buttons=buttons,x=1,y=1.16,xanchor='right',yanchor='top')],margin=dict(l=80,r=30,t=170,b=60))
    interactive=fig.to_html(full_html=False,include_plotlyjs=True,div_id='error-scatter',config={'responsive':True,'displaylogo':False,'toImageButtonOptions':{'format':'png','scale':2}})
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>이미지별 위치·회전 오차 산점도</title><style>body{font-family:sans-serif;margin:15px;color:#183047}h1{font-size:22px}p{line-height:1.6}#error-scatter{width:100%}</style><h1>위치·회전 오차 산점도</h1><p>각 색상에서 점 하나는 이미지 한 장의 개별 오차입니다. 방법별 평균은 같은 색 점선입니다. 점에 마우스를 올리면 이미지 ID와 오차가 표시됩니다. 범례를 누르면 방법을 숨기고, 드래그하면 확대할 수 있습니다. 오른쪽 메뉴에서 전체192장·유지121장·제외71장을 전환합니다. 필터 후에도 x는 원래192장의 이미지 순번을 유지합니다.</p><p>121장/71장 구분은 모든 방법의 위치18.728cm 또는 회전7.222° 초과 영상을 공통 제외한 사후 분석입니다. 원래192장 결과를 대체하지 않습니다. CAL46은 포함하지 않았습니다.</p>'+interactive+'</html>'
    write(output/'index.html',page)
    write(output/'PLOT_DATA.json',dict(points=points,cohorts={k:[r['id']for r in rs]for k,rs in populations.items()},statistics=stats,
        session_blocks=groups,point_definition='one image, one individual error per method',mean_lines='arithmetic mean within selected cohort, separately for each method',
        static_plots=['position_full192.png','rotation_full192.png','position_retained121.png','rotation_retained121.png']))
    write(output/'README_KO.md','# 이미지별 오차 산점도\n\n[대화형 위치·회전 산점도](index.html)\n\n'
        'x는 고정PRIMARY192내 이미지순번, y는 각이미지의 위치(cm) 또는 회전(°) 오차다. 각색에서 이미지한장당점하나이며 방법간오차를 평균내지않았다. '
        '방법별평균은동일색점선이다. 유지121/제외71도모든방법에같은영상집합을적용하고원래x순번을유지한다. '
        '전체192장을기본으로표시하며CAL46은별도다. PNG4개와전체192장SVG2개를함께저장했다. 새추론·PnP·학습·정답변경은없다.\n')
    write(output/'RECEIPT.json',dict(source_sha256=sha(__file__),input_csv_sha256=sha(summary),outlier_definition_sha256=sha(outliers),
        images=192,methods=5,scatter_points_per_full_plot=960,means_per_plot=5,cohort_sizes={k:len(v)for k,v in populations.items()},
        image_index_preserved=True,new_training_or_model_or_PnP_or_reference_fit=0,wall_seconds=time.monotonic()-start))
    write(output/'SEAL.json',dict(assets={p.name:sha(p)for p in output.iterdir()if p.is_file()}))
    return output/'index.html'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('summary','outliers','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(build(a.summary,a.outliers,a.output))

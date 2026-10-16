"""Build a point-only paper working draft from frozen experiment artifacts.

No training/inference. Charts are source-derived. --open verifies a visible
Chrome window after rendering, following the user's persistent preference.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from html.parser import HTMLParser
from urllib.parse import urlparse,unquote
import numpy as np
import markdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/pallet/results/pallet_point_only_paper_v1'
NOTE=ROOT/'_docs/notes/pose_geometry/point_only_paper/manuscript.md'
TITLE='점 기반 팔레트 논문 — 구조·Loss·검증 결과'

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

class Check(HTMLParser):
    def __init__(self):
        super().__init__();self.images=0;self.cells=[];self.active=False;self.buf='';self.links=0
    def handle_starttag(self,tag,attrs):
        self.images+=tag=='img'
        if tag in ['td','th']:self.active=True;self.buf=''
        for k,v in attrs:
            if k not in ['href','src']:continue
            u=urlparse(v)
            if u.scheme=='file':assert Path(unquote(u.path)).exists(),v;self.links+=1
    def handle_data(self,text):
        if self.active:self.buf+=text
    def handle_endtag(self,tag):
        if tag in ['td','th']:self.cells.append(self.buf.strip());self.active=False

def configure_plot():
    from matplotlib import font_manager
    fp='/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    font_manager.fontManager.addfont(fp)
    plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False,'font.size':11,
                         'svg.fonttype':'path','figure.facecolor':'white','savefig.facecolor':'white'})

def draw_architecture():
    fig,ax=plt.subplots(figsize=(14,7));ax.set_xlim(0,14);ax.set_ylim(0,7);ax.axis('off')
    def box(x,y,w,h,text,planned=False):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=.13',
            facecolor='#fff4d5' if planned else '#eaf1fb',edgecolor='#bd8216' if planned else '#557eab',
            linewidth=1.4,linestyle='--' if planned else '-'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11,linespacing=1.6)
    def arrow(a,b,dashed=False):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,linewidth=1.4,
                                    color='#8c6f35' if dashed else '#49627d',linestyle='--' if dashed else '-'))
    ax.text(.15,6.8,'완료한 구조: 기존 YOLO + 후보별 점 보정',fontsize=19,weight='bold',va='top')
    box(.2,3.95,1.25,.9,'RGB\n이미지')
    box(1.95,3.75,2,1.3,'YOLO26n 동결\n기존 점·박스\n45채널 출력 특징')
    box(4.5,3.75,2,1.3,'점·ROI 특징\n코너 대응 H 4개\n좌표·특징 함께 재배열')
    box(7.15,4.65,2.65,1.05,'후보 오차 예측기\nMSE 감독 / 독립 파라미터')
    box(7.15,2.85,2.65,1.05,'후보별 잔차 보정기\n각 x/y 성분 ±16px')
    box(10.6,3.7,3.0,1.45,'예측 오차로 후보 선택\n그 후보의 보정점 출력\nGT 없이 추론')
    box(4.5,5.65,2,0.65,'알려진 치수 W/D/H')
    arrow((1.45,4.4),(1.95,4.4));arrow((3.95,4.4),(4.5,4.4))
    arrow((6.5,4.5),(7.15,5.15));arrow((6.5,4.05),(7.15,3.35))
    arrow((6.5,5.95),(7.8,5.7));arrow((6.5,5.85),(7.08,3.9))
    arrow((9.8,5.15),(10.6,4.7));arrow((9.8,3.35),(10.6,4.0))
    ax.text(.25,3.2,'영상 선/Hough 없음\n기존 backbone·검출기 유지\n추가 모듈 94,331 parameters',fontsize=11,linespacing=1.8,va='top',color='#354e6b')
    ax.axhline(2.35,color='#d5d5d5',linewidth=1)
    ax.text(.25,2.08,'통합 목표 — 현재 보정 구조와 함께 학습한 결과는 아직 없음',fontsize=14,color='#906117',weight='bold')
    box(.3,.65,3.3,.95,'등록된 종류·치수\n허용 대칭 G = C1 / C2 / C4',True)
    box(4.25,.65,4.4,.95,'물체 전체에 같은 g 적용\nL_sym = min(g ∈ G) L_point(g)',True)
    box(9.3,.65,4.05,.95,'평가도 동일 G 사용\n동등한 절대 번호 선택 목표는 재정의',True)
    arrow((3.6,1.12),(4.25,1.12),True);arrow((8.65,1.12),(9.3,1.12),True)
    fig.subplots_adjust(left=.025,right=.99,top=.98,bottom=.025)
    fig.savefig(OUT/'architecture.svg');fig.savefig(OUT/'architecture.png',dpi=160);plt.close(fig)

def draw_comparison(results):
    fig,axes=plt.subplots(1,2,figsize=(13,4.8));colors=['#78889c','#df9c44','#167f83']
    arms=[('stock','기존 YOLO'),('direct','직접 점 보정'),('selected','후보 선택 + 보정')]
    for ax,metric,title in zip(axes,['strict_px','orbit_px'],['번호 고정 오차','번호 대응을 제외한 위치 오차 · 사후 진단']):
        for j,(arm,label) in enumerate(arms):
            values=[]
            for pop in ['seen','unseen']:
                vv=[results['stock'][pop][metric]] if arm=='stock' else [results[f'{arm}_s{s}'][pop][metric] for s in [1,2,3]]
                values.append(vv)
            x=np.arange(2)+(j-1)*.25;means=[np.mean(v) for v in values]
            bars=ax.bar(x,means,.22,label=label,color=colors[j],zorder=2)
            for xx,vv,b in zip(x,values,bars):
                if len(vv)>1:ax.scatter(np.full(3,xx),vv,c='#23394c',s=13,zorder=3)
                ax.text(xx,max(vv)+(.65 if metric=='strict_px' else .28),f'{b.get_height():.2f}',ha='center',fontsize=10)
        ax.set_xticks([0,1],['본 종류 · 216장','새 종류 · 208장']);ax.set_ylabel('평균 코너 오차 (px) ↓')
        ax.set_title(title,pad=15);ax.set_ylim(0,32 if metric=='strict_px' else 11)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2,zorder=0)
    axes[0].legend(loc='upper left',fontsize=9)
    fig.text(.5,.015,'동일한 새 합성 576관측 중 공통 매칭 대상 · 막대: 시드 평균 / 점: 개별 시드 · 물리 C4 자세 평가가 아님',ha='center',fontsize=10,color='#475971')
    fig.tight_layout(rect=(0,.045,1,1));fig.savefig(OUT/'comparison.png',dpi=170);fig.savefig(OUT/'comparison.svg');plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--open',action='store_true');args=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    source=ROOT/'data/pallet/results/pallet_role_symmetry_audit/SUMMARY.json'
    main_source=ROOT/'data/pallet/results/pallet_role_selection_v2/foreground_matched/SUMMARY.json'
    c4=ROOT/'challenge/yolo_pose_one_model/challenge_c4_track/clean_label/RESULT.json'
    dim=ROOT/'data/pallet/results/pallet_dimension_symmetry_v1/SUMMARY.json'
    files=[source,main_source,c4,dim,ROOT/'scripts/research/pallet_role_selection_v2/model.py',
        ROOT/'scripts/research/pallet_role_selection_v2/foreground_train.py',ROOT/'scripts/research/pallet_role_selection_v2/foreground_cache.py',
        ROOT/'challenge/config/SQUARE_PALLET_SYMMETRY_CONTRACT.json']
    locked={str(p.relative_to(ROOT)):sha(p) for p in files}
    summary=read(source);results=summary['results'];original=read(main_source)
    sys.path.insert(0,str(ROOT/'scripts/research/pallet_role_selection_v2'))
    from model import SelectorResidual
    model=SelectorResidual('C_select_residual',1)
    params=sum(p.numel() for p in model.parameters());assert params==94331
    expected=[];matrix={}
    for arm,label in [('stock','기존 YOLO'),('direct','직접 점 보정'),('selected','후보 선택 + 작은 보정')]:
        values=[]
        for pop in ['seen','unseen']:
            for metric in ['strict_px','orbit_px']:
                v=results['stock'][pop][metric] if arm=='stock' else np.mean([results[f'{arm}_s{s}'][pop][metric] for s in [1,2,3]])
                values.append(float(v))
        matrix[arm]=values;expected.extend([label]+[f'{v:.2f}' for v in values])
    assert original['decision']=='NO_STABLE_MATCHED_SUPERVISION_SUPPORT'
    for pop in ['seen','unseen']:
        for name in ['stock','direct_s1','direct_s2','direct_s3','selected_s1','selected_s2','selected_s3']:
            assert abs(results[name][pop]['strict_px']-original['results'][name][pop]['overall']['matched_mean_px'])<1e-8
    evidence=dict(created='2026-09-11',training_steps=0,new_inference=False,scope='point-only paper working draft',
        combined_symmetry_refiner_trained=False,parameters=params,source_sha256=locked,
        table_columns=['seen_strict','seen_orbit','unseen_strict','unseen_orbit'],table=matrix,
        comparison_population={'seen':216,'unseen':208,'all':576},metric='8-corner mean pixel error on common matched objects; orbit is post-hoc ID diagnosis only',
        comparisons=summary['comparisons'],c4_cleanlabel=read(c4)['arms'],
        dimension_symmetry_activation=read(dim)['initial_symmetry_activation'],original_decision=original['decision'])
    write(OUT/'EVIDENCE.json',evidence)
    configure_plot();draw_architecture();draw_comparison(results)
    md=NOTE.read_text()
    def resolve(m):
        ref=m[1]
        if '://' in ref or ref.startswith('#'):return m[0]
        target=(NOTE.parent/ref).resolve()
        assert target.exists() or target==OUT/'index.html',target
        return ']('+target.as_uri()+')'
    md=re.sub(r'\]\(([^)]+)\)',resolve,md)
    renderer=markdown.Markdown(extensions=['tables','fenced_code','toc'],extension_configs={'toc':{'toc_depth':'2-3'}})
    body=renderer.convert(md)
    toc='<details><summary>원고 목차 펼치기</summary>'+renderer.toc+'</details>'
    css='''body{margin:0;background:#edf2f7;color:#203048;font:16px/1.85 system-ui,sans-serif}main{max-width:1260px;margin:25px auto;padding:38px 46px;background:#fff;border-radius:18px}h1{font-size:32px;line-height:1.4}h2{margin-top:48px;padding-bottom:8px;border-bottom:2px solid #dbe7f1}h3{margin-top:30px;color:#285679}strong{color:#116876}a{color:#176eb0}img{max-width:100%;height:auto;border:1px solid #e0e7ef;border-radius:6px}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;margin:22px 0}th,td{padding:10px 12px;border-bottom:1px solid #dce4ed;text-align:left}th{background:#eaf2f8}pre{background:#142a40;color:#edf6ff;padding:18px;overflow:auto;border-radius:8px;line-height:1.7}code{font-size:.9em}p code,li code{background:#eef3f8;padding:2px 4px}.banner{padding:16px 22px;background:#fff3d1;border-left:5px solid #c19032;border-radius:6px;margin-bottom:28px}.nav{display:flex;gap:15px;flex-wrap:wrap;font-size:14px;margin-bottom:24px}@media(max-width:760px){main{padding:18px;margin:0}body{font-size:14px}td,th{padding:6px}table{font-size:12px}h1{font-size:25px}}@media print{body{background:white}main{padding:0;margin:0}.nav{display:none}h2,h3{break-after:avoid}img,table{break-inside:avoid}}'''
    nav='<nav class="nav"><a href="'+NOTE.as_uri()+'">Markdown 원고</a><a href="'+(OUT/'EVIDENCE.json').as_uri()+'">원시 수치·근거</a><a href="'+(OUT/'architecture.svg').as_uri()+'">구조도 SVG</a><a href="'+(OUT/'comparison.svg').as_uri()+'">비교 그림 SVG</a></nav>'
    banner='<div class="banner"><b>현재 증거:</b> 선 없는 점 보정 구조의 합성 위치 개선. <b>미완료:</b> 객체별 대칭 loss와의 통합 학습 및 독립 실물 평가.</div>'
    page=f'<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{TITLE}</title><style>{css}</style><main>{nav}{banner}{toc}{body}</main></html>'
    (OUT/'index.html').write_text(page)
    check=Check();check.feed(page)
    assert check.images==4
    def sequence_in(xs,ys):return any(xs[i:i+len(ys)]==ys for i in range(len(xs)-len(ys)+1))
    assert sequence_in(check.cells,expected),'Main table disagrees with source JSON'
    assert locked=={str(p.relative_to(ROOT)):sha(p) for p in files}
    # Preserve the completed upstream experiment and previous diagnostic report.
    audited=[]
    for folder in ['pallet_role_selection_v2','pallet_role_symmetry_audit']:
        manifest=read(ROOT/'data/pallet/results'/folder/'COMPLETION.json')
        for name,h in manifest['files'].items():assert sha(ROOT/name)==h,name
        audited.append(dict(folder=folder,files=len(manifest['files'])))
    write(OUT/'AUDIT.json',dict(PASS=True,table_matches_source=True,images=check.images,local_links=check.links,
        original_artifacts_unchanged=audited,source_sha256=locked))
    if args.open:
        url=(OUT/'index.html').as_uri()
        with (OUT/'browser.log').open('w') as log,(OUT/'rendered_dom.html').open('w') as dom:
            subprocess.run(['google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-background-networking','--no-first-run',
                '--user-data-dir=/tmp/pallet-point-paper-chrome','--window-size=1550,1450','--screenshot='+str(OUT/'browser_preview.png'),
                '--dump-dom',url],stdout=dom,stderr=log,timeout=40,check=True)
        rendered=Check();rendered.feed((OUT/'rendered_dom.html').read_text())
        assert rendered.images==4 and sequence_in(rendered.cells,expected)
        subprocess.run(['google-chrome','--new-window',url],capture_output=True,timeout=20,check=True)
        windows=[]
        for _ in range(10):
            clients=subprocess.run(['xprop','-root','_NET_CLIENT_LIST'],capture_output=True,text=True).stdout
            for wid in re.findall(r'0x[0-9a-fA-F]+',clients):
                name=subprocess.run(['xprop','-id',wid,'_NET_WM_NAME'],capture_output=True,text=True).stdout
                if TITLE in name:
                    info=subprocess.run(['xwininfo','-id',wid],capture_output=True,text=True).stdout
                    windows.append(dict(id=wid,title=name.strip(),viewable='IsViewable' in info))
            if any(w['viewable'] for w in windows):break
            time.sleep(.5)
        visible=any(w['viewable'] for w in windows)
        write(OUT/'DELIVERY.json',dict(html_open_confirmed=visible,windows=windows,rendered_table_verified=True))
        assert visible
    bound=[p for p in OUT.iterdir() if p.is_file() and p.name not in ['COMPLETION.json','DISCORD_NOTIFICATION.json'] and p.suffix!='.log']
    bound += [NOTE,NOTE.parent/'README.md',Path(__file__)]
    write(OUT/'COMPLETION.json',dict(complete=True,training_steps=0,new_inference=False,combined_model_trained=False,
        html_open_confirmed=bool(args.open),files={str(p.relative_to(ROOT)):sha(p) for p in bound}))
    print('Point-only manuscript, architecture, source-derived chart and evidence verified.',flush=True)

if __name__=='__main__':main()

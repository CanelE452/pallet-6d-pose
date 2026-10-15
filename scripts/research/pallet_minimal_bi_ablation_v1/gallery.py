"""Postseal saved-stage gallery: no inference, reference reader, or pose fit."""
from __future__ import annotations

from pathlib import Path
import gzip
import html
import json
import shutil
import numpy as np

from . import scoring as S
from .diagnostics import changed_case_groups, observation_summary

BRANCHES = (('RAW_YOLO','R0','R1','RAW 입력'),('N3_SUBPIX','N0','N1','N3+SubPix 입력'))
COLORS = dict(B='#00a4c4',I='#8847d9')
EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
FONT = Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
PUBLIC_INVENTORY = S.LEGACY_DOC/'publication_followup_20261015_v11/repro_inventory_current_files_v2.jsonl.gz'
PUBLIC_INVENTORY_SHA256 = '75c5aaa33d534faefc028c3358d9be6b5cbe688599488a29d34478502fac47b5'
STYLE = ('body{font-family:sans-serif;max-width:1800px;margin:24px auto;padding:16px;line-height:1.7;'
         'background:#eef2f6;color:#172633}article,section{background:white;padding:18px;margin:16px 0;border-radius:10px}'
         '.tiles{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}img{width:100%;height:auto}'
         'table{border-collapse:collapse;width:100%}td,th{padding:6px;border-bottom:1px solid #ccc;text-align:left}'
         'nav a{display:inline-block;margin:8px}@media(max-width:1250px){.tiles{grid-template-columns:repeat(2,minmax(0,1fr))}}'
         '@media(max-width:850px){.tiles{grid-template-columns:1fr}}')


def public_image_inventory(source_root):
    path=Path(source_root)/PUBLIC_INVENTORY
    if S.sha(path)!=PUBLIC_INVENTORY_SHA256:raise ValueError('Published V11 image permission inventory changed')
    result={}
    with gzip.open(path,'rt')as stream:
        for line in stream:
            row=json.loads(line);name=row['relpath']
            if Path(name).suffix.lower()in('.png','.jpg','.jpeg'):
                result.setdefault(row['sha256'],name)
    return result


def _projection(frame,pose):
    if not pose.get('available',False):return None
    from scripts.research.pallet_visibility_curriculum_20261013_v1.render_raw_yolo_N3_subpix_gain10_process_v1 import projection
    return projection(pose,frame['K'],frame['xyz'])[0]


def _roi(frame,observations):
    """Shared viewport uses only initial predictions/coarse and measured BI."""
    h,w=frame['raw_hw'];parts=[]
    for branch,_,_,_ in BRANCHES:
        source=frame['arms'][branch]
        q=np.asarray(source['native_points'],float)[:8]
        keep=np.asarray(source['prediction_support'],bool)[:8]&np.isfinite(q).all(1)&~np.all(q==-1,axis=1)
        parts.append(q[keep])
        coarse=_projection(frame,source['coarse_pose'])
        if coarse is not None:parts.append(coarse)
        observed=observations[branch].get('observation')
        if observed is not None:
            for m in observed['frozen_visible_measurements']:
                if observed['states'][m['native_corner']]not in('B','I'):raise ValueError('Excluded measurement found')
                parts.append(np.asarray(m['xy'],float).reshape(1,2))
    q=np.concatenate(parts)if parts else np.empty((0,2));q=q[np.isfinite(q).all(1)]
    q=q[(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)]
    if not len(q):return [0.,float(w),0.,float(h)]
    lo,hi=q.min(0),q.max(0);center=(lo+hi)/2;span=np.maximum(hi-lo,[240.,180.]);half=span/2+np.maximum(32.,span*.2)
    return [max(0.,float(center[0]-half[0])),min(float(w),float(center[0]+half[0])),
            max(0.,float(center[1]-half[1])),min(float(h),float(center[1]+half[1]))]


def draw_case(frame,prediction,observations,poses,score,folder,source_root):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt,font_manager,patheffects
    from matplotlib.lines import Line2D
    from PIL import Image
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=FONT).get_name(),'axes.unicode_minus':False})
    path=Path(source_root)/frame['image']
    if S.sha(path)!=frame['image_sha256']:raise ValueError('Original RGB bytes changed')
    image=np.asarray(Image.open(path).convert('RGB'))
    if list(image.shape[:2])!=frame['raw_hw']:raise ValueError('Same camera canvas required')
    folder=Path(folder);folder.mkdir();shutil.copyfile(path,folder/('source_rgb'+path.suffix.lower()))
    roi=_roi(frame,observations);proof={};tiles=[]
    def canvas(title):
        fig,ax=plt.subplots(figsize=(10,7.4),layout='constrained');ax.imshow(image,interpolation='nearest')
        ax.set_xlim(roi[:2]);ax.set_ylim(roi[3],roi[2]);ax.set_aspect('equal');ax.set_title(title,fontsize=15)
        ax.set_xlabel('원본 x [px]');ax.set_ylabel('원본 y [px]');return fig,ax
    def point(ax,q,k,color,marker='o'):
        ax.scatter(*q,color=color,marker=marker,s=48,zorder=7)
        text=ax.annotate(str(k),q,xytext=(6,10 if k%2 else -14),textcoords='offset points',color=color,fontsize=12,weight='bold')
        text.set_path_effects([patheffects.withStroke(linewidth=2.2,foreground='white')])
    def wire(ax,q,color,label):
        if q is None:return
        for i,(a,b)in enumerate(EDGES):ax.plot(q[[a,b],0],q[[a,b],1],color=color,lw=1.7,label=label if i==0 else None)
    def finish(fig,ax,name):
        handles,_=ax.get_legend_handles_labels()
        if handles:ax.legend(fontsize=10)
        fig.savefig(folder/name,dpi=135);plt.close(fig);tiles.append(name)
    for branch,before,after,label in BRANCHES:
        source=frame['arms'][branch];initial=source['coarse_pose'];initial_xy=_projection(frame,initial)
        result=observations[branch];obs=result.get('observation');pose=prediction['predictions'][after]['actual_pose']
        details=None if obs is None else observation_summary(obs,source['native_points'],poses[branch].get('result'))
        q=np.asarray(source['native_points'],float);support=np.asarray(source['prediction_support'],bool)
        fig,ax=canvas('① '+label+' · 보정 전 예측과 자기 F')
        wire(ax,initial_xy,'#8852b6','저장 초기 F 모델')
        for k in range(8):
            if support[k]and np.isfinite(q[k]).all()and not np.all(q[k]==-1):point(ax,q[k],k,'#cf5938','x')
        finish(fig,ax,branch+'_01_initial.png')
        fig,ax=canvas('② '+label+' · 자기 초기 자세로 다시 판정한 상태')
        wire(ax,initial_xy,'#a9a3b1','초기 모델 — S는 모델 자기 차폐')
        if obs is None:ax.text(.03,.95,'초기 F 미산출: 상태/영상 보정 없음',transform=ax.transAxes,va='top')
        else:
            for k,state in enumerate(obs['states']):
                if state in COLORS:point(ax,q[k],k,COLORS[state])
        finish(fig,ax,branch+'_02_states.png')
        fig,ax=canvas('③ '+label+' · B/I 실제 측정과 원예측 유지')
        movement=[]
        if details is not None:
            measures={m['native_corner']:m for m in obs['frozen_visible_measurements']}
            for m in details['measurements']:
                k=m['native_corner'];a=np.asarray(source['native_points'][k],float);b=np.asarray(measures[k]['xy'],float)
                ax.scatter(*a,color='#777777',marker='x',s=42,zorder=6)
                if m['original_prediction_retained']:point(ax,b,k,'#777777','s')
                else:point(ax,b,k,COLORS[m['state']])
                if m['moved']:ax.annotate('',b,a,arrowprops=dict(arrowstyle='->',color=COLORS[m['state']],lw=2))
                movement.append(dict(native_corner=k,state=m['state'],displacement_px=m['displacement_px'],source=m['source'],reason=m['reason']))
            ax.legend(handles=[Line2D([],[],color='#777777',marker='x',ls='',label='입력 예측'),
                Line2D([],[],color=COLORS['B'],marker='o',ls='',label='B 실제 영상 측정'),
                Line2D([],[],color=COLORS['I'],marker='o',ls='',label='I 실제 영상 측정'),
                Line2D([],[],color='#777777',marker='s',ls='',label='근거 부족: 원예측 유지')],fontsize=9)
        else:ax.text(.03,.95,'B/I 관측 없음',transform=ax.transAxes,va='top')
        finish(fig,ax,branch+'_03_measurements.png')
        fig,ax=canvas('④ '+label+' · 공통 B/I-only PnP 최종 모델')
        final_xy=_projection(frame,pose)
        if final_xy is None:ax.text(.03,.95,'최종 pose 미산출 — 전체 분모의 실패로 유지',transform=ax.transAxes,va='top')
        else:wire(ax,final_xy,'#ef841b','최종 모델 투영 — 실제 측정점 아님')
        finish(fig,ax,branch+'_04_final.png')
        proof[branch]=dict(initial_projection_xy=None if initial_xy is None else initial_xy.tolist(),
            final_projection_xy=None if final_xy is None else final_xy.tolist(),diagnostics=details,movement=movement,
            prediction_status=score['prediction_status'][after],metrics_before=score['metrics'][before],metrics_after=score['metrics'][after])
    return dict(id=frame['id'],ROI_original_xy=roi,branches=proof,tiles=tiles,
                initial_markers='initial predictions, not new S observations',S_O_T_U_measurement_markers=0,
                final_wire='model projection only',new_NN_PnP_or_measurement_calls=0)


def render(*,folder,input_dir,score_dir,source_root,output,expected_prediction_seal_sha256,
           expected_score_seal_sha256,root_gallery_GO=False):
    if root_gallery_GO is not True:raise PermissionError('Postscore gallery GO before IO')
    S._digest(expected_prediction_seal_sha256);S._digest(expected_score_seal_sha256)
    folder,input_dir,score_dir,output=map(Path,(folder,input_dir,score_dir,output))
    if output.exists():raise FileExistsError('Preserve every gallery attempt')
    if S.sha(score_dir/'SCORE_SEAL.json')!=expected_score_seal_sha256:raise ValueError('Score seal changed')
    seal=S.read(score_dir/'SCORE_SEAL.json')
    if seal['prediction_seal_sha256']!=expected_prediction_seal_sha256:raise ValueError('Same complete scored prediction required')
    for name,digest in seal['assets'].items():
        if S.sha(score_dir/name)!=digest:raise ValueError('Saved scoring asset changed')
    S.prediction_gate(folder=folder,input_dir=input_dir,expected_prediction_seal_sha256=expected_prediction_seal_sha256)
    scored=S.read(score_dir/'SCORED_ROWS.json')['rows'];selection=changed_case_groups(scored)
    selected=sorted({v for v in selection['first_id'].values()if v is not None})
    frames=S.read(input_dir/'INPUTS.json')['frames'];indices={f['id']:i for i,f in enumerate(frames)}
    permissions=public_image_inventory(source_root)
    for fid in selected:
        if frames[indices[fid]]['image_sha256']not in permissions:
            raise ValueError('Selected RGB public permission is unverified; do not substitute another example')
    predictions={r['id']:r for r in S._rows(folder/'PREDICTIONS.jsonl.gz')if r['id']in selected}
    scoremap={r['id']:r for r in scored};output.mkdir(parents=True);assets={};cases=[]
    for case,fid in enumerate(selected):
        i=indices[fid];frame=frames[i]
        observed={b:S.read(folder/'observations'/f'{i:03d}_{b}.json')['result']for b,_,_,_ in BRANCHES}
        poses={b:S.read(folder/'poses'/f'{i:03d}_{b}.json')for b,_,_,_ in BRANCHES}
        name=f'case{case:02d}';proof=draw_case(frame,predictions[fid],observed,poses,scoremap[fid],output/name,source_root)
        proof['public_RGB_proof']=dict(inventory_path=str(PUBLIC_INVENTORY),inventory_sha256=PUBLIC_INVENTORY_SHA256,
            original_public_path=permissions[frame['image_sha256']],image_sha256=frame['image_sha256'])
        S.write(output/name/'PROCESS.json',proof)
        sections=[]
        for b,_,_,label in BRANCHES:
            p=proof['branches'][b];m=p['metrics_after'];status=p['prediction_status']
            state='NA'if p['diagnostics']is None else ', '.join(k+':'+str(v)for k,v in p['diagnostics']['state_counts'].items())
            success='참조 회전 미해결'if m['success_le5']is None else'평가 기준 통과'if m['success_le5']else'평가 기준 실패'
            constraint='제약 수락 NA'if status['accepted']is None else'PnP 제약 통과'if status['accepted']else'폴백'if status['fallback']else'미산출/제약 미수락'
            tiles=[x for x in proof['tiles']if x.startswith(b+'_')]
            section=f'<section><h2>{html.escape(label)}</h2><p>{html.escape(state)} · {constraint} · {success} · T={m["translation_cm"]}cm / R={m["rotation_deg"]}°</p><div class="tiles">'
            section+=''.join(f'<a href="{t}" target="_blank"><img src="{t}" alt="{html.escape(t)}"></a>'for t in tiles)+'</div>'
            section+='<table><tr><th>코너</th><th>상태</th><th>실제 이동 px</th><th>출처/유지 사유</th></tr>'
            section+=''.join(f'<tr><td>{v["native_corner"]}</td><td>{v["state"]}</td><td>{v["displacement_px"]:.3f}</td><td>{html.escape(v["source"]+" / "+str(v["reason"]))}</td></tr>'for v in p['movement'])+'</table></section>'
            sections.append(section)
        text=f'<!doctype html><html lang="ko"><meta charset="utf-8"><title>최소 B/I 과정 · {html.escape(fid)}</title><style>{STYLE}</style><a href="../index.html">사례 목록</a><h1>{html.escape(fid)}</h1><p>실제 측정 원/원예측 유지 사각형과 주황색 최종 모델 wire를 구분합니다. S/O/T/U는 측정점으로 표시하지 않습니다. 초기 점은 상태 판정 전 예측입니다. 중간 pose 궤적을 생성하지 않았습니다.</p>'+''.join(sections)+'</html>'
        (output/name/'index.html').write_text(text);cases.append(dict(id=fid,path=name+'/index.html',tiles=[name+'/'+t for t in proof['tiles']]))
    S.write(output/'SELECTION.json',selection)
    nav=''.join(f'<li><a href="{c["path"]}">{html.escape(c["id"])}</a></li>'for c in cases)
    (output/'index.html').write_text(f'<!doctype html><html lang="ko"><meta charset="utf-8"><title>최소 B/I — 회복과 손상 과정</title><style>{STYLE}</style><h1>양쪽 모두 최소 B/I 영상보정 적용</h1><p>전체192장의 사전 고정 범주에서 ID순 첫 사례를 표시합니다. 성공만 고르지 않았습니다. 실제 존재하지 않는 범주는 SELECTION.json에 없음으로 남깁니다. HTML은 clone 후 로컬 서버로 열 수 있고 GitHub에서는 아래 Markdown/PNG를 사용할 수 있습니다.</p><ul>{nav}</ul><p><a href="GALLERY_KO.md">Markdown 그림 목록</a></p></html>')
    md=['# 최소 B/I 과정 갤러리','', '회복·손상·N1만 회복·양쪽 실패에서 고정 ID 순서 첫 사례다. 부류가 없으면 대체 사례를 고르지 않는다. 각 PNG는 클릭해 확대할 수 있다. 최종 모델 wire는 측정점이 아니다.','']
    md+=['존재하지 않는 범주: '+(', '.join(selection['absent_categories'])or'없음'),'']
    for c in cases:
        md += ['## '+c['id'],'',f'[개별 HTML]({c["path"]})','']
        md += [f'![{Path(t).stem}]({t})'for t in c['tiles']]+['']
    (output/'GALLERY_KO.md').write_text('\n'.join(md))
    for p in output.rglob('*'):
        if p.is_file():assets[str(p.relative_to(output))]=S.sha(p)
    S.write(output/'GALLERY_SEAL.json',dict(schema='MINIMAL_BI_SAVED_STAGE_GALLERY_V1',cases=len(cases),stage_images=len(cases)*8,
        image_files=sum(Path(name).suffix.lower()in('.png','.jpg','.jpeg')for name in assets),
        prediction_seal_sha256=expected_prediction_seal_sha256,score_seal_sha256=expected_score_seal_sha256,
        assets=assets,source_sha256={Path(__file__).name:S.sha(__file__)},
        font_binding=dict(name=FONT.name,sha256=S.sha(FONT)),public_RGB_inventory_sha256=PUBLIC_INVENTORY_SHA256,
        GT_wire_or_reference_decodes=0,new_inference_or_posefit_calls=0))
    return S.sha(output/'GALLERY_SEAL.json')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in('folder','input-dir','score-dir','source-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prediction-seal-sha256',required=True);p.add_argument('--score-seal-sha256',required=True)
    p.add_argument('--root-gallery-GO',action='store_true');a=p.parse_args()
    print(render(folder=a.folder,input_dir=a.input_dir,score_dir=a.score_dir,source_root=a.source_root,output=a.output,
        expected_prediction_seal_sha256=a.prediction_seal_sha256,expected_score_seal_sha256=a.score_seal_sha256,
        root_gallery_GO=a.root_gallery_GO))

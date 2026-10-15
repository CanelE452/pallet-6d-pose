"""Sealed, actual four-stage N3-only/SubPix-only comparison, with no fitting.

Only saved image predictions, own observations, own PnP and numeric scores are
read. No label/reference file, new image measurement or predictor is called.
JPEG panels are display copies; separately copied source RGB retains its hash.
"""
from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import html
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.research.pallet_guarded_bi_final_v1 import gallery as GEOMETRY


SCHEMA = 'SINGLE_COMPONENT_BI_ACTUAL_PROCESS_GALLERY_V1'
BRANCHES = (('N3_ONLY','N3_F','N3_BI','N3만 초기화'),
            ('SUBPIX_ONLY','SUBPIX_F','SUBPIX_BI','SubPix만 초기화'))
ARMS = ('R0','R1','N3_F','N3_BI','SUBPIX_F','SUBPIX_BI','N0','N1')
COLORS = {'B':'#009fc3','I':'#8846cb'}
JPEG_QUALITY = 90
STYLE = ('body{font-family:sans-serif;max-width:1750px;margin:22px auto;padding:18px;'
         'line-height:1.65;background:#eef2f6;color:#172633}article,section{background:white;'
         'padding:18px;margin:16px 0;border-radius:10px}.tiles{display:grid;'
         'grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}img{width:100%;height:auto}'
         'table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:6px;'
         'border-bottom:1px solid #ccc;text-align:left;overflow-wrap:anywhere}'
         '@media(max-width:1300px){.tiles{grid-template-columns:repeat(2,minmax(0,1fr))}}'
         '@media(max-width:750px){.tiles{grid-template-columns:1fr}}')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with gzip.open(path,'rt') as stream:
        yield from map(json.loads,stream)


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def selection(scored):
    """Four fixed disagreement groups; absent groups never get substitutes."""
    primary=sorted((r for r in scored if r['partition']!='CAL'),key=lambda r:r['id'])
    categories=[]
    chosen={}
    for arm in ('N3_BI','SUBPIX_BI'):
        for own,combined in ((True,False),(False,True)):
            name=arm+('만 성공 / N1 실패' if own else '실패 / N1만 성공')
            fid=next((r['id'] for r in primary
                if r['metrics'][arm]['success_le5'] is own
                and r['metrics']['N1']['success_le5'] is combined),None)
            categories.append(dict(category=name,id=fid,absent=fid is None))
            if fid is not None:
                chosen.setdefault(fid,dict(id=fid,categories=[]))['categories'].append(name)
    return dict(schema=SCHEMA,categories=categories,cases=list(chosen.values()),
        absent_categories=[r['category'] for r in categories if r['absent']],maximum_cases=4,
        selection_rule='PRIMARY first lexicographic ID in each fixed N3_BI/SUBPIX_BI versus N1 success-disagreement group, deduplicated',
        image_selection_after_scoring_only=True,method_selection_or_inference_effect=0)


def metric_text(m):
    if m['translation_cm'] is None:
        return 'pose 없음 · 전체 분모 실패'
    rotation='NA' if m['rotation_deg'] is None else f'{m["rotation_deg"]:.3f}'
    result='성공' if m['success_le5'] is True else '실패' if m['success_le5'] is False else '회전 참조 미확정'
    return f'T {m["translation_cm"]:.3f}cm / R {rotation}° · {result}'


def draw_case(frame,prediction,observations,solutions,score,folder,source_root):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt,font_manager,patheffects
    from matplotlib.lines import Line2D
    from PIL import Image

    font_manager.fontManager.addfont(str(GEOMETRY.FONT))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=GEOMETRY.FONT).get_name(),
                         'axes.unicode_minus':False})
    source_path=Path(source_root)/frame['image']
    if sha(source_path)!=frame['image_sha256']:
        raise ValueError('Actual source RGB bytes changed')
    image=np.asarray(Image.open(source_path).convert('RGB'))
    if list(image.shape[:2])!=frame['raw_hw']:
        raise ValueError('Original source camera canvas changed')
    height,width=image.shape[:2]
    folder=Path(folder)
    folder.mkdir()
    shutil.copyfile(source_path,folder/('source_rgb'+source_path.suffix.lower()))
    proof=dict(id=frame['id'],image_sha256=frame['image_sha256'],branches={},tiles=[],
        viewport_original_xy=[0,width-1,0,height-1],same_full_RGB_canvas_every_panel=True,
        JPEG_display=dict(quality=JPEG_QUALITY,source_RGB_copy_is_byte_exact=True),
        S_O_T_U_actual_measurement_markers=0,GT_wire_or_reference_file_reads=0,
        new_NN_PnP_or_image_measurement_calls=0,model_projection_is_not_actual_measurement=True)

    def canvas(title):
        fig,ax=plt.subplots(figsize=(10.4,8.2),layout='constrained')
        ax.imshow(image,interpolation='nearest')
        ax.set_xlim(-.5,width-.5)
        ax.set_ylim(height-.5,-.5)
        ax.set_aspect('equal')
        ax.set_title(title,fontsize=14)
        ax.set_xlabel('원영상 x [px]')
        ax.set_ylabel('원영상 y [px]')
        return fig,ax

    def point(ax,xy,k,color,marker='o',state=None,size=36):
        xy=np.asarray(xy,float)
        if not np.isfinite(xy).all() or not (0<=xy[0]<width and 0<=xy[1]<height):
            return
        ax.scatter(*xy,color=color,marker=marker,s=size,zorder=7)
        tag=ax.annotate(str(k)+((':'+state) if state else ''),xy,
            xytext=(5,-12 if k%2 else 10),textcoords='offset points',
            color=color,fontsize=10,weight='bold',clip_on=True)
        tag.set_path_effects([patheffects.withStroke(linewidth=2,foreground='white')])

    def wire(ax,xy,color,label,clipping,token,linestyle='-',linewidth=1.6):
        record=dict(outside_native_vertices=0,clipped_edges=0,invisible_edges=0,
            display='Same full RGB canvas; offcanvas model segments are clipped, never rescaled')
        clipping[token]=record
        if xy is None:
            record['pose_unavailable_or_not_positive_depth']=True
            return
        record['outside_native_vertices']=int(np.sum((xy[:,0]<0)|(xy[:,0]>=width)|(xy[:,1]<0)|(xy[:,1]>=height)))
        labelled=False
        for first,second in GEOMETRY.EDGES:
            original=xy[[first,second]]
            visible=GEOMETRY.clip_segment(*original,width,height)
            if visible is None:
                record['invisible_edges']+=1
                continue
            record['clipped_edges']+=int(not np.array_equal(original,visible))
            ax.plot(visible[:,0],visible[:,1],color=color,lw=linewidth,ls=linestyle,
                    label=label if not labelled else None)
            labelled=True

    def save(fig,ax,name,clipping):
        handles,_=ax.get_legend_handles_labels()
        if handles and ax.get_legend() is None:
            ax.legend(fontsize=9,loc='upper right')
        if any(r.get('outside_native_vertices',0) for r in clipping.values()):
            ax.text(.01,.01,'화면 밖 모델 선은 원영상 경계에서 잘라 표시',
                transform=ax.transAxes,fontsize=9,color='#222',bbox=dict(facecolor='white',alpha=.9))
        fig.savefig(folder/name,dpi=120,format='jpg',
            pil_kwargs={'quality':JPEG_QUALITY,'optimize':True,'subsampling':0})
        plt.close(fig)
        proof['tiles'].append(name)

    for branch,before,after,label in BRANCHES:
        source=frame['arms'][branch]
        initial=GEOMETRY.projection(frame,source['coarse_pose'])
        final=GEOMETRY.projection(frame,prediction['predictions'][after]['actual_pose'])
        obs=observations[branch].get('observation')
        solution=solutions[branch]
        states=['U']*8 if obs is None else obs['states']
        raw=np.asarray(source['native_points'],float)
        support=np.asarray(source['prediction_support'],bool)
        measured=[] if obs is None else obs['frozen_visible_measurements']
        if any(states[m['native_corner']] not in ('B','I') for m in measured):
            raise ValueError('Excluded state provided as a measured 2D corner')
        rays,excluded=([],[]) if obs is None else GEOMETRY._supported_rays(obs)
        clipping={}
        detail=dict(states=states,initial_projection_xy=None if initial is None else initial.tolist(),
            final_projection_xy=None if final is None else final.tolist(),
            projection_basis='Own CF cube from cf_extents with R_cf and centroid; Q authenticates physical vertex set',
            displayed_initial_native_predictions=[dict(native_corner=k,xy=raw[k].tolist())
                for k,s in enumerate(states) if s in ('B','I') and support[k]],
            actual_BI_measurements=deepcopy(measured),supported_observed_rays=[dict(r,native_corner=k) for k,r in rays],
            excluded_rays=excluded,metrics_before=score['metrics'][before],metrics_after=score['metrics'][after],
            initial_pose_source=source['coarse_pose'].get('source'),
            final_pose_source=prediction['predictions'][after]['actual_pose'].get('source'),
            accepted=solution.get('accepted'),fallback=solution.get('fallback'),
            solver_status=(solution.get('solution') or {}).get('status'),
            solver_fallback_reason=(solution.get('solution') or {}).get('fallback_reason'),
            model_clipping=clipping,movement=[],S_O_T_U_measurement_markers=0,
            synthetic_midstage_pose_or_GT_wire=False)

        fig,ax=canvas('① '+label+' · 보정 전 입력과 자기 F\n'+metric_text(score['metrics'][before]))
        wire(ax,initial,'#8053b2','자기 초기 F 모델 투영',clipping,'stage1_initial')
        for k,state in enumerate(states):
            if state in ('B','I') and support[k] and np.isfinite(raw[k]).all() and not np.all(raw[k]==-1):
                point(ax,raw[k],k,'#df653a','x')
        ax.legend(handles=[Line2D([],[],color='#8053b2',label='자기 초기 모델 투영'),
            Line2D([],[],color='#df653a',marker='x',ls='',label='보정 전 native 예측 · B/I만 표시')],fontsize=9)
        save(fig,ax,branch+'_01_initial.jpg',clipping)

        fig,ax=canvas('② '+label+' · 자기 초기 자세로 판정한 상태\n마커는 보정 전 예측 · S/O/T/U는 모델 위치에 글자만 표시')
        wire(ax,initial,'#aaa0b6','자기 초기 모델',clipping,'stage2_initial',linewidth=1.)
        for k,state in enumerate(states):
            if state in ('B','I') and support[k]:
                point(ax,raw[k],k,COLORS[state],'x',state=state)
            elif initial is not None and 0<=initial[k,0]<width and 0<=initial[k,1]<height:
                ax.text(*initial[k],f'모델{k}:{state}',fontsize=8,color='#69616f',clip_on=True)
        state_counts={s:states.count(s) for s in ('B','I','S','O','T','U')}
        ax.text(.01,.99,' · '.join(s+':'+str(n) for s,n in state_counts.items()),
            transform=ax.transAxes,va='top',fontsize=10,bbox=dict(facecolor='white',alpha=.9))
        if obs is None:
            ax.text(.01,.94,'초기 pose 없음: 상태/영상 보정 없음',transform=ax.transAxes,va='top')
        save(fig,ax,branch+'_02_states.jpg',clipping)

        fig,ax=canvas('③ '+label+' · 실제 B/I와 관측 선 · 이동/유지\n원은 영상 측정, 사각형은 원예측 유지 · 모델 wire는 투영')
        wire(ax,initial,'#b7aebe','자기 초기 모델',clipping,'stage3_initial',linewidth=1.)
        for k,ray in rays:
            points=np.asarray(ray['observed_points'],float)
            n=np.asarray(ray['normal'],float);norm=np.linalg.norm(n);n/=norm
            tangent=np.array([-n[1],n[0]])
            interval=points@tangent
            fitted=-float(ray['offset'])/norm*n+np.array([interval.min(),interval.max()])[:,None]*tangent
            ax.scatter(points[:,0],points[:,1],color='#f1c624',s=5,alpha=.9,zorder=4)
            ax.plot(fitted[:,0],fitted[:,1],color='#cfaa00',lw=1.1,zorder=4)
        for m in measured:
            k=m['native_corner'];xy=np.asarray(m['xy'],float)
            retained=m['source']=='VISIBLE_BACKBONE_FALLBACK'
            point(ax,xy,k,'#666666' if retained else COLORS[states[k]],'s' if retained else 'o',states[k])
            movement=float(np.linalg.norm(xy-raw[k]))
            detail['movement'].append(dict(native_corner=k,state=states[k],source=m['source'],
                original_prediction_retained=retained,displacement_px=movement,xy=m['xy']))
            if not retained and movement>1e-8:
                ax.annotate('',xy,raw[k],arrowprops=dict(arrowstyle='->',color=COLORS[states[k]],lw=1.5))
        ax.legend(handles=[Line2D([],[],color='#f1c624',marker='.',label='실제 관측 선 표본/지지구간'),
            Line2D([],[],color=COLORS['B'],marker='o',ls='',label='B 영상 측정'),
            Line2D([],[],color=COLORS['I'],marker='o',ls='',label='I 영상 측정'),
            Line2D([],[],color='#666666',marker='s',ls='',label='원예측 유지')],fontsize=9)
        save(fig,ax,branch+'_03_BI_observations.jpg',clipping)

        fig,ax=canvas('④ '+label+' · 실제 최종 B/I PnP 모델\n'+metric_text(score['metrics'][after]))
        wire(ax,initial,'#96919c','자기 초기 모델',clipping,'stage4_initial','--',1.1)
        wire(ax,final,'#ee861c','최종 PnP 모델 투영',clipping,'stage4_final')
        for m in measured:
            k=m['native_corner'];retained=m['source']=='VISIBLE_BACKBONE_FALLBACK'
            point(ax,m['xy'],k,'#666666' if retained else COLORS[states[k]],'s' if retained else 'o',size=20)
        status='PnP 제약 수락' if solution.get('accepted') is True else 'PnP 미수락 · 저장 fallback'
        note='초기: '+metric_text(score['metrics'][before])+'\n최종: '+metric_text(score['metrics'][after])+'\n'+status
        if detail['solver_fallback_reason']:
            note+='\n'+detail['solver_fallback_reason']
        ax.text(.01,.99,note,transform=ax.transAxes,va='top',fontsize=8.5,
            color='#222',bbox=dict(facecolor='white',alpha=.9))
        save(fig,ax,branch+'_04_final.jpg',clipping)
        proof['branches'][branch]=detail
    return proof


def render(folder,prior,source_root,output):
    folder,prior,source_root,output=map(Path,(folder,prior,source_root,output))
    if output.exists():
        raise FileExistsError('Preserve every gallery attempt')
    pseal=read(folder/'PREDICTION_SEAL.json');sseal=read(folder/'score/SCORE_SEAL.json')
    if (pseal['rows']!=238 or pseal['arm_rows']!=1904 or pseal['arms']!=list(ARMS)
            or pseal['predictions_sha256']!=sha(folder/'PREDICTIONS.jsonl.gz')
            or pseal['independent_bindings_sha256']!=sha(folder/'INDEPENDENT_BINDINGS.json')
            or pseal['observation_seal_sha256']!=sha(folder/'run/OBSERVATION_SEAL.json')
            or sseal['prediction_seal_sha256']!=sha(folder/'PREDICTION_SEAL.json')):
        raise ValueError('Complete sealed predictor/observation/score chain required')
    for name,digest in sseal['artifacts'].items():
        if sha(folder/'score'/name)!=digest:
            raise ValueError('Saved numeric score artifact changed')
    input_seal=read(folder/'inputs_v1/INPUT_SEAL.json')
    if (input_seal['inputs_sha256']!=sha(folder/'inputs_v1/INPUTS.json')
            or input_seal['prior_inputs_sha256']!=sha(prior/'inputs_v2/INPUTS.json')):
        raise ValueError('Own initializer or prior raw metadata changed')
    frames=read(folder/'inputs_v1/INPUTS.json')['frames']
    scored=read(folder/'score/SCORED_ROWS.json')['rows']
    predicted=list(rows(folder/'PREDICTIONS.jsonl.gz'))
    if ([f['id'] for f in frames]!=pseal['ordered_ids'] or [r['id'] for r in scored]!=pseal['ordered_ids']
            or [r['id'] for r in predicted]!=pseal['ordered_ids']):
        raise ValueError('Fixed238 pairing changed')
    permissions=GEOMETRY.public_image_inventory(source_root)
    for frame in frames:
        if frame['image_sha256'] not in permissions or sha(source_root/frame['image'])!=frame['image_sha256']:
            raise ValueError('Original238 public RGB permission or hash unverified')
    selected=selection(scored)
    predicted={r['id']:r for r in predicted};scores={r['id']:r for r in scored}
    indices={f['id']:i for i,f in enumerate(frames)}
    bindings={r['id']:r for r in read(folder/'INDEPENDENT_BINDINGS.json')['records']}
    output.mkdir(parents=True)
    write(output/'SELECTION.json',selected)
    cases=[]
    for index,case in enumerate(selected['cases']):
        fid=case['id'];frame=frames[indices[fid]]
        observed={};solved={};bound={}
        for branch,_,after,_ in BRANCHES:
            bound[branch]={}
            for kind in ('observation','solution'):
                b=bindings[fid]['branches'][branch][kind];p=Path(b['path']).resolve()
                if not p.is_relative_to(folder.resolve()) or sha(p)!=b['sha256']:
                    raise ValueError('Foreign or changed own saved stage')
                value=read(p)
                if value['id']!=fid or value['branch']!=branch:
                    raise ValueError('Foreign own-initialization record')
                bound[branch][kind]=dict(path=str(p.relative_to(folder.resolve())),sha256=b['sha256'])
                if kind=='observation':observed[branch]=value['result']
                else:solved[branch]=value['result']
            if solved[branch]['actual_pose']!=predicted[fid]['predictions'][after]['actual_pose']:
                raise ValueError('Saved actual solver and final prediction differ')
        name=f'case{index:02d}'
        proof=draw_case(frame,predicted[fid],observed,solved,scores[fid],output/name,source_root)
        proof.update(categories=case['categories'],input_bindings=bound,
            public_RGB_proof=dict(image_sha256=frame['image_sha256'],
                public_inventory_original_path=permissions[frame['image_sha256']],
                inventory_path=str(GEOMETRY.PUBLIC_INVENTORY),inventory_sha256=GEOMETRY.PUBLIC_INVENTORY_SHA256,
                entire238_permission_and_original_RGB_hashes_rechecked=True))
        write(output/name/'PROCESS.json',proof)
        sections=[]
        for branch,before,after,label in BRANCHES:
            detail=proof['branches'][branch]
            table='<table><tr><th>경로</th><th>실제 저장 오차</th></tr>'
            for arm in (before,after,'N1'):
                table+='<tr><td>'+arm+'</td><td>'+html.escape(metric_text(scores[fid]['metrics'][arm]))+'</td></tr>'
            table+='</table>'
            tiles=[t for t in proof['tiles'] if t.startswith(branch+'_')]
            section='<section><h2>'+html.escape(label)+'</h2>'+table+'<div class="tiles">'
            section+=''.join('<a href="'+t+'" target="_blank"><img src="'+t+'" alt="'+html.escape(t)+'"></a>' for t in tiles)+'</div>'
            section+='<table><tr><th>native KP</th><th>상태</th><th>실제 이동 px</th><th>측정 출처/유지</th></tr>'
            section+=''.join('<tr><td>'+str(m['native_corner'])+'</td><td>'+m['state']+'</td><td>'+f'{m["displacement_px"]:.3f}'+'</td><td>'+html.escape(m['source'])+'</td></tr>' for m in detail['movement'])+'</table></section>'
            sections.append(section)
        text='<!doctype html><html lang="ko"><meta charset="utf-8"><title>'+html.escape(fid)+'</title><style>'+STYLE+'</style>'
        text+='<a href="../index.html">사례 목록</a><h1>'+html.escape(fid)+'</h1><p>'+html.escape(' / '.join(case['categories']))+'</p>'
        text+='<p>두 초기화 각각 자기 F·상태·프롬프트/마스크·B/I·최종 PnP를 사용했습니다. 모든 단계는 같은 전체 원영상 범위입니다. S/O/T/U는 실제 측정 마커가 없고 최종 wire는 모델 투영입니다. 화면 밖 모델은 경계에서 잘라 표시합니다. JPEG는 quality90 표시본이며 source_rgb 원본 바이트는 따로 보존했습니다. 정답 선이나 만들어낸 중간 자세는 없습니다.</p>'
        text+=''.join(sections)+'<p><a href="PROCESS.json">실제 좌표와 출처</a></p></html>'
        (output/name/'index.html').write_text(text)
        cases.append(dict(**case,path=name+'/index.html',process=name+'/PROCESS.json',tiles=[name+'/'+t for t in proof['tiles']]))
    absent=' / '.join(selected['absent_categories']) or '없음'
    nav=''.join('<li><a href="'+c['path']+'">'+html.escape(c['id'])+'</a> · '+html.escape(' / '.join(c['categories']))+'</li>' for c in cases)
    (output/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>N3/SubPix 개별 B/I 과정</title><style>'+STYLE+'</style><h1>N3만·SubPix만: 각자의 상태별 영상 보정</h1><p>PRIMARY에서 N3_BI/SUBPIX_BI와 기존 결합 N1의 성공 여부가 다른 고정 네 범주의 ID순 첫 사례입니다. 중복을 합쳤고 없는 범주는 대체하지 않았습니다. 사례 선정은 방법과 지표에 영향이 없습니다.</p><p>없는 범주: '+html.escape(absent)+'</p><ul>'+nav+'</ul><p><a href="GALLERY_KO.md">Markdown 그림 목록</a></p></html>')
    md=['# N3만·SubPix만의 실제 B/I 과정', '',
        'PRIMARY에서 N3_BI/SubPix_BI와 결합 N1의 성공 여부가 다른 네 고정 범주의 ID순 첫 사례를 사용했다. 중복을 합쳤고 없는 범주는 대체하지 않았다. 각 초기화는 자기 상태·마스크·B/I와 PnP를 사용한다.', '',
        '모든 그림은 같은 전체 원영상 범위이며 S/O/T/U 실제 측정 마커와 정답 선은 없다. 화면 밖 모델은 원영상 경계에서 잘라 표시한다. JPG는 quality90 표시본이고 원본RGB는 바이트 그대로 복사했다.', '',
        '없는 범주: '+absent, '']
    for case in cases:
        md+=['## '+case['id'], '', ' / '.join(case['categories']), '', '[개별 HTML]('+case['path']+') · [실제 과정/좌표]('+case['process']+')', '',
             '| 경로 | 위치·회전 오차 |','|---|---|']
        for arm in ('N3_F','N3_BI','SUBPIX_F','SUBPIX_BI','N1'):
            md+=['| '+arm+' | '+metric_text(scores[case['id']]['metrics'][arm])+' |']
        md+=['']+['!['+Path(t).stem+']('+t+')' for t in case['tiles']]+['']
    (output/'GALLERY_KO.md').write_text('\n'.join(md))
    write(output/'PROCESS.json',dict(schema=SCHEMA,cases=cases,selection=selected,
        actual_stage_images=len(cases)*8,all238_public_RGB_permission_hashes_rechecked=True,
        GT_reference_file_reads=0,new_NN_PnP_calls=0))
    assets={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}
    write(output/'GALLERY_SEAL.json',dict(schema=SCHEMA,cases=len(cases),stage_images=len(cases)*8,
        JPEG_quality=JPEG_QUALITY,original_RGB_copies=len(cases),
        prediction_seal_sha256=sha(folder/'PREDICTION_SEAL.json'),score_seal_sha256=sha(folder/'score/SCORE_SEAL.json'),
        input_seal_sha256=sha(folder/'inputs_v1/INPUT_SEAL.json'),
        public_RGB_inventory_sha256=GEOMETRY.PUBLIC_INVENTORY_SHA256,font_sha256=sha(GEOMETRY.FONT),
        source_sha256={'gallery.py':sha(__file__),'fixed_projection_display_helpers.py':sha(GEOMETRY.__file__)},
        assets=assets,new_NN_PnP_calls=0,GT_reference_file_reads=0))
    return dict(cases=len(cases),stage_images=len(cases)*8,JPEG_quality=JPEG_QUALITY,
                gallery_seal_sha256=sha(output/'GALLERY_SEAL.json'))


if __name__=='__main__':
    import argparse
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ('folder','prior','source-root','output'):
        cli.add_argument('--'+name,type=Path,required=True)
    args=cli.parse_args()
    print(render(args.folder,args.prior,args.source_root,args.output))

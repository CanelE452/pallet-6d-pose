"""Six fixed real-image cases, rendered only after prediction/metric seals.

No reference/annotation reader, image measurement, network, or pose solver is
called. Pixel residuals compare stored observations and stored model poses;
pose T/R comes solely from sealed numerical score rows.
"""
from __future__ import annotations

from collections import Counter
import csv
import gzip
import hashlib
import html
import json
from pathlib import Path

import numpy as np

SCHEMA='STATEWISE_CORRESPONDENCE_ACTUAL_SIX_CASE_GALLERY_V1'
IDS=('wood_night_01:029428','plastic_day_01:010995','plastic_day_01:011640',
     'wood_day_01:005249','wood_night_01:031426','wood_night_01:031344')
ARMS=('REOBS_MULTI','REOBS_SINGLE','TYPED_MULTI','TYPED_SINGLE')
EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
COLORS={'B':'#02a2c4','I':'#9651cf','N0':'#e3af00','E0':'#188b56',
        'POSEFIX_RAW':'#e17b16','ORIGINAL_WD':'#1b9c87','SWAPPED_WD':'#ce6a28'}
JPEG_QUALITY=88
MAX_PUBLIC_BYTES=8*1024*1024


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def rows(path):
    with gzip.open(path,'rt')as stream:
        yield from map(json.loads,stream)


def wanted(path,ids=IDS):return {r['id']:r for r in rows(path)if r['id']in ids}


def cube(extents):
    a,b,c=np.asarray(extents,float)/2
    return np.array([[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],
                     [-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]])


def projection(pose,K,xyz=None):
    if not isinstance(pose,dict)or pose.get('available',True)is False:return None
    R=pose.get('R_cf',pose.get('R'));t=pose.get('centroid',pose.get('t'))
    if R is None or t is None:return None
    model=np.asarray(xyz,float)if xyz is not None else cube(pose['cf_extents'])
    camera=model@np.asarray(R,float).T+np.asarray(t,float);h=camera@np.asarray(K,float).T
    if not np.isfinite(h).all()or np.any(camera[:,2]<=0):return None
    return h[:,:2]/h[:,2,None]


def family_pose(solution,family):
    if solution.get('R')is None or solution.get('t')is None:return {'available':False}
    return dict(available=True,R_cf=solution['R'],centroid=solution['t'],cf_extents=family['cf_extents'],
                R_physical=(np.asarray(solution['R'])@np.asarray(family['physical_registration_Q'])).tolist())


def shared_roi(record):
    """Fixed viewport uses original N3/coarse predictions, never scores/finals."""
    a=record['A_original_N3'];h,w=a['geometry']['image_hw'];q=np.asarray(a['points'],float)[:8]
    good=np.isfinite(q).all(1)&~np.all(q==-1,axis=1);parts=[q[good]]
    p=projection(a['pose'],a['geometry']['K'])
    if p is not None:parts.append(p)
    points=np.concatenate(parts);points=points[(points[:,0]>=0)&(points[:,0]<w)&(points[:,1]>=0)&(points[:,1]<h)]
    if not len(points):return [0,0,w,h]
    lo,hi=points.min(0),points.max(0);span=np.maximum(hi-lo,[180.,110.]);pad=np.maximum(28.,span*.22)
    return [max(0,int(np.floor(lo[0]-pad[0]))),max(0,int(np.floor(lo[1]-pad[1]))),
            min(w,int(np.ceil(hi[0]+pad[0]))),min(h,int(np.ceil(hi[1]+pad[1])))]


def fitting_residuals(packet,pose):
    s=packet['solver_input'];p=projection(pose,s['K'],s['xyz'])
    if p is None:return dict(available=False)
    by={m['native_corner']:m for m in s['visible']};full=[];partial=[]
    for k in packet['activation']['complete_ids']:
        m=by[k];full.append(dict(native_corner=k,source=m.get('source'),
            measured_xy=m['xy'],observed_covariance_px2=m['covariance_px2'],
            fit_covariance_px2=m.get('fit_covariance_px2'),fit_reliability_weight=m.get('fit_reliability_weight'),
            model_to_stored_point_residual_px=float(np.linalg.norm(p[k]-m['xy']))))
    for b in s['boundary_observations']:
        for i,line in enumerate(b['lines']):
            n=np.asarray(line['normal_unit_xy'],float);d=float(n@p[b['native_corner']]+line['offset_px'])
            partial.append(dict(native_corner=b['native_corner'],line_index=i,normal_residual_px=abs(d),
                                normal_unit_xy=line['normal_unit_xy'],offset_px=line['offset_px'],
                                variance_px2=line['normal_variance_px2'],fit_reliability_weight=b['fit_reliability_weight'],
                                tangent_interval_px=line['support_tangent_interval_px']))
    return dict(available=True,full=full,partial=partial,
                meaning='model-to-stored-measurement fit residual, not GT keypoint localization error')


def _metrics(path,ids=IDS):
    out={}
    with Path(path).open(newline='')as stream:
        for r in csv.DictReader(stream):
            if r['id']not in ids:continue
            m={k:(float(r[k])if r.get(k)not in('',None)else None)for k in('translation_cm','rotation_deg')}
            m.update(success_le5=True if r.get('success_le5')=='True'else False if r.get('success_le5')=='False'else None,
                     status=r.get('status'),reason=r.get('reason'),pose_available=r.get('pose_available')=='True',
                     accepted=r.get('accepted'),fallback=r.get('fallback'))
            out.setdefault(r['id'],{})[r['arm']]=m
    return out


def metric_text(m):
    if not m or m['translation_cm']is None:return 'pose 없음 / 전체 분모 실패'
    r='NA'if m['rotation_deg']is None else f"{m['rotation_deg']:.2f}°"
    state='성공'if m['success_le5']is True else'실패'if m['success_le5']is False else'회전 참조 미확정'
    return f"T {m['translation_cm']:.2f}cm / R {r} · {state}"


def _seal_artifact(seal,filename):
    value=seal['artifacts'][filename]
    return value['sha256']if isinstance(value,dict)else value


def _selection_panels(cases,records,packets,baselines,metrics,root,output,permission):
    """Privileged postseal explanation; never a new operational selector."""
    from PIL import Image
    from matplotlib import pyplot as plt
    from scripts.research.pallet_guarded_bi_final_v1.gallery import clip_segment
    sections=[];proofs=[]
    for i,value in enumerate(cases,1):
        ident=value['id'];packet=packets[ident];record=records[ident];roi=shared_roi(record);x0,y0,x1,y1=roi
        if packet['image_sha256']not in permission:raise ValueError('Oracle case RGB permission absent')
        source=root/packet['image']
        if sha(source)!=packet['image_sha256']:raise ValueError('Oracle case actual RGB changed')
        image=Image.open(source).convert('RGB').crop(tuple(roi));case=output/f'oracle_case_{i:02d}';case.mkdir()
        thumbnail=image.copy();thumbnail.thumbnail((1200,900));thumbnail.save(case/'source_crop.jpg',quality=JPEG_QUALITY,optimize=True)
        candidate=value['candidate_pose'];selected=baselines[ident]['predictions']['E0']['actual_pose'];cm=value['candidate_metric'];K=packet['solver_input']['K']
        fig,axes=plt.subplots(1,3,figsize=(17,max(3.,min(7.,17/3*(y1-y0)/(x1-x0)+1.1))),layout='constrained')
        fig.suptitle(ident+' · 오프라인 참조를 본 후보 진단 — 새 추론 결과 아님',fontsize=14)
        titles=('실제 같은 RGB 확대', '실제 선택한 기존 e\n'+metric_text(metrics[ident]['E0']),
                '보관돼 있었던 공동성공 후보\n'+metric_text(cm))
        for ax,title in zip(axes,titles):
            ax.imshow(image,extent=(x0,x1,y1,y0),interpolation='nearest');ax.set_xlim(x0,x1);ax.set_ylim(y1,y0);ax.set_aspect('equal');ax.set_title(title,fontsize=11);ax.tick_params(labelsize=7)
        for ax,pose,color in ((axes[1],selected,'#cf512b'),(axes[2],candidate,'#168f5d')):
            q=projection(pose,K)
            if q is None:raise ValueError('Selection-case retained pose unavailable')
            for a,b in EDGES:
                line=clip_segment(q[a]-[x0,y0],q[b]-[x0,y0],x1-x0,y1-y0)
                if line is not None:
                    line+=np.array([x0,y0]);ax.plot(line[:,0],line[:,1],color=color,lw=1.7)
            for m in packet['solver_input']['visible']:
                if m['native_corner']not in packet['activation']['complete_ids']:continue
                xy=m['xy'];fallback=m.get('source')=='VISIBLE_BACKBONE_FALLBACK'
                ax.scatter(*xy,s=24,marker='x'if fallback else'o',color='#666666'if fallback else COLORS[m['state']])
        fig.savefig(case/'selection.jpg',dpi=120,pil_kwargs={'quality':JPEG_QUALITY,'optimize':True});plt.close(fig)
        proof=dict(id=ident,selection_reason=value['failure_stage'],candidate_id=value['candidate_id'],
            source_RGB_sha256=packet['image_sha256'],source_crop=f'oracle_case_{i:02d}/source_crop.jpg',viewport_original_xyxy=roi,
            selected_e_pose=selected,selected_e_metric=metrics[ident]['E0'],candidate_pose=candidate,candidate_metric=cm,
            candidate_provenance=value.get('source',{}),semantic_owner_status='SEMANTIC_OWNER_UNVERIFIED',
            candidate_objective=value.get('candidate_objective'),selected_e_objective=value.get('selected_e_objective'),
            candidate_objective_components=value.get('candidate_objective_components'),
            selected_e_objective_components=value.get('selected_e_objective_components'),
            G10_family_objectives=value.get('G10_family_objectives'),G13_family_objectives=value.get('G13_family_objectives'),
            privileged_offline_reference_oracle=True,operational_inference_or_routing=False,new_pose_or_candidate_generation=0,
            original_activation=packet['activation'])
        proofs.append(proof)
        objective=value.get('candidate_objective');objective_text='NA'if objective is None else str(objective)
        sections.append(f'<article id="oracle{i}"><h2>추가 {i}. {html.escape(ident)} · {html.escape(value["failure_stage"])}</h2><p><strong>봉인 뒤 기존 참조를 본 오프라인 oracle입니다. 새 추론 결과나 운영 selector가 아닙니다.</strong> PRIMARY의 해당 실패 단계에서 ID 사전순 첫 사례를 고정했습니다. 저장 후보 <code>{html.escape(value["candidate_id"])}</code>를 그대로 그렸습니다.</p><img src="oracle_case_{i:02d}/selection.jpg" alt="실제 선택과 보관된 성공 후보의 오프라인 비교"><p>후보의 원 목적함수: {html.escape(objective_text)}. 실제 선택의 목적함수와 비교할 때 G10/G13 관측 패키지를 구분합니다. source 종류·full/partial/region과 physical owner 가설은 아래 출처에 보존하며 covariance가 owner 정답을 증명하지 않습니다.</p><details><summary>후보·관측·물리 대응의 원 출처</summary><pre>{html.escape(json.dumps(proof,ensure_ascii=False,indent=2))}</pre></details></article>')
    return ''.join(sections),proofs


def render(folder,source_root,output,*,predictions_path=None,metrics_path=None):
    """Requires final seals; callers must wait for the root's explicit render GO."""
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt,font_manager,patheffects
    from matplotlib.patches import Ellipse,Rectangle
    from scripts.research.pallet_minimal_bi_ablation_v1.gallery import public_image_inventory,FONT
    from scripts.research.pallet_guarded_bi_final_v1.gallery import clip_segment

    folder,root,output=Path(folder),Path(source_root),Path(output)
    if output.exists():raise FileExistsError('Preserve previous actual gallery')
    pp=Path(predictions_path)if predictions_path else folder/'REPLAY_PREDICTIONS.jsonl.gz'
    mp=Path(metrics_path)if metrics_path else folder/'score_v1/ROW_METRICS.csv'
    ps=read(pp.parent/'PREDICTION_SEAL.json');ss=read(mp.parent/'SCORE_SEAL.json')
    if sha(pp)!=ps['predictions_sha256']or sha(mp)!=_seal_artifact(ss,mp.name):raise ValueError('Prediction/score seal changed')
    baseline_path=mp.parent/'BASELINE_PREDICTIONS.jsonl.gz'
    if sha(baseline_path)!=_seal_artifact(ss,baseline_path.name):raise ValueError('Baseline pose seal changed')
    ep=folder/'e_trace_v2';es=read(ep/'EXTRACTION_SEAL.json')
    for name in ('OBSERVATION_LINEAGE.jsonl.gz','TYPED_PACKETS.jsonl.gz'):
        if sha(ep/name)!=es['files'][name]['sha256']:raise ValueError('Lineage/packet bytes changed')
    selection_path=folder/'SELECTION_CASES.json'
    selection_doc=read(selection_path)if selection_path.exists()else{'cases':[]}
    selection_cases=selection_doc['cases'];allids=IDS+tuple(c['id']for c in selection_cases)
    oracle_bindings={}
    if selection_cases:
        oracle_seal_path=folder/'oracle_v1/ORACLE_SEAL.json';oracle_seal=read(oracle_seal_path)
        if sha(oracle_seal_path)!=selection_doc['oracle_seal_sha256']:raise ValueError('Selection explanation oracle seal changed')
        for name in ('ORACLE_ROWS.jsonl.gz','ORACLE_FRAME_TABLE.csv'):
            if sha(oracle_seal_path.parent/name)!=_seal_artifact(oracle_seal,name):raise ValueError('Original oracle artifact changed')
        archive_shas={c['source']['archive_sha256']for c in selection_cases}
        if len(archive_shas)!=1:raise ValueError('Mixed original candidate archives')
        archive_path=folder/'original_candidate_archive_v1/ORIGINAL_E_CANDIDATE_BANK.jsonl.gz'
        if sha(archive_path)not in archive_shas:raise ValueError('Retained candidate archive changed')
        if [(c['failure_stage'],c['id'])for c in selection_cases]!=[('family_selection','plastic_day_01:011891'),('within_family_final_selection','plastic_day_01:007881')]:raise ValueError('Frozen first-ID selection diagnostics changed')
        oracle_bindings=dict(selection_cases_sha256=sha(selection_path),original_oracle_seal_sha256=sha(oracle_seal_path),original_candidate_archive_sha256=sha(archive_path))
    records=wanted(ep/'OBSERVATION_LINEAGE.jsonl.gz',allids);packets=wanted(ep/'TYPED_PACKETS.jsonl.gz',allids)
    predictions=wanted(pp,allids);baselines=wanted(baseline_path,allids);metrics=_metrics(mp,allids);permission=public_image_inventory(root)
    for name,values in [('lineage',records),('typed packets',packets),('new prediction',predictions),('baseline',baselines),('sealed metrics',metrics)]:
        if set(values)!=set(allids):raise ValueError(name+' does not contain all exact requested IDs')
    font_manager.fontManager.addfont(str(FONT));plt.rcParams.update({'font.family':font_manager.FontProperties(fname=FONT).get_name(),'axes.unicode_minus':False})
    output.mkdir(parents=True);proofs=[];articles=[]
    for number,ident in enumerate(IDS,1):
        record,packet=records[ident],packets[ident];s=packet['solver_input'];K=s['K'];roi=shared_roi(record);x0,y0,x1,y1=roi
        if packet['image_sha256']not in permission:raise ValueError('Source RGB is outside previous public image permission')
        source=root/packet['image']
        if sha(source)!=packet['image_sha256']:raise ValueError('Actual source RGB hash changed')
        full=Image.open(source).convert('RGB');image=full.crop(tuple(roi));case=output/f'case_{number:02d}';case.mkdir()
        thumb=image.copy();thumb.thumbnail((1200,900));thumb.save(case/'source_crop.jpg',quality=JPEG_QUALITY,optimize=True)
        states=record['B_original_pre_e']['states'];clipped=[]
        def canvas(ax,title):
            ax.imshow(image,extent=(x0,x1,y1,y0),interpolation='nearest');ax.set_xlim(x0,x1);ax.set_ylim(y1,y0);ax.set_aspect('equal');ax.set_title(title,fontsize=11);ax.tick_params(labelsize=7)
        def wire(ax,pose,color,label,xyz=None,ls='-',lw=1.6,vertices=False,vertex_offset=3):
            q=projection(pose,K,xyz)
            if q is None:
                ax.text(.02,.04,label+': pose 없음',transform=ax.transAxes,fontsize=8,color=color);return
            outside=0
            for a,b in EDGES:
                seg=clip_segment(q[a]-[x0,y0],q[b]-[x0,y0],x1-x0,y1-y0)
                if seg is None:outside+=1;continue
                seg=seg+[x0,y0];ax.plot(seg[:,0],seg[:,1],color=color,ls=ls,lw=lw,label=label if a==0 and b==1 else None)
                outside+=int(not np.allclose(seg,q[[a,b]]))
            clipped.append(dict(label=label,clipped_or_invisible_edges=outside))
            for k in range(8)if vertices else():
                if x0<=q[k,0]<=x1 and y0<=q[k,1]<=y1:
                    tag=ax.annotate('M'+str(k),q[k],xytext=(2,vertex_offset),textcoords='offset points',fontsize=7,color=color,clip_on=True)
                    tag.set_path_effects([patheffects.withStroke(linewidth=1.8,foreground='white')])
        def physical_axes(ax,pose,color,label):
            if not pose.get('available',False):return
            R=np.asarray(pose['R_physical']);t=np.asarray(pose['centroid']);length=.18
            xyz=np.array([[0.,0.,0.],[length,0.,0.],[0.,0.,length]])
            camera=xyz@R.T+t;h=camera@np.asarray(K).T
            if np.any(camera[:,2]<=0):return
            q=h[:,:2]/h[:,2,None]
            for i,name in ((1,'x'),(2,'z')):
                ax.annotate('',q[i],q[0],arrowprops={'arrowstyle':'->','lw':1.4,'color':color,'linestyle':'-'if i==1 else'--'})
                tag=ax.annotate(label+' '+name,q[i],xytext=(2,3 if i==1 else -9),textcoords='offset points',fontsize=7,color=color,clip_on=True)
                tag.set_path_effects([patheffects.withStroke(linewidth=1.8,foreground='white')])
        def marker(ax,xy,k,state,source='measured',color=None):
            if state not in('B','I'):return
            xy=np.asarray(xy,float)
            if not np.isfinite(xy).all()or not(x0<=xy[0]<=x1 and y0<=xy[1]<=y1):return
            fallback=source=='VISIBLE_BACKBONE_FALLBACK' or source=='predictor'
            col='#666666'if fallback else color or COLORS[state]
            ax.scatter(*xy,c=col,marker='x'if fallback else'o',s=28,zorder=6)
            ax.annotate(f'{k}:{state}'+(' 예측'if fallback else''),xy,xytext=(4,-12),textcoords='offset points',fontsize=7,color=col,clip_on=True)
        def observations(ax,typed=True):
            if typed:
                for m in s['visible']:
                    if m['native_corner']in packet['activation']['complete_ids']:marker(ax,m['xy'],m['native_corner'],m['state'],m.get('source'))
                for b in s['boundary_observations']:
                    for line in b['lines']:
                        n=np.asarray(line['normal_unit_xy']);t=np.array([-n[1],n[0]]);a,bound=line['support_tangent_interval_px'];c=-line['offset_px']*n
                        seg=clip_segment(c+t*a-[x0,y0],c+t*bound-[x0,y0],x1-x0,y1-y0)
                        if seg is not None:
                            seg+=np.array([x0,y0]);ax.plot(seg[:,0],seg[:,1],c='#cc36a2',lw=1.4,ls='--');ax.annotate('P'+str(b['native_corner']),seg.mean(0),fontsize=8,color='#a3197b',clip_on=True)
                        ray=line.get('provenance',{}).get('line_record',{});q=np.asarray(ray.get('observed_points',[]))
                        if q.ndim==2 and q.shape[1]==2:ax.scatter(q[:,0],q[:,1],s=3,c='#c647a5',alpha=.5)
                for d in s['domains']+s['u_regions']:
                    center=d.get('center_xy');axes=d.get('half_axes_px');color='#e39d22'if d['state']=='O'else'#8997aa'
                    if center is not None and axes is not None:ax.add_patch(Ellipse(center,2*axes[0],2*axes[1],angle=d.get('angle_deg',0),fill=False,edgecolor=color,lw=1.,ls=':'))
                    bounds=d.get('bounds_xyxy',d.get('box_xyxy'))
                    if bounds is not None:ax.add_patch(Rectangle(bounds[:2],bounds[2]-bounds[0],bounds[3]-bounds[1],fill=False,edgecolor=color,lw=1.,ls=':'))
            else:
                obs=record['D_post_e']['observation']
                for m in obs['frozen_visible_measurements']:marker(ax,m['xy'],m['native_corner'],m['state'],m.get('source'))
                for r in obs['visible_correction']['records']:
                    if r['state']not in('B','I'):continue
                    for ray in r.get('rays',[]):
                        q=np.asarray(ray.get('observed_points',[]))
                        if ray.get('supported')and q.ndim==2 and q.shape[1]==2:ax.scatter(q[:,0],q[:,1],s=3,c=COLORS[r['state']],alpha=.45)
        aspect=(y1-y0)/(x1-x0)
        process_height=max(3.8,min(9.6,2*(17/3*aspect+.65)))
        fig,axes=plt.subplots(2,3,figsize=(17,process_height),layout='constrained');fig.suptitle(ident+' · 원 e 이전/이후를 분리한 실제 저장 과정',fontsize=15)
        canvas(axes[0,0],'① 원 N3·SubPix + F\n'+metric_text(metrics[ident].get('N0')))
        wire(axes[0,0],record['A_original_N3']['pose'],COLORS['N0'],'N0',vertices=True)
        for k,xy in enumerate(record['A_original_N3']['points'][:8]):marker(axes[0,0],xy,k,states[k],'predictor')
        canvas(axes[0,1],'② 원 e의 활성 full / partial / region\nS/O/T/U 정밀 관측점 없음');observations(axes[0,1]);wire(axes[0,1],packet['e0_pose'],'#458a58','e 모델',ls=':',lw=.9)
        canvas(axes[0,2],'③ 원 G10의 두 family 후보\n저장된 family winner, 전체 생성 bank 아님')
        candidate_summary=[]
        for name,bank in packet['original_e_stage1_family_seed_bank'].items():
            old=bank['initial_solutions'][2];pose=family_pose(old,bank['family']);wire(axes[0,2],pose,COLORS[name],name,lw=1.3,vertices=True,vertex_offset=4 if name=='ORIGINAL_WD'else -10)
            candidate_summary.append(dict(stage='original_G10',family=name,accepted=old.get('accepted'),objective=old.get('objective'),selected_family=name==packet['registered_family']['name']))
        canvas(axes[1,0],'④ G10 family 선택 → 같은 family의 d/e 보정\n'+metric_text(metrics[ident].get('E0')))
        selected=packet['original_e_stage1_family_seed_bank'][packet['registered_family']['name']]
        wire(axes[1,0],family_pose(selected['initial_solutions'][2],selected['family']),'#d0a125','G10 선택 전',ls='--');wire(axes[1,0],packet['e0_pose'],COLORS['E0'],'최종 e',vertices=True)
        canvas(axes[1,1],'⑤ e 이후 새 자기 상태·B/I 재관측\n원 e의 partial/region을 재사용한 단계가 아님');observations(axes[1,1],False);wire(axes[1,1],packet['e0_pose'],'#448960','초기 e',ls=':',lw=.9)
        canvas(axes[1,2],'⑥ 저장된 기존 비교 자세\n모두 예측, GT 상자 없음')
        for arm in('POSEFIX_RAW','N0','E0'):
            pose=baselines[ident]['predictions'][arm]['actual_pose'];wire(axes[1,2],pose,COLORS[arm],arm,lw=1.2)
            if arm in('N0','E0'):physical_axes(axes[1,2],pose,COLORS[arm],arm)
        for ax in axes.flat:
            if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=7,loc='upper right')
        fig.savefig(case/'process.jpg',dpi=120,pil_kwargs={'quality':JPEG_QUALITY,'optimize':True});plt.close(fig)
        diagnostic_height=max(2.6,min(7.,19/4*aspect+.85))
        fig,axes=plt.subplots(1,4,figsize=(19,diagnostic_height),layout='constrained');fig.suptitle(ident+' · 네 고정 진단 결과',fontsize=14)
        diagnostic_records={}
        for ax,arm in zip(axes,ARMS):
            value=predictions[ident]['predictions'][arm];canvas(ax,arm+'\n'+metric_text(metrics[ident].get(arm)))
            observations(ax,arm.startswith('TYPED'));wire(ax,packet['e0_pose'],'#999999','E0 시작',ls='--',lw=.8);wire(ax,value.get('actual_pose',{}),'#d84b33','새 결과',vertices=True)
            diagnostic_records[arm]=dict(actual_pose=value.get('actual_pose'),accepted=value.get('accepted'),fallback=value.get('fallback'),status=value.get('status'),reason=value.get('reason'),
                generated_stage_counts=dict(Counter(r.get('stage','unknown')for r in value.get('generated_candidates',[]))))
        fig.savefig(case/'diagnostics.jpg',dpi=120,pil_kwargs={'quality':JPEG_QUALITY,'optimize':True});plt.close(fig)
        fit=fitting_residuals(packet,packet['e0_pose']);proof=dict(id=ident,source_RGB_sha256=packet['image_sha256'],permission_inventory_match=True,
            source_crop='case_'+f'{number:02d}'+'/source_crop.jpg',viewport_original_xyxy=roi,viewport_uses_original_initial_predictions_only=True,
            source_RGB_original_size=list(full.size),model_CF_projection=True,model_lines_clipped_to_common_viewport=clipped,
            actual_S_O_T_U_point_markers=0,all_model_corner_labels_prefixed_M=True,
            original_activation=packet['activation'],owner_audit=record['audit'],original_G10_candidates=candidate_summary,
            selected_family=packet['registered_family'],original_e_fit_residuals_px=fit,metrics=metrics[ident],diagnostics=diagnostic_records,
            full_point_and_partial_line_distinct_glyphs=True,GT_coordinate_or_reference_file_reads=0,new_NN_PnP_or_pixel_search=0)
        proofs.append(proof)
        metricrows=''.join('<tr><td>'+html.escape(a)+'</td><td>'+html.escape(metric_text(metrics[ident].get(a)))+'</td></tr>'for a in('N0','E0','POSEFIX_RAW')+ARMS)
        factorrows=''.join('<tr><td>'+str(a['native_corner'])+'</td><td>'+html.escape(a['original_state'])+'</td><td>'+html.escape(a['original_observation_type'])+'</td><td>'+html.escape(str(a['original_source']))+'</td><td>'+html.escape(a['post_e_same_owner_state'])+'</td><td>'+str(a['post_e_injected_N3_source_id'])+'</td></tr>'for a in record['audit'])
        articles.append(f'<article id="case{number}"><h2>{number}. {html.escape(ident)}</h2><p>사용자가 지목한 기존 진단 사례. 개선·악화·실패 여부로 대체하지 않았습니다. 선택 W/D: {html.escape(packet["registered_family"]["name"])}. 영상 의미 owner는 미검증입니다.</p><a href="case_{number:02d}/source_crop.jpg">실제 RGB 확대</a><img src="case_{number:02d}/process.jpg" alt="원래 e 이전과 이후 과정"><img src="case_{number:02d}/diagnostics.jpg" alt="네 진단 최종 자세"><table><tr><th>방법</th><th>기존 참조 기준 pose 오차</th></tr>{metricrows}</table><details><summary>native owner·관측 종류·후단 원예측 주입</summary><table><tr><th>원 e native</th><th>원 상태</th><th>원 관측 종류</th><th>출처</th><th>새 상태</th><th>주입한 N3 source ID</th></tr>{factorrows}</table></details><details><summary>실제 full/partial fitting residual [px] 및 원 factor 값</summary><pre>{html.escape(json.dumps(fit,ensure_ascii=False,indent=2))}</pre><p>위 px는 측정→모델 잔차이며 GT 키포인트 위치 오차가 아닙니다. 전체 자세 T/R과 단위를 혼합하지 않습니다.</p></details></article>')
    oracle_html,oracle_proofs=_selection_panels(selection_cases,records,packets,baselines,metrics,root,output,permission)
    process=dict(schema=SCHEMA,cases=proofs,offline_oracle_cases=oracle_proofs,case_selection=list(IDS),selection='Six exact user-specified diagnostic IDs; no replacement; two separate postseal first-ID oracle selection diagnostics',
                 prediction_sha256=sha(pp),score_seal_sha256=sha(mp.parent/'SCORE_SEAL.json'),lineage_seal_sha256=sha(ep/'EXTRACTION_SEAL.json'),
                 offline_oracle_bindings=oracle_bindings,
                 no_new_inference_or_fit=True,source_RGB_permission_verified=True,display_only_JPEG_quality=JPEG_QUALITY)
    (output/'PROCESS.json').write_text(json.dumps(process,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    nav=' '.join(f'<a href="#case{i}">{i}. {html.escape(v)}</a>'for i,v in enumerate(IDS,1))
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>상태별 대응 진단 · 실제 사례</title><style>body{font-family:sans-serif;max-width:1800px;margin:20px auto;padding:15px;background:#eef2f6;line-height:1.65}article,header{background:white;padding:18px;margin-bottom:18px;border-radius:10px}img{width:100%;height:auto}table{border-collapse:collapse;width:100%}th,td{padding:5px;border-bottom:1px solid #ddd;text-align:left}nav a{display:inline-block;margin:6px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><header><h1>상태별 관측·대응·자세 선택: 지정 여섯 사례와 오프라인 후보 진단</h1><p>○ 실제 full B/I · × 상속 예측 fallback · 자홍색 점선 partial 선/범위 · 점선 타원 O/U 영역 · M번호와 상자선은 모델 투영입니다. S/O/T/U를 정밀 영상점으로 그리지 않습니다. 모든 패널은 같은 초기예측 기반 확대 범위를 사용하며, 화면 밖 모델 선은 잘라 표시합니다. 전체 RGB CNN의 수용영역이 S와 독립이라는 뜻은 아닙니다.</p><p>G10 후보와 원 e 선택, 그 뒤의 새 재관측을 구분합니다. 새 진단의 좋은 결과가 원 e보다 향상됐다는 뜻인지는 전체238/PRIMARY192 집계로 판단해야 합니다. 비교패널의 x/z 화살표도 저장 pose에서 투영한 physical 모델축입니다.</p><nav>'+nav+''.join(f'<a href="#oracle{i}">오프라인 선택실패 {i}</a>'for i in range(1,len(selection_cases)+1))+'</nav><a href="PROCESS.json">저장 좌표·수치·출처</a></header>'+''.join(articles)+oracle_html+'</html>'
    (output/'index.html').write_text(page)
    (output/'GALLERY_KO.md').write_text('# 실제 여섯 사례 과정 갤러리\n\n[HTML 열기](index.html) · [저장 과정·수치·SHA](PROCESS.json)\n\n원 e 이전의 complete/partial/region/owner, 실제 G10 두 family winner와 선택 후 refinement, e 이후 새 재관측, 네 고정 진단 결과를 구분했습니다. ○는 full 영상점, ×는 predictor fallback, 자홍 점선은 partial 선/범위, 점선 영역은 기존 O/U 제약입니다. M번호와 상자는 저장 pose의 CF 모델 투영이며 GT가 아닙니다.\n\n측정점→모델 잔차는 px, 기존 참조에 대한 자세 오차는 cm/°입니다. 새 GT·새 모델·새 관측은 만들지 않았습니다. 지정 여섯 장은 반복 개발의 진단 사례이며 전체 성능을 대신하지 않습니다.\n')
    files={str(p.relative_to(output)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in output.rglob('*')if p.is_file()}
    total=sum(v['bytes']for v in files.values())
    if total>MAX_PUBLIC_BYTES:raise ValueError(f'Public gallery exceeds fixed8MiB: {total}')
    seal=dict(schema=SCHEMA,cases=6,offline_oracle_cases=len(selection_cases),JPEG_panels=12+len(selection_cases),source_crop_JPEGs=6+len(selection_cases),total_bytes_before_seal=total,files=files,
              source_sha256=sha(__file__),prediction_sha256=sha(pp),score_seal_sha256=sha(mp.parent/'SCORE_SEAL.json'),offline_oracle_bindings=oracle_bindings,new_NN_PnP=0)
    (output/'GALLERY_SEAL.json').write_text(json.dumps(seal,ensure_ascii=False,indent=2)+'\n')
    return seal

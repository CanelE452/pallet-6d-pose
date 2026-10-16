"""Draw saved candidates/observations on already permitted RGB, without fitting.

All candidate comparisons use one image crop derived from the old initial N3
coordinates. Cost rows and reference numbers are read from diagnostic ledgers;
this renderer neither scores references nor discovers new image measurements.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
from pathlib import Path
import time

import numpy as np

from .baseline import read_rows, read_metrics, sha, write_json, REPRESENTATIVE_RULE

EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
LOSS_FAMILIES=('full_BI','partial_normal','outline','prior')
GROUPS=('FAMILY_SELECTION_LOSS','WITHIN_FAMILY_SELECTION_LOSS','FINAL_SUCCESS_CONTROL')
NAMES={'full_BI':'full B/I','partial_normal':'partial normal','outline':'outline','prior':'prior'}


def projection(R,t,xyz,K):
    camera=np.asarray(xyz,float)@np.asarray(R,float).T+np.asarray(t,float)
    if not np.isfinite(camera).all()or np.any(camera[:,2]<=0):
        return None
    q=camera@np.asarray(K,float).T
    return q[:,:2]/q[:,2,None]


def select_representatives(frames,metrics,eligible_controls):
    """Precommitted E0 T ordering, with per-group recording diversity first."""
    output=[]
    for group in GROUPS:
        pool=[r for r in frames if r['partition']!='CAL'and r['group']==group
              and(group!='FINAL_SUCCESS_CONTROL'or r['id']in eligible_controls)]
        pool.sort(key=lambda r:(-metrics[r['id']]['E0']['T_cm'],r['id']))
        used=set();chosen=[]
        for r in pool:
            rec=metrics[r['id']]['E0']['cluster_id']
            if rec not in used:
                chosen.append(r);used.add(rec)
            if len(chosen)==2:break
        if len(chosen)<2:
            chosen.extend(r for r in pool if r not in chosen)
            chosen=chosen[:2]
        for r in chosen:
            output.append(dict(id=r['id'],group=group,
                original_E0_translation_cm=metrics[r['id']]['E0']['T_cm'],
                recording=metrics[r['id']]['E0']['cluster_id'],
                unavoidable_recording_repeat=len({metrics[x['id']]['E0']['cluster_id']for x in chosen})<len(chosen)))
    return output


def compact_cost(cost):
    """Keep every observation-level contribution; dense outline samples stay private."""
    result={k:cost[k]for k in('id','stage','family','candidate_id','objective','objective_recomputed',
        'objective_stored','weighted_components','term_status','minimum_constraint_slack',
        'recomputed_joint_feasible','stored_objective_parity','stored_component_parity')if k in cost}
    rows=[]
    for r in cost['per_observation']:
        row={k:v for k,v in r.items()if k not in('provenance','association_provenance','sample_xy','residual_xy_px',
                'robust_losses','weighted_sample_contributions','standardized_radius','arc_or_sample_weights',
                'whitening','assumed_physical_owner_Q')}
        if 'lines'in row:
            row['lines']=[{k:v for k,v in line.items()if k!='provenance'}for line in row['lines']]
        rows.append(row)
    result['per_observation']=rows
    result['constraints']=cost['constraints']
    result['dense_outline_samples_and_original_provenance']='Bound full private cost ledger; no recomputation or truncation in analysis.'
    return result


def obs_table(s,g):
    ss={r['observation_id']:r for r in s['per_observation']}
    gg={r['observation_id']:r for r in g['per_observation']}
    rows=[]
    for oid in sorted(set(ss)|set(gg)):
        a,b=ss.get(oid),gg.get(oid)
        # Across W/D families association may differ, so preserve missing as NA.
        d=None if a is None or b is None else b['weighted_contribution']-a['weighted_contribution']
        base=a or b
        def raw(v):
            if v is None:return None
            if v['loss_family']=='partial_normal':return [x['normal_residual_px']for x in v['lines']]
            return v.get('raw_residual_xy_px',v.get('residual_mean_px'))
        rows.append(dict(observation_id=oid,loss_family=base['loss_family'],
            state=base.get('state'),native_corner=base.get('native_corner'),
            source_s=None if a is None else a.get('measurement_source'),
            source_other=None if b is None else b.get('measurement_source'),
            active_s=None if a is None else a.get('active',True),
            active_other=None if b is None else b.get('active',True),
            weighted_s=None if a is None else a['weighted_contribution'],
            weighted_other=None if b is None else b['weighted_contribution'],delta=d,
            raw_residual_units='px: full xy vector; partial per-line signed normal; outline mean distance',
            raw_s=raw(a),raw_other=raw(b)))
    return rows


def load_candidate_metrics(path):
    result={}
    for r in read_rows(path):
        key=(r['id'],r['candidate_id'])
        if key in result:raise ValueError('Duplicate candidate metric row')
        result[key]=r.get('metric',r.get('reference_metric',r))
    return result


def fmt(x):
    return 'NA'if x is None else f'{x:.8g}'if isinstance(x,(int,float))else str(x)


def metric_text(value):
    t=value.get('translation_cm',value.get('T_cm'));r=value.get('rotation_deg',value.get('R_deg'))
    success=value.get('success_le5',value.get('success'))
    return f"T {fmt(t)}cm / R {fmt(r)}° · {'참조 공동 성공'if success is True else'참조 실패'if success is False else'미확정'}"


def render(private,old_private,source_root,output,*,costs_path,candidate_metrics_path,selection_folder=None):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt,font_manager
    from matplotlib.patches import Ellipse,Rectangle
    from PIL import Image
    from scripts.research.pallet_minimal_bi_ablation_v1.gallery import public_image_inventory,FONT
    from scripts.research.pallet_guarded_bi_final_v1.gallery import clip_segment
    from scripts.research.pallet_statewise_correspondence_diagnosis_v1.gallery import shared_roi

    start=time.monotonic();cpu=time.process_time()
    private,old,root,out=map(Path,(private,old_private,source_root,output))
    if out.exists():raise FileExistsError('Preserve every gallery attempt')
    sf=Path(selection_folder)if selection_folder else private/'selection_attempt01'
    cost_seal_path=Path(costs_path).parent/'COST_SEAL.json'
    cost_seal=json.loads(cost_seal_path.read_text())
    if not cost_seal['all_complete']or cost_seal['frames']!=238 or cost_seal['rows']!=11603:
        raise ValueError('Full fixed-candidate cost seal required')
    if sha(costs_path)!=cost_seal['candidate_costs_sha256']:
        raise ValueError('Fixed candidate cost bytes changed')
    frames=list(read_rows(sf/'FRAME_SELECTIONS.jsonl.gz'))
    refs=load_candidate_metrics(candidate_metrics_path)
    cost_keys=set()
    for f in frames:
        if f['partition']=='CAL'or f['group']not in GROUPS:continue
        for cid in(f['selected_candidate_id'],f['representative_failure_candidate_id']if f['group']=='FINAL_SUCCESS_CONTROL'else f['representative_success_candidate_id']):
            if cid is not None:cost_keys.add((f['id'],cid))
        if f['group']=='FAMILY_SELECTION_LOSS':
            cost_keys.update((f['id'],cid)for cid in f['stage_winner_ids']['G10'].values()if cid is not None)
    costs={}
    for r in read_rows(costs_path):
        key=(r['id'],r['candidate_id'])
        if key not in cost_keys:continue
        if key in costs:raise ValueError('Duplicate cost row')
        costs[key]=r.get('cost',r)
    metrics=read_metrics(old/'score_v2/ROW_METRICS.csv')
    # A control is eligible only if an actual failing retained candidate exists,
    # and an active additive family that favors bad s in a within-loss pair
    # instead protects successful s against that control's bad candidate.
    loss_favoring=set()
    for f in frames:
        if f['partition']=='CAL'or f['group']!='WITHIN_FAMILY_SELECTION_LOSS':continue
        a=costs[(f['id'],f['selected_candidate_id'])];b=costs[(f['id'],f['representative_success_candidate_id'])]
        loss_favoring.update(k for k in LOSS_FAMILIES if b['weighted_components'][k]-a['weighted_components'][k]>0)
    eligible=set()
    for f in frames:
        if f['partition']=='CAL'or f['group']!='FINAL_SUCCESS_CONTROL'or not f['representative_failure_candidate_id']:continue
        a=costs[(f['id'],f['selected_candidate_id'])];b=costs[(f['id'],f['representative_failure_candidate_id'])]
        if any(b['weighted_components'][k]-a['weighted_components'][k]>0 for k in loss_favoring):eligible.add(f['id'])
    reps=select_representatives(frames,metrics,eligible)
    selected=[f for f in frames if f['partition']!='CAL'and f['group']in GROUPS[:2]]
    if len(selected)!=32:raise ValueError('All32 selection-loss cases required')
    controls=[f for f in frames if f['id']in{r['id']for r in reps if r['group']=='FINAL_SUCCESS_CONTROL'}]
    shown=selected+controls;ids={f['id']for f in shown}
    no_candidate_frames=[f for f in frames if f['group']=='NO_SUCCESS_CANDIDATE']
    image_ids=ids|{f['id']for f in no_candidate_frames}
    candidates={(r['id'],r['candidate_id']):r for r in read_rows(sf/'RETAINED_CANDIDATES.jsonl.gz')if r['id']in ids}
    packets={r['id']:r for r in read_rows(old/'e_trace_v2/TYPED_PACKETS.jsonl.gz')if r['id']in image_ids}
    lineages={r['id']:r for r in read_rows(old/'e_trace_v2/OBSERVATION_LINEAGE.jsonl.gz')if r['id']in image_ids}
    permission=public_image_inventory(root)
    for f in shown+no_candidate_frames:
        if packets[f['id']]['image_sha256']not in permission:raise PermissionError('New RGB publication not authorized '+f['id'])
    font_manager.fontManager.addfont(str(FONT));plt.rcParams.update({'font.family':font_manager.FontProperties(fname=FONT).get_name(),'axes.unicode_minus':False})
    out.mkdir(parents=True);write_json(out/'REPRESENTATIVE_SELECTION.json',{'rule':REPRESENTATIVE_RULE,'eligible_controls':sorted(eligible),'examples':reps})
    articles=[];proofs=[];no_candidate_proofs=[]
    (out/'no_candidate_rgb').mkdir()
    for f in no_candidate_frames:
        ident=f['id'];p=packets[ident];roi=shared_roi(lineages[ident]);source=root/p['image']
        if sha(source)!=p['image_sha256']:raise ValueError('No-candidate RGB changed '+ident)
        cropped=Image.open(source).convert('RGB').crop(tuple(roi));cropped.thumbnail((900,650))
        filename='no_candidate_rgb/'+ident.replace(':','__')+'.jpg';cropped.save(out/filename,quality=85,optimize=True)
        no_candidate_proofs.append(dict(id=ident,partition=f['partition'],source_RGB_sha256=p['image_sha256'],
            permission_inventory_match=True,source_crop=filename,viewport_original_xyxy=roi,
            display_size=list(cropped.size),new_pose_or_observation_drawn=False))
    for number,f in enumerate(shown,1):
        ident=f['id'];p=packets[ident];s=p['solver_input'];roi=shared_roi(lineages[ident]);x0,y0,x1,y1=roi
        source=root/p['image']
        if sha(source)!=p['image_sha256']:raise ValueError('Original RGB changed '+ident)
        image=Image.open(source).convert('RGB').crop(tuple(roi));case=out/f'case_{number:02d}';case.mkdir()
        image.save(case/'source_crop.jpg',quality=87,optimize=True)
        sid=f['selected_candidate_id'];oid=f['representative_failure_candidate_id']if f['group']=='FINAL_SUCCESS_CONTROL'else f['representative_success_candidate_id']
        panels=[]
        if f['group']=='FAMILY_SELECTION_LOSS':
            for fam in('ORIGINAL_WD','SWAPPED_WD'):
                cid=f['stage_winner_ids']['G10'][fam]
                if cid is None:raise ValueError('Family10 G10 accepted winners required')
                panels.append((cid,'G10 실제 family winner '+fam+(' [선택]'if fam==f['selected_family']else'')))
        panels.extend([(sid,'G13 실제 선택 s'),(oid,'G13 참조 실패 대조 b'if f['group']=='FINAL_SUCCESS_CONTROL'else'G13 원순위 최선 참조 성공 g')])
        rows=len(panels)//2;aspect=(y1-y0)/(x1-x0);fig,axes=plt.subplots(rows,2,figsize=(14,rows*max(2.7,min(6.5,7*aspect+.8))),layout='constrained',squeeze=False)
        fig.suptitle(ident+' · '+f['group'],fontsize=12)
        panelproof=[]
        for ax,(cid,title)in zip(axes.flat,panels):
            c=candidates[(ident,cid)];cost=costs[(ident,cid)];stage=c['stage'];bound=s['boundary_observations']if stage=='G13'else p['original_e_legacy_boundary_observations']
            revoked={b['native_corner']for b in bound};complete={v['native_corner']for v in s['visible']}-revoked
            ax.imshow(image,extent=(x0,x1,y1,y0),interpolation='nearest');ax.set_xlim(x0,x1);ax.set_ylim(y1,y0);ax.set_aspect('equal')
            ax.set_title(title+'\n'+metric_text(refs[(ident,cid)])+f" / J={cost['objective']:.8g}",fontsize=9)
            ax.tick_params(labelsize=7);ax.set_xlabel('원본 x [px]',fontsize=7);ax.set_ylabel('원본 y [px]',fontsize=7)
            stored_outline=s['outline_observation'].get('observed_boundary_segments_xy',[])
            for segment in stored_outline:
                z=np.asarray(segment,float)
                if z.shape!=(2,2):raise ValueError('Stored outline segment shape changed')
                ax.plot(z[:,0],z[:,1],c='#36b7ca',ls=':',lw=1.,alpha=.8)
            for v in s['visible']:
                k=v['native_corner']
                if k not in complete:continue
                xy=np.asarray(v['xy'],float);fallback=v.get('source')=='VISIBLE_BACKBONE_FALLBACK'
                color='#666666'if fallback else'#00a4c4';ax.scatter(*xy,c=color,marker='x'if fallback else'o',s=24,zorder=6)
                ax.annotate(f'{k}:{v["state"]}'+(' 예측'if fallback else''),xy,xytext=(3,-10),textcoords='offset points',fontsize=6,color=color,clip_on=True)
            for b in bound:
                for line in b['lines']:
                    n=np.asarray(line['normal_unit_xy'],float);tan=np.array([-n[1],n[0]]);lo,hi=line['support_tangent_interval_px'];center=-line['offset_px']*n
                    seg=clip_segment(center+tan*lo-[x0,y0],center+tan*hi-[x0,y0],x1-x0,y1-y0)
                    if seg is not None:
                        seg+=np.array([x0,y0]);ax.plot(seg[:,0],seg[:,1],c='#cd3d9e',lw=1.35,ls='--');ax.annotate('P'+str(b['native_corner']),seg.mean(0),fontsize=7,color='#a3197b',clip_on=True)
                    ray=line.get('provenance',{}).get('line_record',{});q=np.asarray(ray.get('observed_points',[]))
                    if q.ndim==2 and q.shape[1]==2:ax.scatter(q[:,0],q[:,1],s=2,c='#c647a5',alpha=.5)
            for d in s['domains']+s['u_regions']:
                center=d.get('center_xy');half=d.get('half_axes_px');col='#bf8d00'if d['state']=='O'else'#8997aa'
                if center is not None and half is not None:ax.add_patch(Ellipse(center,2*half[0],2*half[1],angle=d.get('angle_deg',0),fill=False,edgecolor=col,lw=.8,ls=':'))
                bounds=d.get('bounds_xyxy',d.get('box_xyxy'))
                if bounds is not None:ax.add_patch(Rectangle(bounds[:2],bounds[2]-bounds[0],bounds[3]-bounds[1],fill=False,edgecolor=col,lw=.8,ls=':'))
            q=projection(c['R'],c['t'],c['native_xyz'],s['K']);color='#218a62'if cid==sid or(stage=='G10'and c['family']==f['selected_family'])else'#e18a16'
            if q is None:raise ValueError('Retained pose projection unavailable')
            for a,b in EDGES:ax.plot(q[[a,b],0],q[[a,b],1],color=color,lw=1.25)
            for k,xy in enumerate(q):
                ax.annotate('M'+str(k)+(' S'if s['states'][k]=='S'else''),xy,xytext=(2,5),textcoords='offset points',fontsize=6,color=color,clip_on=True)
            panelproof.append(dict(candidate_id=cid,stage=stage,family=c['family'],R=c['R'],t=c['t'],native_xyz=c['native_xyz'],
                physical_registration_Q=c['physical_registration_Q'],model_projection_xy=q.tolist(),active_full_ids=sorted(complete),active_partial_ids=sorted(revoked),
                stored_observed_outline_segments=len(stored_outline),
                reference_metric=refs[(ident,cid)],weighted_components=cost['weighted_components'],objective=cost['objective']))
        fig.savefig(case/'comparison.jpg',dpi=115,pil_kwargs={'quality':87,'optimize':True});plt.close(fig)
        sc,gc=costs[(ident,sid)],costs[(ident,oid)];delta={k:gc['weighted_components'][k]-sc['weighted_components'][k]for k in LOSS_FAMILIES}
        delta_total=gc['objective']-sc['objective']
        if abs(sum(delta.values())-delta_total)>1e-9:raise ValueError('Gallery Delta sum parity failed')
        details=dict(id=ident,group=f['group'],selected_candidate_id=sid,comparison_candidate_id=oid,
            pair_role='successful_selected_vs_reference_failure_control'if f['group']=='FINAL_SUCCESS_CONTROL'else'selected_vs_comparator_best_reference_success',
            same_stage='G13',same_family=sc['family']==gc['family'],Delta_weighted_components=delta,Delta_total=delta_total,
            meaning='Delta = comparison minus actual selected; positive favors actual selected. Across family10 use original family-specific packages; do not mix G10 and G13 costs.',
            selected=compact_cost(sc),comparison=compact_cost(gc),observation_deltas=obs_table(sc,gc))
        if f['group']=='FAMILY_SELECTION_LOSS':
            g10=[costs[(ident,cid)]for cid,_ in panels[:2]]
            details['G10_actual_family_decision']=[compact_cost(v)for v in g10]
            details['G10_frozen_family']=f['selected_family'];details['G13_own_reselected_family']=f['g13_reselected_family']
        write_json(case/'cost_and_observations.json',details)
        weighted=''.join(f'<tr><td>{NAMES[k]}</td><td>{fmt(sc["weighted_components"][k])}</td><td>{fmt(gc["weighted_components"][k])}</td><td>{fmt(delta[k])}</td></tr>'for k in LOSS_FAMILIES)
        observed=''.join('<tr>'+''.join(f'<td>{html.escape(fmt(row[k]))}</td>'for k in('observation_id','state','source_s','raw_s','raw_other','weighted_s','weighted_other','delta'))+'</tr>'for row in details['observation_deltas'])
        g10html=''
        if f['group']=='FAMILY_SELECTION_LOSS':
            g10html='<h3>G10 실제 family 결정 — G13 비용과 별도</h3><table><tr><th>family</th><th>G10 원 winner 비용</th><th>full</th><th>partial</th><th>outline</th><th>prior</th></tr>'+''.join('<tr><td>'+v['family']+'</td><td>'+fmt(v['objective'])+'</td>'+''.join('<td>'+fmt(v['weighted_components'][k])+'</td>'for k in LOSS_FAMILIES)+'</tr>'for v in details['G10_actual_family_decision'])+'</table><p>G10 고정 family: '+f['selected_family']+'; G13 자체 재선택: '+f['g13_reselected_family']+'. G13 자체 선택도 이 실패10장을 자동 회복하지 않습니다.</p>'
        proof=dict(id=ident,group=f['group'],source_RGB_sha256=p['image_sha256'],permission_inventory_match=True,
            source_crop=f'case_{number:02d}/source_crop.jpg',viewport_original_xyxy=roi,
            common_crop_and_scale=True,viewport_source='original_initial_N3_prediction_and_coarse_only',panels=panelproof,
            no_GT_wire=True,S_precise_observation_markers=0,OTU_precise_observation_markers=0)
        proofs.append(proof)
        articles.append(f'<article id="case{number}"><h2>{number}. {html.escape(ident)} · {f["group"]}</h2><p>○ full B/I 실제 측정, × 상속 예측 fallback, 자홍색 점선 partial normal·유한 tangent 지지, 점선 영역 O/T/U, 하늘색 점선은 저장 mask의 convex outline(physical owner 인증 아님), M번호·실선은 모델 투영입니다. S는 모델 투영만 표시합니다. 두 후보는 같은 RGB·crop·축척입니다. semantic owner는 미검증입니다.</p><img loading="lazy" src="case_{number:02d}/comparison.jpg" alt="실제 선택과 원순위 참조 대조 후보"><p><a href="case_{number:02d}/source_crop.jpg">저장 RGB crop</a> · <a href="case_{number:02d}/cost_and_observations.json">후보·비용·전체 관측 장부</a></p>{g10html}<h3>G13 후보 비용 기여 — Delta = 대조 − 실제 선택</h3><p>양수는 실제 선택 s를 선호합니다. 이 경우 {"s는 성공·b는 실패인 반례"if f["group"]=="FINAL_SUCCESS_CONTROL"else"g는 참조 성공이나 원비용은 s를 선택"}입니다. 각 full/partial의 raw px·whitened·강건화 값은 아래 JSON에 별도 단위로 있습니다. hard constraints는 합산 비용이 아닙니다.</p><table><tr><th>실제 가중 후 항</th><th>s</th><th>대조</th><th>Delta</th></tr>{weighted}<tr><th>전체</th><th>{fmt(sc["objective"])}</th><th>{fmt(gc["objective"])}</th><th>{fmt(delta_total)}</th></tr></table><details><summary>전체 관측별 raw px와 가중 후 양·음 비용 기여</summary><p>raw는 full의 [x,y] 잔차, partial의 각 선 signed normal 잔차, outline의 평균 거리(px)입니다. 정규화·가중 후 cost와 단위를 섞지 않습니다.</p><table><tr><th>관측</th><th>상태</th><th>출처 s</th><th>raw s [px]</th><th>raw 대조 [px]</th><th>가중 s</th><th>가중 대조</th><th>가중 Delta</th></tr>{observed}</table></details></article>')
    nav=''.join(f'<a href="#case{i}">{i}. {html.escape(p["id"])}</a> 'for i,p in enumerate(proofs,1))
    rephtml=''.join(f'<li>{r["group"]}: <a href="#case{next(i for i,p in enumerate(proofs,1)if p["id"]==r["id"])}">{html.escape(r["id"])}</a>, 기존 E0 T={r["original_E0_translation_cm"]:.3f}cm'+(' (같은 녹화 반복 불가피)'if r['unavoidable_recording_repeat']else'')+'</li>'for r in reps)
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>원 e 선택 실패32장과 성공 반례</title><style>body{font-family:sans-serif;max-width:1700px;margin:20px auto;padding:16px;line-height:1.7;background:#eef2f6;color:#172633}header,article{background:white;padding:18px;margin-bottom:20px;border-radius:9px}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:5px;border-bottom:1px solid #ddd;text-align:left;overflow-wrap:anywhere}nav a{display:inline-block;margin:5px}details{overflow-x:auto}</style><header><h1>원 e 선택 실패32장: 실제 선택과 원순위 참조 성공 후보</h1><p>이는 원 목적함수로 생성한 고정 후보 은행의 사후 진단입니다. 참조를 사용해 비교 후보를 고른 것이며 새로운 선택기의 검증 성능이 아닙니다. 원32장은 모두 추가143에 속합니다. 새 model forward·PnP·최적화·후보·관측 생성은0회입니다.</p><p><a href="../RESULTS_KO.md">전체192·CAL46 결과</a> · <a href="../NO_CANDIDATE_INDEX_KO.md">후보부재46장 전량·기존 이미지</a> · <a href="GALLERY_PROOF.json">실제 저장 R/t·XYZ·투영 검산</a></p><h2>사전 고정 규칙으로 고른 본문 대표 ≤6장</h2><ul>'+rephtml+'</ul><p>대표는 그룹별 기존 E0 최종 위치오차 내림차순·녹화 중복 회피이며, 성공 반례는 실제 실패 후보와 동일 활성 비용항이 존재할 때만 택했습니다.</p><nav>'+nav+'</nav></header>'+''.join(articles)+'</html>'
    (out/'index.html').write_text(page)
    (out/'INDEX_KO.md').write_text('# 원 e 선택32장과 성공 반례 갤러리\n\n[HTML 열기](index.html) · [원 R/t·투영 장부](GALLERY_PROOF.json) · [본문 대표 선택 규칙](REPRESENTATIVE_SELECTION.json) · [후보부재46장](../NO_CANDIDATE_INDEX_KO.md)\n\n실제 선택32장 전량과 성공 반례2장을 같은 원사진·crop·축척으로 비교합니다. G10 결정 표와 G13 결과 표는 분리하며, raw px와 정규화 cost를 섞지 않습니다. 그림의 M번호·상자는 모델 투영이고 S는 영상 관측이 아닙니다. 새 RGB는 게시하지 않았습니다. 기존 공개 허가 SHA를 확인한 원사진 crop만 재사용했습니다.\n')
    write_json(out/'GALLERY_PROOF.json',{'schema':'SELECTION_SCORE_GALLERY_PROOF_V1','cases':proofs,'no_candidate_RGB_links':no_candidate_proofs})
    receipt={'schema':'SELECTION_SCORE_GALLERY_V1','selection_loss_images':32,'success_control_images':len(controls),
        'family10_images':sum(p['group']=='FAMILY_SELECTION_LOSS'for p in proofs),'within22_images':sum(p['group']=='WITHIN_FAMILY_SELECTION_LOSS'for p in proofs),
        'no_candidate_primary_RGB_crops':sum(p['partition']!='CAL'for p in no_candidate_proofs),
        'no_candidate_CAL_RGB_crops':sum(p['partition']=='CAL'for p in no_candidate_proofs),
        'representative_examples':len(reps),'all_RGB_previously_permitted':True,'new_NN_PnP_optimizer_candidate_measurement_training_calls':0,
        'elapsed_seconds':time.monotonic()-start,'process_CPU_seconds':time.process_time()-cpu,
        'inputs':{str(Path(v)):sha(v)for v in(costs_path,cost_seal_path,candidate_metrics_path,sf/'FRAME_SELECTIONS.jsonl.gz',sf/'RETAINED_CANDIDATES.jsonl.gz',old/'e_trace_v2/TYPED_PACKETS.jsonl.gz',old/'e_trace_v2/OBSERVATION_LINEAGE.jsonl.gz')},
        'assets':{str(p.relative_to(out)):sha(p)for p in sorted(out.rglob('*'))if p.is_file()},
        'public_bytes_before_seal':sum(p.stat().st_size for p in out.rglob('*')if p.is_file())}
    write_json(out/'GALLERY_RECEIPT.json',receipt);return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in('private','old-private','source-root','output','costs-path','candidate-metrics-path'):p.add_argument('--'+k,required=True)
    p.add_argument('--selection-folder');a=p.parse_args()
    print(json.dumps(render(a.private,a.old_private,a.source_root,a.output,costs_path=a.costs_path,candidate_metrics_path=a.candidate_metrics_path,selection_folder=a.selection_folder),ensure_ascii=False))

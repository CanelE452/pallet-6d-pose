"""Export71 true saved histories for synchronized side-by-side HTML canvases.

No model, image-feature search, solver, reference reader, or optimizer runs.
The original e history must come from the audited pre-e observation archive.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import hashlib
import json
import shutil
import time

import numpy as np
from PIL import Image

from .gallery import read,rows,sha,projection,PF95,PF143


def write(p,obj):
    with Path(p).open('x') as f:
        f.write(obj if isinstance(obj,str) else json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def native_pose(R,t,coarse):
    R=np.asarray(R,float);Q=np.asarray(coarse['R_cf']).T@np.asarray(coarse['R_physical'])
    return dict(available=True,R_cf=R.tolist(),R_physical=(R@Q).tolist(),centroid=t,cf_extents=coarse['cf_extents'])


def projected(frame,pose):
    p=projection(frame,pose)
    return None if p is None else p.tolist()


def state_projection(frame,initial,final):
    """Map initial-native state IDs through fixed physical vertices, for display."""
    def vertices(p):
        a,b,c=np.asarray(p['cf_extents'])/2
        X=np.array([[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],[-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]])
        Q=np.asarray(p['R_cf']).T@np.asarray(p['R_physical'])
        return X@Q
    xy=projected(frame,final)
    if xy is None:return None
    dist=np.linalg.norm(vertices(initial)[:,None]-vertices(final)[None,:],axis=2)
    match=dist<1e-8
    if not np.all(match.sum(0)==1) or not np.all(match.sum(1)==1):raise ValueError('State/model physical basis mismatch')
    return [xy[int(np.flatnonzero(row)[0])] for row in match]


def supported_rays(observation):
    out=[];X=np.asarray(observation['initial_geometry']['native_xyz'])
    for record in observation['visible_correction']['records']:
        k=record['native_corner']
        if observation['states'][k] not in ('B','I') or record.get('pixel_queried') is not True:continue
        for ray in record.get('rays',[]):
            j=ray.get('native_neighbor');points=np.asarray(ray.get('observed_points',[]))
            if (not ray.get('supported') or not isinstance(j,int) or not 0<=j<8 or np.count_nonzero(X[k]!=X[j])!=1
                    or points.ndim!=2 or points.shape[1]!=2 or len(points)<2):continue
            out.append({key:deepcopy(ray.get(key)) for key in ('native_corner','native_neighbor','observed_points','normal','offset','measurement_source','supported','inlier_mask')})
    return out


def compact_e_regions(regions):
    result=[];partial_rays=[]
    for kind in ('OT','U'):
        for r in regions.get(kind,[]):
            result.append({key:deepcopy(r.get(key)) for key in ('native_corner','state','shape','center_xy','half_axes_px','angle_deg','halfplanes','source')}|dict(kind=kind))
    for record in regions.get('partial_BI',[]):
        for line in record.get('lines',[]):
            result.append(dict(kind='partial_BI',native_corner=record['native_corner'],state=record.get('state'),shape='tangent_interval',
                normal=line['normal_unit_xy'],offset=line['offset_px'],interval=line['support_tangent_interval_px']))
            ray=line.get('provenance',{}).get('line_record')
            if ray and ray.get('supported') and len(ray.get('observed_points',[]))>=2:
                partial_rays.append({key:deepcopy(ray.get(key)) for key in ('native_corner','native_neighbor','observed_points','normal','offset','measurement_source','supported','inlier_mask')})
    return result,partial_rays


def build(source_root,private_root,trace,output):
    start=time.monotonic();source_root,private_root,trace,output=map(Path,(source_root,private_root,trace,output))
    if output.exists():raise FileExistsError('Preserve earlier galleries')
    trial=private_root/'pallet-posefix-gap-audit-20261016-v1';minimal=private_root/'minimal-bi-ablation-20261016-v1'
    single=private_root/'pallet-single-component-bi-20261016-v1';guarded=private_root/'pallet-guarded-bi-final-20261016-v1'
    excluded=read(trial/'excluded_gallery_v1/CARDS.json');ids=excluded['ordered_ids'];cards={r['id']:r for r in excluded['cards']}
    if len(ids)!=len(set(ids)) or len(ids)!=71:raise ValueError('Same71 excluded IDs required')
    frame_list=read(guarded/'e_inputs/INPUTS.json')['frames'];frames={r['id']:r for r in frame_list}
    components={r['id']:r for r in read(single/'inputs_v1/INPUTS.json')['frames']}
    single_score={r['id']:r for r in read(single/'score/SCORED_ROWS.json')['rows']}
    guarded_score={r['id']:r for r in read(guarded/'score/SCORED_ROWS.json')['rows']}
    e_trace=read(trace)
    if set(e_trace)!=set(ids):raise ValueError('Audited e trace must cover exactly71')
    revoked=sum(m['used_as_complete_final_likelihood'] is False for e in e_trace.values() for m in e['measurements'])
    if revoked!=107:raise ValueError('Use corrected V2 e trace with explicit107 point-likelihood revocations')
    pf={r['id']:r for r in rows(source_root/PF95/'SCORED_ROWS.jsonl.gz') if r['method']=='POSEFIX_RAW'}
    pf.update({r['id']:r for r in rows(source_root/PF143/'PREDICTIONS.jsonl.gz')})
    predictions={r['id']:r for r in rows(minimal/'run_v2/PREDICTIONS.jsonl.gz')}
    source_bindings={'PRIVATE/e_trace_V2':sha(trace),'PRIVATE/e_inputs':sha(guarded/'e_inputs/INPUTS.json'),
        'PRIVATE/component_inputs':sha(single/'inputs_v1/INPUTS.json'),'PRIVATE/single_scores':sha(single/'score/SCORED_ROWS.json'),
        'PRIVATE/guarded_scores':sha(guarded/'score/SCORED_ROWS.json'),'PRIVATE/minimal_predictions':sha(minimal/'run_v2/PREDICTIONS.jsonl.gz'),
        'PRIVATE/excluded_cards':sha(trial/'excluded_gallery_v1/CARDS.json'),
        PF95+'/SCORED_ROWS.jsonl.gz':sha(source_root/PF95/'SCORED_ROWS.jsonl.gz'),PF143+'/PREDICTIONS.jsonl.gz':sha(source_root/PF143/'PREDICTIONS.jsonl.gz')}
    output.mkdir(parents=True);(output/'rgb').mkdir();(output/'masks').mkdir()
    assets=[];mask_sources=[]
    def export_masks(path,expected,number,key,frame):
        if sha(path)!=expected:raise ValueError('Actual mask bytes changed')
        name_map={}
        with np.load(path,allow_pickle=False) as packet:
            default_field='pallet_mask_clean' if key=='E0' else 'pallet_mask'
            for field,alias in [(default_field,'pallet'),('state_pallet_mask','state')]:
                if field not in packet:continue
                a=packet[field]
                if list(a.shape)!=frame['raw_hw'] or not np.isfinite(a).all() or a.min()<0 or a.max()>1:raise ValueError('Stored probability mask/canvas changed')
                rgba=np.zeros((*a.shape,4),np.uint8);rgba[:,:,:3]=[38,132,235];rgba[:,:,3]=np.rint(a*88).astype(np.uint8)
                name=f'masks/{number:03d}_{key}_{alias}.png';Image.fromarray(rgba).save(output/name);name_map[alias]=name
        mask_sources.append(dict(id=frame['id'],method=key,source_sha256=expected,source_fields={alias:('pallet_mask_clean' if key=='E0' else 'pallet_mask') if alias=='pallet' else 'state_pallet_mask' for alias in name_map},probabilities_displayed_as_alpha=True))
        return name_map
    for number,fid in enumerate(ids,1):
        frame=frames[fid];e=e_trace[fid];index=next(i for i,r in enumerate(frame_list) if r['id']==fid);record=cards[fid]
        if e['image_sha256']!=frame['image_sha256'] or e['K']!=frame['K']:raise ValueError('e actual image/camera mismatch')
        image_source=trial/'excluded_gallery_v1'/record['full_display'];image_name=f'rgb/{number:03d}.jpg'
        sealed=read(trial/'excluded_gallery_v1/SEAL.json')['assets'][record['full_display']]
        if sha(image_source)!=sealed:raise ValueError('Already-verified display RGB changed')
        shutil.copyfile(image_source,output/image_name)
        observation_path=minimal/f'run_v2/observations/{index:03d}_N3_SUBPIX.json';pose_path=minimal/f'run_v2/poses/{index:03d}_N3_SUBPIX.json'
        observation_packet=read(observation_path);solved_packet=read(pose_path)
        if observation_packet['id']!=fid or solved_packet['id']!=fid:raise ValueError('Own N3 actual stage ID mismatch')
        result=observation_packet['result'];obs=result['observation'];solved=solved_packet['result'];solution=solved.get('solution') or {}
        source_bindings[f'PRIVATE/observation/{observation_path.name}']=sha(observation_path);source_bindings[f'PRIVATE/pose/{pose_path.name}']=sha(pose_path)
        raw=frame['arms']['RAW_YOLO'];n0=frame['arms']['N3_SUBPIX'];n1=predictions[fid]['predictions']['N1'];p=pf[fid]
        if n1['actual_pose']!=solved['actual_pose']:raise ValueError('N1 final pose must exactly match its saved solver')
        if e['final_pose']!=frame['arms']['E_FROZEN']['coarse_pose']:
            for k in ('R_cf','R_physical','centroid','cf_extents'):
                if e['final_pose'][k]!=frame['arms']['E_FROZEN']['coarse_pose'][k]:raise ValueError('e final numeric bridge differs')
        if any(e['initial_points'][k]!=n0['native_points'][k] for k in range(8) if e['states'][k]!='S'):
            raise ValueError('e non-S input points must be actual original N3 coordinates')
        nm=result['mask_binding'];n1_masks=export_masks(minimal/'run_v2'/nm['path'],nm['sha256'],number,'N1',frame)
        em=e['provenance']['mask_binding'];e_masks=export_masks(source_root/e['mask_path_or_none'],em['sha256'],number,'E0',frame)
        n3_points=components[fid]['arms']['N3_ONLY']['native_points']
        methods={}
        for key,points,pose in [('R0',raw['native_points'],raw['coarse_pose']),('POSEFIX_RAW',p['native_points'],p['actual_pose']),('N0',n0['native_points'],n0['coarse_pose'])]:
            methods[key]=dict(input_points=raw['native_points'],points=points,coarse_xy=projected(frame,pose),final_xy=projected(frame,pose),states=None,
                intermediate_points=n3_points if key=='N0' else points,
                intermediate_label='N3 단독의 저장 좌표 · 아직 SubPix 전' if key=='N0' else 'PoseFix 신경망의 저장 보정 좌표' if key=='POSEFIX_RAW' else '원 YOLO 좌표 유지',
                notes={'2':'N3→SubPix 최종 좌표와 자기 F' if key=='N0' else '추가 상태별 판정은 사용하지 않습니다.'})
        n1_candidates=[]
        for c in solution.get('candidates',[]):
            if c.get('joint_feasible') and c.get('R') is not None and c.get('t') is not None:
                n1_candidates.append(projected(frame,native_pose(c['R'],c['t'],n0['coarse_pose'])))
        for c in (solution.get('ambiguity') or {}).get('alternatives',[]):
            if c.get('R') is not None:n1_candidates.append(projected(frame,native_pose(c['R'],c['t'],n0['coarse_pose'])))
        methods['N1']=dict(input_points=raw['native_points'],intermediate_points=n3_points,intermediate_label='N3 단독의 저장 좌표 · 아직 SubPix 전',
            points=n0['native_points'],coarse_xy=projected(frame,n0['coarse_pose']),final_xy=projected(frame,n1['actual_pose']),final_state_xy=state_projection(frame,n0['coarse_pose'],n1['actual_pose']),states=obs['states'],
            measurements=obs['frozen_visible_measurements'],rays=supported_rays(obs),masks=n1_masks,candidate_xy=n1_candidates,regions=[],
            solver_label='동시점 B/I만으로 다시 푼 저장 자세 · 회색 점선은 저장된 대안',
            solver=dict(accepted=n1['accepted'],fallback=n1['fallback'],status=solution.get('status'),reason=n1.get('reason'),
                ambiguity=(solution.get('ambiguity') or {}).get('detected'),region_domain_count=solved.get('OT_U_region_pose_constraints'),
                saved_candidate_count=len(solution.get('candidates',[])),feasible_candidate_count=sum(bool(c.get('joint_feasible')) for c in solution.get('candidates',[])),
                visible_point_count=len(obs['frozen_visible_measurements']),objective=solution.get('objective'),rank=solution.get('rank'),ambiguity_record=solution.get('ambiguity')),
            notes={'3':'S/O/T/U는 B/I 점 관측에 넣지 않습니다. 이 최소 실험의 O/T/U 영역 제약은 0입니다.',
                   '4':'최적화 궤적을 생성하지 않았습니다. 저장된 후보/모호성 및 채택 여부를 표시합니다.'})
        regions,partial_rays=compact_e_regions(e['regions']);e_rays=[r for r in e['rays'] if r.get('supported') and len(r.get('observed_points',[]))>=2]+partial_rays
        audit=e['selected_candidate_audit'];fallback=audit.get('fallback_audit') or {};ambiguity=audit.get('ambiguity') or {}
        methods['E0']=dict(input_points=raw['native_points'],intermediate_points=e['initial_points'],intermediate_label='원 e에 입력된 실제 N3→SubPix 좌표',
            points=e['initial_points'],coarse_xy=projected(frame,e['initial_pose']),final_xy=projected(frame,e['final_pose']),final_state_xy=state_projection(frame,e['initial_pose'],e['final_pose']),states=e['states'],
            measurements=e['measurements'],rays=e_rays,masks=e_masks,candidate_xy=[],regions=regions,
            solver_label='기존 W/D family와 partial B/I·영역·외곽 제약으로 선택한 저장 e',
            solver={**audit,'fallback':fallback.get('applied'),'ambiguity':ambiguity.get('detected'),'ambiguity_record':ambiguity,
                'region_domain_count':len(e['regions'].get('OT',[]))+len(e['regions'].get('U',[]))},
            notes={'3':'빈 원은 최종 완전 2D 점 우도를 철회한 원 관측입니다. 주황 구간/영역은 partial 제약입니다. e 관측 마스크는 최종 외곽에 사용한 저장 clean mask이며 상태용 마스크는 따로 선택합니다.',
                   '4':'원래 e가 사용한 저장 기록입니다. e 이후 새로 관측한 guarded trial의 자료가 아닙니다.'})
        metrics={key:(guarded_score[fid] if key=='E0' else single_score[fid])['metrics'][key] for key in ('R0','N0','N1','E0')}
        metrics['POSEFIX_RAW']={k:record['metrics']['POSEFIX_RAW'][k] for k in ('translation_cm','rotation_deg')}
        r=record['crop_xyxy'];reasons=[r['arm']+':'+('T' if r['translation_exceeds'] else '')+('R' if r['rotation_exceeds'] else '') for r in record['reasons']]
        assets.append(dict(id=fid,number=number,image=image_name,source_image_sha256=frame['image_sha256'],raw_hw=frame['raw_hw'],roi=r,
            raw_points=raw['native_points'],metrics=metrics,reasons=reasons,methods=methods,
            source_files=dict(N1_observation=observation_path.name,N1_pose=pose_path.name,E0=e['provenance']),
            no_GT_overlay=True,no_new_model_solver_or_image_measurements=True))
    data=dict(schema='SYNCHRONIZED_ACTUAL_PARALLEL_STAGES71_V1',cases=assets,ordered_ids=ids,method_order=['R0','POSEFIX_RAW','N0','N1','E0'],stage_count=6,
        same_images_and_viewports=True,all_original_e_traces=True,source_bindings=source_bindings,mask_sources=mask_sources,
        original_e_complete_point_likelihood_revocations=revoked,new_NN_or_PnP_or_line_search_or_reference_fit=0)
    write(output/'DATA.json',data)
    template=Path(__file__).with_suffix('.html').read_text();embedded=json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('</','<\\/')
    write(output/'index.html',template.replace('__PAYLOAD__',embedded))
    write(output/'README_KO.md','# 제외71장 병렬 과정\n\n[HTML 열기](index.html)\n\n'
        '같은71장·같은확대범위에서 RAW/PoseFix/N3→SubPix/최소B/I/기존e의 저장단계를 병렬로 표시한다. '
        '전체과정과 단일단계 전환, 이전·다음, ID검색, 마스크·관측선·대안 자세 표시, 각패널 확대를 지원한다. '
        'S는 해당상태별경로의 영상관측에서제외하며 모델투영으로만 표시한다. 원e의107개 관측은 완전2D점 우도철회가 명시돼 빈원으로구분한다. '
        'RAW/PF/N0에 사용하지 않은 상태나 마스크를 다른방법에서 빌려넣지 않았다. 최적화 중간궤적·가짜관측·새모델계산은 없다.\n')
    write(output/'RECEIPT.json',dict(cases=71,methods=5,stage_rows=6,canvas_panels_per_case=30,
        source_sha256=sha(__file__),template_sha256=sha(Path(__file__).with_suffix('.html')),trace_sha256=sha(trace),
        stage_availability=dict(N1_original_observations=71,E0_actual_pre_e_observations=71,N3_preSubPix_points=71),
        mask_packets_read=len(mask_sources),probability_masks_displayed_as_alpha=True,original_e_revoked_point_likelihoods=revoked,
        wall_seconds=time.monotonic()-start,new_inference_or_solver_or_line_search=0))
    write(output/'SEAL.json',dict(assets={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}))
    return output/'index.html'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','private-root','trace','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(build(a.source_root,a.private_root,a.trace,a.output))

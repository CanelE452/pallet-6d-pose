"""Draw actual saved PoseFix/N3/BI/e stages without inference or pose fitting."""
from pathlib import Path
import argparse
import gzip
import hashlib
import html
import json
import time

import numpy as np

LEGACY='_docs/experiments/pallet_visibility_curriculum_20261013_v1'
PF95=LEGACY+'/nonsevere_v1/posefix_sub_comparison_v1'
PF143=LEGACY+'/nonsevere_v1/remainder_after_test49_v1/posefix_baseline_supplement_v4_14'
INVENTORY=LEGACY+'/publication_followup_20261015_v11/repro_inventory_current_files_v2.jsonl.gz'
INVENTORY_SHA='75c5aaa33d534faefc028c3358d9be6b5cbe688599488a29d34478502fac47b5'
EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
COLORS={'POSEFIX_RAW':'#c33396','N0':'#ef8f13','N1':'#ea4f36','E0':'#126ba0'}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with gzip.open(p,'rt') as f:
        yield from map(json.loads,f)


def write(p,obj):
    with Path(p).open('x') as f:
        f.write(obj if isinstance(obj,str) else json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def projection(frame,pose):
    if not pose.get('available',False):return None
    a,b,c=np.asarray(pose['cf_extents'],float)/2
    cube=np.array([[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],[-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]])
    R=np.asarray(pose['R_cf']);Q=R.T@np.asarray(pose['R_physical'])
    x,y,z=np.asarray(frame['xyz'],float)/2
    physical=np.array([[-x,-y,-z],[x,-y,-z],[x,y,-z],[-x,y,-z],[-x,-y,z],[x,-y,z],[x,y,z],[-x,y,z]])
    matches=np.linalg.norm((cube@Q)[:,None]-physical[None,:],axis=2)<1e-8
    if not np.all(matches.sum(0)==1) or not np.all(matches.sum(1)==1):raise ValueError('Physical model registration differs')
    camera=cube@R.T+np.asarray(pose['centroid'])
    if np.any(camera[:,2]<=0):return None
    h=camera@np.asarray(frame['K']).T
    return h[:,:2]/h[:,2,None]


def build(source_root,private_root,output):
    start=time.monotonic();source_root,private_root,output=map(Path,(source_root,private_root,output))
    if output.exists():raise FileExistsError('Keep prior gallery and attempts')
    guarded=private_root/'pallet-guarded-bi-final-20261016-v1'
    single=private_root/'pallet-single-component-bi-20261016-v1'
    minimal=private_root/'minimal-bi-ablation-20261016-v1'
    input_path=guarded/'e_inputs/INPUTS.json';single_path=single/'score/SCORED_ROWS.json'
    guarded_path=guarded/'score/SCORED_ROWS.json'
    frames={r['id']:r for r in read(input_path)['frames']}
    current={r['id']:r for r in read(single_path)['rows']}
    frozen={r['id']:r for r in read(guarded_path)['rows']}
    pf={r['id']:r for r in rows(source_root/PF95/'SCORED_ROWS.jsonl.gz') if r['method']=='POSEFIX_RAW'}
    old143=read(source_root/PF143/'POSTSEAL_SCORE.json')
    predictions={r['id']:r for r in rows(source_root/PF143/'PREDICTIONS.jsonl.gz')}
    for r in old143['rows']:
        p=predictions[r['id']];pf[r['id']]={**p,'metric':r['metrics']['POSEFIX_RAW']}
    if not set(pf)==set(frames)==set(current)==set(frozen):raise ValueError('Same238 required')
    for fid in current:
        for metric in ('translation_cm','rotation_deg','ADDsym_cm'):
            if abs(current[fid]['metrics']['N1'][metric]-frozen[fid]['metrics']['N1'][metric])>1e-9:
                raise ValueError('Old and current minimal BI differ')
    combined={fid:{**current[fid]['metrics'],'E0':frozen[fid]['metrics']['E0'],'POSEFIX_RAW':pf[fid]['metric']} for fid in current}
    primary=[fid for fid in current if current[fid]['partition']!='CAL']
    def ok(fid,arm):
        m=combined[fid][arm];return m['translation_cm']<=5 and m['rotation_deg']<=5
    def delta(fid,a,b):return combined[fid][a]['translation_cm']-combined[fid][b]['translation_cm']
    categories=[
        ('PoseFix 실패 → 기존 e 성공 · 위치 개선 최대',[f for f in primary if not ok(f,'POSEFIX_RAW') and ok(f,'E0')],lambda f:(delta(f,'E0','POSEFIX_RAW'),f)),
        ('PoseFix 성공 → 기존 e 실패 · 위치 손상 최대',[f for f in primary if ok(f,'POSEFIX_RAW') and not ok(f,'E0')],lambda f:(-delta(f,'E0','POSEFIX_RAW'),f)),
        ('N3→SubPix 성공 → 최소 BI 실패 · 위치 손상 최대',[f for f in primary if ok(f,'N0') and not ok(f,'N1')],lambda f:(-delta(f,'N1','N0'),f)),
        ('N3→SubPix 실패 → 최소 BI 성공 · 위치 개선 최대',[f for f in primary if not ok(f,'N0') and ok(f,'N1')],lambda f:(delta(f,'N1','N0'),f)),
        ('기존 e의 최대 위치 오차',primary,lambda f:(-combined[f]['E0']['translation_cm'],f)),
        ('PoseFix 실패 → N3→SubPix 성공 · 위치 개선 최대',[f for f in primary if not ok(f,'POSEFIX_RAW') and ok(f,'N0')],lambda f:(delta(f,'N0','POSEFIX_RAW'),f)),
    ]
    chosen={};selection=[]
    for label,candidates,key in categories:
        fid=min(candidates,key=key) if candidates else None
        selection.append(dict(category=label,id=fid,empty=not candidates))
        if fid is not None:chosen.setdefault(fid,[]).append(label)
    actual={r['id']:r for r in rows(minimal/'run_v2/PREDICTIONS.jsonl.gz') if r['id'] in chosen}
    inventory=source_root/INVENTORY
    if sha(inventory)!=INVENTORY_SHA:raise ValueError('Published RGB permission inventory changed')
    permitted={r['sha256'] for r in rows(inventory) if Path(r['relpath']).suffix.lower() in ('.png','.jpg','.jpeg')}
    output.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt,font_manager,patheffects
    from PIL import Image
    font=Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False})
    evidence=[];cards=[]
    bindings={
        'PRIVATE/guarded/e_inputs/INPUTS.json':sha(input_path),
        'PRIVATE/single/score/SCORED_ROWS.json':sha(single_path),
        'PRIVATE/guarded/score/SCORED_ROWS.json':sha(guarded_path),
        'PRIVATE/minimal/run_v2/PREDICTIONS.jsonl.gz':sha(minimal/'run_v2/PREDICTIONS.jsonl.gz'),
        PF95+'/SCORED_ROWS.jsonl.gz':sha(source_root/PF95/'SCORED_ROWS.jsonl.gz'),
        PF143+'/POSTSEAL_SCORE.json':sha(source_root/PF143/'POSTSEAL_SCORE.json'),
        PF143+'/PREDICTIONS.jsonl.gz':sha(source_root/PF143/'PREDICTIONS.jsonl.gz'),
        INVENTORY:INVENTORY_SHA}
    for number,(fid,labels) in enumerate(chosen.items(),1):
        frame=frames[fid];m=combined[fid];n1=actual[fid]['predictions']['N1'];p=pf[fid]
        index=list(frames).index(fid)
        obs_path=minimal/f'run_v2/observations/{index:03d}_N3_SUBPIX.json'
        obs=read(obs_path)['result']['observation']
        if read(obs_path)['id']!=fid:raise ValueError('Own N3 observation mismatch')
        bindings[f'PRIVATE/minimal/run_v2/observations/{obs_path.name}']=sha(obs_path)
        source=source_root/frame['image']
        if sha(source)!=frame['image_sha256'] or frame['image_sha256'] not in permitted:
            raise ValueError('Only unchanged already-public RGB can be republished')
        image=np.asarray(Image.open(source).convert('RGB'));h,w=image.shape[:2]
        if [h,w]!=frame['raw_hw']:raise ValueError('Image canvas differs')
        n0=frame['arms']['N3_SUBPIX'];raw=frame['arms']['RAW_YOLO'];e0=frame['arms']['E_FROZEN']
        box=np.asarray(frame['candidate_metadata']['box_xyxy'],float).reshape(2,2)
        q=np.vstack([box,np.asarray(n0['native_points'])[:8],np.asarray(p['native_points'])[:8]])
        q=q[np.isfinite(q).all(1)&(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)]
        lo=q.min(0);hi=q.max(0);margin=np.maximum((hi-lo)*.15,[25,25]);lo=np.maximum(lo-margin,0);hi=np.minimum(hi+margin,[w-1,h-1])
        poses={'R0':raw['coarse_pose'],'POSEFIX_RAW':p['actual_pose'],'N0':n0['coarse_pose'],'N1':n1['actual_pose'],'E0':e0['coarse_pose']}
        projections={name:projection(frame,pose) for name,pose in poses.items()}
        fig,axs=plt.subplots(2,3,figsize=(15.8,9.3),layout='constrained')
        fig.suptitle(fid+'\n모든 상자·점은 저장 예측/관측입니다. GT 표시는 없습니다.',fontsize=15)
        for ax in axs.flat:
            ax.imshow(image,interpolation='nearest');ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.set_aspect('equal');ax.tick_params(labelsize=7)
        def wire(ax,arm,label=None,alpha=1):
            q=projections[arm]
            if q is None:return
            for i,(a,b) in enumerate(EDGES):
                ax.plot(q[[a,b],0],q[[a,b],1],c=COLORS.get(arm,'#999999'),lw=1.7,alpha=alpha,label=label if i==0 else None)
        def points(ax,q,color):
            for k,xy in enumerate(np.asarray(q)[:8]):
                if not np.isfinite(xy).all() or np.all(xy==-1):continue
                ax.scatter(*xy,c=color,marker='x',s=26)
                if lo[0]<=xy[0]<=hi[0] and lo[1]<=xy[1]<=hi[1]:
                    t=ax.annotate(str(k),xy,xytext=(4,5),textcoords='offset points',color=color,fontsize=8)
                    t.set_path_effects([patheffects.withStroke(linewidth=1.5,foreground='white')])
        def title(ax,number,label,arm):
            metric=m[arm];ax.set_title(f'{number}. {label}\nT {metric["translation_cm"]:.2f}cm / R {metric["rotation_deg"]:.2f}°',fontsize=11)
        ax=axs[0,0];wire(ax,'R0');points(ax,raw['native_points'],'#b44831');title(ax,1,'RAW YOLO → 공통 F','R0')
        ax=axs[0,1];wire(ax,'POSEFIX_RAW');points(ax,p['native_points'],COLORS['POSEFIX_RAW']);title(ax,2,'별도 RGB PoseFix → 자기 F','POSEFIX_RAW')
        ax=axs[0,2];wire(ax,'N0');points(ax,n0['native_points'],COLORS['N0']);title(ax,3,'N3→SubPix → 공통 F','N0')
        ax=axs[1,0];wire(ax,'N1');movement=[]
        for measurement in obs['frozen_visible_measurements']:
            k=measurement['native_corner'];state=obs['states'][k]
            if state not in ('B','I'):raise ValueError('Non-BI measurement')
            a=np.asarray(n0['native_points'][k]);b=np.asarray(measurement['xy']);distance=float(np.linalg.norm(a-b));color='#00a0bd' if state=='B' else '#8847d9'
            ax.scatter(*a,color='#888888',marker='x',s=25);ax.scatter(*b,color=color,s=24)
            if distance>1e-6:ax.annotate('',b,a,arrowprops=dict(arrowstyle='->',color=color,lw=1.4))
            if lo[0]<=b[0]<=hi[0] and lo[1]<=b[1]<=hi[1]:ax.annotate(f'{k}:{state}',b,xytext=(4,6),textcoords='offset points',fontsize=8,color=color)
            movement.append(dict(native_id=k,state=state,source=measurement['source'],distance_px=distance,original_xy=a.tolist(),observed_xy=b.tolist()))
        title(ax,4,'최소 B/I 실제 관측 → 최종 자세','N1')
        ax=axs[1,1];wire(ax,'E0');title(ax,5,'기존 전체 상태별 방법 e · 저장 최종 자세','E0')
        ax=axs[1,2]
        for arm,label in [('POSEFIX_RAW','PoseFix'),('N0','N3→SubPix'),('E0','기존 e')]:wire(ax,arm,label)
        ax.legend(fontsize=8);ax.set_title('6. 같은 영상에서 최종 예측 비교\n각 상자는 자기 CF 축·모델 크기로 투영',fontsize=11)
        filename=f'case_{number:02d}.jpg';fig.savefig(output/filename,dpi=115);plt.close(fig)
        item=dict(id=fid,categories=labels,partition=frame['partition'],image_sha256=frame['image_sha256'],image_already_public=True,
            viewport_xyxy=[*lo.tolist(),*hi.tolist()],metrics={k:m[k] for k in ('R0','POSEFIX_RAW','N0','N1','E0')},
            actual_saved_pose_keys=list(poses),N1_states=obs['states'],N1_measurements=movement,
            N1_accepted=n1['accepted'],N1_fallback=n1['fallback'],N1_reason=n1['reason'],asset=filename,
            no_GT_overlay=True,no_new_observation_or_pose_or_optimizer=True,
            E0_intermediate_reconstructed=False,N1_measurements_not_E0_measurements=True,
            corner_IDs_are_each_pipeline_native_not_proof_of_shared_physical_semantic_identity=True)
        evidence.append(item)
        text=' · '.join(labels)
        body=f'<article id="case-{number}"><h2>{html.escape(fid)}</h2><p>{html.escape(text)}</p><img src="{filename}" alt="저장 단계 비교"><p>최소 B/I 상태: {html.escape(" ".join(f"{k}:{s}" for k,s in enumerate(obs["states"])))}, accepted={n1["accepted"]}, fallback={n1["fallback"]}</p></article>'
        cards.append(body)
    write(output/'PROCESS.json',dict(selection=selection,cases=evidence,posthoc_diagnostic_selection=True,selection_did_not_change_inference=True,
        source_bindings=bindings,new_neural_or_mask_or_line_detection_or_PnP_calls=0,new_reference_or_annotation_reads=0))
    intro='공통 영상의 저장된 실제 단계입니다. e는 비교 예측이며 GT가 아닙니다. 4번은 최소 B/I 관측이고 5번은 기존 전체 e의 최종 자세로 서로 다른 경로입니다. e의 저장되지 않은 중간 과정을 만들어 연결하지 않았습니다. S는 B/I 관측에 넣지 않으며 최종 모델에서 투영합니다. 0–7 번호는 각 경로의 native 코너 번호입니다.'
    write(output/'index.html','<!doctype html><html lang="ko"><meta charset="utf-8"><title>PoseFix와 현재 방법 차이</title><style>body{font-family:sans-serif;background:#eef2f6;max-width:1800px;margin:20px auto;padding:15px;line-height:1.7}article{background:white;padding:18px;margin:18px 0;border-radius:8px}img{width:100%;height:auto}</style><h1>PoseFix·N3→SubPix·최소 B/I·기존 e</h1><p>'+intro+'</p>'+''.join(cards)+'</html>')
    md=['# 실제 이미지의 저장 단계 비교',intro,'사례 선택은 점수를 확인한 뒤의 진단용이며 추론이나 방법 선택에 영향을 주지 않았다. 전체 분모와 모든 승패는 상위 보고서와 CSV에 보존한다.']
    for i,item in enumerate(evidence,1):
        md.extend([f'## {item["id"]}',' · '.join(item['categories']),f'![실제 저장 단계 비교](case_{i:02d}.jpg)'])
    write(output/'GALLERY_KO.md','\n\n'.join(md)+'\n')
    write(output/'RECEIPT.json',dict(cases=len(evidence),assets=len(evidence),source_sha256=sha(__file__),wall_seconds=time.monotonic()-start,
        image_decodes=len(evidence),same_actual_RGB_SHA_verified=True,already_public_RGB_only=True,new_pose_or_measurement_or_NN_calls=0))
    write(output/'SEAL.json',{'assets':{p.name:sha(p) for p in output.iterdir() if p.is_file()}})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','private-root','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();build(a.source_root,a.private_root,a.output)

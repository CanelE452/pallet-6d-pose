"""Explain a saved width/depth-family difference, without fitting new poses."""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np

from .gallery import read,rows,projection,EDGES


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    with Path(p).open('x')as f:f.write(obj if isinstance(obj,str)else json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def angle(R):return float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1))))


def build(source_root,private_root,output):
    start=time.monotonic();source_root,private_root,output=map(Path,(source_root,private_root,output))
    if output.exists():raise FileExistsError('Keep earlier actual-case diagrams')
    trial=private_root/'pallet-posefix-gap-audit-20261016-v1';guarded=private_root/'pallet-guarded-bi-final-20261016-v1';minimal=private_root/'minimal-bi-ablation-20261016-v1'
    fid='plastic_day_01:011640'
    frame=next(r for r in read(guarded/'e_inputs/INPUTS.json')['frames']if r['id']==fid)
    data=next(r for r in read(trial/'parallel_gallery_v2/DATA.json')['cases']if r['id']==fid)
    prediction=next(r for r in rows(minimal/'run_v2/PREDICTIONS.jsonl.gz')if r['id']==fid)
    poses={'N0':frame['arms']['N3_SUBPIX']['coarse_pose'],'N1':prediction['predictions']['N1']['actual_pose'],'E0':frame['arms']['E_FROZEN']['coarse_pose']}
    dims=np.asarray(frame['xyz']);R0=np.asarray(poses['N0']['R_physical']);C2=np.diag([-1.,1.,-1.])
    audit={}
    for key,p in poses.items():
        R=np.asarray(p['R_physical']);Rcf=np.asarray(p['R_cf']);Q=Rcf.T@R
        audit[key]=dict(cf_extents=p['cf_extents'],physical_registration_Q=Q.tolist(),R_cf=p['R_cf'],R_physical=p['R_physical'],centroid_m=p['centroid'],
            CF_rotation_from_N0_deg=angle(np.asarray(poses['N0']['R_cf']).T@Rcf),
            physical_C2_rotation_from_N0_deg=min(angle((R0@q).T@R)for q in (np.eye(3),C2)),
            width_direction_components_in_N0_W_D=[float(R0[:,0]@R[:,0]),float(R0[:,2]@R[:,0])],
            depth_direction_components_in_N0_W_D=[float(R0[:,0]@R[:,2]),float(R0[:,2]@R[:,2])])
    output.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt,font_manager,patheffects
    from PIL import Image
    font=Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf');font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False})
    image_path=source_root/frame['image']
    if sha(image_path)!=frame['image_sha256']:raise ValueError('Actual RGB changed')
    rgb=Image.open(image_path).convert('RGB');roi=data['roi'];K=np.asarray(frame['K'])
    labels={'N0':'N3→SubPix + F','N1':'N3→SubPix + 최소 B/I','E0':'기존 전체 방법 e'}
    fig,axs=plt.subplots(2,3,figsize=(16,9.5),layout='constrained')
    fig.suptitle('같은 상자처럼 보여도 폭·깊이 대응은 다를 수 있습니다\n'+fid+' · 실제 등록 크기 W=110cm, D=130cm · 모든 상자는 예측, GT 표시는 없음',fontsize=16)
    for column,key in enumerate(('N0','N1','E0')):
        p=poses[key];R=np.asarray(p['R_physical']);t=np.asarray(p['centroid']);ax=axs[0,column]
        m=data['metrics'][key];ax.imshow(rgb,interpolation='nearest');ax.set_xlim(roi[0],roi[2]);ax.set_ylim(roi[3],roi[1]);ax.set_aspect('equal')
        ax.set_title(f'{labels[key]}\n위치 {m["translation_cm"]:.2f}cm / 회전 {m["rotation_deg"]:.2f}°',fontsize=14)
        xy=projection(frame,p)
        for a,b in EDGES:ax.plot(xy[[a,b],0],xy[[a,b],1],c='#657681',lw=1.4,alpha=.8)
        top=np.array([0.,-dims[1]/2,0.])
        def project(q):
            v=K@(R@q+t);return v[:2]/v[2]
        for axis,length,color,label in [(0,dims[0],'#e54832','W 폭110'),(2,dims[2],'#186db8','D 깊이130')]:
            d=np.zeros(3);d[axis]=length*.44;a,b=project(top-d),project(top+d)
            ax.annotate('',b,a,arrowprops=dict(arrowstyle='<->',color=color,lw=3))
            txt=ax.annotate(label,b,xytext=(5,-15 if axis==0 else 13),textcoords='offset points',fontsize=12,color=color,weight='bold')
            txt.set_path_effects([patheffects.withStroke(linewidth=3,foreground='white')])
        ax.tick_params(labelsize=8)
        ax=axs[1,column];ax.axhline(0,c='#ccd5de',lw=1);ax.axvline(0,c='#ccd5de',lw=1)
        ax.plot([-.55,.55,.55,-.55,-.55],[-.65,-.65,.65,.65,-.65],c='#bcc7d0',ls='--',label='N0 기준 사각형')
        for axis,length,color,label in [(0,dims[0],'#e54832','W 폭110cm'),(2,dims[2],'#186db8','D 깊이130cm')]:
            direction=np.array([R0[:,0]@R[:,axis],R0[:,2]@R[:,axis]])*length/2
            ax.annotate('',direction,-direction,arrowprops=dict(arrowstyle='<->',color=color,lw=3.3))
            ax.text(direction[0]+.035,direction[1]+.025,label,color=color,fontsize=12,weight='bold')
        ax.set(aspect='equal',xlim=(-.9,.9),ylim=(-.9,.9),xlabel='N0의 폭 방향 [m]',ylabel='N0의 깊이 방향 [m]')
        ax.set_title(f'중심을 겹쳐 방향만 비교 · N0와의 물리 회전 차이 {audit[key]["physical_C2_rotation_from_N0_deg"]:.2f}°',fontsize=11)
        ax.text(.02,.02,'N0 기준 평면에 표시 · GT 또는 보정된 결과 아님',transform=ax.transAxes,fontsize=9,color='#556574')
        ax.grid(alpha=.2)
    fig.savefig(output/'PHYSICAL_AXES_COMPARE.png',dpi=145);fig.savefig(output/'PHYSICAL_AXES_COMPARE.svg');plt.close(fig)
    write(output/'AXIS_DIAGNOSTIC.json',dict(id=fid,physical_dimensions_m=dims.tolist(),current_fixed_reference_metrics=data['metrics'],poses=audit,
        known_rectangular_C2=True,C4_not_used_in_scores=True,point_or_pose_or_reference_values_modified=False,
        diagram_orthographic_panel='centers aligned, directions projected onto original N0 width-depth plane; not GT',
        source_sha256=sha(__file__),source_image_sha256=frame['image_sha256'],new_model_or_PnP_or_reference_fit=0,wall_seconds=time.monotonic()-start))
    write(output/'README_KO.md','# 011640 물리 폭·깊이 대응 비교\n\n![폭·깊이 축 비교](PHYSICAL_AXES_COMPARE.png)\n\n'
        '빨강은 등록 폭W=110cm, 파랑은 깊이D=130cm다. 위는 같은 실제영상·같은ROI에 저장된 예측모델과 물리축을 투영했다. '
        '아래는 중심을 겹치고 N0의 W/D평면에서 방향만 비교하며 GT가 아니다. 기존e는 N0와 native CF회전이약5.59°만다르지만 '
        '폭·깊이 모델가족/기저등록이다르므로 물리C2회전차는약84.68°다. 직사각형에서90°는평가대칭이아니다. '
        '표시오차는기존고정참조에대한값이며 원주석의독립적물리축GT를새로확정한것이아니다.\n')
    write(output/'SEAL.json',dict(assets={p.name:sha(p)for p in output.iterdir()if p.is_file()}))
    return output/'PHYSICAL_AXES_COMPARE.png'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','private-root','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(build(args.source_root,args.private_root,args.output))

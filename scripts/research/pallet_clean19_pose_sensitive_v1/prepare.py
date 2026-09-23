"""Preflight, candidate-preserving D8 scoring, exact geometry/projection gates."""
import argparse
import inspect
import subprocess
import json
import math
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from ultralytics.data.base import BaseDataset
from ultralytics.data.dataset import YOLODataset
from . import common as E
from .common import D,C

SIDETABLE=E.ROOT/'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz'

def preflight():
    assert all(not p.exists() for p in [E.DOC,E.RAW,E.OUT])
    for p in [E.DOC,E.RAW,E.OUT]:p.mkdir(parents=True)
    data=D.load();C.immutable()
    files=data['paths']+[SIDETABLE,E.ROOT/'scripts/research/pallet_translation_loss_v1/build_geometry_sidetable.py',E.ROOT/'scripts/self_training_yolo/v3/true_ignore_pose_loss.py',E.ROOT/'scripts/research/pallet_clean19_structured_easyhard_v1/train.py']
    for mat in C.MATERIALS:files.append(E.ROOT/C.read(C.DOC/f'FIT_{mat}_S1.json')['checkpoint']['path'])
    files.append(C.H.C.N.E.R0)
    binding=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),files=[E.bind(p) for p in sorted(set(files))])
    E.save(E.DOC/'INPUT_BINDINGS.json',binding)
    E.save(E.RAW/'WORKTREE_BEFORE.json',dict(status=subprocess.check_output(['git','status','--short','--branch'],text=True)))
    args=E.read(C.DOC/'PREFLIGHT.json')['args']
    E.save(E.DOC/'PREFLIGHT_AUDIT.md','# Pose-sensitive preflight\n\nHEAD: `'+binding['head']+'`; branch: '+binding['branch']+'\n\nOriginal artifacts verified; old worktree preserved. Phase A no training, D8 changes only residual point set. Phase B requires exact512 binding and <=0.05px projection agreement before any fit. Four fits max, original R0, seed42, 320 steps each. No rescues.\n\nExisting args:\n```json\n'+json.dumps(args,indent=2)+'\n```\n')
    E.save(E.DOC/'INSTALLED_DATA_CONTRACT.json',dict(load_image_source=inspect.getsource(BaseDataset.load_image),collate_source=inspect.getsource(YOLODataset.collate_fn),note='stock collate stacks only known fields; geo_H/enabled require explicit custom stack',args=args))
    print('PREFLIGHT_READY',flush=True)

def phase_a():
    data=D.load();previous=E.read(D.DOC/'FRAME_DIAGNOSTICS.json');rows={a:{} for a in ['R0','S0','S1','S2']};cfg=D.Selector.SelectorConfig()
    for arm in rows:
        for r in data['records']:
            fid=r['id'];q=D.points(data['preds'][arm][fid]);m=data['meta'][fid]
            sr=D.select(q,np.array(m['K']),np.array(m['xyz']));scores=E.d8_scores(sr,q)
            d9=E.rank(sr['hypotheses'],{k:v['D9'] for k,v in scores.items()});d8=E.rank(sr['hypotheses'],{k:v['D8'] for k,v in scores.items()})
            assert d9==sr['selected_hypothesis']==previous[arm][fid]['selector']['selected_hypothesis']
            # Reuse already validated hypothesis poses and posthoc metrics. D8 NEVER re-solves.
            hh=previous[arm][fid]['hypotheses'];get=lambda n:next((h['metric'] for h in hh if h['name']==n),dict(available=False))
            p9=get(d9);p8=get(d8);D.close(p9,data['pm'][arm][fid])
            rows[arm][fid]=dict(frame_id=fid,severity=r['severity'],material=r['object_type'],session=r['session'],D9_selected=d9,D8_selected=d8,D9=p9,D8=p8,scores=scores,
                changed=d9!=d8,wrong_to_correct=p9.get('axis_correct') is False and p8.get('axis_correct') is True,correct_to_wrong=p9.get('axis_correct') is True and p8.get('axis_correct') is False,
                ADD_delta=p8['ADDsym_normalized']-p9['ADDsym_normalized'] if p8['available'] and p9['available'] else None)
        print('PHASE_A',arm,len(rows[arm]),flush=True)
    groups={}
    for group in data['results']['groups']:
        rr=[r for r in data['records'] if group=='ALL300' or group==r['severity'] or group==r['object_type'].upper() or group==r['object_type'].upper()+'_'+r['severity'] or group=='SESSION_'+r['session']]
        if not rr:continue
        groups[group]={}
        for arm in rows:
            xx=[rows[arm][r['id']] for r in rr]
            groups[group][arm]=dict(D9=D.aggregate([r['D9'] for r in xx]),D8=D.aggregate([r['D8'] for r in xx]),selection_changed=sum(r['changed'] for r in xx),
                wrong_to_correct=sum(r['wrong_to_correct'] for r in xx),correct_to_wrong=sum(r['correct_to_wrong'] for r in xx),
                changed_ADD_improved=sum(r['changed'] and r['ADD_delta'] is not None and r['ADD_delta']<0 for r in xx),changed_ADD_worsened=sum(r['changed'] and r['ADD_delta'] is not None and r['ADD_delta']>0 for r in xx))
    E.save(E.DOC/'CENTER_P8_DIAGNOSTIC.json',dict(status='DIAGNOSTIC_CANDIDATE_ONLY',production_changed=False,groups=groups,rows=rows,selector_config=vars(cfg),same_candidate_poses=True,D8_new_threshold=False))

def raw_label(stem):
    if stem.startswith('G38__G__'):
        return E.ROOT/'data/pallet/training_data/paper_release/v2_prod40k_clean_merged/labels'/(stem.removeprefix('G38__G__')+'_label.json')
    source,ident=stem.split('__',1);shard,frame=ident.rsplit('_',1)
    root='_raw_legacy_v1v2_p0_10k' if source=='P0' else '_raw_legacy_v1v2_p0_tex10k'
    assert source in ['P0','TEX']
    return E.ROOT/'challenge/yolo_pose_one_model/datasets'/root/shard/'labels'/(frame+'_label.json')

def project(K,R,t,X):
    P=X@R.T+t;u=P@K.T
    return u[:,:2]/u[:,2:]

def geometry():
    pp=E.plans();unique={Path(r['image']).stem.removeprefix('syn__'):r for r in pp if not r['real']}
    assert len(unique)==512
    table=dict(np.load(SIDETABLE));ix={str(s):i for i,s in enumerate(table['stems'])};bound=[];missing=[];geos={}
    from challenge.yolo_pose_one_model.spatial_concat_scratch.build_probe_metadata import fixed_dimensions
    for stem,r in sorted(unique.items()):
        label=raw_label(stem)
        if stem not in ix or not label.exists():missing.append(dict(stem=stem,reason='sidetable or renderer label missing',label=str(label)));continue
        i=ix[stem];ann=E.read(label);obj=ann['objects'][0];cam=ann['camera_data'];intr=cam['intrinsics'];pad=float(table['pad'][i]);T=np.array(obj['pose_transform'])
        _,dims,perm,case=fixed_dimensions(obj,stem)
        K4=np.array([intr['fx'],intr['fy'],intr['cx']+pad,intr['cy']+pad]);K=np.array([[K4[0],0,K4[2]],[0,K4[1],K4[3]],[0,0,1.]])
        np.testing.assert_allclose(K4,table['K'][i],atol=1e-10);np.testing.assert_allclose(dims,table['dims'][i],atol=1e-10)
        np.testing.assert_allclose(T[:3,:3],table['R'][i],atol=1e-10);np.testing.assert_allclose(T[:3,3],table['t'][i],atol=1e-10)
        image=Path(r['image']);w,h=Image.open(image).size
        assert (w,h)==(int(cam['width']+2*pad),int(cam['height']+2*pad)),(stem,w,h,cam)
        training_label=image.parent.parent/'labels'/image.with_suffix('.txt').name
        tokens=np.array(training_label.read_text().split(),float);kp=tokens[5:].reshape(9,3);q=project(K,table['R'][i],table['t'][i],table['Xcf'][i])
        mask=(kp[:8,2]==2)&(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)
        error=float(np.linalg.norm(q[mask]-kp[:8,:2][mask]*[w,h],axis=1).max()) if mask.any() else 0.
        renderer_error=float(np.linalg.norm(q-(np.array(obj['projected_cuboid'])[:8]+pad),axis=1).max())
        assert error<=.05 and renderer_error<=.05,(stem,error,renderer_error)
        # Exact source image and label hashes, not a stem-only join claim.
        row=dict(stem=stem,image=E.bind(image),label=E.bind(training_label),renderer_label=E.bind(label),table_index=i,K=K.tolist(),dimensions=dims,R=table['R'][i].tolist(),t=table['t'][i].tolist(),
            Xcf=table['Xcf'][i].tolist(),perm_v4=perm,pad=pad,hw=[h,w],native_projection_error_px=error,renderer_projection_error_px=renderer_error,symmetry='not required: fixed renderer 3D correspondence',pose_source='stored renderer pose_transform; no PnP-generated GT')
        bound.append(row);geos[stem]=row
    E.save(E.DOC/'SYNTH_GEOMETRY_BINDING.json',dict(expected=512,bound=len(bound),missing=missing,side_table=E.bind(SIDETABLE),builder=E.bind(E.ROOT/'scripts/research/pallet_translation_loss_v1/build_geometry_sidetable.py'),records=bound))
    if missing:
        E.save(E.DOC/'STOP.json',dict(status='POSE_SENSITIVE_HYPOTHESIS_NOT_TESTED',reason='synthetic512 exact binding incomplete',fits=0));return
    # Actual installed call uses the default rect_mode=True, even with dataset.rect=False.
    src=inspect.getsource(BaseDataset.get_image_and_label)
    assert 'self.load_image(index)' in src
    assert inspect.signature(BaseDataset.load_image).parameters['rect_mode'].default is True
    E.save(E.DOC/'RESIZE_CONTRACT.json',dict(source=src,dataset_rect=False,load_image_rect_mode=True,resize='long side640, ceil each dimension; M=stored RandomPerspective M @ diag(resized_w/w,resized_h/h,1)',native_check_correction='Initial helper included visibility0 (0,0) sentinel targets; corrected to required supervised/in-frame-only before geometry binding completed. No supervised target mismatch found at that stage.'))
    parity=[];maxerr=0.;failing=[];occ=[]
    for r in pp:
        E.verify(r['cache'])
        if r['real']:continue
        stem=Path(r['image']).stem.removeprefix('syn__');g=geos[stem];h,w=g['hw'];ratio=640/max(h,w)
        rw=min(math.ceil(w*ratio),640);rh=min(math.ceil(h*ratio),640)
        M=np.array(r['affine'])@np.diag([rw/w,rh/h,1.])
        assert np.array_equal(M[2],[0,0,1]) and abs(np.linalg.det(M))>1e-10
        with np.load(E.ROOT/r['cache']['path']) as z:kp=z['keypoints'][0]
        q=project(M@np.array(g['K']),np.array(g['R']),np.array(g['t']),np.array(g['Xcf']))
        valid=(kp[:8,2]==2)&(q[:,0]>=0)&(q[:,0]<640)&(q[:,1]>=0)&(q[:,1]<640)
        err=float(np.linalg.norm(q[valid]-kp[:8,:2][valid]*640,axis=1).max()) if valid.any() else 0.
        maxerr=max(maxerr,err)
        row=dict(material=r['material'],epoch=r['epoch'],slot=r['slot'],stem=stem,M=M.tolist(),valid_corners=valid.tolist(),max_error_px=err,cache=r['cache'])
        occ.append(row)
        if err>.05:failing.append(row)
    E.save(E.DOC/'SYNTH_PROJECTION_PARITY.json',dict(tolerance_px=.05,occurrences=len(occ),max_error_px=maxerr,failed=len(failing),failures=failing))
    if failing:
        E.save(E.DOC/'STOP.json',dict(status='POSE_SENSITIVE_HYPOTHESIS_NOT_TESTED',reason='projection parity failed',fits=0));return
    E.save(E.RAW/'SYNTH_OCCURRENCES.json',occ)
    # Fixed-reference reconstruction tolerance, set before any sensitivity calculation.
    tolerance=1e-6;Hs={};audit=[]
    for stem,g in geos.items():
        K=np.array(g['K']);R=np.array(g['R']);t=np.array(g['t']);X=np.array(g['Xcf']);q=project(K,R,t,X);diam=np.linalg.norm(g['dimensions'])
        def solve(v):
            rr,tt,_=D.Pose.solve(X,np.asarray(v,np.float64),K,np.ones(8,bool));return rr,tt
        rr,tt=solve(q);recon=float(np.max(np.abs((X@rr.T+tt)-(X@R.T+t))))
        assert recon<=tolerance,(stem,'GT reconstruction',recon)
        A=np.zeros((24,16))
        for k in range(16):
            plus=q.copy();minus=q.copy();plus.flat[k]+=1.;minus.flat[k]-=1.
            rp,tp=solve(plus);rm,tm=solve(minus);A[:,k]=(((X@rp.T+tp)-(X@rm.T+tm))/diam/2).ravel()
        H=A.T@A;H=(H+H.T)/2;eig=np.linalg.eigvalsh(H)
        assert np.isfinite(H).all() and eig.min()>=-1e-10*max(1.,eig.max())
        Hs[stem]=H;audit.append(dict(stem=stem,reconstruction_max_m=recon,min_eigenvalue=float(eig.min()),max_eigenvalue=float(eig.max()),trace=float(np.trace(H))))
    dest=E.RAW/'H_NATIVE.npz';assert not dest.exists();np.savez(dest,**Hs)
    E.save(E.DOC/'POSE_SENSITIVITY_AUDIT.json',dict(valid=len(Hs),invalid=0,epsilon_px=1.,GT_reconstruction_tolerance_m=tolerance,psd_tolerance='-1e-10*max(1,maxeig)',records=audit,cache=E.bind(dest),description='finite-difference local pose-sensitivity quadratic; not full Linear-Covariance reproduction'))
    print('GEOMETRY_READY',len(bound),'max_projection',maxerr,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','phase_a','geometry']);a=p.parse_args();globals()[a.stage]()

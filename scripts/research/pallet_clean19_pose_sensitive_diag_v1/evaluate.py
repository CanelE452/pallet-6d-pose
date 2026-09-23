"""Freeze student predictions first; reference-dependent scoring is a separate stage."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from scripts.research.pallet_clean19_structured_easyhard_v1.diagnostics import source_model
from scripts.research.pallet_clean19_structured_easyhard_v1.evaluate import transitions
from . import common as E
from .geo_diag_loss import criterion,diag_total
from .trainer import DiagDataset,DiagModel,collate
C=E.C;D=E.D

def infer():
    E.deterministic();E.immutable();files=[]
    records=E.read(C.H.P.DOC/'SPLIT.json')['evaluation']
    for mat in E.MATS:
        for arm in E.ARMS:assert E.read(E.DOC/f'FIT_{mat}_{arm}.json')['complete']
    for mat in E.MATS:
        for arm in E.ARMS:
            fit=E.read(E.DOC/f'FIT_{mat}_{arm}.json');E.verify(fit['checkpoint'])
            path=E.RAW/f'EVAL_{mat}_{arm}.json'
            if path.exists():
                saved=E.read(path);assert saved['checkpoint']==fit['checkpoint'] and saved['GT_input'] is False
                assert set(saved['predictions'])=={r['id'] for r in records if r['object_type'].upper()==mat}
                files.append(E.bind(path));continue
            model=YOLO(str(E.ROOT/fit['checkpoint']['path']),task='pose');pp={}
            for r in records:
                if r['object_type'].upper()!=mat:continue
                E.verify(r['image']);pp[r['id']]=C.predict(model,cv2.imread(str(E.ROOT/r['image']['path'])),100)
            path=E.RAW/f'EVAL_{mat}_{arm}.json';E.save(path,dict(predictions=pp,checkpoint=fit['checkpoint'],GT_input=False));files.append(E.bind(path))
            del model;torch.cuda.empty_cache();print('EVAL_FROZEN',mat,arm,len(pp),flush=True)
    E.save(E.DOC/'EVAL_PREDICTIONS_LOCK.json',dict(files=files,before_scoring=True,students_alone=True,frames_per_arm=300))

def pose_row(args):
    fid,row,g=args
    current=D.metric(fid,row['current'],g);hh=[]
    for h in row['hypotheses']:hh.append(dict(**h,metric=D.metric(fid,h['pose'],g)))
    ok=[h for h in hh if h['metric']['available']]
    best=min(ok,key=lambda h:(h['metric']['ADDsym_normalized'],h['name'])) if ok else None
    return fid,dict(current=current,hypotheses=hh,oracle=best['metric'] if best else dict(available=False),oracle_name=best['name'] if best else None)

def score():
    E.immutable();lock=E.read(E.DOC/'EVAL_PREDICTIONS_LOCK.json')
    for b in lock['files']:E.verify(b)
    records=E.read(C.H.P.DOC/'SPLIT.json')['evaluation'];ids=[r['id'] for r in records]
    preds={a:{} for a in E.ARMS}
    for mat in E.MATS:
        for a in E.ARMS:preds[a].update(E.read(E.RAW/f'EVAL_{mat}_{a}.json')['predictions'])
    assert all(set(pp)==set(ids) for pp in preds.values()) and len(ids)==300
    meta={r['id']:r for r in E.read(C.H.V.RAW/'INFERENCE_METADATA.json')};poses={}
    for a,pp in preds.items():
        poses[a]={}
        for fid,p in pp.items():
            q=D.points(p);m=meta[fid];K=np.array(m['K']);xyz=np.array(m['xyz'])
            current=D.Pose.infer(q,K,xyz,False);sel=D.select(q,K,xyz)
            assert sel['selected_hypothesis']==current.get('selected_hypothesis')
            hh=[dict(name=h['name'],pose=D.hyp_pose(h,q,K,xyz),score=h['score']) for h in sel['hypotheses']]
            poses[a][fid]=dict(current=current,hypotheses=hh)
    E.save(E.RAW/'POSE_PREDICTIONS.json',poses);E.save(E.DOC/'POSE_PREDICTIONS_LOCK.json',dict(file=E.bind(E.RAW/'POSE_PREDICTIONS.json'),before_6D_reference=True,selector='unchanged production D9',oracle_not_used=True))
    truth=E.read(C.H.P.RAW/'TRUTH_FOR_DISPLAY_ONLY.json');_,gt=D.Pose.metadata('REAL_DEV')
    fm={a:{} for a in E.ARMS};pm={}
    for a,pp in preds.items():
        for fid,p in pp.items():
            t=truth[fid];c=C.H.P.C.selected(p);matched=c is not None and C.H.E.O.iou(c['box_xyxy'],t['box'])>=.5
            q=np.full((9,2),np.nan) if c is None else c['keypoints_xy']
            fm[a][fid]=dict(id=fid,**C.H.P.M.measure(q,t['gt'],t['valid'],t['permutations'],t['hw'],matched,c is not None))
        with ProcessPoolExecutor(max_workers=4) as pool:pm[a]=dict(pool.map(pose_row,[(fid,poses[a][fid],gt[fid]) for fid in ids],chunksize=8))
        print('SCORED',a,flush=True)
    groups={'ALL300':ids}
    groups.update({s:[r['id'] for r in records if r['severity']==s] for s in C.H.P.SEVERITIES})
    for mat in E.MATS:
        groups[mat]=[r['id'] for r in records if r['object_type'].upper()==mat]
        for s in C.H.P.SEVERITIES:groups[mat+'_'+s]=[r['id'] for r in records if r['object_type'].upper()==mat and r['severity']==s]
    for ss in sorted({r['session'] for r in records}):groups['SESSION_'+ss]=[r['id'] for r in records if r['session']==ss]
    r0=E.read(C.RAW/'FRAME_METRICS.json')['R0'];results={};tt={};common={}
    for group,ii in groups.items():
        if not ii:continue
        results[group]={};tt[group]={};common[group]={}
        commonids=[i for i in ii if all(fm[a][i]['matched'] for a in E.ARMS) and r0[i]['matched']]
        for a in E.ARMS:
            rows=[fm[a][i] for i in ii];s=C.H.P.M.summary(rows);s['correct']={str(t):sum(e<=t for r in rows for e in r['errors']) for t in (5,10,20)}
            current=D.aggregate([pm[a][i]['current'] for i in ii]);current['coverage']=current['available']/len(ii)
            oracle=D.aggregate([pm[a][i]['oracle'] for i in ii])
            results[group][a]=dict(twoD=s,current=current,oracle=oracle,selection_loss=oracle['ADDsym_AUC']-current['ADDsym_AUC'],vs_R0=transitions([r0[i] for i in ii],rows))
            common[group][a]=dict(twoD=C.H.P.M.summary([fm[a][i] for i in commonids]),current=D.aggregate([pm[a][i]['current'] for i in commonids]))
        tt[group]=transitions([fm['M0'][i] for i in ii],[fm['M1'][i] for i in ii])
    assert [len(groups[s]) for s in C.H.P.SEVERITIES]==[132,87,81]
    E.save(E.RAW/'FRAME_METRICS.json',fm);E.save(E.RAW/'POSE_METRICS.json',pm)
    E.save(E.DOC/'RESULTS.json',dict(groups=results,common_matched_supplement=common,oracle_label='POSTHOC GT ORACLE_WD — NONDEPLOYABLE'))
    E.save(E.DOC/'TRANSITIONS.json',tt)
    E.save(E.DOC/'ORACLE_CANDIDATE_AUDIT.json',dict(label='POSTHOC GT ORACLE_WD — NONDEPLOYABLE',rule='minimum ADD among existing two W/D hypotheses, name breaks ties',selector_unchanged=True,groups={g:{a:dict(candidate_quality=v['oracle']['ADDsym_AUC'],selection_loss=v['selection_loss']) for a,v in arms.items()} for g,arms in results.items()}))
    parity={}
    for mat in E.MATS:
        fits=[E.read(E.DOC/f'FIT_{mat}_{a}.json') for a in E.ARMS]
        assert fits[0]['initial_state']==fits[1]['initial_state'] and fits[0]['inventory']==fits[1]['inventory']
        traces=[E.read(E.RAW/f'TRACE_{mat}_{a}.json') for a in E.ARMS];assert traces[0]==traces[1]
        parity[mat]=dict(passed=True,occurrences=5120,steps=[f['steps'] for f in fits],same_initial_state=True,same_trainable_inventory=True,RGB_order_target_mask_exact=True)
    E.save(E.DOC/'TRAINING_PARITY.json',parity)

def diagnostics():
    E.deterministic();E.immutable();E.read(E.DOC/'EVAL_PREDICTIONS_LOCK.json')
    targets=E.read(C.DOC/'TARGETS.json');source={};train={};syn={};cross=[]
    occ={(r['material'],r['slot']):r for r in E.read(E.V.RAW/'SYNTH_OCCURRENCES.json') if r['epoch']==0}
    geometry={r['stem']:r for r in E.read(E.V.DOC/'SYNTH_GEOMETRY_BINDING.json')['records']}
    for mat in E.MATS:
        source[mat]={};train[mat]={};syn[mat]={}
        source[mat]['R0']=C.summary(E.read(C.RAW/'PROBE_BEFORE_PREDICTIONS.json')[mat]['R0']['source'])
        source[mat]['old_S1']=C.summary(E.read(C.RAW/f'DIAGNOSTICS_{mat}_S1.json')['source'])
        for arm in E.ARMS:
            path=E.ROOT/E.read(E.DOC/f'FIT_{mat}_{arm}.json')['checkpoint']['path'];model=YOLO(str(path),task='pose')
            sr=source_model(model);source[mat][arm]=C.summary(sr);tr=[]
            for r in targets:
                if r['material']!=mat:continue
                pred=C.predict(model,cv2.imread(str(E.ROOT/r['image']['path'])),100)
                tr.append(dict(id=r['id'],prediction=pred,manual=C.native(pred,r['manual'],r['manual_mask'],r['hw']),teacher=C.native(pred,r['target'],r['mask'],r['hw'])))
            train[mat][arm]={k:C.summary([r[k] for r in tr]) for k in ('manual','teacher')}
            ds=DiagDataset(mat);idx=[i for i,r in enumerate(ds.records[:1024]) if not r['real']][:16];probe=[];H=np.load(E.V.RAW/f'H_MODEL_{mat}.npz')['H']
            for i in idx:
                sample=ds[i];rgb=sample['img'].numpy().transpose(1,2,0);pred=C.predict(model,rgb[:,:,::-1].copy(),0);q=D.points(pred)
                z=sample['keypoints'][0].numpy();mask=z[:8,2]==2;w=sample['diag_weight'].numpy();weighted=full=None;pose=None
                if q is not None:
                    residual=((q[:8]-z[:8,:2]*640)*mask[:,None]).ravel();weighted=float((w*residual**2).sum()/max(1,2*mask.sum()));full=float(residual@H[i]@residual/max(1,2*mask.sum()))
                    o=occ[mat,i];g=geometry[o['stem']];native=np.c_[q[:8],np.ones(8)]@np.linalg.inv(o['M']).T;native=native[:,:2]/native[:,2:]
                    if mask.sum()>=6:
                        solved=D.Pose.solve(np.array(g['Xcf']),native,np.array(g['K']),mask)
                        if solved is not None:
                            R,t,_=solved;X=np.array(g['Xcf']);gtcam=X@np.array(g['R']).T+np.array(g['t']).reshape(1,3);camera=X@R.T+t.reshape(1,3)
                            pose=float(np.linalg.norm(camera-gtcam,axis=1).mean()/np.linalg.norm(g['dimensions']))
                probe.append(dict(slot=i,weighted_coordinate_error=weighted,full_quadratic=full,pose_camera_corner_error_normalized=pose,prediction=pred))
                cross.append(dict(material=mat,arm=arm,slot=i,full=full,diag=weighted))
            E.save(E.RAW/f'DIAGNOSTICS_{mat}_{arm}.json',dict(source=sr,train=tr,synthetic=probe))
            del model;torch.cuda.empty_cache();E.deterministic()
            m=YOLO(str(path),task='pose').model.float().cuda().train();m.args=get_cfg(overrides=dict(E.read(C.DOC/'PREFLIGHT.json')['args'],task='pose'));m.lambda_diag=0.;m.collect_diag=True
            b=collate([ds[i] for i in idx])
            for k,v in b.items():
                if isinstance(v,torch.Tensor):b[k]=v.cuda()
            b['img']=b['img'].float()/255;loss=criterion(m)
            with torch.no_grad():total,items=loss(m(b['img']),b);dg=diag_total(loss,len(idx))
            syn[mat][arm]=dict(slots=idx,base_total=float(total.sum()),base_components=items.cpu().tolist(),existing_keypoint_component=float(items[1]),L_diag_effective=float(dg),L_diag_branches=[float(loss.one2many.latest_diag),float(loss.one2one.latest_diag)],weighted_coordinate_error_mean=float(np.mean([p['weighted_coordinate_error'] for p in probe if p['weighted_coordinate_error'] is not None])),pose_mean=float(np.mean([p['pose_camera_corner_error_normalized'] for p in probe if p['pose_camera_corner_error_normalized'] is not None])),no_optimizer=True,TRAIN_NOT_GENERALIZATION=True)
            del m,b,loss;torch.cuda.empty_cache();print('DIAGNOSTICS',mat,arm,flush=True)
    E.save(E.DOC/'SYNTH_HELDOUT_RESULTS.json',dict(materials=source,frames=256,existing_exact_source=True,pose_status='See separate source_pose stage: SOURCE_GEOMETRY_BINDING.json and SYNTH_HELDOUT_POSE_RESULTS.json; no pose GT inferred'))
    E.save(E.DOC/'TRAIN_DIAGNOSTICS.json',dict(real19=train,synthetic_fixed_probe=syn,diag_reduced={m:syn[m]['M1']['L_diag_effective']<syn[m]['M0']['L_diag_effective'] for m in E.MATS}))
    E.save(E.DOC/'CROSS_TERM_AUDIT.json',dict(label='LOSS PROPERTY ONLY; no full-quadratic training',rows=cross,cancellation_full_less_than_half_diag=sum(r['full'] is not None and r['full']<.5*r['diag'] for r in cross)))

def source_pose():
    from .common import V
    from scripts.research.pallet_clean19_pose_sensitive_v1.prepare import raw_label,project,SIDETABLE
    table=dict(np.load(SIDETABLE));ix={str(s):i for i,s in enumerate(table['stems'])}
    _,gt=D.Pose.metadata('SYNTH_HELDOUT');plan=E.read(C.DOC/'SOURCE_PROBE_PLAN.json')['records'];binding=[];meta={}
    for r in plan:
        fid=r['id'];i=ix[fid];label=raw_label(fid);ann=E.read(label);T=np.array(ann['objects'][0]['pose_transform'])
        np.testing.assert_allclose(T[:3,:3],table['R'][i],atol=1e-10);np.testing.assert_allclose(T[:3,3],table['t'][i],atol=1e-10)
        h,w=r['hw'];fx,fy,cx,cy=table['K'][i];K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
        assert [h,w]==[ann['camera_data']['height']+2*table['pad'][i],ann['camera_data']['width']+2*table['pad'][i]]
        kp=np.array(r['targets'][0]['keypoints_normalized']);q=project(K,table['R'][i],table['t'][i],table['Xcf'][i]);mask=kp[:8,2]>0
        err=float(np.linalg.norm(q[mask]-kp[:8,:2][mask]*[w,h],axis=1).max());assert err<=.05,(fid,err)
        E.verify(r['image']);E.verify(r['label']);assert fid in gt
        binding.append(dict(id=fid,image=r['image'],label=r['label'],renderer=E.bind(label),projection_error_px=err));meta[fid]=(K,table['dims'][i])
    E.save(E.DOC/'SOURCE_GEOMETRY_BINDING.json',dict(bound=len(binding),records=binding,table=E.bind(SIDETABLE),K='prepared image: includes existing pad100',new_GT_estimation=False))
    output={}
    for mat in E.MATS:
        output[mat]={}
        for arm in ('R0','old_S1','M0','M1'):
            rr=E.read(C.RAW/'PROBE_BEFORE_PREDICTIONS.json')[mat]['R0']['source'] if arm=='R0' else E.read(C.RAW/f'DIAGNOSTICS_{mat}_S1.json')['source'] if arm=='old_S1' else E.read(E.RAW/f'DIAGNOSTICS_{mat}_{arm}.json')['source']
            poses={r['id']:D.Pose.infer(D.points(r['prediction']),*meta[r['id']],True) for r in rr}
            E.save(E.RAW/f'SOURCE_POSES_{mat}_{arm}.json',poses)
            with ProcessPoolExecutor(max_workers=4) as pool:metrics=list(pool.map(D.Pose.metric,[(fid,p,gt[fid]) for fid,p in poses.items()],chunksize=8))
            for m in metrics:
                if m['available']:m['axis_correct']=bool(abs(poses[m['id']]['cf_extents'][0]-gt[m['id']]['body_xyz'][0])<1e-6)
            output[mat][arm]=D.aggregate(metrics)
    E.save(E.DOC/'SYNTH_HELDOUT_POSE_RESULTS.json',dict(materials=output,frames=256,geometry=E.bind(E.DOC/'SOURCE_GEOMETRY_BINDING.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['infer','score','diagnostics','source_pose']);a=p.parse_args();globals()[a.stage]()

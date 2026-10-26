"""Saved-output SYNDEV235 classification/location and three-fit scope replay.

Default mode is blocked until all three final300 checkpoints and the training
COMPLETE are sealed. Replay consumes only compact numeric labels/predictions,
logs and small head-state checkpoints. No RGB, M4, current223, model forward,
optimizer, policy search or threshold change is performed.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import math
import shutil

import numpy as np

HERE=Path(__file__).absolute().parent
ARMS=('XY_ONLY','VIS_ONLY','JOINT')
SEMANTIC=('VIS_ONLY','JOINT','CONSTANT_FIT_PRIOR')
RESULT_NAMES=('CLASSIFICATION.csv','LOCATION_SUMMARY.csv','PAIRED_LOCATION.csv',
              'PAIRED_VISIBILITY.csv','NONDE_DISTRIBUTIONS.csv','TRAINING_AUDIT.json','RESULTS.json')


def safe(value):
    p=Path(value)
    if not p.is_absolute() or any(x in str(p).lower() for x in ('v112','cache','.pyc','..')):
        raise ValueError(str(p))
    for a in reversed((p,*p.parents)):
        if a.is_symlink():raise ValueError(str(a))
    return p


def binding(path):
    p=safe(path);h=hashlib.sha256()
    with p.open('rb')as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return {'path':str(p),'sha256':h.hexdigest(),'size_bytes':p.stat().st_size}


def checked(bound):
    assert binding(bound['path'])==bound,bound
    return safe(bound['path'])


def json_read(path):return json.loads(safe(path).read_text())


def savej(path,x):
    p=safe(path);assert not p.exists(),p
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n')


def load_npz(path):
    with np.load(safe(path),allow_pickle=False)as z:return {k:z[k] for k in z.files}


def copy_bound(bound,destination):
    source=checked(bound);destination=safe(destination);assert not destination.exists()
    shutil.copyfile(source,destination)
    b=binding(destination);assert b['sha256']==bound['sha256'] and b['size_bytes']==bound['size_bytes']
    return {'file':destination.name,'sha256':b['sha256'],'size_bytes':b['size_bytes'],'original_binding':bound}


def build_compact(destination):
    destination=safe(destination);assert not destination.exists(),destination
    # This is the first read of any trained output in this program.
    training_path=safe(HERE/'syn_training_v1/COMPLETE.json')
    training=json_read(training_path);assert training['completed'] and training['PASS']
    assert set(training['checkpoints'])==set(ARMS) and set(training['SYNDEV_predictions'])==set(ARMS)
    barrier_path=HERE/'syn_training_v1/ALL_CHECKPOINTS_FROZEN_BEFORE_DEV.json'
    barrier=json_read(barrier_path);assert barrier['PASS'] and barrier['steps_per_arm']==300
    assert barrier['checkpoints']==training['checkpoints']
    for bound in training['checkpoints'].values():checked(bound)
    protocol=json.loads(checked(training['protocol']).read_text())
    assert protocol['PASS'] and protocol['completed']
    assert protocol['final_decision_threshold']['E_logit_ge']==0 and protocol['final_decision_threshold']['E_logit_equal_zero']=='E'
    feature=json.loads(checked(training['source_feature_complete']).read_text())
    raw_complete=json_read(HERE/'raw_model_targets9_v2/COMPLETE.json')
    native_complete=json_read(HERE/'syn_joint_targets_v1/COMPLETE.json')
    frame_rows=[json.loads(x) for x in checked(native_complete['outputs']['FRAME_TARGETS1003.jsonl']).read_text().splitlines()]
    destination.mkdir()
    sources={
        'TRAINING_COMPLETE.json':binding(training_path),'ALL_CP_BEFORE_DEV.json':binding(barrier_path),
        'PROTOCOL_BEFORE_FIT.json':training['protocol'],'FEATURE_COMPLETE.json':training['source_feature_complete'],
        'TRAIN_INPUTS.npz':feature['outputs']['TRAIN_INPUTS.npz'],
        'RAW_MODEL_TARGETS9.npz':raw_complete['outputs']['RAW_MODEL_TARGETS9.npz'],
        'NATIVE_PROXY_TARGETS8.npz':native_complete['outputs']['TARGETS.npz'],
        'INITIAL_HEAD_STATE.pt':feature['outputs']['INITIAL_HEAD_STATE.pt'],
        'FIT_DRAWS300x8.npy':feature['outputs']['FIT_DRAWS300x8.npy'],
    }
    for arm in ARMS:
        sources[f'{arm}_FINAL300.pt']=training['checkpoints'][arm]
        sources[f'{arm}_TRAIN_LOG.json']=training['logs'][arm]
        sources[f'{arm}_SYNDEV235.npz']=training['SYNDEV_predictions'][arm]
    entries={name:copy_bound(bound,destination/name) for name,bound in sources.items()}
    # Metadata only; no RGB/mask paths or pixels are required for replay.
    metadata=[{k:r[k] for k in ('frame_index','id','split','pallet_type','material_variant')} for r in frame_rows]
    savej(destination/'FRAME_METADATA1003.json',metadata)
    b=binding(destination/'FRAME_METADATA1003.json');entries['FRAME_METADATA1003.json']={
        'file':'FRAME_METADATA1003.json','sha256':b['sha256'],'size_bytes':b['size_bytes']}
    savej(destination/'COMPACT_INPUTS.json',{'completed':True,'PASS':True,
        'schema':'V270_PORTABLE_SYNDEV_METRICS_AND_CP_SCOPE_INPUTS_V1','files':entries,
        'RGB_or_M4_or_current_inputs':0,'model_forward_required':False,
        'teacher_warning':'Virtual cuboid source external-ray proxy, not physical human native visibility gold.'})


def inputs_from(directory):
    directory=safe(directory);manifest=json_read(directory/'COMPACT_INPUTS.json')
    assert manifest['completed'] and manifest['PASS']
    paths={}
    for name,b in manifest['files'].items():
        assert Path(b['file']).name==b['file']
        p=safe(directory/b['file']);actual=binding(p)
        assert actual['sha256']==b['sha256'] and actual['size_bytes']==b['size_bytes']
        paths[name]=p
    return paths


def state_sha(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        a=v.detach().cpu().contiguous().numpy();h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def training_audit(paths):
    import torch
    torch.set_num_threads(1)
    training=json_read(paths['TRAINING_COMPLETE.json']);protocol=json_read(paths['PROTOCOL_BEFORE_FIT.json'])
    barrier=json_read(paths['ALL_CP_BEFORE_DEV.json']);features=json_read(paths['FEATURE_COMPLETE.json'])
    initial=torch.load(paths['INITIAL_HEAD_STATE.pt'],map_location='cpu',weights_only=True)
    initial_flat={f'{k}.{j}':v for k,d in initial.items() for j,v in d.items()};initial_sha=state_sha(initial_flat)
    logs={};cps={};frozen=set();bn=set();model_code=set();protocol_sha=set()
    for arm in ARMS:
        cp=torch.load(paths[f'{arm}_FINAL300.pt'],map_location='cpu',weights_only=True)
        history=json_read(paths[f'{arm}_TRAIN_LOG.json'])
        assert cp['arm']==arm and cp['steps']==300 and len(history)==300
        assert [x['step'] for x in history]==list(range(1,301))
        assert cp['initial_head_sha256']==initial_sha and cp['RLE_loss']==0
        assert cp['frozen_before_sha256']==cp['frozen_after_sha256']
        assert cp['BN_before_sha256']==cp['BN_after_sha256']
        assert cp['semantic_E_trained']==(arm!='XY_ONLY')
        assert cp['protocol']==training['protocol'] and cp['draws']==barrier['draws']
        frozen.add(cp['frozen_before_sha256']);bn.add(cp['BN_before_sha256'])
        model_code.add(cp['model_code']['sha256']);protocol_sha.add(cp['protocol']['sha256'])
        shared_changed=0;e_changed=0;bn_changed=0
        for k,v in initial['cv4'].items():
            actual=cp['head_state']['cv4'][k]
            assert actual.shape==v.shape and actual.dtype==v.dtype
            different=not torch.equal(actual,v)
            if any(token in k for token in ('running_mean','running_var','num_batches_tracked')):
                bn_changed+=int(different)
            else:shared_changed+=int(different)
        for k,v in initial['E_heads'].items():e_changed+=int(not torch.equal(cp['head_state']['E_heads'][k],v))
        assert shared_changed>0 and bn_changed==0
        assert (e_changed==0) if arm=='XY_ONLY' else (e_changed>0)
        max_error=0.0
        for item in history:
            assert all(math.isfinite(float(item[key])) for key in ('total','canvas','xy','E'))
            expected=item['canvas']+(0 if arm=='VIS_ONLY' else 12*item['xy'])+(0 if arm=='XY_ONLY' else item['E'])
            max_error=max(max_error,abs(item['total']-expected))
            assert abs(item['total']-expected)<2e-4
        logs[arm]={'steps':300,'first':history[0],'last':history[-1],
            'maximum_logged_total_vs_prescribed_component_sum_error':max_error}
        cps[arm]={'steps':300,'semantic_E_trained':cp['semantic_E_trained'],
            'trainable_parameters':cp['trainable_parameters'],'shared_state_tensors_changed':shared_changed,
            'E_state_tensors_changed':e_changed,'BN_state_tensors_changed':bn_changed,
            'initial_state_sha256':initial_sha,'frozen_state_before_after_equal':True,'BN_before_after_equal':True,
            'head_state_sha256':state_sha({f'{k}.{j}':v for k,d in cp['head_state'].items() for j,v in d.items()})}
    assert len(frozen)==len(bn)==len(model_code)==len(protocol_sha)==1
    assert training['total_optimizer_updates']==900 and training['current223_reads']==0
    assert barrier['checkpoints']==training['checkpoints']
    return {'completed':True,'PASS':True,'arms':cps,'logs':logs,'successful_fits':3,'updates_each':300,
        'total_optimizer_updates':900,'all_checkpoints_sealed_before_DEV_prediction':True,
        'initial_state_identical_across_arms':True,'baseline_frozen_state_identical_across_arms':True,
        'BN_running_buffers_identical_and_unchanged':True,'XY_ONLY_new_E_unchanged_and_unreported':True,
        'current223_fit_reads':0,'audit_optimizer_steps':0,'audit_model_forwards':0,
        'protocol_sha256':next(iter(protocol_sha)),'model_code_sha256':next(iter(model_code)),
        'assigned_frames':features['assigned_count'],'unassigned_ids':features['unassigned_ids'],
        'teacher_warning':'SYN native virtual-cuboid external-ray blockage proxy only.'}


def sigmoid(logit):
    return 1/(1+np.exp(-np.clip(np.asarray(logit,np.float64),-60,60)))


def div(a,b):return float(a/b) if b else None


def classify(y,logit,available):
    assert y.ndim==logit.ndim==available.ndim==1 and y.size==logit.size==available.size
    pred=np.where(available,np.where(logit>=0,1,0),2)
    cm=[[int(((y==a)&(pred==b)).sum()) for b in (0,1,2)] for a in (0,1)]
    d0,e0=cm;N=len(y);u=d0[2]+e0[2]
    dp=div(d0[0],d0[0]+e0[0]);dr=div(d0[0],sum(d0))
    ep=div(e0[1],e0[1]+d0[1]);er=div(e0[1],sum(e0))
    finite=np.isfinite(logit)&available
    p=sigmoid(logit[finite]);yy=y[finite].astype(np.float64)
    bce=float(-(yy*np.log(np.clip(p,1e-12,1))+(1-yy)*np.log(np.clip(1-p,1e-12,1))).mean()) if len(yy) else None
    return {'N':N,'D':int((y==0).sum()),'E':int((y==1).sum()),
        'D_to_D':d0[0],'D_to_E':d0[1],'D_to_U':d0[2],
        'E_to_D':e0[0],'E_to_E':e0[1],'E_to_U':e0[2],
        'D_precision':dp,'D_recall':dr,'D_F1':div(2*d0[0],2*d0[0]+d0[1]+d0[2]+e0[0]),
        'E_precision':ep,'E_recall':er,'E_F1':div(2*e0[1],2*e0[1]+e0[0]+e0[2]+d0[1]),
        'balanced_accuracy':float((dr+er)/2) if dr is not None and er is not None else None,
        'U':u,'U_rate':div(u,N),'accuracy':div(d0[0]+e0[1],N),
        'covered_BCE':bce,'covered_Brier':float(np.mean((p-yy)**2)) if len(yy) else None}


def error_summary(error):
    error=np.asarray(error,np.float64);assert np.isfinite(error).all()
    return {'N':len(error),'mean_px':float(error.mean()) if len(error) else None,
        'median_px':float(np.median(error)) if len(error) else None,
        'p90_px':float(np.percentile(error,90)) if len(error) else None,
        'LE3':int((error<=3).sum()),'LE6':int((error<=6).sum())}


def csv_write(path,rows):
    assert rows
    with safe(path).open('x')as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def analyze(compact,out):
    paths=inputs_from(compact)
    labels=load_npz(paths['RAW_MODEL_TARGETS9.npz']);native=load_npz(paths['NATIVE_PROXY_TARGETS8.npz'])
    trained=load_npz(paths['TRAIN_INPUTS.npz']);metadata=json_read(paths['FRAME_METADATA1003.json'])
    feature=json_read(paths['FEATURE_COMPLETE.json'])
    ix=np.flatnonzero(~labels['fit']);assert len(ix)==235 and [metadata[i]['split'] for i in ix]==['validation']*235
    assert np.array_equal(labels['E9'],trained['E9']) and np.array_equal(labels['Emask9'],trained['Emask9'])
    assert int((labels['E9'][ix]==0).sum())==733 and int((labels['E9'][ix]==1).sum())==192
    gt=labels['gt_xy9_raw'][ix].astype(np.float64);loc=labels['coordvalid9'][ix]
    state=labels['E9'][ix,:8];fraction=native['external_ray_fraction'][ix]
    sv=native['source_convex_self_visible'][ix]
    strict=(state>=0);self_inside=loc[:,:8]&~sv
    ambiguous=loc[:,:8]&sv&(fraction>0)&(fraction<.5)
    assert int(strict.sum())==925 and int(self_inside.sum())==294 and int(ambiguous.sum())==531
    pred={arm:load_npz(paths[f'{arm}_SYNDEV235.npz']) for arm in ARMS}
    affine=trained['raw_to_input_affine'][ix].astype(np.float64)
    rawxy={};logits={};available={};errors={}
    for arm,z in pred.items():
        assert np.array_equal(z['frame_index'],ix) and z['xy_px'].shape==(235,9,2)
        assert np.isfinite(z['xy_px']).all() and z['canvas_logit'].shape==(235,9)
        # Source affine has uniform axis scale and translation, no rotation/shear.
        assert (affine[:,0,1]==0).all() and (affine[:,1,0]==0).all()
        rawxy[arm]=(z['xy_px'].astype(np.float64)-affine[:,None,:,2])/np.stack([affine[:,0,0],affine[:,1,1]],-1)[:,None,:]
        errors[arm]=np.linalg.norm(rawxy[arm]-gt,axis=-1)
        if arm!='XY_ONLY':
            logits[arm]=z['E_logit'][:,:8].astype(np.float64);assert np.isfinite(logits[arm]).all()
            available[arm]=np.repeat((z['detection_confidence']>=.001)[:,None],8,axis=1)
        else:assert np.isnan(z['E_logit']).all()
    assert all(np.array_equal(pred['VIS_ONLY'][k],pred['JOINT'][k]) for k in ('detection_anchor','detection_confidence'))
    prevalence=feature['E_counts_FIT_assigned'][1]/sum(feature['E_counts_FIT_assigned'])
    logits['CONSTANT_FIT_PRIOR']=np.full((235,8),math.log(prevalence/(1-prevalence)),np.float64)
    available['CONSTANT_FIT_PRIOR']=np.ones((235,8),bool)
    groups=[('ALL',np.ones(235,bool))]
    for material in sorted({metadata[i]['material_variant'] for i in ix}):
        groups.append(('MATERIAL:'+material,np.asarray([metadata[i]['material_variant']==material for i in ix])))
    for pallet in sorted({metadata[i]['pallet_type'] for i in ix}):
        groups.append(('PALLET_TYPE:'+pallet,np.asarray([metadata[i]['pallet_type']==pallet for i in ix])))
    crows=[];prows=[];nonde=[];lrows=[];pairedloc=[]
    for group,gmask in groups:
        pointmask=np.repeat(gmask[:,None],8,1)
        for cohort,mask in [('STRICT_D_E',strict&pointmask)]+[
            ('STRICT_FRACTION:'+f'{value:.4f}',strict&pointmask&np.isclose(fraction,value,rtol=0,atol=1e-7))
            for value in sorted(np.unique(fraction[strict&pointmask]).tolist())]:
            if not mask.any():continue
            for arm in SEMANTIC:
                for policy in ('ALL_SAVED_DETECTION_ANCHORS','FIXED_DETECTION_GE_0.001'):
                    av=np.ones_like(mask) if policy=='ALL_SAVED_DETECTION_ANCHORS' else available[arm]
                    crows.append({'group':group,'cohort':cohort,'method':arm,'availability_policy':policy,
                        **classify(state[mask],logits[arm][mask],av[mask])})
            a=(logits['VIS_ONLY']>=0);b=(logits['JOINT']>=0)
            prows.append({'group':group,'cohort':cohort,'N':int(mask.sum()),
                'D_VISwrong_JOINTcorrect':int((mask&(state==0)&a&~b).sum()),
                'D_VIScorrect_JOINTwrong':int((mask&(state==0)&~a&b).sum()),
                'E_VISwrong_JOINTcorrect':int((mask&(state==1)&~a&b).sum()),
                'E_VIScorrect_JOINTwrong':int((mask&(state==1)&a&~b).sum()),
                'both_correct':int((mask&(a==state)&(b==state)).sum()),
                'both_wrong':int((mask&(a!=state)&(b!=state)).sum())})
        for cohort,mask in [('SELF_INSIDE_NO_D_E_GOLD',self_inside&pointmask),('PARTIAL_RAY_UNKNOWN',ambiguous&pointmask),
                            ('OUTSIDE_NATIVE',~loc[:,:8]&pointmask)]:
            for arm in ('VIS_ONLY','JOINT'):
                av=available[arm];p=logits[arm]>=0
                nonde.append({'group':group,'cohort':cohort,'method':arm,'N':int(mask.sum()),
                    'pred_D':int((mask&av&~p).sum()),'pred_E':int((mask&av&p).sum()),'pred_U':int((mask&~av).sum()),
                    'mean_P_E':float(sigmoid(logits[arm][mask]).mean()) if mask.any() else None,
                    'not_confusion_matrix':'No native external D/E gold for this source state.'})
        native9mask=np.repeat(gmask[:,None],9,1)
        cohorts=[('ALL_NATIVE_INFRAME',loc&native9mask&np.asarray([1]*8+[0],bool)),
            ('CENTROID_INFRAME',loc&native9mask&np.asarray([0]*8+[1],bool))]
        for name,smask in [('D_CLEAR',state==0),('E_BLOCKED',state==1),('SELF_INSIDE',self_inside),('PARTIAL_UNKNOWN',ambiguous)]:
            cohorts.append((name,np.concatenate([smask,np.zeros((235,1),bool)],1)&native9mask))
        for cohort,mask in cohorts:
            if not mask.any():continue
            for arm in ARMS:lrows.append({'group':group,'cohort':cohort,'method':arm,**error_summary(errors[arm][mask])})
            before=errors['VIS_ONLY'][mask];after=errors['JOINT'][mask];delta=after-before
            displacement=np.linalg.norm(rawxy['JOINT']-rawxy['VIS_ONLY'],axis=-1)[mask]
            damage=(before<=3)&(after>3)
            pairedloc.append({'group':group,'cohort':cohort,'N':int(mask.sum()),
                'JOINT_smaller_error':int((delta<-1e-6).sum()),'JOINT_larger_error':int((delta>1e-6).sum()),
                'equal_within_1e_6':int((abs(delta)<=1e-6).sum()),
                'median_JOINT_minus_VIS_error_px':float(np.median(delta)),
                'VIS_LE3_to_JOINT_GT3':int(damage.sum()),
                'damage_final_3_to_6':int((damage&(after<=6)).sum()),
                'damage_final_6_to_10':int((damage&(after>6)&(after<=10)).sum()),
                'damage_final_GT10':int((damage&(after>10)).sum()),
                'VIS_GT3_to_JOINT_LE3':int(((before>3)&(after<=3)).sum()),
                'median_intermodel_coordinate_shift_px':float(np.median(displacement)),
                'p90_intermodel_coordinate_shift_px':float(np.percentile(displacement,90))})
    audit=training_audit(paths)
    out=safe(out);assert not out.exists(),out;out.mkdir()
    for name,rows in [('CLASSIFICATION.csv',crows),('LOCATION_SUMMARY.csv',lrows),('PAIRED_LOCATION.csv',pairedloc),
                      ('PAIRED_VISIBILITY.csv',prows),('NONDE_DISTRIBUTIONS.csv',nonde)]:csv_write(out/name,rows)
    savej(out/'TRAINING_AUDIT.json',audit)
    primary=[r for r in crows if r['group']=='ALL' and r['cohort']=='STRICT_D_E']
    result={'completed':True,'PASS':True,'schema':'V270_SAVED_SYNDEV235_PROXY_LEARNABILITY_V1',
        'frames':235,'native_points':1880,'strict_D':733,'strict_E':192,
        'self_inside_unscored':294,'partial_unscored':531,'outside_native_unscored':130,
        'primary_classification':primary,
        'all_location_rows':[r for r in lrows if r['group']=='ALL'],
        'paired_visibility':[r for r in prows if r['group']=='ALL' and r['cohort']=='STRICT_D_E'],
        'paired_location':[r for r in pairedloc if r['group']=='ALL'],
        'classification_rows':len(crows),'location_rows':len(lrows),'material_groups':len([g for g,_ in groups if g.startswith('MATERIAL:')]),
        'E_threshold':'E_logit>=0 / P_E>=.5 means E; otherwise D. No fitting/tuning in analysis.',
        'location_scope':'Secondary raw-pixel GT error/drift of saved model-chosen detector-anchor outputs. No R0 raw DEV coordinates were saved by extractor, so do not claim R0-relative location gain. XY_ONLY is the position/control comparison.',
        'classification_scope':'Source virtual-cuboid external9ray proxy labels only; this does not certify human native D/E or independent real transfer.',
        'development_exposure':'Existing repeatedly used SYNDEV235, not independent blind validation.',
        'model_forward_count':0,'optimizer_step_count':0,'RGB_M4_reads':0,'current223_reads':0}
    savej(out/'RESULTS.json',result)
    savej(out/'COMPLETE.json',{'completed':True,'PASS':True,'code':binding(Path(__file__).absolute()),
        'compact_inputs':binding(Path(compact)/'COMPACT_INPUTS.json'),'outputs':{n:binding(out/n) for n in RESULT_NAMES},
        'model_forwards':0,'optimizer_steps':0,'current223_reads':0})
    print(json.dumps({'COMPLETE':binding(out/'COMPLETE.json'),'primary':primary,
        'paired':result['paired_visibility'],'training_audit_PASS':True},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='syndev_analysis_v1')
    parser.add_argument('--replay',default=None)
    args=parser.parse_args();output=safe(HERE/args.out)
    if args.replay is None:
        compact=safe(HERE/'syndev_compact_inputs_v1');build_compact(compact)
    else:compact=safe(Path(args.replay) if Path(args.replay).is_absolute() else HERE/args.replay)
    analyze(compact,output)

"""Tiny invented-only V261 contract checks; no SOURCE or official asset access."""
import copy, json, sys, types
from pathlib import Path
import numpy as np
import torch

H=Path(__file__).absolute().parent

def module(p,name):
    m=types.ModuleType(name);m.__file__=str(p);sys.modules[name]=m
    exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__);return m

def run():
    t=module(H/'train_final_public_aux.py','v261_invented_trainer')
    helper=t.load_code(t.UTILITY,[t.UTILITY],'v261_invented_old_functions')
    crop=t.load_code(t.CROP,[t.CROP],'v261_invented_crop')
    checks=[]
    # Old8 rows=two each family/class; public four extra AUX rows, plus masked/DEV.
    s={'target':np.array([0,0,1,1,0,0,1,1,0,0,1,1,-1,0],np.int8),
       'family':np.array([0,0,0,0,1,1,1,1,1,1,1,1,1,1],np.int8),
       'fit':np.array([True]*13+[False]),'available':np.array([True]*12+[False,True]),
       'row_id':np.array([f'invented{i}' for i in range(14)])}
    ec=np.array([True]*8+[False]*6);et=np.array([True]*12+[False]*2)
    dc,dt,u,pc,pt=t.common_uniform_draws(s,ec,et)
    assert dc.shape==dt.shape==(1000,32) and u.shape==(4,1000,8)
    assert dc[:,:16].tobytes()==dt[:,:16].tobytes() and (dt[:,16:]>=8).any()
    assert not np.isin(dt,[12,13]).any()
    for draws in (dc,dt):
        for f in (0,1):
            for y in (0,1):assert np.all(np.sum((s['family'][draws]==f)&(s['target'][draws]==y),1)==8)
    for i,(f,y) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
        expected=np.random.Generator(np.random.PCG64(np.random.SeedSequence([17,f,y]))).random((1000,8))
        assert np.array_equal(u[i],expected)
        assert np.array_equal(dc[:,i*8:(i+1)*8],pc[i][np.floor(expected*len(pc[i])).astype(np.int64)])
        assert np.array_equal(dt[:,i*8:(i+1)*8],pt[i][np.floor(expected*len(pt[i])).astype(np.int64)])
    checks.append('Same four U streams/floor remap; SYN exact, AUX expanded,32=8D8E/family, no U/DEV draws')
    torch.manual_seed(17);head=t.build_model(crop);initial=copy.deepcopy(head.state_dict())
    assert helper.state_sha(initial)==t.INITIAL_SHA and sum(p.numel()for p in head.parameters())==76098
    x=torch.from_numpy(np.random.default_rng(17).normal(size=(32,1186)).astype(np.float32))
    assert not head(x).detach().numpy().any()
    y=torch.tensor([0]*8+[1]*8+[0]*8+[1]*8);f=torch.tensor([0]*16+[1]*16)
    loss,_=crop.family_loss(head(x),y,f);loss.backward();assert torch.isfinite(loss)
    checks.append('Exact historical seed17 initial SHA76098/finalzero/two family-loss outputs; one invented backward, optimizer0')
    # Empty first frame, all284 frame roles fixed and coarse-group consistent.
    frames=[];records=[];target=[];fit=[];ids=[];frame_ids=[];group_ids=[]
    for i in range(284):
        frame=f'public_frame_{i:04d}';group=f'group_{i:04d}';role=i<168
        counts={'D':0 if i==0 else 1,'E':0 if i==0 else 1,'U':4800 if i==0 else 4798}
        frames.append({'publisher_frame_path':frame,'coarse_group_id':group,'fit':role,'label_counts':counts,'selected_counts':{k:counts[k]for k in ('D','E')}})
        if i==0:continue
        for label in (0,1):
            j=len(records);rowid=f'{frame}_r{label}'
            records.append({'sample_index':j,'row_id':rowid,'publisher_frame_path':frame,'coarse_group_id':group,'reference_grid_index':label,'q_xy':[4+8*label,4],'r_xy':[4+8*label,4],'q_offset_index':0,'L_pixels':96,'native_slot':None,'SELF_state':'UNKNOWN'})
            target.append(label);fit.append(role);ids.append(rowid);frame_ids.append(frame);group_ids.append(group)
    samples={'target':np.array(target,np.int8),'family':np.ones(len(target),np.int8),'fit':np.array(fit,bool),'available':np.ones(len(target),bool),'row_id':np.array(ids),'frame_id':np.array(frame_ids),'coarse_group_id':np.array(group_ids)}
    _,_,census=t.validate_public_manifest(samples,b'\n'.join(json.dumps(r).encode()for r in records),json.dumps({'frames':frames}).encode())
    assert census['frames']==284 and census['selected_rows']==566 and census['teacher_U_excluded_from_sampling_and_loss']==4800+283*4798
    checks.append('All284 frames including empty/teacherU retained;168TRAIN116DEV, exact row/role/q=r/L96/nativeNone joins')
    wrong=copy.deepcopy(records);wrong[0]['q_xy']=[5,4]
    try:t.validate_public_manifest(samples,b'\n'.join(json.dumps(r).encode()for r in wrong),json.dumps({'frames':frames}).encode())
    except ValueError:pass
    else:raise AssertionError('Displaced query label accepted')
    checks.append('Displaced q rejected without coordinate repair or label copying')
    import tempfile
    with tempfile.TemporaryDirectory(prefix='v261_invented_')as d:
        e=Path(d)/'false.json';e.write_text(json.dumps({'approved_actual_final_public_aux_fit':False}))
        try:t.run(e)
        except ValueError:pass
        else:raise AssertionError('False entry accepted')
    checks.append('False entry rejected before data/model/output access')
    return {'completed':True,'PASS':True,'invented_checks':checks,'actual_SOURCE_or_public_features_or_labels_or_RGB_or_CP_reads':0,'encoder_forwards':0,'optimizer_updates':0,'invented_backward_calls':1}

if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False,sort_keys=True,indent=2))

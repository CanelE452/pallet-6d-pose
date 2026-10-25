"""독립 invented tensor/row 검산. 실제 checkpoint/features/샘플을 열지 않음."""
import ast,hashlib,json,os,stat,tempfile,types
from pathlib import Path
from collections import Counter
import numpy as np
import torch

R=Path(__file__).absolute().parent.parent
SOURCE=R/'V244_OWN_BRANCH_DIAGNOSTIC_A/diagnose_own_branches.py'
HEAD=R/'V237_RELATIONAL_QUERY_DESIGN_B/relational_query_head.py'

def code(p,sha):
    assert not any(x in str(p).casefold() for x in ('v112','__pycache__','.pyc'))
    for q in reversed(p.parents):assert stat.S_ISDIR(os.lstat(q).st_mode)
    with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:b=f.read()
    assert hashlib.sha256(b).hexdigest()==sha
    return b

def reject(f):
    try:f()
    except (ValueError,PermissionError):return
    raise AssertionError('negative guard failed')

def run():
    src=code(SOURCE,'96202dc4f3a1e18ee3d5bb3f649e4243859820c5651d80346745022ed204b358')
    m=types.ModuleType('v244_C_CODE_only');m.__file__=str(SOURCE);exec(compile(src,str(SOURCE),'exec'),m.__dict__)
    rawhead=code(HEAD,'dfd26ab7ccfa7037f0c3dca548d59c8574aa8b78c50699f70f01e03bf48935a9')
    hm=types.ModuleType('v244_C_invented_head');exec(compile(rawhead,str(HEAD),'exec'),hm.__dict__)
    torch.set_num_threads(1);torch.manual_seed(913)
    head=hm.RelationalQueryHead().eval().requires_grad_(False)
    with torch.no_grad():
        head.base.head[-1].weight.fill_(.002);head.relational.readout[-1].weight.fill_(-.003)
    n=5;x=torch.randn(n,1186);node=torch.randn(n,8,384);context=torch.randn(n,384)
    slot=torch.arange(n,dtype=torch.int64);avail=torch.ones(n,8,dtype=torch.bool);support=avail.clone();support[:,6]=False
    args=(node,context,slot,avail,support,torch.ones(n,dtype=torch.bool),torch.zeros(n,8,2))
    with torch.no_grad():
        b,r,t,_=m.own_branches(head,x,args);full,_=head(x,*args)
    assert torch.equal(t,b+r) and torch.equal(t,full) and sum(p.numel() for p in head.parameters())==86054
    assert not any(p.requires_grad for p in head.parameters())
    # One row per family × role × target. FIT recovers, DEV is damaged.
    combos=[(f,fit,y) for f in (0,1) for fit in (True,False) for y in (0,1)]
    family=np.array([f for f,_,_ in combos],np.int8);target=np.array([y for _,_,y in combos],np.int8)
    fit=np.array([a for _,a,_ in combos]);base=np.full((8,2),100.,np.float32);delta=np.zeros((8,2),np.float32)
    for i,(f,isfit,y) in enumerate(combos):
        sign=2*y-1;base[i,f]=np.float32((-sign) if isfit else sign);delta[i,f]=np.float32((2*sign) if isfit else (-2*sign))
    total=base+delta;data={'family':family,'target':target,'fit':fit,'available':np.ones(8,bool),'row_id':np.array(['invented:'+str(i) for i in range(8)])}
    saved=[{'selected_family_logit':float(total[i,f]),'prediction':'E' if total[i,f]>=0 else 'D'} for i,(f,_,_) in enumerate(combos)]
    rows=m.rows_from_branches(data,saved,base,delta,total);metrics,strata=m.summarize(rows)
    for i,(f,isfit,y) in enumerate(combos):
        row=rows[i];truth='E' if y else 'D';z=float(total[i,f]);bp='E' if base[i,f]>=0 else 'D';tp='E' if z>=0 else 'D'
        assert row['selected_own_total_logit']==z and row['own_total_prediction']==tp
        assert row['target_aligned_residual']==(2*y-1)*float(delta[i,f])
        assert row['target_effect']==('RECOVERED' if isfit else 'DAMAGED')
        assert row['base_correct']==(bp==truth) and row['total_correct']==(tp==truth)
    for branch,field in [('own_base','own_base_prediction'),('own_residual_sign_only','own_residual_sign_prediction'),('own_total','own_total_prediction')]:
        for f in (0,1):
            for role in ('FIT','DEV'):
                cc=Counter((r['target'],r[field]) for r in rows if r['family']==f and r['role']==role)
                expected=[[cc[y,p] for p in ('D','E','U')] for y in (0,1)]
                assert metrics[branch][f'family{f}_{role}']['CM_D_E_by_D_E_U']==expected
    assert len(strata)==8 and sum(s['rows'] for s in strata)==8
    assert all(sum(s['crossings'].values())==s['rows'] and sum(s['effects'].values())==s['rows'] for s in strata)
    assert all(s['target_aligned_residual']['positive']==1 if s['role']=='FIT' else s['target_aligned_residual']['negative']==1 for s in strata)
    # Different output column is not an alternative class; never argmax.
    assert any(np.argmax(total[i])!=int(total[i,f]>=0) for i,(f,_,_) in enumerate(combos))
    boundary=np.array([[0.,0.]],np.float32);dd={'family':np.array([1],np.int8),'target':np.array([1],np.int8),'fit':np.ones(1,bool),'available':np.ones(1,bool),'row_id':np.array(['tie'])}
    rr=m.rows_from_branches(dd,[{'selected_family_logit':0.,'prediction':'E'}],boundary,boundary,boundary)
    assert rr[0]['own_total_prediction']=='E' and rr[0]['target_aligned_residual']==0
    changed=total.copy();changed[0,0]+=.5;reject(lambda:m.rows_from_branches(data,saved,base,delta,changed))
    assert m.ATOL==m.RTOL==1e-5 and np.allclose(total,total+np.float32(1e-6),atol=m.ATOL,rtol=m.RTOL)
    assert not np.allclose(total,total+.1,atol=m.ATOL,rtol=m.RTOL)
    with tempfile.TemporaryDirectory(prefix='v244_C_CODE_only_') as td:
        p=Path(td)/'entry.json';p.write_text('{"approved_frozen_SOURCE_own_branch_diagnosis":false}')
        reject(lambda:m.run(p))
    reject(lambda:m.safe('/absent/V112/no-read'));reject(lambda:m.safe('/absent/__pycache__/no-read.pyc'))
    functions={a.name:{'line':a.lineno,'end_line':a.end_lineno} for a in ast.parse(src).body if isinstance(a,ast.FunctionDef)}
    return {'completed':True,'PASS':True,'groups':4,'checks':['same frozen invented head own base+residual byteequals original fullforward; CONTROL absent','independent family/role/target CM and signed residual/crossing/recovery/damage recount','binary selected-family threshold0 including exactzero; additive drift rejected; inherited atol=rtol1e-5','false entry before payload and forbidden lexical rejection'],'functions':functions,'actual_trained_checkpoint_feature_samples_RGB_GT_or_current_held_reads':0,'invented_random_head_forward_only':True,'optimizer_updates':0}

if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False))

"""Independent small Boolean scalar vs production FFT/alpha fixtures; no source payloads."""
from pathlib import Path
import importlib.util,hashlib,json,os,stat
import numpy as np
H=Path(__file__).absolute().parent;B=H.parent/'raw96_control_preparation_b';path=B/'raw96_control_geometry.py'
assert 'v112'not in str(path).casefold()
for p in (path,*path.parents):assert not stat.S_ISLNK(os.lstat(p).st_mode)
assert hashlib.sha256(path.read_bytes()).hexdigest()=='b0927c898b181eb2bdc298f0a1bd69a1d968a9524674bdeda4c522e746e80703'
spec=importlib.util.spec_from_file_location('frozen_pure_geometry',path);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
checks=[]
def add(n,x):assert x,n;checks.append({'name':n,'PASS':True})
def rejected(fn):
 try:fn()
 except AssertionError:return True
 return False
rng=np.random.default_rng(207)
for index,(Hh,W,h,w)in enumerate([(7,8,3,4),(8,9,4,2),(6,6,3,3),(4,7,1,5)]):
 valid=rng.random((Hh,W))>.2;feature=np.zeros((Hh,W),bool);feature[Hh//2,W//2]=True;feature[1,1]=True
 alpha=rng.random((h,w))>.35
 if index==2:alpha[:]=True;alpha[1,1]=False
 cv,sh=g.feasible_positions(alpha,valid,feature);oracle_c=[];oracle_s=[];corr=np.empty((Hh-h+1,W-w+1),np.int64)
 for y in range(Hh-h+1):
  for x in range(W-w+1):
   taps={(x+xx,y+yy)for yy in range(h)for xx in range(w)if alpha[yy,xx]}
   corr[y,x]=sum(bool(valid[yy,xx])for xx,yy in taps)
   if taps and all(valid[yy,xx]for xx,yy in taps):
    f={(xx,yy)for yy in range(Hh)for xx in range(W)if feature[yy,xx]}
    if f<=taps:oracle_c.append((x,y))
    if not f&taps:oracle_s.append((x,y))
 add('independent_scalar_FFT_and_COVER_SHAM_positions_'+str(index),np.array_equal(g.correlation_counts(valid,alpha),corr)and cv==oracle_c and sh==oracle_s)
case={'original_query_available':True,'state_A':'YES','state_C':'YES','ROI_A':[10,10,11,11],'ROI_C':[12,11,13,12]}
f,status=g.obligation(case);bbox=g.rectangle([10,10,13,12]);add('union_two_ROIs_not_enclosing_bbox',int(f.sum())==8 and int(bbox.sum())==12 and np.all(f<=bbox))
before=(np.arange(96*96*3).reshape(96,96,3)%231).astype(np.uint8);texture=np.full((4,5,3),253,np.uint8);alpha=np.ones((4,5),bool);valid=np.ones((96,96),bool)
cover,ca=g.paste(before,texture,alpha,(9,9));sham,sa=g.paste(before,texture,alpha,(50,50));proof=g.verify_actual(before,cover,sham,ca,sa,valid,case)
add('same_transform_mass_outside_exact_and_feature_SHAM_preservation',proof['actual_change_PASS']and proof['pasted_alpha_pixels']==20 and np.array_equal(cover[ca],sham[sa])and np.array_equal(before,((np.arange(96*96*3).reshape(96,96,3)%231).astype(np.uint8))))
zero=g.verify_actual(before,before.copy(),sham,ca,sa,valid,case);add('actual_RGB_change_failure_is_false_not_forced_success',not zero['actual_change_PASS'])
bad=sham.copy();bad[10,10]^=np.uint8(1);add('SHAM_changed_feature_rejected',rejected(lambda:g.verify_actual(before,cover,bad,ca,sa,valid,case)))
alpha_hole=np.ones((3,3),bool);alpha_hole[1,1]=False;feature=np.ones((3,3),bool);cv,sh=g.feasible_positions(alpha_hole,np.ones((3,3),bool),feature)
add('alpha_hole_not_filled_and_no_legal_pair_preserved',cv==[]and sh==[])
valid_pad=np.ones((96,96),bool);valid_pad[10,10]=False;cv,sh=g.feasible_positions(np.ones((4,5),bool),valid_pad,f);add('feature_padding_cannot_be_force_COVERed',cv==[])
expected={'scope':{'path':'/invented/scope.json','sha256':'a'*64}}
entry={'approved':True,'actual_payload_access_authorized':True,'no_new_E_or_training_admission':True,'attempt_count':80,**expected}
add('actual_approval_gate_accepts_exact_contract',g.validate_gate(entry,expected))
add('actual_gate_false_and_binding_drift_refused',rejected(lambda:g.validate_gate(dict(entry,approved=False),expected))and rejected(lambda:g.validate_gate(dict(entry,scope={'path':'/invented/other.json','sha256':'b'*64}),expected)))
result={'completed':True,'PASS':True,'checks':checks,'checks_count':len(checks),'invented_only':True,'actual_source_PNG_NPZ_GT_stat_hash_header_decode':0,'operational_model_or_fit_calls':0}
with (H/'INVENTED_CRITICAL_CHECKS.json').open('x',encoding='utf-8')as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'PASS':True,'checks':len(checks)}))

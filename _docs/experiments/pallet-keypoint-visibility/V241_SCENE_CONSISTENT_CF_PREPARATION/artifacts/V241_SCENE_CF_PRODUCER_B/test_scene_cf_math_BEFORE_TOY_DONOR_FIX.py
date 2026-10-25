from pathlib import Path
import ast,hashlib,json,types
import numpy as np
H=Path(__file__).parent;R=H.parent;P=R.parent
ns={'__name__':'toy_scene_math'};exec(compile((H/'scene_cf_math.py').read_bytes(),'<new pure scene math>','exec'),ns)
def module(p,sha):
 z=p.read_bytes();assert hashlib.sha256(z).hexdigest()==sha;m=types.SimpleNamespace();d={'__name__':'toy_pinned_pure'};exec(compile(z,str(p),'exec'),d);m.__dict__.update(d);return m
# Only pure crop AST is executed from V90; no model or full historical run.
p=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py';z=p.read_bytes();assert hashlib.sha256(z).hexdigest()=='6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b'
a=[x for x in ast.parse(z).body if isinstance(x,ast.FunctionDef) and x.name=='crop'];d={'np':np};exec(compile(ast.Module(body=a,type_ignores=[]),'<V90 crop only>','exec'),d);crop=d['crop']
context=module(P/'query_visibility_target_context_rgb_20261023_v107/target_context_math.py','aa7cd8e2568c8b7d30827e8681b4b87899944bd23cfa1b27948ceda5b55d07f3')
scene=module(R/'V237_CONTEXT_COMPOSITION_A/context_scene_math.py','f48545c907cf0705a1cd9ee408e54f520f7eef4563fb166f20a1821b3fa9b30f')
old=module(R/'V235_CF_COMPOSITOR_A/compositor.py','d940ff8cb901482a77761dfac638ab473a5841d682ef0150f1ece0ab8c5fe4f4')
y,x=np.mgrid[:256,:256];base=np.stack((x%256,y%256,(x+y)%256),2).astype(np.uint8)
q8=np.array([[128,128],[160,128],[160,160],[128,160],[100,100],[120,100],[100,120],[120,120]],np.float64)
group={'q8':q8,'L8':np.full(8,96.,np.float64),'original_available8':np.array([1,1,1,1,1,1,1,0],bool),'box_xyxy':[100.,100.,160.,160.]}
before=ns['resample_all'](base,group,crop,context);drgb=np.zeros((32,28,3),np.uint8);drgb[:]=[12,221,31];mask=np.ones((32,28),bool);mask[:2,:2]=False
q=q8[0];ref=np.array([128.25,128.25],np.float64)
proposal=old.compose_counterfactual_pair(before['NODE_RGB96'][0],before['NODE_VALID96'][0],drgb,mask,q,np.float64(96),ref,'V241_INVENTED_0');assert proposal['success']
parent={'active_slot':0,'original_input':{'q':q.tolist(),'L':96.,'row_id':'V241_INVENTED_0','fit':True,'target':0,'original_query_available':True,'crop_available':True},'same_V235_donor_id':'TOY_FIT_DONOR','same_V235_donor_plan':ns['plain'](proposal['donor_plan']),'expected_BEFORE_RGB96_CHW_sha256':ns['sha'](before['NODE_RGB96'][0]),'expected_BEFORE_valid96_bool_sha256':ns['sha'](before['NODE_VALID96'][0]),'teacher_reference_xy':ref.tolist(),'teacher_authority':'INVENTED_ASSOCIATED_REFERENCE_NOT_GOLD'}
result=ns['compose_scene_pair'](base,group,parent,drgb,mask,'TOY_FIT_DONOR',crop,context,old,scene)
assert result['constructed'] and result['proof']['PASS'] and result['proposed_reference_targets']=={'BEFORE':0,'COVER':1,'ADJACENT_SHAM':0} and not result['training_admission'] and not result['new_native_gold']
assert not result['old5367_proof_inherited'] and np.array_equal(base,np.stack((x%256,y%256,(x+y)%256),2).astype(np.uint8))
for role in ('COVER','ADJACENT_SHAM'):
 native=result['native_scenes'][role];alpha=result['native_alpha'][role];packet=result['inputs'][role]
 assert np.array_equal(native[~alpha],base[~alpha])
 for j in range(8):
  rgb,v=crop(native,q8[j],np.float64(96),96);assert np.array_equal(rgb,packet['NODE_RGB96'][j]) and np.array_equal(v[0],packet['NODE_VALID96'][j])
 cc,vv=context.sample_target_context(native,group['box_xyxy']);assert np.array_equal(cc.transpose(2,0,1),packet['CONTEXT_RGB224_F32']) and np.array_equal(vv,packet['CONTEXT_VALID224'])
 assert not packet['original_available8'][7] and not packet['node_available'][7]
assert not np.array_equal(result['inputs']['COVER']['NODE_RGB96'][0],proposal['COVER']) # Native resampling is genuinely renewed.
# Unsupported original context is zero/flag, not an extra local-pair label gate.
g2=dict(group,box_xyxy=None);r2=ns['compose_scene_pair'](base,g2,parent,drgb,mask,'TOY_FIT_DONOR',crop,context,old,scene)
assert r2['proof']['PASS'] and r2['proposed_reference_targets']==result['proposed_reference_targets'] and not r2['inputs']['COVER']['context_supported'] and not r2['inputs']['COVER']['CONTEXT_RGB224_F32'].any()
# Missing/insufficient reference never becomes E; originalD preserved.
p2=dict(parent,teacher_reference_xy=None);r3=ns['compose_scene_pair'](base,group,p2,drgb,mask,'TOY_FIT_DONOR',crop,context,old,scene)
assert r3['proposed_reference_targets']['BEFORE']==0 and r3['proposed_reference_targets']['COVER'] is None and r3['status']=='UNKNOWN_REFERENCE'
a=np.ones((8,8),bool);a[3,3]=False;zero=np.zeros((8,8),bool);bb=np.zeros((8,8,3),np.uint8)
proof=ns['renewed_teacher_proof'](bb,bb,bb,a,zero,np.array([3.25,3.25],np.float64));assert not proof['PASS'] and proof['status']=='CONSTRUCTED_UNKNOWN'
support=ns['positive_support_pixels'](np.array([3.,3.],np.float64),(8,8));assert len(support['continuous_reference']['pixels_xy'])==1 and len(support['nearest_half_up_3x3']['pixels_xy'])==9
edge=ns['renewed_teacher_proof'](bb,bb,bb,np.ones((8,8),bool),zero,np.array([0.,0.],np.float64));assert not edge['PASS']
# Parity failure stops before renewed composition; never fix/recenter a mismatch.
p3=dict(parent,expected_BEFORE_RGB96_CHW_sha256='0'*64)
try:ns['compose_scene_pair'](base,group,p3,drgb,mask,'TOY_FIT_DONOR',crop,context,old,scene)
except ValueError:pass
else:raise AssertionError('BEFORE parity failure accepted')
summary={'completed':True,'PASS':True,'invented_scene_groups':4,'all8_and_context_same_native_composite_exact':True,'V90_U8_and_V107_F32_used_not_helper_localF32':True,'original_geometry_flags_and_valid_unchanged':True,'native_positive_tap_continuous_and_halfup3x3_renewed':True,'SHAM_reference_RGB_unchanged':True,'unsupported_context_no_new_label_gate':True,'reference_unknown_or_native_proof_failure_never_E':True,'BEFORE_active_SHA_failure_rejected':True,'old_post96_crop_not_used_as_final_input':True,'actual_SOURCE_RGB_GT_arrays_weights_model_or_fit':0}
p=H/'INVENTED_SCENE_CHECKS.json';assert not p.exists();p.write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))

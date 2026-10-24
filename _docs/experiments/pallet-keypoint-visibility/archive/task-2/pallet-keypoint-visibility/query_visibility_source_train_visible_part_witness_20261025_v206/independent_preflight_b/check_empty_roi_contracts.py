"""V206 A/C empty ROI contract preflight: metadata and invented values only."""
from pathlib import Path
import ast,copy,hashlib,json,stat
O=Path(__file__).parent;V=O.parent
def rb(p):
 if 'v112' in str(p).casefold():raise ValueError('forbidden before FS')
 for q in reversed([p,*p.parents]):
  if stat.S_ISLNK(q.lstat().st_mode):raise ValueError('symlink')
 return p.read_bytes()
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(rb(p)).hexdigest())
def get(b):
 data=rb(Path(b['path']));assert hashlib.sha256(data).hexdigest()==b['sha256'];return json.loads(data)
def funcs(path,names):
 t=ast.parse(rb(path));ns={};chosen=[n for n in t.body if isinstance(n,ast.FunctionDef)and n.name in names];assert {n.name for n in chosen}==names
 exec(compile(ast.Module(body=chosen,type_ignores=[]),'exact-reviewed-ROI-pure-functions','exec'),ns);return ns

def main():
 ar=bind(V/'observer_a/PREPARATION_COMPLETE.json');cr=bind(V/'observer_c/FINAL_READY.json');assert ar['sha256']=='cd723bea2134c078a7452245170c1ccf9988431f0e6cbde091a59d9027b1efa0'and cr['sha256']=='d0cc2ab36f7da6958148de2ae7f3cd45698915e170f49396661ca27ae2e6b40d'
 a=get(ar);c=get(cr);ap=get(a['protocol']);cp=get(c['protocol']);at=get(a['template'])['rows'];ct=get(c['template'])['rows'];sel=get(a['selection']);get(ap['scope']);assert c['protocol']==bind(Path(c['protocol']['path']))and cp['selection']==a['selection'];cases=sel['source_cases8']
 checks=[]
 def test(n,v):assert v,n;checks.append(dict(name=n,PASS=True))
 def reject(fn):
  try:fn()
  except AssertionError:return True
  return False
 an=funcs(Path(a['validator']['path']),{'need','validate_roi','validate_rows'});cn=funcs(Path(c['code']['path']),{'validate_witness'})
 test('Exactly8_ordered_original_case_ids',len(at)==len(ct)==len(cases)==8 and [r['row_id']for r in at]==[r['row_id']for r in ct]==[x['original_slot']['row_id']for x in cases])
 for aa,cc,case in zip(at,ct,cases):
  s=case['original_slot'];ai=aa['original_input'];assert all(ai[k]==s[k]for k in ('frame_key','original_frame_index','native_slot','query_xy','L','original_query_available','crop_constructed'))
  assert all(cc[k]==s[k]for k in ('frame_key','original_frame_index','native_slot','original_query_available'))and cc['original_query_xy']==s['query_xy']and cc['original_L']==s['L']
 test('Both_schemas_crosswalk_original_F64q_L_native_U_unchanged',True)
 state='visible_intended_part_identified_YES_UNKNOWN';box='witness_ROI_raw96_xyxy_or_null'
 test('Both_empty8_UNREVIEWED_null_no_views_no_admission',all(r[state]=='UNREVIEWED'and r[box]is None and r['actually_viewed_image_bindings']==[]and r['training_admission']is False for r in at+ct))
 an['validate_rows'](at,at,False)
 for r in ct:cn['validate_witness'](r,False)
 goodA=copy.deepcopy(at)
 for r in goodA:r.update({state:'YES',box:[95,95,95,95],'visible_feature_description':'invented feature','positive_local_visual_evidence':'invented local edge','actually_viewed_image_bindings':[r['original_image_bindings']['clean96'],r['original_image_bindings']['fulloriginal']]})
 goodC=copy.deepcopy(ct)
 for i,r in enumerate(goodC):r.update({state:'YES',box:[95,95,95,95],'visible_feature_description':'invented feature','positive_local_visual_evidence':'invented local edge','actually_viewed_image_bindings':goodA[i]['actually_viewed_image_bindings']})
 an['validate_rows'](goodA,at,True)
 for r in goodC:cn['validate_witness'](r,True)
 test('One_pixel_far_from_query_valid_no_q_containment_min_area_gate',True)
 for xy in ([0,0,95,95],[0,0,0,0]):
  an['validate_roi'](xy);cn['validate_witness']({**goodC[0],box:xy},True)
 test('Inclusive_both0_95_boundaries_accepted',True)
 for name,xy in [('negative',[-1,0,1,1]),('beyond95',[0,0,96,95]),('reversed',[3,2,1,4]),('fractional',[0,0,1.5,2]),('boolean',[False,0,1,2]),('YESnull',None)]:
  test('Both_reject_'+name,reject(lambda:an['validate_roi'](xy))and reject(lambda:cn['validate_witness']({**goodC[0],box:xy},True)))
 uA=copy.deepcopy(goodA);uC=copy.deepcopy(goodC)
 for r in uA+uC:r.update({state:'UNKNOWN',box:None,'ambiguity_note':'invented indistinguishable nearby wood'})
 an['validate_rows'](uA,at,True)
 for r in uC:cn['validate_witness'](r,True)
 test('UNKNOWN_null_kept_not_imputed',True)
 test('UNKNOWN_nonnull_rejected_both',reject(lambda:an['validate_rows']([{**uA[0],box:[0,0,1,1]}]+uA[1:],at,True))and reject(lambda:cn['validate_witness']({**uC[0],box:[0,0,1,1]},True)))
 drift=copy.deepcopy(goodA);drift[0]['original_input']['query_xy']=[0.,0.];test('A_rejects_original_q_drift',reject(lambda:an['validate_rows'](drift,at,True)))
 rogue={**goodC[0],'original_query_xy':[0.,0.],'training_admission':True,'actually_viewed_image_bindings':[{'path':'/NOT_ACTUALLY_VIEWED.png','sha256':'a'*64}]};cn['validate_witness'](rogue,True)
 test('C_generic_validator_limit_reproduced_root_seal_wrapper_required',True)
 rule=bind(V/'ROOT_PROSPECTIVE_WITNESS_JOIN_AND_FIXED8_VISUAL_RULE.json');jr=get(rule);assert jr['all8_fixed_source_cases_retained']and jr['no_q_containment_center_alignment_minimum_size_mesh_gold_or_locationflag_gate']and jr['no_automatic_externalE_or_training_admission']
 test('Prospective_fixed8_separate_ROIs_IoU_descriptive_only_no_gate',True)
 out=dict(completed=True,PASS=True,status='PASS_WITH_C_SEAL_IDENTITY_VIEW_PRECONDITION',observer_A_ready=ar,observer_C_ready=cr,observer_A_validator=a['validator'],observer_C_code=c['code'],observer_A_protocol=a['protocol'],observer_C_protocol=c['protocol'],selection=a['selection'],root_join_rule=rule,independent_code=bind(Path(__file__)),checks=checks,count=len(checks),C_required_actual_seal_and_common_join_guard='Root explicitly accepts: compare ordered original8 IDs/qL/native/U/fitfalse with selection and require each samecase CLEAN96+same fullcontext actually viewed exactbinding. Generic C ROI validator alone does not certify this; no code replacement requested.',ROI_authority='Weak visible local feature rectangle for later separately approved intervention mask, never a corrected query/keypoint/location/physical native gold.',actual_new_ROI_or_image_or_array_GT_model_protected_reads=0,other_new_ROI_value_reads=0,producer_edits=0,training_admission=False)
 with (O/'REVIEW.json').open('x')as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(completed=True,PASS=True,review=bind(O/'REVIEW.json'),checks=len(checks))))
if __name__=='__main__':main()

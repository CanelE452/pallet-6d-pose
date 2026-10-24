"""Exact join classification on invented rows; no observer values read."""
from pathlib import Path
import ast,hashlib,json,stat
O=Path(__file__).parent;V=O.parent
def rb(p):
 if 'v112' in str(p).casefold():raise ValueError('forbidden')
 for q in reversed([p,*p.parents]):
  if stat.S_ISLNK(q.lstat().st_mode):raise ValueError('symlink')
 return p.read_bytes()
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(rb(p)).hexdigest())
def main():
 code=V/'join_independent_source_observations.py';tree=ast.parse(rb(code));f=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='classification')
 ns={'STATE':'part_visibility_D_E_S_UNKNOWN','RELATION':'query_relation_on_near_off_UNKNOWN','IDENTITY':('target_instance_identified','intended_part_identified')};exec(compile(ast.Module(body=[f],type_ignores=[]),'exact-pure-classification','exec'),ns);fn=ns['classification']
 s=ns['STATE'];r=ns['RELATION'];base={s:'D',r:'ON','target_instance_identified':'YES','intended_part_identified':'YES','location_error_possible':'NO','short_visual_evidence':'visible queried part','external_occluder_evidence':''}
 tests=[('U_first_never_E',False,{**base,s:'E'},{**base,s:'E'},('ORIGINAL_INPUT_U',None)),('D_ON_agrees',True,base,base,('ALIGNED_WEAK_CONSENSUS_CANDIDATE','D')),('D_NEAR_agrees',True,{**base,r:'NEAR'},{**base,r:'NEAR'},('ALIGNED_WEAK_CONSENSUS_CANDIDATE','D')),('E_external_both_positive_text',True,{**base,s:'E','external_occluder_evidence':'foreground object blocks intendedpart'},{**base,s:'E','external_occluder_evidence':'identified external layer'},('ALIGNED_WEAK_CONSENSUS_CANDIDATE','E')),('E_empty_external_not_admitted',True,{**base,s:'E'},{**base,s:'E'},('INSUFFICIENT_ALIGNED_EVIDENCE',None)),('ON_NEAR_disagree',True,base,{**base,r:'NEAR'},('INSUFFICIENT_ALIGNED_EVIDENCE',None)),('Wrong_target_not_E',True,{**base,s:'E','target_instance_identified':'NO'},{**base,s:'E','external_occluder_evidence':'foreign'},('INSUFFICIENT_ALIGNED_EVIDENCE',None)),('OFF_before_Dagreement',True,{**base,r:'OFF'},base,('LOCATION_ERROR_OR_OFF',None)),('Location_error_before_self',True,{**base,s:'S','location_error_possible':'YES'},{**base,s:'S'},('LOCATION_ERROR_OR_OFF',None)),('SELF_separate',True,{**base,s:'S'},{**base,s:'S'},('SELF_CONSENSUS',None)),('UNKNOWN_separate',True,{**base,s:'UNKNOWN'},{**base,s:'UNKNOWN'},('UNKNOWN_CONSENSUS',None)),('D_E_disagree',True,base,{**base,s:'E'},('DISAGREEMENT_OR_UNRESOLVED',None))]
 checks=[]
 for name,av,a,c,expected in tests:
  assert fn(av,a,c)==expected,name;checks.append(dict(name=name,PASS=True))
 try:fn(True,{**base,s:'UNREVIEWED'},base)
 except AssertionError:checks.append(dict(name='Available_UNREVIEWED_refused',PASS=True))
 else:raise AssertionError('UNREVIEWED')
 result=dict(completed=True,PASS=True,invented_only=True,count=len(checks),checks=checks,code=bind(code),actual_observer_values_images_GT_model_score_reads=0)
 with (O/'INVENTED_CHECKS.json').open('x')as out:json.dump(result,out,indent=2);out.write('\n')
 print(json.dumps(dict(completed=True,PASS=True,count=len(checks),checks=bind(O/'INVENTED_CHECKS.json'))))
if __name__=='__main__':main()

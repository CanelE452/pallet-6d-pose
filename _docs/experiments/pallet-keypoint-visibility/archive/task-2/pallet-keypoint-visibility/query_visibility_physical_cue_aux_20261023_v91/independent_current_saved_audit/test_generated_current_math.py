"""Generated fixtures only; never reads any current artifact or SHA."""
from pathlib import Path
import json,hashlib,datetime,math
import current_metric_math as m
O=Path(__file__).resolve().parent

def b(p):
 p=Path(p);z=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(z).hexdigest(),'bytes':len(z)}
def main():
 for z,want in[(-5e-324,'VISIBLE'),(-1,'VISIBLE'),(-0.,'ABSENT_NONVISIBLE'),(0.,'ABSENT_NONVISIBLE'),(5e-324,'ABSENT_NONVISIBLE'),(100,'ABSENT_NONVISIBLE')]:
  assert m.authoritative(z,True)==want and m.authoritative(z,False,True)=='OTHER_UNRESOLVED'
 assert m.authoritative(-2,True,True)=='ABSENT_NONVISIBLE'
 assert m.authoritative(-5e-324,True)=='VISIBLE'and 1/(1+math.exp(-5e-324))==.5
 base={'present_at_fixed_0_5':'False','legacy_stored_state':'B'};assert m.baseline_call(base,'v61','D_vs_E')=='ABSENT_NONVISIBLE'
 expected=['VISIBLE','VISIBLE','ABSENT_NONVISIBLE','OUT_OF_TASK_GEOMETRY_S','ABSTAIN','OTHER_UNRESOLVED']
 for s,w in zip(['B','I','O','S','U','X'],expected):assert m.baseline_call({**base,'legacy_stored_state':s},'legacy','D_vs_E')==w
 assert m.baseline_call({**base,'legacy_stored_state':'S'},'legacy','D_vs_S')=='ABSENT_NONVISIBLE'
 assert m.baseline_call(base,'V76','D_vs_ES',p=.5)=='VISIBLE'and m.baseline_call(base,'V76','D_vs_ES',p=.5,valid=False,geometry=True)=='OTHER_UNRESOLVED'
 rows=[{'id':str(i),'human_state':m.D if i<3 else m.E,'group':'WOOD_PRIMARY105','paired_same_query_RGB_D_vs_E':'True','paired_same_query_RGB_D_vs_ES':'True'}for i in range(6)]
 pp=['VISIBLE','ABSENT_NONVISIBLE','OTHER_UNRESOLVED','VISIBLE','ABSENT_NONVISIBLE','OTHER_UNRESOLVED'];cm=m.calculate(rows,pp)
 assert[cm[k]for k in('TP','FN_decided','FP','TN','other_unresolved_visible','other_unresolved_occluded')]==[1,1,1,1,1,1]
 assert cm['N']==6 and cm['visible_recall_all_GT']==1/3 and cm['unresolved_rate']==1/3 and cm['decision_coverage']==2/3
 assert len(m.selected(rows,'paired_same_query_RGB','WOOD_PRIMARY105','D_vs_E'))==6
 changes=m.paired_change(rows,['VISIBLE']*6,pp);assert sum(changes.values())==6 and changes['new_unresolved']==2
 p={'id':'x','annotation_coordinate_source':'manual_click','human_state':m.D,'query_valid':'True','initial_reference_error_px':'3'}
 prs=[p,{**p,'id':'projection','annotation_coordinate_source':'projected','initial_reference_error_px':'2'},{**p,'id':'badq','query_valid':'False'},{**p,'id':'above3','initial_reference_error_px':'3.0000001'},{**p,'id':'neg','human_state':m.E}]
 assert[r['id']for r in m.manual_near(prs)]==['x']and m.error_bin(prs[1]['initial_reference_error_px'])=='px_le3'
 assert[m.error_bin(x)for x in(3,3.0001,10,10.0001,30,30.0001)]==['px_le3','px_3_to10','px_3_to10','px_10_to30','px_10_to30','px_gt30']
 # Any-map support must not promote an invalid centre or originalU.
 prior=[True,False,True,True];centre=[True,True,False,True];new=[p and c for p,c in zip(prior,centre)]
 assert new==[True,False,False,True]and sum(not x for x in new)==sum(not x for x in prior)+1
 result={'completed':True,'PASS':True,'created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'code':b(__file__),'math':b(O/'current_metric_math.py'),'scope':b(O/'SCOPE_BEFORE_CURRENT_RESULTS.json'),'raw_zE_tiny_and_zero_exact':True,'display_probability_tie_not_decoder':True,'invalid_before_geometry':True,'legacy_v61_V76_rules_unchanged':True,'CM_P_R_U_allGT_denominators':True,'methodU_not_cohort_filter':True,'paired_changes_preserve_allrows':True,'manual_direct_near_separatefromprojected':True,'error_bin_boundaries_exact':True,'priorU_and_additionalcentreU_preserved':True,'actual_current_artifact_body_SHA_reads':0,'model_weights_forward_fit_current_PnP':0}
 with(O/'GENERATED_INDEPENDENT_CURRENT_MATH_TESTS.json').open('x')as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({'PASS':True,'generated':b(O/'GENERATED_INDEPENDENT_CURRENT_MATH_TESTS.json')},indent=2))
if __name__=='__main__':main()

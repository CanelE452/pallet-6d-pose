"""Independent saved scalar V81/V82 metrics, criteria and paired transitions."""
from pathlib import Path
from collections import Counter
import ast
import csv
import hashlib
import json
import math
import numpy as np

R=Path(__file__).resolve().parent;V=R.parents[1];P=V.parent;OLD=P/'query_visibility_grounded_target_20261023_v81'
NEW=V/'source_evaluation/REQUESTED_CATEGORY_SAM21_FIXED'
PRIOR=OLD/'source_evaluation/GROUNDED_SAM21_FIXED'
COM=V/'source_evaluation/REQUESTED_CATEGORY_SAM21_FIXED_COMPARISON'
POLICY=R/'READER_POLICY_BEFORE_PACKETS.json';POLICY_SHA='c71ec2195c7efe6511c3755a3c942826587613a9d18387e2eb5ad1e2d4ffc496'
READS=[]

def safe(path):
    p=Path(path);assert p.suffix in {'.json','.jsonl','.csv','.npz','.py'}
    assert 'private_evaluation' not in p.parts and '/segmentation/outputs/masks/' not in str(p) and 'current223' not in str(p).lower()
    return p
def sha(path):p=safe(path);READS.append(str(p));return hashlib.sha256(p.read_bytes()).hexdigest()
def binding(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def check(b):assert sha(b['path'])==b['sha256'],b['path']
def read(path):p=safe(path);READS.append(str(p));return json.loads(p.read_text())
def packet(path):
    p=safe(path);READS.append(str(p))
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

# Reuse the already independently verified V81 scalar definitions verbatim;
# do not import or call either producer/evaluator or its GT-mask readers.
PURE=OLD/'source_audit/metrics_reconciliation/reconcile_saved_metrics.py'
tree=ast.parse(PURE.read_text());functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'frac','confusion','cross','compare'}]
assert len(functions)==4
exec(compile(ast.Module(body=functions,type_ignores=[]),str(PURE),'exec'),globals())

def derive(z):
    y=z['reference_local_label8'];q=z['q_pixel_state8'];pred=z['prediction_CVU'];primary=z['original_primary8'];strict=z['strict535_mask8']
    have=z['mask_available'];assoc=z['detector_associated'];a=z['GT_visible_pixels'];b=z['predicted_visible_pixels'];inter=z['intersection_pixels'];union=z['union_pixels'];saved_iou=z['IoU_allrows_nomask0']
    assert np.array_equal(have,assoc) and np.array_equal(union,a+b-inter)
    assert np.all((inter>=0)&(inter<=a)&(inter<=b))
    iou=np.zeros(1035,np.float64)
    for i in range(1035):
        if have[i]:iou[i]=float(inter[i])/int(union[i]) if union[i] else 1.
        else:assert inter[i]==0 and b[i]==0
    assert np.array_equal(iou,saved_iou)
    result=[]
    for src in ['ALL','SYNTHETIC','YCB_PUBLIC_REAL']:
        f=np.ones(1035,bool) if src=='ALL' else z['source']==src
        s=np.repeat(f[:,None],8,axis=1);n=int(f.sum());v=iou[f];h=have[f];ac=assoc[f]
        iom=dict(all_rows_N=n,all_rows_no_mask0_mean=float(sum(v)/n),usable_selected_mask_N=int(h.sum()),usable_selected_mask_conditional_mean=frac(float(sum(v[h])),int(h.sum())),detector_associated_N=int(ac.sum()),detector_associated_with_failedmask0_mean=frac(float(sum(v[ac])),int(ac.sum())),nomatch_N=int((~ac).sum()),associated_missing_mask_N=int((ac&~h).sum()),mask_hold_rate=float((~h).sum()/n),empty_GT_N=int((a[f]==0).sum()),actual_both_empty_mask_N=int((h&(a[f]==0)&(b[f]==0)).sum()),all_rows_quantiles_0_25_50_75_100=np.quantile(v,[0,.25,.5,.75,1]).tolist())
        def dist(mask):return dict(N=int(mask.sum()),pred_CVU_counts=[int((mask&(pred==j)).sum()) for j in [0,1,2]])
        result.append(dict(source=src,target_instances=n,mask_IoU=iom,REFERENCE_LOCAL_PRIMARY_UNCHANGED_mask_at_q_component=confusion(y,pred,s&primary),SEPARATE_QUERY_PIXEL_CV_given_GT_M0_and_native_valid=confusion(q,pred,s&np.isin(q,[0,1])),query_pixel_O_separate_not_E_or_S=dist(s&(q==2)),query_pixel_invalid_separate=dist(s&(q==-1)),strict535_same_frozen_IDs=confusion(y,pred,s&strict),PRIMARY_reference_CVU_vs_query_GT_invalid_C_V_O=cross(y,q,s&primary)))
    return result

def csv_check(folder,z):
    READS.append(str(folder/'PER_QUERY.csv'))
    with (folder/'PER_QUERY.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==8280
    for i,row in enumerate(rows):
        j,k=divmod(i,8)
        expected=dict(input_id=str(z['input_id'][j]),original_id=str(z['original_id'][j]),source=str(z['source'][j]),native_k=str(k),reference_label=str(z['reference_local_label8'][j,k]),primary=str(z['original_primary8'][j,k]),qGTstate_invalid_C_V_O=str(z['q_pixel_state8'][j,k]),qGT_valid=str(z['q_pixel_valid8'][j,k]),pred_CVU=str(z['prediction_CVU'][j,k]),strict535=str(z['strict535_mask8'][j,k]),mask_status='OK' if z['mask_available'][j] else 'U')
        assert all(row[key]==value for key,value in expected.items())

def transitions(old,new,mask):return [[int((mask&(old==i)&(new==j)).sum()) for j in range(3)] for i in range(3)]

def main():
    assert sha(POLICY)==POLICY_SHA
    policy=read(POLICY);bound=policy['bound_before_packet_reads']
    assert sha(NEW/'EVALUATION_SEAL.json')==bound['evaluation_seal_sha256']
    assert sha(COM/'COMPARISON_AND_SCREEN_SEAL.json')==bound['comparison_seal_sha256']
    for f in [NEW/'EVALUATION_SEAL.json',PRIOR/'EVALUATION_SEAL.json',COM/'COMPARISON_AND_SCREEN_SEAL.json']:
        x=read(f);assert x['completed'] is True and x['PASS'] is True
        for b in x['files']:check(b)
    oldreport=read(PRIOR/'SOURCE_MASK_RESULTS.json');newreport=read(NEW/'SOURCE_MASK_RESULTS.json')
    old=packet(PRIOR/'SAVED_MASK_EVALUATION_PER_QUERY.npz');new=packet(NEW/'SAVED_MASK_EVALUATION_PER_QUERY.npz')
    assert len(old['input_id'])==len(new['input_id'])==1035
    invariant=['input_id','original_id','source','original_primary8','reference_local_label8','q_pixel_state8','q_pixel_valid8','strict535_mask8']
    for k in invariant:assert np.array_equal(old[k],new[k]),k
    assert Counter(new['source'])=={'SYNTHETIC':235,'YCB_PUBLIC_REAL':800}
    gt=packet(OLD/'source_evaluation_preparation/GT_EVALUATION_PACK_DO_NOT_FORWARD.npz')
    for k in ['input_id','original_id','source','original_primary8','reference_local_label8','q_pixel_valid8','strict535_mask8']:assert np.array_equal(gt[k],new[k]),k
    assert np.array_equal(gt['q_pixel_state8_CVO_invalid'],new['q_pixel_state8'])
    assert np.array_equal(gt['original_primary8'],np.isin(gt['reference_local_label8'],[0,1])&gt['input_available8']&gt['reference_in_local_square8']&gt['q_pixel_valid8'])
    csv_check(PRIOR,old);csv_check(NEW,new)
    assert newreport['source_GT_requested_category_metadata_provided_to_detector'] is True
    assert newreport['GT_masks_reference_pose_depth_visibility_or_instance_token_to_models'] is False and newreport['GT_used_to_choose_detector_or_SAM_mask_after_prediction'] is False
    assert newreport['no_automatic_current223_admission'] is True and newreport['root_source_pilot_only_no_automatic_current_admission'] is True
    differences=[];checks=[]
    oldmetrics=derive(old);newmetrics=derive(new)
    compare(oldreport['benchmarks'],oldmetrics,'V81.benchmarks',differences,checks);compare(newreport['benchmarks'],newmetrics,'V82.benchmarks',differences,checks)
    comparison=read(COM/'SAME_IDS_V81_V82_COMPARISON_KO.json');screen=read(COM/'PROSPECTIVE_COMPONENT_SCREEN_KO.json');criterion=read(V/'SOURCE_COMPONENT_CRITERIA_BEFORE_FORWARD.json')
    check(screen['criteria']);assert comparison['not_same_input_ablation'] is True
    y=new['reference_local_label8'];s=new['strict535_mask8'];op=old['prediction_CVU'];np_=new['prediction_CVU']
    assert s.sum()==535 and (s&(y==0)).sum()==76 and (s&(y==1)).sum()==459
    strict_transition=[dict(GT_label=j,N=int((s&(y==j)).sum()),V81_to_V82_CVU_counts=transitions(op,np_,s&(y==j))) for j in [0,1]]
    compare(comparison['same_strict535_prediction_transitions'],strict_transition,'paired.strict535',differences,checks)
    availability=[dict(V81=a,V82=b,N=int(((old['mask_available']==a)&(new['mask_available']==b)).sum())) for a in [False,True] for b in [False,True]]
    compare(comparison['mask_availability_transitions_all1035'],availability,'paired.mask_availability',differences,checks)
    idx={str(i):j for j,i in enumerate(new['original_id'])}
    for case in comparison['same_fixed_four_cases']:
        i=idx[case['original_id']];k=case['native_k']
        assert case['GT_reference_label']==int(y[i,k]) and case['V81_prediction_CVU']==int(op[i,k]) and case['V82_prediction_CVU']==int(np_[i,k])
        assert case['V81_whole_mask_IoU_no_mask0']==old['IoU_allrows_nomask0'][i] and case['V82_whole_mask_IoU_no_mask0']==new['IoU_allrows_nomask0'][i]
    st=next(x for x in newmetrics if x['source']=='YCB_PUBLIC_REAL')['strict535_same_frozen_IDs']
    assert st['CM_rows_GT_CV_cols_pred_CVU']==[[22,24,30],[47,251,161]]
    fieldmap={'C_end_to_end_recall_at_least':('C_end_to_end_recall_U_is_miss','ge'),'V_end_to_end_recall_at_least':('V_end_to_end_recall_U_is_miss','ge'),'C_precision_known_CV_at_least':('C_precision_known_CV_conditional','ge'),'hold_rate_at_most':('hold_rate','le')}
    calculated_checks={}
    for name,(field,op_) in fieldmap.items():
        val=st[field];limit=criterion['criteria'][name];passed=val is not None and ((val>=limit) if op_=='ge' else (val<=limit))
        calculated_checks[name]=dict(value=val,limit=limit,comparison=op_,PASS=passed)
    compare(screen['checks'],calculated_checks,'criteria',differences,checks)
    sourcepass=all(x['PASS'] for x in calculated_checks.values());assert screen['component_screen_PASS']==sourcepass==False and screen['자동현재223평가승인'] is False
    old_falseV=s&(y==0)&(op==1);endpoints=[int((old_falseV&(np_==j)).sum()) for j in range(3)]
    assert old_falseV.sum()==56 and endpoints==[17,17,22]
    new_falseV=s&(y==0)&(np_==1)
    introduced=[int((new_falseV&(op==j)).sum()) for j in range(3)]
    assert int(new_falseV.sum())==24 and introduced==[5,17,2]
    cm_old=next(x for x in oldmetrics if x['source']=='YCB_PUBLIC_REAL')['strict535_same_frozen_IDs']['CM_rows_GT_CV_cols_pred_CVU'];cm_new=st['CM_rows_GT_CV_cols_pred_CVU']
    assert cm_new[0][0]-cm_old[0][0]==7 and cm_new[0][2]-cm_old[0][2]==25
    common_iou=[]
    for source in ['ALL','SYNTHETIC','YCB_PUBLIC_REAL']:
        f=np.ones(1035,bool) if source=='ALL' else new['source']==source
        masks=[]
        for a in [False,True]:
            for b in [False,True]:
                keep=f&(old['mask_available']==a)&(new['mask_available']==b);n=int(keep.sum())
                masks.append(dict(V81_mask=a,V82_mask=b,N=n,V81_mean_IoU_nomask0=None if not n else float(old['IoU_allrows_nomask0'][keep].mean()),V82_mean_IoU_nomask0=None if not n else float(new['IoU_allrows_nomask0'][keep].mean())))
        old_iou=next(z for z in oldmetrics if z['source']==source)['mask_IoU'];new_iou=next(z for z in newmetrics if z['source']==source)['mask_IoU']
        common_iou.append(dict(source=source,V81_allmean=old_iou['all_rows_no_mask0_mean'],V82_allmean=new_iou['all_rows_no_mask0_mean'],allmean_delta=new_iou['all_rows_no_mask0_mean']-old_iou['all_rows_no_mask0_mean'],availability_subsets=masks))
    result=dict(completed=True,PASS=not differences,discrepancies=differences,published_scalar_metric_leaves_compared=len(checks),reader_policy=binding(POLICY),audit_code=binding(__file__),inherited_independent_pure_formulas=binding(PURE),inputs=dict(V81_evaluation_seal=binding(PRIOR/'EVALUATION_SEAL.json'),V82_evaluation_seal=binding(NEW/'EVALUATION_SEAL.json'),comparison_screen_seal=binding(COM/'COMPARISON_AND_SCREEN_SEAL.json'),V81_saved_packet=binding(PRIOR/'SAVED_MASK_EVALUATION_PER_QUERY.npz'),V82_saved_packet=binding(NEW/'SAVED_MASK_EVALUATION_PER_QUERY.npz'),GT_packet=binding(OLD/'source_evaluation_preparation/GT_EVALUATION_PACK_DO_NOT_FORWARD.npz'),criteria=binding(V/'SOURCE_COMPONENT_CRITERIA_BEFORE_FORWARD.json')),
        reconciled_V81_benchmarks=oldmetrics,reconciled_V82_benchmarks=newmetrics,source_component_screen=dict(PASS=sourcepass,passed=sum(x['PASS'] for x in calculated_checks.values()),total=4,checks=calculated_checks,current223_allowed=False),strict535_transition_tables=strict_transition,mask_availability_transitions=availability,paired_IoU_by_availability=common_iou,
        old56_falsevisible_matched_endpoints=dict(N=56,new_C=17,new_V=17,new_U=22,new_total_falseV24_origins=dict(old_C=5,old_V=17,old_U=2),interpretation='Of the same old56 false-visible points,17 become correct covered,17 remain visible,22 become U. Total false-visible falls56->24 because39 leave false-visible but7 new false-visible errors enter. It is incorrect to call the net32 reduction32 corrected-C points.'),
        aggregate_strict_GT_C_tradeoff=dict(old_CVU=cm_old[0],new_CVU=cm_new[0],net_correct_C=7,net_U=25,old15_correctC_retained=4,old15_correctC_to_wrongV=5,old15_correctC_to_U=6,oldU_to_correctC=1),
        aggregate_strict_all_U=dict(V81=122,V82=191,net_change=69,GT_C_U_change=25,GT_V_U_change=44),
        category_scope=dict(YCB_category_GTmetadata_provided=True,predicted_class=False,same_input_ablation=False,GT_masks_reference_pose_depth_visibility_instance_tokens_forward=False,posthoc_GT_output_selection=False),
        conclusions='All published metrics/criteria/matched tables reconcile. Source component screen remains0/4, not goal completion. Conditional mask-IoU means have different accepted sets; common/new/lost availability subsets are reported separately. All-row YCB IoU changes only slightly downward despite its higher matched-only mean. Source support/pixel tasks remain distinct from human D/E/S.',
        limitations='IoU algebra reuses saved intersection/union counts and no-mask convention. No native GT/prediction mask pixels were reread, so pixel-to-count derivation was not independently rerun.',scope=dict(RGB_reads=0,native_GT_or_prediction_mask_reads=0,weights_reads=0,new_model_or_forward=0,current223_reads=0,fit=0,PnP=0,prior_artifacts_modified=0),saved_metadata_packet_paths=sorted(set(READS)))
    out=R/'INDEPENDENT_SAVED_METRICS_CRITERIA_TRANSITIONS_RECONCILIATION.json'
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    with (R/'RECONCILIATION_SEAL.json').open('x') as f:json.dump(dict(completed=True,PASS=not differences,files=[binding(POLICY),binding(__file__),binding(out)],source_component_screen_PASS=False,old_artifacts_modified=False),f,indent=2);f.write('\n')
    print(json.dumps(dict(PASS=not differences,path=str(out),sha256=sha(out),scalar_leaves=len(checks),discrepancies=differences,old56_newCVU=endpoints,criteria_passed=0,criteria_total=4)))

if __name__=='__main__':main()

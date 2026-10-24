"""Saved scalar GT/count/score reconciliation; no image/mask/model reader."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math
import numpy as np

R=Path(__file__).resolve().parent;V=R.parents[1];P=V.parent
EP=V/'source_evaluation/GROUNDED_SAM21_FIXED'
GP=V/'source_evaluation_preparation'
POLICY=R/'READER_POLICY_BEFORE_PACKETS.json'
POLICY_SHA='c30b28407e10032657c0b0a33cb267af3a3de68d3c09511facee4b459bd2370d'
COVERAGE=P/'query_visibility_natural_occlusion_audit_20261023_v79/coverage/results/COVERAGE_PER_QUERY.npz'
READS=[]

def allowed(path):
    p=Path(path)
    assert p.suffix.lower() in {'.json','.jsonl','.csv','.npz','.py'},'Only saved scalar packets/metadata/code allowed'
    assert 'private_evaluation' not in p.parts and 'current223' not in str(p).lower()
    assert '/segmentation/outputs/masks/' not in str(p)
    assert str(p).startswith(str(V)) or p==COVERAGE
    return p
def sha(path):p=allowed(path);READS.append(str(p));return hashlib.sha256(p.read_bytes()).hexdigest()
def binding(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def read(path):p=allowed(path);READS.append(str(p));return json.loads(p.read_text())
def check(b):assert sha(b['path'])==b['sha256'],b['path']
def packet(path,keys=None):
    p=allowed(path);READS.append(str(p))
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in (z.files if keys is None else keys)}
def frac(a,b):return a/b if b else None

def confusion(y,p,mask):
    cm=[[0,0,0],[0,0,0]]
    for truth,pred,keep in zip(y.reshape(-1),p.reshape(-1),mask.reshape(-1)):
        if keep:
            assert truth in (0,1) and pred in (0,1,2);cm[int(truth)][int(pred)]+=1
    C,V=cm;n=sum(C)+sum(V);hold=C[2]+V[2]
    return dict(N=n,CM_rows_GT_CV_cols_pred_CVU=cm,
        C_precision_known_CV_conditional=frac(C[0],C[0]+V[0]),V_precision_known_CV_conditional=frac(V[1],C[1]+V[1]),
        C_end_to_end_recall_U_is_miss=frac(C[0],sum(C)),V_end_to_end_recall_U_is_miss=frac(V[1],sum(V)),
        C_accepted_only_recall_diagnostic=frac(C[0],C[0]+C[1]),V_accepted_only_recall_diagnostic=frac(V[1],V[0]+V[1]),
        hold_N=hold,hold_rate=frac(hold,n),C_hold_rate=frac(C[2],sum(C)),V_hold_rate=frac(V[2],sum(V)))

def cross(y,q,mask):
    cm=[[0]*4 for _ in range(3)]
    for a,b,keep in zip(y.reshape(-1),q.reshape(-1),mask.reshape(-1)):
        if keep:cm[int(a)+1][int(b)+1]+=1
    return cm

def compare(expected,actual,path,diffs,checks):
    if isinstance(expected,dict):
        if set(expected)!=set(actual):diffs.append(dict(path=path,kind='keys',expected=sorted(expected),actual=sorted(actual)));return
        for k in expected:compare(expected[k],actual[k],path+'.'+k,diffs,checks)
    elif isinstance(expected,list):
        if len(expected)!=len(actual):diffs.append(dict(path=path,kind='length'));return
        for i,(a,b) in enumerate(zip(expected,actual)):compare(a,b,path+'['+str(i)+']',diffs,checks)
    else:
        checks.append(path)
        same=(expected==actual) if not isinstance(expected,float) else (actual is not None and math.isclose(expected,actual,rel_tol=1e-12,abs_tol=1e-12))
        if not same:diffs.append(dict(path=path,published=expected,recomputed=actual))

def main():
    assert sha(POLICY)==POLICY_SHA and read(POLICY)['frozen'] is True
    es=read(EP/'EVALUATION_SEAL.json');gs=read(GP/'GT_PACK_SEAL.json')
    assert es['completed'] is True and es['PASS'] is True and gs['completed'] is True and gs['PASS'] is True
    for b in es['files']+gs['files']:check(b)
    reported=read(EP/'SOURCE_MASK_RESULTS.json');check(reported['GT_pack_seal']);check(reported['SAM_output_seal'])
    assert reported['GT_pack_seal']['sha256']=='f2548ddb36e8c5cdd0fa734509a36b6dd38bff35c35d89d9b980273bf86e5ced'
    gt=packet(GP/'GT_EVALUATION_PACK_DO_NOT_FORWARD.npz');saved=packet(EP/'SAVED_MASK_EVALUATION_PER_QUERY.npz')
    for k in ['input_id','original_id','source','original_primary8','reference_local_label8','q_pixel_valid8','strict535_mask8']:assert np.array_equal(gt[k],saved[k]),k
    assert np.array_equal(gt['q_pixel_state8_CVO_invalid'],saved['q_pixel_state8'])
    assert len(gt['input_id'])==1035 and len(set(gt['input_id']))==1035
    assert Counter(gt['source'])=={'SYNTHETIC':235,'YCB_PUBLIC_REAL':800}
    y=gt['reference_local_label8'];q=gt['q_pixel_state8_CVO_invalid'];pred=saved['prediction_CVU'];primary=gt['original_primary8'];strict=gt['strict535_mask8'];valid=gt['q_pixel_valid8']
    assert y.shape==q.shape==pred.shape==primary.shape==strict.shape==(1035,8)
    assert set(np.unique(y))=={-1,0,1} and set(np.unique(q))=={-1,0,1,2} and set(np.unique(pred))=={0,1,2}
    rebuilt_q=np.full((1035,8),-1,np.int8)
    for i in range(1035):
        h,w=gt['raw_hw'][i]
        for k,xy in enumerate(gt['query8_xy'][i]):
            finite=np.isfinite(xy).all();px,py=(math.floor(xy[0]+.5),math.floor(xy[1]+.5)) if finite else (-1,-1)
            native=bool(finite and 0<=xy[0]<w and 0<=xy[1]<h and 0<=px<w and 0<=py<h)
            assert native==valid[i,k]
            if native:
                m0,m4=int(gt['q_pixel_M08'][i,k]),int(gt['q_pixel_M48'][i,k]);assert m0 in [0,1] and m4 in [0,1] and m4<=m0
                rebuilt_q[i,k]=1 if m4 else (0 if m0 else 2)
    assert np.array_equal(rebuilt_q,q)
    assert np.array_equal(primary,np.isin(y,[0,1])&gt['input_available8']&gt['reference_in_local_square8']&valid)
    assert strict.sum()==535 and not (strict&~primary).any() and np.array_equal(q[strict],y[strict])
    # Join V79 only saved scalar coordinate/reference/GT-state arrays, no masks.
    covkeys=['id','source','split','domain','query8_xy','L','raw_hw','q_center_M0','q_center_M4','q_center_support_state','q_center_mask_valid',
        'teacher_label8','rule8','primary_loss_mask','input_available','reference_in_local_square_GT_only','reference_offset_L_GT_only','q_reference_error_px_GT_only']
    cov=packet(COVERAGE,covkeys)
    mapping={(str(i),str(s)):j for j,(i,s,d) in enumerate(zip(cov['id'],cov['source'],cov['domain'])) if d==0}
    ci=np.asarray([mapping[(str(i),str(s))] for i,s in zip(gt['original_id'],gt['source'])])
    assert np.all(cov['split'][ci]=='val')
    for a,b in [('query8_xy','query8_xy'),('L','L'),('raw_hw','raw_hw'),('q_center_M0','q_pixel_M08'),('q_center_M4','q_pixel_M48'),('q_center_support_state','q_pixel_state8_CVO_invalid'),('q_center_mask_valid','q_pixel_valid8'),('teacher_label8','reference_local_label8'),('primary_loss_mask','original_primary8'),('input_available','input_available8'),('reference_in_local_square_GT_only','reference_in_local_square8')]:assert np.array_equal(cov[a][ci],gt[b]),(a,b)
    reconstructed_strict=(gt['source']=='YCB_PUBLIC_REAL')[:,None]&primary&np.char.startswith(cov['rule8'][ci],'STRONG_3X3_')&(cov['q_reference_error_px_GT_only'][ci]<=3)&(np.max(np.abs(cov['reference_offset_L_GT_only'][ci]),axis=-1)<=1/18)
    assert np.array_equal(reconstructed_strict,strict)
    with (EP/'PER_QUERY.csv').open() as f:csvrows=list(csv.DictReader(f))
    assert len(csvrows)==8280
    for i,row in enumerate(csvrows):
        j,k=divmod(i,8)
        expected=dict(input_id=str(gt['input_id'][j]),original_id=str(gt['original_id'][j]),source=str(gt['source'][j]),native_k=str(k),reference_label=str(y[j,k]),primary=str(primary[j,k]),qGTstate_invalid_C_V_O=str(q[j,k]),qGT_valid=str(valid[j,k]),pred_CVU=str(pred[j,k]),strict535=str(strict[j,k]))
        assert all(row[a]==b for a,b in expected.items())
        assert row['mask_status']==('OK' if saved['mask_available'][j] else 'U')
    have=saved['mask_available'];assoc=saved['detector_associated'];inter=saved['intersection_pixels'];union=saved['union_pixels'];a=saved['GT_visible_pixels'];b=saved['predicted_visible_pixels'];ious=saved['IoU_allrows_nomask0']
    assert np.array_equal(have,assoc) and have.sum()==823
    assert np.all((inter>=0)&(a>=0)&(b>=0)&(inter<=a)&(inter<=b)) and np.array_equal(union,a+b-inter)
    independent_iou=np.zeros(1035,np.float64)
    for i in range(1035):
        if have[i]:independent_iou[i]=float(inter[i])/int(union[i]) if union[i] else 1.
        else:assert inter[i]==0 and b[i]==0
    assert np.array_equal(independent_iou,ious)
    assert not ((~valid)&(pred!=2)).any()
    derived=[];oracles=[]
    for source in ['ALL','SYNTHETIC','YCB_PUBLIC_REAL']:
        frames=np.ones(1035,bool) if source=='ALL' else gt['source']==source
        slots=np.repeat(frames[:,None],8,axis=1)
        io=independent_iou[frames];h=have[frames];as_=assoc[frames];ga=a[frames];pb=b[frames];n=int(frames.sum())
        iom=dict(all_rows_N=n,all_rows_no_mask0_mean=float(sum(io)/n),usable_selected_mask_N=int(h.sum()),usable_selected_mask_conditional_mean=frac(float(sum(io[h])),int(h.sum())),detector_associated_N=int(as_.sum()),detector_associated_with_failedmask0_mean=frac(float(sum(io[as_])),int(as_.sum())),nomatch_N=int((~as_).sum()),associated_missing_mask_N=int((as_&~h).sum()),mask_hold_rate=float((~h).sum()/n),empty_GT_N=int((ga==0).sum()),actual_both_empty_mask_N=int((h&(ga==0)&(pb==0)).sum()),all_rows_quantiles_0_25_50_75_100=np.quantile(io,[0,.25,.5,.75,1]).tolist())
        def distribution(mask):return dict(N=int(mask.sum()),pred_CVU_counts=[int(np.sum(mask&(pred==z))) for z in [0,1,2]])
        derived.append(dict(source=source,target_instances=n,mask_IoU=iom,
            REFERENCE_LOCAL_PRIMARY_UNCHANGED_mask_at_q_component=confusion(y,pred,slots&primary),
            SEPARATE_QUERY_PIXEL_CV_given_GT_M0_and_native_valid=confusion(q,pred,slots&np.isin(q,[0,1])),
            query_pixel_O_separate_not_E_or_S=distribution(slots&(q==2)),query_pixel_invalid_separate=distribution(slots&(q==-1)),
            strict535_same_frozen_IDs=confusion(y,pred,slots&strict),PRIMARY_reference_CVU_vs_query_GT_invalid_C_V_O=cross(y,q,slots&primary)))
        qbit=gt['q_pixel_M48'];selected=slots&primary
        assert np.all(np.isin(qbit[selected],[0,1]))
        oracle=confusion(y,qbit,selected);strict_oracle=confusion(y,qbit,slots&strict)
        oracles.append(dict(source=source,axes='Rows older reference-local C/V; columns exact GT q-M4 bit0/bit1 (third U column kept zero). This is not a learned model or semantic E oracle.',primary=oracle,strict535=strict_oracle,reference_vs_q_C_V_O_invalid=cross(y,q,selected)))
    differences=[];checks=[]
    compare(reported['benchmarks'],derived,'benchmarks',differences,checks)
    ycb=next(x for x in oracles if x['source']=='YCB_PUBLIC_REAL')
    assert [r[:2] for r in ycb['primary']['CM_rows_GT_CV_cols_pred_CVU']]==[[278,9],[660,1071]]
    assert ycb['strict535']['CM_rows_GT_CV_cols_pred_CVU']==[[76,0,0],[0,459,0]]
    strict_sam=next(x for x in derived if x['source']=='YCB_PUBLIC_REAL')['strict535_same_frozen_IDs']
    assert strict_sam['CM_rows_GT_CV_cols_pred_CVU']==[[15,56,5],[83,259,117]]
    result=dict(completed=True,PASS=not differences,reader_policy=binding(POLICY),audit_code=binding(__file__),inputs=dict(evaluation_seal=binding(EP/'EVALUATION_SEAL.json'),GT_pack_seal=binding(GP/'GT_PACK_SEAL.json'),results=binding(EP/'SOURCE_MASK_RESULTS.json'),saved_packet=binding(EP/'SAVED_MASK_EVALUATION_PER_QUERY.npz'),PER_QUERY=binding(EP/'PER_QUERY.csv'),GT_packet=binding(GP/'GT_EVALUATION_PACK_DO_NOT_FORWARD.npz'),V79_saved_q_reference_packet=binding(COVERAGE)),published_metric_scalar_leaves_compared=len(checks),discrepancies=differences,recomputed_benchmarks=derived,diagnostic_exact_q_M4_oracles=oracles,
        alignment=dict(all1035_ids_sources_order_and_8280_CSV_rows_match=True,all_saved_q_GT_bits_to_states_replayed=True,native_coordinate_validity_replayed=True,primary_mask_from_saved_knownlabel_inputavailable_reference_square_qvalid_exact=True,V79_saved_q_L_HW_reference_states_primary_all1035_exact=True,strict535_predicate_from_V79_metadata_exact=True),
        IoU_scope='Recomputed IoU = saved intersection / saved union for available masks, union = GTvisible + predictedvisible - intersection. Available both-empty =>1; no mask =>0. Count identities/aggregates verified; no mask pixels reread, so pixel-to-count derivation remains covered by original sealed evaluator rather than independently repeated here.',
        diagnostic_interpretation=dict(YCB_primary_qbit_vs_reference_disagreements=669,YCB_primary_N=2018,strict535_qbit_vs_reference_disagreements=0,strict535_N=535,warning='GT q-M4 bit is a counterfactual diagnostic, not a model/representation upper bound. It must not relabel qO as semantic C/E or change reference-primary labels/gates. A reference-aware crop model can use different evidence than the exact q bit.',strict535_SAM=dict(correct=274,decided_incorrect=139,U=122),uncertainty='Reference-local support and pixel-M4 target ownership remain proxy tasks; no human D/E/S inference from these results.'),scope=dict(raw_RGB_reads=0,mask_pixel_reads=0,weights_reads=0,model_import_or_forward=0,fit=0,current223_reads=0,PnP=0,existing_artifact_edits=0),read_packet_paths=sorted(set(READS)))
    out=R/'INDEPENDENT_SAVED_METRICS_RECONCILIATION.json'
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    with (R/'RECONCILIATION_SEAL.json').open('x') as f:json.dump(dict(completed=True,PASS=not differences,files=[binding(POLICY),binding(__file__),binding(out)],existing_artifacts_modified=False),f,indent=2);f.write('\n')
    print(json.dumps(dict(PASS=not differences,path=str(out),sha256=sha(out),compared_scalar_leaves=len(checks),discrepancies=differences,YCB_qbit_oracle=ycb['primary']['CM_rows_GT_CV_cols_pred_CVU'],strict535=strict_sam['CM_rows_GT_CV_cols_pred_CVU'])))

if __name__=='__main__':main()

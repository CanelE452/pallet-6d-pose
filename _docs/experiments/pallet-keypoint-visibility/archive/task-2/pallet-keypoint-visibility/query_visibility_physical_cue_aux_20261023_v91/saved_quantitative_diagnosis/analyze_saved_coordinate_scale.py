"""Post-freeze saved CSV/JSON diagnosis only. No model/feature/RGB libraries."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import itertools
import json
import math

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=ROOT/'contract/FROZEN_CURRENT_COMPARISON'
ENTRY=ROOT/'contract/ROOT_APPROVED_CURRENT_EVALUATION_ENTRY_V1.json'
EVAL_SEAL=OUT/'EVALUATION_SEAL.json'
SOURCE=ROOT/'runs/PHYSICAL_CUE_AUXILIARY_V1/SOURCE_RESULTS.json'
GROUPS=['WOOD_PRIMARY105','SQUARE118']
ERRORS=['LE3','GT3_LE10','GT10_LE30','GT30','MISSING_OR_NONFINITE']
REFS=['INSIDE','OUTSIDE','MISSING_OR_NONFINITE']
LS=['LT32','GE32_LT64','GE64_LT128','GE128','MISSING_OR_NONFINITE']
D='DIRECT_VISIBLE';E='EXTERNAL_OCCLUDED';S='SELF_OCCLUDED'


def bind(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def read(p):return json.loads(Path(p).read_text())


def check(b):assert bind(b['path'])['sha256']==b['sha256'],b['path']


def readcsv(p):
    with Path(p).open(newline='')as f:return list(csv.DictReader(f))


def write(name,value):
    with(HERE/name).open('x')as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def writecsv(name,rows):
    with(HERE/name).open('x',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def number(x):
    try:v=float(x);return v if math.isfinite(v)else None
    except(ValueError,TypeError):return None


def yes(x):return str(x).lower()=='true'


def key(r):return r['id'],int(r['native_corner'])


def eb(e):return ERRORS[4]if e is None else ERRORS[0]if e<=3 else ERRORS[1]if e<=10 else ERRORS[2]if e<=30 else ERRORS[3]


def lb(L):return LS[4]if L is None else LS[0]if L<32 else LS[1]if L<64 else LS[2]if L<128 else LS[3]


def ratio(a,b):return a/b if b else None


def stats(rows,route):
    c=Counter((r['human_state'],r[route])for r in rows)
    d=[c[D,p]for p in('VISIBLE','NONVISIBLE','U')];e=[c[E,p]for p in('VISIBLE','NONVISIBLE','U')];s=[c[S,p]for p in('VISIBLE','NONVISIBLE','U')]
    return dict(N=len(rows),D_N=sum(d),E_N=sum(e),S_N=sum(s),OTHER_HUMAN_N=len(rows)-sum(d+e+s),
        D_visible=d[0],D_false_absence=d[1],D_U=d[2],E_false_visible=e[0],E_nonvisible=e[1],E_U=e[2],
        S_visible=s[0],S_nonvisible=s[1],S_U=s[2],
        D_recall_all_GT=ratio(d[0],sum(d)),E_nonvisible_recall_all_GT=ratio(e[1],sum(e)),
        D_precision_conditional_known_DE=ratio(d[0],d[0]+e[0]),E_precision_conditional_known_DE=ratio(e[1],e[1]+d[1]),
        unresolved_N=sum(r[route]=='U'for r in rows),unresolved_rate=ratio(sum(r[route]=='U'for r in rows),len(rows)))


def main():
    assert (HERE/'SCOPE_BEFORE_ROW_ANALYSIS.json').exists()
    entry=read(ENTRY);seal=read(EVAL_SEAL);assert seal['completed']and seal['PASS']
    check(seal['report']);check(seal['inference'])
    result=read(seal['report']['path']);assert result['completed']and result['no_current_fit_threshold_checkpoint_reselection']
    for name in('scores','evaluation_manifest','manual_provenance','inference_seal','preparation_seal'):check(entry[name])
    truth=readcsv(entry['evaluation_manifest']['path']);sc=readcsv(entry['scores']['path']);prov=readcsv(entry['manual_provenance']['path'])
    assert len(truth)==len(sc)==1784 and len({key(r)for r in truth})==1784 and len({key(r)for r in sc})==1784
    sm={key(r):r for r in sc};pm={key(r):r for r in prov};assert len(pm)==len(prov)
    assert [key(r)for r in sc]==[key(r)for r in truth]
    joined=[];error_diffs=[];inside_disagreement=[]
    for t in truth:
        s=sm[key(t)]
        assert s['image_sha256']==t['image_sha256']and float(s['query_x'])==float(t['query_x'])and float(s['query_y'])==float(t['query_y'])
        L=number(s['roi_side']);assert L==number(t['v61_roi_side_raw_px'])
        qx,qy,rx,ry=map(number,[t['query_x'],t['query_y'],t['human_reference_x'],t['human_reference_y']])
        error=number(t['initial_reference_error_px']);numeric=None
        if all(x is not None for x in(qx,qy,rx,ry,L)):
            inside='INSIDE'if qx-L/2<=rx<qx+L/2 and qy-L/2<=ry<qy+L/2 else'OUTSIDE'
            numeric=math.hypot(qx-rx,qy-ry)
            if error is not None:error_diffs.append(abs(error-numeric))
            old=t['human_reference_inside_nominal_crop']
            if old and yes(old)!=(inside=='INSIDE'):inside_disagreement.append(key(t))
        else:inside='MISSING_OR_NONFINITE'
        valid=yes(s['valid']);raw=number(s['raw_physical_external_logit']);assert raw is not None
        pred='U'if not valid else'VISIBLE'if raw<0 else'NONVISIBLE'
        assert s['authoritative_decision']==('UNKNOWN'if pred=='U'else'D'if pred=='VISIBLE'else'E')
        geo=valid and yes(s['geometry_self_hidden_if_valid']);hybrid='U'if not valid else'NONVISIBLE'if geo else pred
        p=pm.get(key(t));manual=bool(p and p.get('annotation_coordinate_source')=='manual_click'and t['human_state']==D and yes(t['query_valid'])and error is not None and error<=3)
        if p:
            assert p['image_sha256']==t['image_sha256']and p['query_x']==t['query_x']and p['query_y']==t['query_y']
        joined.append(dict(id=t['id'],native_corner=int(t['native_corner']),group=t['group'],human_state=t['human_state'],recording=t['recording'],
            image_sha256=t['image_sha256'],query_x=qx,query_y=qy,reference_x_GT_only=rx,reference_y_GT_only=ry,
            original_error_px=error,numeric_q_reference_error_audit_px=numeric,error_bin=eb(error),reference_crop=inside,
            original_manifest_reference_inside=t['human_reference_inside_nominal_crop'],L=L,L_bin=lb(L),
            RAW=pred,HYBRID_OLD_GEOMETRY=hybrid,raw_zE=raw,valid=valid,prior_valid=yes(s['prior_valid']),centre_valid=yes(s['center_valid']),
            paired_D_E=yes(t['paired_same_query_RGB_D_vs_E'])and t['human_state']in(D,E),
            same_slot_D_E=yes(t['same_slot_pipeline_D_vs_E'])and t['human_state']in(D,E),
            manual_near_D=manual,coordinate_provenance=(p.get('annotation_coordinate_source')if p else'NOT_PROVEN_BY_THIS_SIDE_CAR'),
            v61_saved_visible=('VISIBLE'if yes(t['present_at_fixed_0_5'])else'NONVISIBLE')if t['present_at_fixed_0_5']else'U',
            model_or_query_changed=False))
    assert sum(r['manual_near_D']and r['group']==GROUPS[0]for r in joined)==153
    assert sum(r['manual_near_D']and r['group']==GROUPS[1]for r in joined)==223
    scopes={'ALL_NATIVE_SLOTS':joined,'PAIRED_D_VS_E':[r for r in joined if r['paired_D_E']],
            'SAME_SLOT_D_VS_E':[r for r in joined if r['same_slot_D_E']], 'MANUAL_NEAR_D':[r for r in joined if r['manual_near_D']]}
    primary=[];marginal=[];cross=[]
    for scope,rr in scopes.items():
        for group,route in itertools.product(GROUPS,('RAW','HYBRID_OLD_GEOMETRY')):
            rows=[r for r in rr if r['group']==group]
            primary.append(dict(scope=scope,group=group,route=route,**stats(rows,route)))
            for dimension,levels in [('error_bin',ERRORS),('reference_crop',REFS),('L_bin',LS)]:
                for level in levels:marginal.append(dict(scope=scope,group=group,route=route,dimension=dimension,level=level,**stats([r for r in rows if r[dimension]==level],route)))
            for er,ref,Lbin in itertools.product(ERRORS,REFS,LS):
                cross.append(dict(scope=scope,group=group,route=route,error_bin=er,reference_crop=ref,L_bin=Lbin,**stats([r for r in rows if r['error_bin']==er and r['reference_crop']==ref and r['L_bin']==Lbin],route)))
    expected={GROUPS[0]:(586,616,11,29),GROUPS[1]:(472,531,30,61)}
    for group,values in expected.items():
        p=next(r for r in primary if r['scope']=='PAIRED_D_VS_E'and r['group']==group and r['route']=='RAW')
        assert (p['D_visible'],p['D_N'],p['E_nonvisible'],p['E_N'])==values
    manual=[]
    for group in GROUPS:
        rr=[r for r in scopes['MANUAL_NEAR_D']if r['group']==group]
        transitions=Counter(('v61_correct'if r['v61_saved_visible']=='VISIBLE'else'v61_wrong_or_U',
                             'V91_correct'if r['RAW']=='VISIBLE'else'V91_wrong_or_U')for r in rr)
        manual.append(dict(group=group,N=len(rr),V91_RAW=stats(rr,'RAW'),V91_HYBRID=stats(rr,'HYBRID_OLD_GEOMETRY'),
            v61_visible=sum(r['v61_saved_visible']=='VISIBLE'for r in rr),v61_nonvisible=sum(r['v61_saved_visible']=='NONVISIBLE'for r in rr),
            transitions={a+'__'+b:n for(a,b),n in transitions.items()},all_reference_inside=all(r['reference_crop']=='INSIDE'for r in rr),
            meaning='Manual-click human D only; not source emulated-q near3 or dense material-pixel GT.'))
    writecsv('ALL1784_SAVED_COORDINATE_SCALE_ROWS.csv',joined)
    writecsv('PRIMARY_AND_CONTEXT_COUNTS.csv',primary);writecsv('MARGINAL_BINS.csv',marginal);writecsv('CROSS_ERROR_CROP_L_BINS.csv',cross)
    source=read(SOURCE);prep=read(entry['preparation_seal']['path']);inference=read(entry['inference_seal']['path'])
    fit=read(ROOT/'FIT_PROTOCOL_BEFORE_SOURCE_ARRAYS.json');packpath=fit['bindings']['source_pack_manifest'];check(packpath);pack=read(packpath['path'])
    source_dtype=pack['chunks'][0]['schema']['p3_roi']['dtype']
    assert all(c['schema']['p3_roi']['dtype']==source_dtype for c in pack['chunks'])
    by_group_L={g:dict(Counter(r['L_bin']for r in joined if r['group']==g))for g in GROUPS}
    salient=[]
    for group in GROUPS:
        rr=[r for r in scopes['PAIRED_D_VS_E']if r['group']==group]
        for label,name in[(D,'humanD'),(E,'humanE')]:
            allrows=[r for r in rr if r['human_state']==label]
            correct=lambda r:r['RAW']==('VISIBLE'if label==D else'NONVISIBLE')
            wrong=[r for r in allrows if r['RAW']!='U'and not correct(r)]
            salient.append(dict(group=group,label=name,N=len(allrows),decided_errors=len(wrong),
                errors_LE3=sum(r['error_bin']=='LE3'for r in wrong),errors_LE3_and_inside=sum(r['error_bin']=='LE3'and r['reference_crop']=='INSIDE'for r in wrong),
                errors_outside=sum(r['reference_crop']=='OUTSIDE'for r in wrong),errors_inside=sum(r['reference_crop']=='INSIDE'for r in wrong),
                errors_by_L=dict(Counter(r['L_bin']for r in wrong)),all_by_L=dict(Counter(r['L_bin']for r in allrows))))
    limits=[
        'Source physical structural cue D/E teachers and current human native-corner D/E/S are different annotation targets; source native8 mapping remains UNKNOWN.',
        'Source validation constructs a native mosaic from every129 fixed tile before scoring at fixed r or9-point uncertainty region. Current inference is one unchanged initializer q crop, centre(4,4), no mosaic or GT-snapped q.',
        'Source query centres are a GT-free uniform grid; a source cue may be scored away from any tile centre. No GT-besttile/source-centre label metric is invented here.',
        'Current error bins use the original stored reference error, and mix coordinate provenance unless explicitly marked manual_click. A point inside nominal crop does not certify that its physical cue is visible or discernible.',
        'Current E references may be hidden/projected semantic locations. A small q-r distance is not dense target-pixel GT, observed foreign ownership, or proof that RGB contains the same source cue.',
        'Associations with L, error or crop containment are observational, with differing object type, scene, human state, recording and label composition. No isolated scale, dtype, field-of-view or target-identity causal effect is established.',
        'Stored source P3 ROI dtypeF16 vs current preservedF32 is an input-storage fact; historical values may already contain FP16 rounding. No additional current quantization was applied and dtype alone is not diagnosed as the cause.',
        'All results are post-freeze diagnostics of a historically reused current evaluation. No threshold/checkpoint/coordinate changes, alternative inference or performance-based selection occurred.',
    ]
    report=dict(completed=True,PASS=True,scope=bind(HERE/'SCOPE_BEFORE_ROW_ANALYSIS.json'),analysis_code=bind(__file__),
        bindings=dict(entry=bind(ENTRY),evaluation_seal=bind(EVAL_SEAL),evaluation_report=seal['report'],scores=entry['scores'],truth=entry['evaluation_manifest'],
                      manual_provenance=entry['manual_provenance'],inference=entry['inference_seal'],preparation=entry['preparation_seal'],source_results=bind(SOURCE),source_input_pack=packpath),
        all_native_rows=1784,all_native_valid=int(sum(r['valid']for r in joined)),all_native_U=int(sum(not r['valid']for r in joined)),
        primary_counts=primary,manual_near_D=manual,paired_error_salient_counts=salient,
        coordinate_audit=dict(stored_error_used_authoritatively=True,numeric_recompute_N=len(error_diffs),max_abs_error_difference=max(error_diffs,default=0),
            original_manifest_vs_exact_halfopen_crop_disagreement_N=len(inside_disagreement),disagreement_keys=inside_disagreement),
        source_vs_current=dict(source_VAL=source['source_metrics']['VAL_HALL'],
            source_operator='All129tile native mosaic then fixed9 uncertainty region mean logit; r-only separate',
            current_operator='One original q/L tile; raw map[4,4] only; old factualU and newcentreU unresolved first',
            source_L_values=[128,256],source_nominal_map_cell_spacing=[128/9,256/9],
            current_L_min=prep['original_L_min'],current_L_max=prep['original_L_max'],current_L_percentiles=prep['original_L_percentiles'],
            current_L_below_source_min_N=prep['native_L_below_source_min_N'],current_L_above_source_max_N=prep['native_L_above_source_max_N'],
            current_L_bins_by_group=by_group_L,current_nominal_map_cell_spacing_range=[prep['original_L_min']/9,prep['original_L_max']/9],
            source_stored_P3_ROI_dtype=source_dtype,current_P3_dtype_preserved=prep['original_P3_dtype'],current_P3_requantized=prep['P3_input_requantization'],
            source_independent_object_generalization=False,source_one_physical_bin_and_weakAI_conditional_cues=True,
            common_source_vs_current_label_or_operator_assumption=False),
        limitations=limits,RGB_pixels_or_feature_tensors_or_model_weights_read=0,new_forward_training_thresholdsearch_PnP=0,
        checkpoint_query_L_decision_or_source_order_modified=False,
        outputs=[bind(HERE/n)for n in ['ALL1784_SAVED_COORDINATE_SCALE_ROWS.csv','PRIMARY_AND_CONTEXT_COUNTS.csv','MARGINAL_BINS.csv','CROSS_ERROR_CROP_L_BINS.csv']])
    write('SAVED_COORDINATE_SCALE_DIAGNOSIS.json',report)
    write('FINAL_SEAL.json',dict(completed=True,PASS=True,report=bind(HERE/'SAVED_COORDINATE_SCALE_DIAGNOSIS.json'),
        files=[bind(p)for p in sorted(HERE.iterdir())if p.is_file()],new_forward_fit_thresholdsearch_PnP=0,source_current_targets_equivalent=False))
    print(json.dumps(dict(report=bind(HERE/'SAVED_COORDINATE_SCALE_DIAGNOSIS.json'),manual=manual,salient=salient),ensure_ascii=False,indent=2))


if __name__=='__main__':main()

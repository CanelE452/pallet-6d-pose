"""Root-authorized completed-summary/CM-CSV QA only; no raw rows/arrays/GT/model."""
from pathlib import Path
from collections import Counter
import csv,hashlib,io,json,math,os,re,stat
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
SOURCE=P/'query_visibility_dense_point_weak_bridge_source_training_20261025_v212'
TAGS=('C62','C63')
ARMS=('C62_DENSE_POINT_RAW_Q8','C63_DENSE_POINT_WEAK_BRIDGE_RAW_Q8')
OPENED=[]
def need(ok,msg):
    if not bool(ok):raise AssertionError(msg)
def path(p):
    p=Path(p).absolute();need(P in p.parents and '..'not in p.parts and all('v112'not in x.casefold()for x in p.parts),'Lexical path before FS')
    need(p.suffix.lower()not in('.npz','.npy','.pt','.png','.jpg','.jpeg','.svg'),'No binary/source/image access');return p
def opened(p,create=False):
    p=path(p);fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            n=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        n=os.open(p.name,(os.O_WRONLY|os.O_CREAT|os.O_EXCL if create else os.O_RDONLY)|os.O_NOFOLLOW,0o644,dir_fd=fd)
    finally:os.close(fd)
    need(stat.S_ISREG(os.fstat(n).st_mode),'No symlink/regular explicit file');return os.fdopen(n,'wb'if create else'rb')
def raw(p):
    p=path(p);OPENED.append(str(p))
    with opened(p)as f:return f.read()
def bind(p):return dict(path=str(path(p)),sha256=hashlib.sha256(raw(p)).hexdigest())
def read(p):return json.loads(raw(p))
def checked(b):
    need(bind(b['path'])=={k:b[k]for k in('path','sha256')},'Completed report/table binding');return path(b['path'])
def write(p,d):
    with opened(p,True)as f:f.write((json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def ratio(a,b):return a/b if b else None
def cm_check(cm,d,e,metrics):
    need(len(cm)==2 and all(len(r)==3 and all(type(v)is int and v>=0 for v in r)for r in cm),'2x3 nonnegative integer CM')
    need(sum(cm[0])==d and sum(cm[1])==e,'Original D/E denominators include U')
    expected=dict(D_precision=ratio(cm[0][0],cm[0][0]+cm[1][0]),D_recall=ratio(cm[0][0],d),E_precision=ratio(cm[1][1],cm[1][1]+cm[0][1]),E_recall=ratio(cm[1][1],e))
    for k,v in expected.items():need(metrics[k]==v,'Saved metric algebra '+k)
    return expected
def sumcm(rows):return [[sum(r[i][j]for r in rows)for j in range(3)]for i in range(2)]
def local_links(base,md):
    need('/home/'not in md and 'file://'not in md,'Portable rendered text')
    refs=re.findall(r'\]\(([^)]+)\)',md)
    for ref in refs:
        need(ref and not ref.startswith(('/','//'))and not re.match(r'^[A-Za-z]+:',ref)and'v112'not in ref.casefold(),'Relative portable target')
        # Existence only. Do not read linked per-row prediction tables or logs.
        with opened(os.path.normpath(base/ref)):pass
    return len(refs)
def main():
    tests=[]
    def test(n,ok):need(ok,n);tests.append(n)
    source_done=read(SOURCE/'source_fit_report_preparation_b/SOURCE_REPORT_BUILD_COMPLETE.json')
    test('Source_report_completed',source_done['completed']and source_done['PASS'])
    source=read(checked(source_done['outputs']['ROOT_RESULTS_KO.json']));source_md=raw(checked(source_done['outputs']['PORTABLE_REPORT_KO.md'])).decode()
    source_links=local_links(SOURCE,source_md)
    test('Source_portable_and_root_MD_same_bound_bytes',source_done['outputs']['PORTABLE_REPORT_KO.md']['sha256']==source_done['outputs']['ROOT_REPORT_KO.md']['sha256'])
    test('Source_model_and_raw_read_scope_claims',source['actual_prediction_NPZs_opened']==4 and source['model_checkpoint_or_source_input_or_target_NPZ_GT_RGB_or_private_reads']==0 and source['claims']['source_apparent_not_validation']and source['claims']['weak44_not_independent_for_C62'])
    csvrows=list(csv.DictReader(io.StringIO(raw(checked(source_done['tables']['SOURCE_APPARENT_CM_P_R_U.csv'])).decode())))
    sm={(r['domain'],r['model']):r for r in source['metrics']};test('Four_source_rows_and_CSV',len(csvrows)==len(sm)==4)
    table_rows=[]
    for c in csvrows:
        r=sm[(c['domain'],c['model'])];cm=r['CM_D_E_by_D_E_U']
        test('Source_CSV_JSON_CM_'+r['domain']+'_'+r['model'],json.loads(c['CM_D_E_by_D_E_U'])==cm and int(c['known_N'])==r['known_N']and int(c['D_N'])==r['D_N']and int(c['E_N'])==r['E_N']and all(float(c[k])==r[k]for k in('D_precision','D_recall','E_precision','E_recall')))
        cm_check(cm,r['D_N'],r['E_N'],r)
        test('Source_full_known_unknown_conserved_'+r['domain']+'_'+r['model'],r['known_N']+r['UNKNOWN_target_rows']==r['all_rows']and sum(r['UNKNOWN_target_predictions'].values())==r['UNKNOWN_target_rows'])
        rr=[x['CM_D_E_by_D_E_U']for x in source['native8']if x['domain']==r['domain']and x['model']==r['model']]
        test('Source_native8_sums_'+r['domain']+'_'+r['model'],len(rr)==8 and sumcm(rr)==cm)
        if r['domain']=='POINT7040':test('All7040_6096_D5853_E243_UNKNOWN944_inputU64_'+r['model'],(r['all_rows'],r['known_N'],r['D_N'],r['E_N'],r['UNKNOWN_target_rows'],r['original_input_U'])==(7040,6096,5853,243,944,64))
        else:test('Weak44_D26_E18_not_validation_'+r['model'],(r['known_N'],r['D_N'],r['E_N'])==(44,26,18))
    for row in source['paired']:
        test('Source_paired_conservation_'+row['domain'],sum(row[k]for k in('both_correct','recovered_by_C63','damaged_by_C63','both_wrong'))==row['known_N']and row['recovered_by_C63']-row['damaged_by_C63']==row['net_correct_change'])
    hold_done=read(V/'source_point_holdout_evaluation_v2/COMPLETE.json');test('HOLD_V2_completed',hold_done['completed']and hold_done['PASS'])
    hold=read(checked(hold_done['outputs']['SUMMARY.json']))
    test('SOURCE_HOLD18groups_72scenes_1152rows',hold['cohort_counts']['parents']==18 and hold['cohort_counts']['scenes']==72 and hold['cohort_counts']['eligible_inputs']==1152 and len(hold['strata']['base_group'])==18)
    test('HOLD_known938_D899_E39_UNKNOWN214',hold['known_evaluation_targets']=={'0':899,'-1':214,'1':39})
    real_done_path=V/'saved_real_head_evaluation_v1/COMPLETE.json';real_done=read(real_done_path);qa=read(V/'independent_saved_result_review_a/actual_review_v1/REVIEW.json')
    test('Real_saved_and_independent_exact_complete',real_done['completed']and real_done['PASS']and qa['completed']and qa['PASS']and qa['evaluation_complete']==bind(real_done_path))
    realmetrics=list(csv.DictReader(io.StringIO(raw(checked(real_done['outputs']['current1784/METRICS.csv'])).decode())))
    for tag,arm in zip(TAGS,ARMS):
        tr=sm[('POINT7040',tag)];wr=sm[('WEAK44',tag)];hr=hold['models'][arm];rr=qa['current']['primary']['ALL'][tag]
        hm=dict(zip(('D_precision','E_precision'),hr['precision_D_E']));hm.update(zip(('D_recall','E_recall'),hr['recall_D_E']))
        cm_check(hr['confusion_D_E_by_D_E_U'],899,39,hm)
        test('HOLD_group_CMs_sum_'+tag,sumcm([g[arm]['confusion_D_E_by_D_E_U']for g in hold['strata']['base_group'].values()])==hr['confusion_D_E_by_D_E_U'])
        test('HOLD1152_known938_UNKNOWN214_'+tag,(hr['all_rows'],hr['known_truth'],hr['unknown_truth'])==(1152,938,214))
        cm_check(rr['CM_D_E_by_D_E_U'],1147,90,rr);test('Real1237_D1147_E90_'+tag,(rr['N'],rr['D_N'],rr['E_N'])==(1237,1147,90))
        old=[x for x in realmetrics if x['model']==tag and x['cohort']=='paired_same_query_RGB'and x['task']=='D_vs_E']
        def n(r,k):return int(r[k])
        cm=[[sum(n(x,'TP')for x in old),sum(n(x,'FN_decided')for x in old),sum(n(x,'U_visible')+n(x,'out_of_task_S_visible')+n(x,'other_unresolved_visible')for x in old)],[sum(n(x,'FP')for x in old),sum(n(x,'TN')for x in old),sum(n(x,'U_occluded')+n(x,'out_of_task_S_occluded')+n(x,'other_unresolved_occluded')for x in old)]]
        test('Real_material_CSV_equals_numeric_CM_'+tag,len(old)==2 and cm==rr['CM_D_E_by_D_E_U'])
        for cohort,row,cmkey,denom in [('SOURCE TRAIN point',tr,'CM_D_E_by_D_E_U',(5853,243)),('SOURCE weak44',wr,'CM_D_E_by_D_E_U',(26,18)),('SOURCE HOLD point',dict(hr,**hm),'confusion_D_E_by_D_E_U',(899,39)),('실사 paired',rr,'CM_D_E_by_D_E_U',(1147,90))]:
            cm=row[cmkey];table_rows.append(dict(cohort=cohort,model=tag,N=sum(denom),D=denom[0],E=denom[1],CM=cm,D_precision=row['D_precision'],D_recall=row['D_recall'],E_precision=row['E_precision'],E_recall=row['E_recall'],known_U=cm[0][2]+cm[1][2]))
    test('Same_source_fit_across_train_and_hold',source['source_complete']==hold_done['fit_complete'])
    test('Source_report_not_HOLD_or_real_scores',source['claims']['point_HOLDOUT18_not_read_or_merged']and source['claims']['selected_model']is None and source['claims']['adopted']is False)
    pct=lambda x:'%.2f%%'%(100*x)if x is not None else'UNKNOWN'
    lines=['# SOURCE TRAIN·SOURCE HOLDOUT·실사 평가의 간격','', '학습에 사용한 SOURCE point에서는 거의 모두 맞혔지만, SOURCE HOLDOUT과 실사에서는 특히 E recall이 낮다. 이 수치 간격은 관측됐으나 감독 의미·입력 분포·노출 차이가 함께 있어 원인이나 실사 개선 효과를 분리해 입증하지는 못한다. C63은 결과 전 지정한 주 모델이고 C62는 point-only 대조 팔이며, 두 모델을 모두 보고한다.','', 'CM 행은 D/E, 열은 예측 D/E/U다. Recall은 각 기존 known 클래스의 전체 분모를 사용한다. 서로 다른 집합의 정확도를 합치지 않는다.','', '|집합|모델|N (D/E)|CM [D행; E행]|D precision / recall|E precision / recall|known U|','|---|---|---|---|---|---|---|']
    for r in table_rows:lines.append('|%s|%s|%d (%d/%d)|`%s`|%s / %s|%s / %s|%d|'%(r['cohort'],r['model'],r['N'],r['D'],r['E'],json.dumps(r['CM'],separators=(',',':')),pct(r['D_precision']),pct(r['D_recall']),pct(r['E_precision']),pct(r['E_recall']),r['known_U']))
    lines+=['','SOURCE TRAIN point는 원7,040행을 보존하고, fit-eligible6,096(D5,853/E243)과 UNKNOWN-loss944를 분리한 apparent fit이다. 전체 원 inputU64는 UNKNOWN 쪽에 남는다. Weak44는 공개 SOURCE TRAIN 3frame의 factual8·SHAM18 D와 COVER18 weak E다. C63의44/44는 직접 학습한 약한 감독 적합도이며, C62도 같은 원frame을 old auxiliary에서 보았으므로 C62의32/44 역시 독립 검증으로 부르지 않는다.','', 'SOURCE HOLDOUT은18 component/base groups·72장면·576개 own native 상태에서 나온1,152 input rows다. Known938(D899/E39)과 UNKNOWN214를 구분하며, 제외사유는 중복될 수 있다. 원 inputU80도 제외사유에 보존된다. 같은 base·variant·native와 anchor 문맥에서 반복된 행이므로938개 독립 실사 사례가 아니다. 기존에 알려진 SOURCE 설계와 역사적 노출이 있는 합성 내부 holdout으로, fresh blind 검증이나 공개 실사 holdout으로 일반화하지 않는다.','', '실사 paired는 기존 human D/E1,237(D1,147/E90)이다. C62→C63에서 D 정답은709→843(+134), E 정답은26→19(−7)로 바뀌었다. SOURCE HOLDOUT에서는 D881→885(+4), E30→30이었다. SOURCE TRAIN의 E100%가 실사 E100%를 뜻하지 않으며, C63의 D 증가와 E 감소를 함께 보아야 한다. Legacy보다 개선됐다는 결론이나 더 좋은 팔의 사후 채택은 이 문서에서 하지 않는다.','', '이전 finite-end327에서 dense point6,096으로 바뀔 때 target 정의와 데이터 노출도 함께 바뀌었다. 이를 순수 라벨 하나의 효과로 해석하지 않는다. C62/C63는 같은 구조·fresh init·old192/point32 draw·4,000회 갱신에 대해 C63만 추가 weak branch(.25)를 쓰는 사전 대조다. SOURCE procedural point와 public local-feature weak target은 인간 native visibility gold와 다르며, source→real 간격만으로 shortcut·국소 정보 소실·라벨 문제 등 특정 원인을 확정하지 않는다.','', '표의 known U0는 availability/유한 logit 이후 고정0 문턱으로 이진 응답을 했다는 뜻이다. Confidence·calibration·불확실성 해결의 증거가 아니다. 실사 전체1784의 원 inputU45와 주평가 U0도 별개다. 반복 노출된 실사/HOLDOUT 수치로 모델·데이터·문턱·checkpoint를 조정하거나 선택하지 않았고, 이 문서도 새로운 최적화를 수행하지 않는다.','', '## 저장 보고서·표','']
    refs=[('SOURCE TRAIN/weak 보고서',SOURCE/'PORTABLE_REPORT_KO.md'),('SOURCE CM/P/R/U 표',source_done['tables']['SOURCE_APPARENT_CM_P_R_U.csv']['path']),('SOURCE native8 표',source_done['tables']['SOURCE_NATIVE8_CM_P_R_U.csv']['path']),('SOURCE HOLDOUT V2 요약',V/'source_point_holdout_evaluation_v2/SUMMARY.json'),('SOURCE HOLDOUT 완료',V/'source_point_holdout_evaluation_v2/COMPLETE.json'),('실사 전체 모델 저장 metric',real_done['outputs']['current1784/METRICS.csv']['path']),('실사 독립 수치 QA',V/'independent_saved_result_review_a/actual_review_v1/REVIEW.json'),('고정12 갤러리',V/'fixed12_dense_point_real_visual_v1/PORTABLE_GALLERY_KO.md')]
    for label,p in refs:lines.append('- [%s](%s)'%(label,Path(os.path.relpath(p,V)).as_posix()))
    md='\n'.join(lines)+'\n';linkcount=local_links(V,md)
    textpath=V/'ROOT_SOURCE_TRAIN_HOLDOUT_REAL_GAP_KO.md'
    with opened(textpath,True)as f:f.write(md.encode())
    source_checks=dict(completed=True,PASS=True,scope='SOURCE generated report scalar/text QA plus cross-cohort summary consistency; no raw prediction/GT/model read',source_report_build_complete=bind(SOURCE/'source_fit_report_preparation_b/SOURCE_REPORT_BUILD_COMPLETE.json'),source_ROOT_RESULTS=source_done['outputs']['ROOT_RESULTS_KO.json'],source_PORTABLE=source_done['outputs']['PORTABLE_REPORT_KO.md'],source_CM_CSV=source_done['tables']['SOURCE_APPARENT_CM_P_R_U.csv'],source_portable_link_occurrences=source_links,source_native8_CM_totals_match=True,source_unknown_and_inputU_separate=True)
    qaout=dict(completed=True,PASS=True,code=bind(Path(__file__)),document=bind(textpath),source_report_QA=source_checks,source_hold_complete=bind(V/'source_point_holdout_evaluation_v2/COMPLETE.json'),source_hold_summary=hold_done['outputs']['SUMMARY.json'],real_complete=bind(real_done_path),real_numeric_QA=bind(V/'independent_saved_result_review_a/actual_review_v1/REVIEW.json'),count=len(tests),checks=tests,table_rows=table_rows,portable_link_occurrences=linkcount,files_read_for_content=sorted(set(OPENED)),raw_rows_or_arrays_or_GT_or_weights_or_original_images_read=0,new_model_forward_fit_or_tuning=0,selected_model=None,adopted=False,causal_or_legacy_improvement_claim=False)
    write(H/'REVIEW.json',qaout)
    print(json.dumps(dict(completed=True,PASS=True,checks=len(tests),document=bind(textpath),review=bind(H/'REVIEW.json'))))
if __name__=='__main__':main()

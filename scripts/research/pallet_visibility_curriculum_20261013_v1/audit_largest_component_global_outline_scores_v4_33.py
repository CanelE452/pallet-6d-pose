"""Score-JSON-only recount for the uniform V33 mask control.

No reference, prediction, image, mask, optimizer or model values are loaded.
The frozen V32 score accounting provides independent arithmetic. Native/e
metric invariance and both gains/losses of the changed C arms are reported.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json

from . import audit_clean_fresh_BI_scores_v4_32 as A

ROOT=A.ROOT
NEW=A.B/'largest_component_global_outline_pose_eval_v4_33'
OLD=A.NEW
OUTPUT=A.B/'independent_largest_component_global_outline_score_review_v4_33'
OLD_SEAL_SHA='7bcc9bce3e8b14a87021955c513e3061dcb561cc309fabd61964e06efbcc6c80'
ARMS=A.ARMS
INVARIANT_ARMS=(ARMS[0],ARMS[1],ARMS[3])


def valid_sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def compare_invariants(before,after,checks):
    checks.check([r['id'] for r in before]==[r['id'] for r in after],'V32/V33 exact ordered IDs')
    for old,new in zip(before,after):
        for arm in INVARIANT_ARMS:
            checks.check(old['metrics'][arm]==new['metrics'][arm],'exact unchanged metrics '+new['id']+'/'+arm)
            checks.check(old['accepted'][arm]==new['accepted'][arm],'exact unchanged accepted '+new['id']+'/'+arm)
            checks.check(old['pose_available'][arm]==new['pose_available'][arm],'exact unchanged pose availability '+new['id']+'/'+arm)


def paired_C_changes(before,after,arm):
    if [r['id'] for r in before]!=[r['id'] for r in after]:
        raise ValueError('Exact paired cohort order required')
    gains=[];losses=[];changes=[]
    for old,new in zip(before,after):
        a=old['metrics'][arm];b=new['metrics'][arm]
        item=dict(id=new['id'],old_translation_cm=a['translation_cm'],translation_cm=b['translation_cm'],
            old_rotation_deg=a['rotation_deg'],rotation_deg=b['rotation_deg'])
        oldpass=bool(a['pose_available'] and a['translation_cm'] is not None and a['translation_cm']<=5.)
        newpass=bool(b['pose_available'] and b['translation_cm'] is not None and b['translation_cm']<=5.)
        if newpass and not oldpass:gains.append(item)
        if oldpass and not newpass:losses.append(item)
        if a!=b:changes.append(item)
    return dict(translation_threshold_gains=gains,translation_threshold_losses=losses,metric_changes=changes,
        scope='Fixed paired control; postscore reporting only, never row routing or variant selection')


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def run(*,root_score_review_GO=False,expected_score_seal_sha256=None,
        expected_prediction_seal_sha256=None,output=OUTPUT):
    if root_score_review_GO is not True:raise PermissionError('ROOT V33 postscore review GO before IO')
    if not all(valid_sha(v) for v in (expected_score_seal_sha256,expected_prediction_seal_sha256)):
        raise ValueError('Explicit score and prediction SHA before IO')
    output=Path(output)
    if output.exists():raise FileExistsError('Preserve all score recount attempts')
    if (A.sha(NEW/'SCORE_SEAL.json')!=expected_score_seal_sha256
            or A.sha(NEW/'PREDICTION_SEAL.json')!=expected_prediction_seal_sha256
            or A.sha(OLD/'SCORE_SEAL.json')!=OLD_SEAL_SHA):
        raise ValueError('Exact current new and frozen V32 seals required')
    score=A.read(NEW/'SCORE_SEAL.json');oldseal=A.read(OLD/'SCORE_SEAL.json')
    if score['prediction_seal']['sha256']!=expected_prediction_seal_sha256 or score['all_rows_kept'] is not True:
        raise ValueError('Complete sealed all-row prediction/score required')
    sources={str(p.relative_to(ROOT)):A.sha(p) for p in (Path(__file__).resolve(),
        Path(__file__).with_name('test_audit_largest_component_global_outline_scores_v4_33.py').resolve(),Path(A.__file__).resolve(),
        Path(A.__file__).with_name('test_audit_clean_fresh_BI_scores_v4_32.py').resolve())}
    inputs={str(p.relative_to(ROOT)):A.sha(p) for folder in (NEW,OLD)
        for p in [folder/'SCORE_SEAL.json',*[folder/(part+'_SCORE.json') for part in A.COUNTS]]}
    inputs[str((NEW/'PREDICTION_SEAL.json').relative_to(ROOT))]=expected_prediction_seal_sha256
    output.mkdir(parents=True)
    write(output/'PLAN.json',dict(source_sha256=sources,input_sha256=inputs,
        scope='Score JSON arithmetic only; no raw reference/prediction/mask/RGB or producer import',numeric_score_rows_read_before_plan=0))
    reports,bindings=A.load_score_files(NEW,score,A.COUNTS)
    oldreports,oldbindings=A.load_score_files(OLD,oldseal,A.COUNTS)
    checks=A.Checks();populations={}
    checks.check(score['default']=='SAME_FROZEN_E unchanged','default unchanged')
    for part,count in A.COUNTS.items():
        report=reports[part]
        checks.check(report['count']==count,'expected cohort '+part)
        parts=Counter(TEST=49,REMAINDER=143) if part=='PRIMARY' else Counter({part:count})
        checks.check(Counter(r['partition'] for r in report['rows'])==parts,'full cohort '+part)
        populations[part]=A.aggregate(report,ARMS,checks)
        compare_invariants(oldreports[part]['rows'],report['rows'],checks)
    checks.check(reports['PRIMARY']['rows']==reports['TEST']['rows']+reports['REMAINDER']['rows'],'PRIMARY exact concatenation')
    checks.check(len({r['id'] for p in ('CAL','TEST','REMAINDER') for r in reports[p]['rows']})==238,'all unique238')
    for part in ('CAL','TEST','REMAINDER'):
        checks.check(A._identity(score['reference_bindings'][part])==A._identity(oldseal['reference_bindings'][part]),'same fixed reference '+part)
        checks.check(A._identity(reports[part]['reference_binding'])==A._identity(score['reference_bindings'][part]),'report reference '+part)
        checks.check(A._identity(reports['PRIMARY']['reference_binding'][part])==A._identity(score['reference_bindings'][part]),'PRIMARY reference '+part)
    for path,expected in {**inputs,**sources}.items():
        checks.check(A.sha(ROOT/path)==expected,'current bound bytes '+path)
    data=dict(schema='INDEPENDENT_SCORE_ONLY_V33_LCC_ACCOUNTING_V1',UTC=datetime.now(timezone.utc).isoformat(),
        checks=checks.count,failure_count=len(checks.failures),failures=checks.failures,max_numeric_difference=checks.max_numeric_difference,
        populations=populations,old_V32_populations={p:r['summary'] for p,r in oldreports.items()},
        paired_C_changes={p:{arm:paired_C_changes(oldreports[p]['rows'],r['rows'],arm) for arm in (ARMS[2],ARMS[4])} for p,r in reports.items()},
        TEST49_regressions={arm:A.regressions(reports['TEST']['rows'],arm) for arm in ARMS[1:]},
        largest_errors={p:{arm:A.largest_errors(r['rows'],arm) for arm in ARMS} for p,r in reports.items()},
        source_sha256=sources,input_sha256=inputs,exact_native_and_e_score_invariance=not checks.failures,
        partitions=dict(CAL=46,TEST=49,REMAINDER=143,PRIMARY=192),arms=list(ARMS),
        default='SAME_FROZEN_E unchanged',new_reference_RGB_model_PnP_optimizer_or_objective_calls=0,
        independent_numerical_pose_cost_audit_performed_here=False,no_runtime_method_or_parameter_selection=True,
        missing_or_undefined_rows_preserve_full_denominator=True,repeated_exploratory_data_not_new_holdout=True)
    write(output/('FAILED_AUDIT.json' if checks.failures else 'AUDIT.json'),data)
    if checks.failures:raise AssertionError('Independent score checks failed '+str(len(checks.failures)))
    report=A.markdown(data).replace('V32 전체 점수 독립 집계','V33 연결 영역 제어 전체 점수 독립 집계').replace('e 수치·수용 여부는 V30과 동일','e와 native 두 방법의 수치·수용 여부는 V32와 동일')
    (output/'REPORT_KO.md').write_text(report,encoding='utf-8')
    write(output/'AUDIT_SEAL.json',dict(receipt_sha256=A.sha(output/'AUDIT.json'),report_sha256=A.sha(output/'REPORT_KO.md'),source_sha256=sources,input_sha256=inputs))
    return dict(checks=checks.count,failures=0,audit_sha256=A.sha(output/'AUDIT.json'))

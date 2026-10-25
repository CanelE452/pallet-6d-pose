"""V253 저장 SOURCE 진단을 독립 NumPy/stdlib 산술로 검산한다. 모델/픽셀 접근 없음."""
import argparse, csv, hashlib, io, json, math, os, stat
from pathlib import Path
import numpy as np

COMPLETE_SHA = '5429030009c4016a056b71543348846228361a5f703bfdc00bf44eb68e218e6b'
MODELS = ('BASE', 'CONTROL', 'TREAT')
VIEWS = ('full_CLS', 'fixed_centre_Q', 'centre_CLS')
ERRORS = ('ALL', 'D_TREATfalseE', 'D_BASEcorrect_TREATfalseE', 'D_BASEfalseE_TREATfalseE', 'D_TREATcorrect', 'E_TREATcorrect', 'E_TREATmiss')
CELLS = ('D_BASEcorrect_TREATfalseE', 'D_TREATcorrect', 'E_TREATcorrect', 'E_TREATmiss')
KEYS = ('target', 'family', 'fit', 'available', 'row_id', 'condition')
checked = []
numeric_fields = 0
max_numeric_difference = 0.0


def guard(value):
    p = Path(value)
    assert p.is_absolute() and '..' not in p.parts
    assert not any(t in str(p).casefold() for t in ('v112', 'cache', '.pyc', 'current223', 'publicheld', 'source_holdout'))
    for a in reversed((p, *p.parents)):
        assert not stat.S_ISLNK(os.lstat(a).st_mode), str(a)
    assert stat.S_ISREG(os.lstat(p).st_mode)
    return p


def read(value, binding=None):
    p = guard(value)
    with os.fdopen(os.open(p, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as f:
        raw = f.read()
    got = {'path': str(p), 'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}
    if binding is not None:
        assert got == binding, (got, binding)
    checked.append(got)
    return raw


def compare(a, b, where=''):
    global numeric_fields, max_numeric_difference
    if isinstance(b, dict):
        assert isinstance(a, dict) and a.keys() == b.keys(), (where, a.keys(), b.keys())
        for k in b:
            compare(a[k], b[k], where + '/' + str(k))
    elif isinstance(b, (list, tuple)):
        assert isinstance(a, (list, tuple)) and len(a) == len(b), where
        for i, v in enumerate(b):
            compare(a[i], v, where + '/' + str(i))
    elif isinstance(b, bool) or b is None or isinstance(b, str):
        assert a == b and type(a) == type(b), (where, a, b)
    elif isinstance(b, (int, float, np.integer, np.floating)):
        assert isinstance(a, (int, float)) and not isinstance(a, bool), (where, a, b)
        assert math.isfinite(float(a)) and math.isfinite(float(b)), where
        d = abs(float(a) - float(b))
        numeric_fields += 1
        max_numeric_difference = max(max_numeric_difference, d)
        assert d <= 2e-12 * max(1., abs(float(b))), (where, a, b, d)
    else:
        raise AssertionError((where, type(b)))


def parse_like(s, wanted):
    if wanted is None:
        assert s == ''
        return None
    if isinstance(wanted, bool):
        assert s in ('True', 'False')
        return s == 'True'
    if isinstance(wanted, (list, dict)):
        return json.loads(s)
    if isinstance(wanted, int):
        return int(s)
    if isinstance(wanted, float):
        return float(s)
    return s


def table(raw, expected, name):
    rows = list(csv.DictReader(io.StringIO(raw.decode(), newline='')))
    assert len(rows) == len(expected), (name, len(rows), len(expected))
    for i, (got, want) in enumerate(zip(rows, expected)):
        assert got.keys() == want.keys(), (name, i, got.keys(), want.keys())
        compare({k: parse_like(got[k], v) for k, v in want.items()}, want, name + '/' + str(i))
    return len(rows)


def statistics(values):
    x = sorted(float(v) for v in values)
    names = ('min', 'p10', 'p25', 'median', 'p75', 'p90', 'max')
    if not x:
        return dict(count=0, mean=None, std=None, **dict.fromkeys(names))
    def percentile(p):
        t = (len(x) - 1) * p
        lo = math.floor(t); hi = math.ceil(t)
        return x[lo] + (x[hi] - x[lo]) * (t - lo)
    mean = math.fsum(x) / len(x)
    std = math.sqrt(math.fsum((v - mean) ** 2 for v in x) / len(x))
    return dict(count=len(x), mean=mean, std=std,
                **dict(zip(names, (percentile(p) for p in (0, .1, .25, .5, .75, .9, 1)))))


def run(complete_path):
    cb = read(complete_path)
    assert hashlib.sha256(cb).hexdigest() == COMPLETE_SHA
    c = json.loads(cb)
    assert c['completed'] is True and c['PASS'] is True and c['current_admission'] is False
    assert set(c['inputs']) == {'BASE_CSV','CONTROL_CSV','TREAT_CSV','generic_ref9','target_ref9','samples','fit_complete'}
    source = {k: read(b['path'], b) for k, b in c['inputs'].items()}
    result = {k: read(b['path'], b) for k, b in c['outputs'].items()}
    fit_complete = json.loads(source['fit_complete'])
    assert fit_complete['completed'] is True and fit_complete['PASS'] is True
    with np.load(io.BytesIO(source['samples']), allow_pickle=False) as z:
        s = {k: z[k] for k in KEYS}
    y, family, fit, available = (s[k] for k in ('target','family','fit','available'))
    n = len(y)
    assert n == 7617 and int(fit.sum()) == 5876 and int((~fit).sum()) == 1741
    assert y.dtype == np.int8 and family.dtype == np.int8 and fit.dtype == bool and available.dtype == bool
    assert all(a.shape == (n,) for a in (y, family, fit, available, s['row_id']))
    assert np.isin(y, [0,1]).all() and np.isin(family,[0,1]).all() and len(set(s['row_id'])) == n
    condition = s['condition']
    assert condition.dtype == np.float32 and condition.shape == (n,32) and np.isfinite(condition).all()
    assert not np.count_nonzero(condition[family == 1])
    assert np.isin(condition[family == 0,:8],[0,1]).all() and (condition[family == 0,:8].sum(1) == 1).all()
    native = np.where(family == 0, condition[:,:8].argmax(1), -1)
    role = np.where(fit, 'FIT', 'DEV')
    truth = np.where(y == 0, 'D', 'E')
    refs = {name: np.load(io.BytesIO(source[key]), allow_pickle=False) for name, key in [('GENERIC','generic_ref9'),('TARGET','target_ref9')]}
    for x in refs.values():
        assert x.dtype == np.float32 and x.shape == (n,9) and np.isfinite(x).all() and np.isin(x[:,6:],[0,1]).all()
        for v in range(3):
            assert not np.count_nonzero(x[x[:,6+v] == 0][:,[v,3+v]])
    assert np.array_equal(refs['GENERIC'][:,:3], refs['GENERIC'][:,3:6])
    predictions = {}; original = {}; logits = {}
    for name in MODELS:
        rows = list(csv.DictReader(io.StringIO(source[name + '_CSV'].decode(), newline='')))
        assert len(rows) == n
        by_index = {int(r['sample_index']): r for r in rows}
        assert len(by_index) == n and set(by_index) == set(range(n))
        original[name] = [by_index[i] for i in range(n)]
        z = np.empty((n,2),np.float64)
        pred = []
        for i in range(n):
            r = by_index[i]
            assert r['row_id'] == str(s['row_id'][i]) and int(r['family']) == int(family[i]) and int(r['target']) == int(y[i])
            assert r['role'] == role[i] and r['available'] == str(bool(available[i]))
            z[i] = [float(r['logit0']),float(r['logit1'])]
            chosen = float(r['selected_family_E_logit'])
            assert np.isfinite(z[i]).all() and chosen == z[i,family[i]]
            p = 'U' if not available[i] else ('E' if chosen >= 0 else 'D')
            assert p == r['prediction'];pred.append(p)
        predictions[name] = np.asarray(pred); logits[name] = z
    diag = []
    for i in range(n):
        r = {'sample_index':i,'row_id':str(s['row_id'][i]),'family':int(family[i]),'role':str(role[i]),'target':int(y[i]),'truth':str(truth[i]),'available':bool(available[i]),'original_native_address':int(native[i]) if family[i] == 0 else None,'native_authority':'ORIGINAL_INITIALIZER_ADDRESS_NOT_CANONICAL_GOLD' if family[i] == 0 else 'UNKNOWN'}
        for name in MODELS:
            r.update({name+'_prediction':str(predictions[name][i]),name+'_E_logit':float(logits[name][i,family[i]]),name+'_logit0':float(logits[name][i,0]),name+'_logit1':float(logits[name][i,1])})
        r['generic_ref9'] = refs['GENERIC'][i].astype(float).tolist();r['target_ref9'] = refs['TARGET'][i].astype(float).tolist()
        for ref,x in refs.items():
            for j,v in enumerate(VIEWS):
                a,b = float(x[i,j]),float(x[i,j+3])
                r.update({ref+'_'+v+'_selectedcos':a,ref+'_'+v+'_nextcos':b,ref+'_'+v+'_margin':a-b,ref+'_'+v+'_supported':bool(x[i,j+6])})
        diag.append(r)
    output_rows = {'ALL7617_SAVED_PREDICTION_AND_REFERENCE_JOIN.csv':table(result['ALL7617_SAVED_PREDICTION_AND_REFERENCE_JOIN.csv'],diag,'ALL7617')}
    b,t = predictions['BASE'],predictions['TREAT']
    memberships = {'ALL':np.ones(n,bool),'D_TREATfalseE':(y==0)&(t=='E'),'D_BASEcorrect_TREATfalseE':(y==0)&(b=='D')&(t=='E'),'D_BASEfalseE_TREATfalseE':(y==0)&(b=='E')&(t=='E'),'D_TREATcorrect':(y==0)&(t=='D'),'E_TREATcorrect':(y==1)&(t=='E'),'E_TREATmiss':(y==1)&(t!='E')}
    def metric(mask,name):
        p = predictions[name][mask];a = y[mask]
        cm = [[int(np.sum((a == yy)&(p == pp))) for pp in ('D','E','U')] for yy in (0,1)]
        denD,denE = sum(cm[0]),sum(cm[1]);dP=cm[0][0]+cm[1][0];eP=cm[0][1]+cm[1][1]
        rd=cm[0][0]/denD if denD else None;re=cm[1][1]/denE if denE else None
        return dict(model=name,rows=int(mask.sum()),D=denD,E=denE,CM_D_E_by_D_E_U=cm,precisionD=cm[0][0]/dP if dP else None,precisionE=cm[1][1]/eP if eP else None,recallD=rd,recallE=re,BR=(rd+re)/2 if rd is not None and re is not None else None,U=cm[0][2]+cm[1][2])
    def transition(mask,old,new):
        a=predictions[old][mask];d=predictions[new][mask];tr=truth[mask];ok=a==tr;nk=d==tr
        return dict(old_model=old,new_model=new,rows=int(mask.sum()),transition_D_E_U_by_D_E_U=[[int(np.sum((a==u)&(d==v)))for v in ('D','E','U')]for u in ('D','E','U')],correct_both=int((ok&nk).sum()),old_correct_new_wrong=int((ok&~nk).sum()),old_wrong_new_correct=int((~ok&nk).sum()),wrong_both=int((~ok&~nk).sum()),prediction_changes=int((a!=d).sum()))
    def events(mask):
        d=mask&(y==0);e=mask&(y==1);total=int((d&(t=='E')).sum());new=int((d&(b=='D')&(t=='E')).sum());existing=int((d&(b=='E')&(t=='E')).sum());other=int((d&(b=='U')&(t=='E')).sum())
        assert total == new+existing+other
        o=dict(rows=int(mask.sum()),D=int(d.sum()),E=int(e.sum()),D_TREATfalseE_total=total,D_BASEcorrect_to_TREATfalseE_new=new,D_BASEfalseE_to_TREATfalseE_existing=existing,D_BASEU_to_TREATfalseE=other,D_TREATfalseE_rate_originalD=total/int(d.sum()) if d.any() else None)
        for old in ('BASE','CONTROL'):
            o['E_'+old+'miss_to_TREATcorrect']=int((e&(predictions[old]!='E')&(t=='E')).sum());o['E_'+old+'correct_to_TREATmiss']=int((e&(predictions[old]=='E')&(t!='E')).sum())
        return o
    groups=[];trans=[];dist=[];ev=[]
    for f in (0,1):
        for k in ([None,*range(8)] if f==0 else [None]):
            for rr in ('FIT','DEV'):
                for yy in ('ALL','D','E'):
                    for error in ERRORS:
                        spec=dict(family=f,role=rr,original_native_address=k,native_group='UNKNOWN' if f==1 else ('ALL_ADDRESSES' if k is None else str(k)),truth_filter=yy,failure_group=error)
                        mask=(family==f)&(role==rr)&memberships[error]
                        if k is not None:mask &= native==k
                        if yy!='ALL':mask &= truth==yy
                        for name in MODELS:groups.append({**spec,**metric(mask,name)})
                        for old,new in [('BASE','CONTROL'),('BASE','TREAT'),('CONTROL','TREAT')]:trans.append({**spec,**transition(mask,old,new)})
                        for j,view in enumerate(VIEWS):
                            for score in ('selectedcos','nextcos','margin'):
                                gx=refs['GENERIC'][:,j].astype(float) if score=='selectedcos' else refs['GENERIC'][:,j+3].astype(float) if score=='nextcos' else refs['GENERIC'][:,j].astype(float)-refs['GENERIC'][:,j+3].astype(float)
                                tx=refs['TARGET'][:,j].astype(float) if score=='selectedcos' else refs['TARGET'][:,j+3].astype(float) if score=='nextcos' else refs['TARGET'][:,j].astype(float)-refs['TARGET'][:,j+3].astype(float)
                                for reference in ('GENERIC','TARGET','TARGET_MINUS_GENERIC'):
                                    support=mask & (refs['GENERIC' if reference=='GENERIC' else 'TARGET'][:,6+j] != 0)
                                    if reference=='TARGET_MINUS_GENERIC':support &= refs['GENERIC'][:,6+j] != 0
                                    values=(gx if reference=='GENERIC' else tx if reference=='TARGET' else tx-gx)[support]
                                    dist.append({**spec,'reference':reference,'view':view,'score':score,'original_rows':int(mask.sum()),'supported_rows':int(support.sum()),'unsupported_rows':int(mask.sum()-support.sum()),**statistics(values)})
                        if yy=='ALL' and error=='ALL':ev.append({**spec,**events(mask)})
    for name,rows in [('GROUP_METRICS.csv',groups),('TRANSITION_COUNTS.csv',trans),('REFERENCE_DISTRIBUTIONS.csv',dist),('FIXED_D_DAMAGE_COUNTS.csv',ev)]:output_rows[name]=table(result[name],rows,name)
    keyfields=('family','original_native_address','truth_filter','failure_group','reference','view','score','role')
    index={tuple(d[k]for k in keyfields):d for d in dist};shifts=[]
    for a in dist:
        if a['role']!='FIT':continue
        d=index[tuple(a[k]for k in keyfields[:-1])+('DEV',)]
        o={k:a[k]for k in ('family','original_native_address','native_group','truth_filter','failure_group','reference','view','score')}
        o.update(FIT_original_rows=a['original_rows'],DEV_original_rows=d['original_rows'],FIT_supported=a['count'],DEV_supported=d['count'],FIT_mean=a['mean'],DEV_mean=d['mean'],DEV_minus_FIT_mean=None if a['mean'] is None or d['mean'] is None else d['mean']-a['mean'],FIT_median=a['median'],DEV_median=d['median'],DEV_minus_FIT_median=None if a['median'] is None or d['median'] is None else d['median']-a['median'])
        shifts.append(o)
    output_rows['FIT_DEV_REFERENCE_SHIFTS.csv']=table(result['FIT_DEV_REFERENCE_SHIFTS.csv'],shifts,'FIT_DEV_REFERENCE_SHIFTS.csv')
    summary=json.loads(result['SUMMARY.json'])
    compare(summary['counts'],dict(known=n,FIT=int(fit.sum()),DEV=int((~fit).sum()),original_rows=15040,masked_S_U_not_E=7423),'summary/counts')
    compare(summary['fixed_D_falseE_and_E_change_events'],{r:events(role==r)for r in ('FIT','DEV')},'summary/events')
    compare(summary['pooled_metrics'],{name:{f'family{f}_{r}':metric((family==f)&(role==r),name)for f in (0,1)for r in ('FIT','DEV')}for name in MODELS},'summary/metrics')
    assert summary['current_admission'] is False and summary['no_threshold_view_or_model_selection'] is True
    selection=json.loads(result['FIT_TILE_SELECTION.json']);cells=[];selected=set()
    for f in (0,1):
        for k in (range(8)if f==0 else [None]):
            for group in CELLS:
                mask=fit&(family==f)&memberships[group]
                if k is not None:mask &= native==k
                ids=np.flatnonzero(mask).tolist();chosen=ids[:2];selected.update(chosen)
                cells.append(dict(family=f,original_native_address=k,failure_group=group,eligible_count=len(ids),selected=[diag[i]for i in chosen],empty=not chosen))
    compare(selection['groups'],cells,'selection/groups')
    assert selection['selected_sample_indices']==sorted(selected) and selection['unique_selected_rows']==len(selected) and len(cells)==36
    assert selection['empty_cells']==sum(c['empty']for c in cells) and all(fit[i]for i in selected) and len(selected)<=72
    compare(c['counts'],summary['counts'],'complete/counts');assert c['selected_FIT_rows']==len(selected)
    pooled={r:events(role==r)for r in ('FIT','DEV')}
    return {'completed':True,'PASS':True,'diagnosis_complete':checked[0],'verified_bindings':checked,'output_row_counts':output_rows,'numeric_fields_compared':numeric_fields,'max_numeric_absolute_difference':max_numeric_difference,'selection':{'cells':36,'empty_cells':sum(c['empty']for c in cells),'selected_FIT_rows':len(selected),'selected_sample_indices':sorted(selected),'selected_DEV_rows':0},'fixed_events':pooled,'family_role_events':{f'family{f}_{r}':events((family==f)&(role==r))for f in (0,1)for r in ('FIT','DEV')},'independence':'생산자 모듈 import/call 없이 원 CSV·6 nonpixel samples members·두 ref9에서 별도 vector masks/CM/전이/정렬 선형 quantile/분포/first2 선택을 재계산했다.','source_teacher_derived_labels_condition_and_existing_reference_scores_read':True,'raw_RGB_GT_archives_CP_models_or_current_held_payloads_read':0,'new_fitting_forward_or_threshold_selection':0,'current_admission':False,'limitations':['저장 SOURCE 기술 진단이며 인과·라벨 오류·physical/native gold·실사 개선을 입증하지 않는다.','temporal freeze는 sealed CODE/바인딩/Root 실행 기록 범위이며 이 사후 산술 검산 자체로 시간 순서를 증명하지 않는다.','7423 masked S/U는 E 또는 성공적인 모델 abstention으로 집계하지 않는다.']}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--complete',required=True);parser.add_argument('--out',required=True);args=parser.parse_args()
    out=Path(args.out);assert out.is_absolute() and '..'not in out.parts and not any(x in str(out).casefold()for x in ('v112','cache','.pyc'))
    for a in reversed((out,*out.parents)):
        try:s=os.lstat(a)
        except FileNotFoundError:continue
        assert not stat.S_ISLNK(s.st_mode)
    out.mkdir(exist_ok=False)
    try:
        review=run(args.complete)
        code=Path(__file__).read_bytes();review['checker']={'path':str(Path(__file__)),'sha256':hashlib.sha256(code).hexdigest(),'size_bytes':len(code)}
        p=out/'REVIEW.json';raw=(json.dumps(review,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode();p.write_bytes(raw)
        print(json.dumps({'completed':True,'PASS':True,'review':{'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)},'selection':review['selection'],'family_role_events':review['family_role_events'],'numeric_fields':numeric_fields,'max_difference':max_numeric_difference},ensure_ascii=False))
    except BaseException as error:
        (out/'FAILURE.json').write_text(json.dumps({'completed':False,'PASS':False,'error':str(error),'error_type':type(error).__name__,'automatic_retry':False},ensure_ascii=False,indent=2)+'\n')
        raise

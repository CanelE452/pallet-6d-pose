"""V212 SOURCE apparent readout. Prepare: code/protocol/invented scalars only.

Actual root entry after both4000/terminal/audit may open four owned prediction
NPZs (META6 + direct raw/pred only), saved loss-target JSONL and V211 metadata.
No model/checkpoint/input/target NPZ, original GT/RGB or protected evaluation.
"""
from pathlib import Path
from collections import Counter
from html import escape
from html.parser import HTMLParser
import argparse,csv,hashlib,io,json,math,os,re,stat,traceback
import numpy as np

H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
FIT_PROTOCOL=V/'fit_preparation_a/PROTOCOL_BEFORE_DENSE_POINT_WEAK_BRIDGE_SOURCE_FIT.json'
FIT_READY=V/'fit_preparation_a/FINAL_READY.json'
RUN=V/'runs/C62_C63_DENSE_POINT_WEAK_BRIDGE_SOURCE_FIXED4000'
OUT=V/'source_fit_readout_b_v1'
PROTOCOL=H/'PROTOCOL_BEFORE_SOURCE_FIT_OUTPUT_OR_LOG_BODIES.json'
CHECKS=H/'INVENTED_SCALAR_CHECKS.json'
ARMS=('C62_DENSE_POINT_RAW_Q8','C63_DENSE_POINT_WEAK_BRIDGE_RAW_Q8')
TAGS=('C62','C63')
WEAK={ARMS[0]:(False,0.),ARMS[1]:(True,.25)}
META=('row_id','original_global_index','native_slot','query_xy','L','original_query_available')
READ_MEMBERS=set(META)|{'raw_direct_end_external_logit','direct_end_prediction_D0_E1_Uminus1'}
PACKET_MEMBERS=READ_MEMBERS|{'raw_support_logits','raw_owner_logits','support_prediction','owner_prediction','regional_support_proxy_D0_E1_Uminus1','support_order','owner_order','direct_end_order'}
LOSS_FIELDS=('old_five_mean','point_loss','weak_loss','weak_loss_weight','total_loss','grad_norm')

def need(ok,msg):
    if not bool(ok):raise AssertionError(msg)

def safe(path):
    p=Path(path).absolute();need(P in p.parents and '..'not in p.parts and all('v112'not in s.casefold()for s in p.parts),'Lexical project path before FS');return p

def opened(path,create=False):
    p=safe(path);fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for s in p.parts[1:-1]:
            n=os.open(s,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL if create else os.O_RDONLY
        n=os.open(p.name,flags|os.O_NOFOLLOW,0o644,dir_fd=fd)
    finally:os.close(fd)
    need(stat.S_ISREG(os.fstat(n).st_mode),'Regular explicit file, no symlink');return os.fdopen(n,'wb'if create else'rb')

def sha(path):
    h=hashlib.sha256()
    with opened(path)as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def bind(path):return dict(path=str(safe(path)),sha256=sha(path))

def check(b):
    need(isinstance(b,dict)and bind(b['path'])=={k:b[k]for k in('path','sha256')},'Exact sealed binding');return safe(b['path'])

def read(path):
    with opened(path)as f:return json.load(f)

def lines(path):
    with opened(path)as f:return [json.loads(x)for x in f if x.strip()]

def text(path,s):
    with opened(path,True)as f:f.write(s.encode('utf8'))

def write(path,d):text(path,json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def csvwrite(path,rows):
    if not rows:text(path,'EMPTY\n');return
    f=io.StringIO();w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);text(path,f.getvalue())

def ratio(a,b):return a/b if b else None

def metric(target,pred,available):
    need(target.ndim==pred.ndim==available.ndim==1 and target.shape==pred.shape==available.shape,'Full original source rows')
    need(np.isin(target,[-1,0,1]).all()and np.isin(pred,[-1,0,1]).all()and available.dtype==bool,'D/E/UNKNOWN typed')
    cm=[[int(((target==t)&(pred==q)).sum())for q in(0,1,-1)]for t in(0,1)]
    dn,en=[int((target==t).sum())for t in(0,1)];u=cm[0][2]+cm[1][2]
    unknown=target<0;uc=Counter(int(v)for v in pred[unknown])
    return dict(all_rows=len(target),known_N=dn+en,D_N=dn,E_N=en,CM_D_E_by_D_E_U=cm,D_precision=ratio(cm[0][0],cm[0][0]+cm[1][0]),D_recall=ratio(cm[0][0],dn),E_precision=ratio(cm[1][1],cm[1][1]+cm[0][1]),E_recall=ratio(cm[1][1],en),U_known=u,U_known_rate=ratio(u,dn+en),UNKNOWN_target_rows=int(unknown.sum()),UNKNOWN_target_predictions=dict(D=uc[0],E=uc[1],U=uc[-1]),original_input_U=int((~available).sum()),all_prediction_U=int((pred<0).sum()),input_U_in_known=int(((~available)&~unknown).sum()))

def decode(raw,available):
    need(raw.ndim==available.ndim==1 and raw.shape==available.shape and available.dtype==bool,'Direct scalar inputs')
    out=np.full(len(raw),-1,np.int8);ok=available&np.isfinite(raw);out[ok]=(raw[ok]>=0).astype(np.int8);return out

def paired(target,a,b):
    known=target>=0;ac=a==target;bc=b==target
    return dict(known_N=int(known.sum()),both_correct=int((known&ac&bc).sum()),recovered_by_C63=int((known&~ac&bc).sum()),damaged_by_C63=int((known&ac&~bc).sum()),both_wrong=int((known&~ac&~bc).sum()),same_prediction=int((known&(a==b)).sum()),different_prediction=int((known&(a!=b)).sum()),net_correct_change=int((known&bc).sum()-(known&ac).sum()),direction='C62 to predeclared C63; source apparent only, no adoption')

def meta_equal(packet,rows):
    n=len(rows);need(len(packet['row_id'])==n and len({r['row_id']for r in rows})==n,'Unique exact source row count')
    need(packet['query_xy'].shape==(n,2)and packet['L'].shape==(n,)and packet['original_query_available'].dtype==bool,'META shapes')
    for i,r in enumerate(rows):
        for k in ('row_id','original_global_index','native_slot','original_query_available'):need(packet[k][i]==r[k],'Original source META '+k)
        q=[np.nan,np.nan]if r['query_xy']is None else[np.nan if v is None else float(v)for v in r['query_xy']]
        L=np.nan if r['L']is None else float(r['L'])
        need(np.array_equal(packet['query_xy'][i].astype(np.float64),np.asarray(q,np.float64),equal_nan=True)and np.array_equal(np.asarray(packet['L'][i],np.float64),np.asarray(L,np.float64),equal_nan=True),'Original q/L exact, including inputU null/NaN')
    need(np.isin(packet['native_slot'],np.arange(8)).all(),'Original native0..7')

def output_packet(binding,rows):
    need(safe(binding['path']).is_relative_to(RUN),'Only new owned source output')
    with opened(check(binding))as f:
        with np.load(f,allow_pickle=False)as z:
            need(set(z.files)==PACKET_MEMBERS,'Exact own16-key packet, no inputs')
            packet={k:z[k]for k in READ_MEMBERS}
    meta_equal(packet,rows);raw=packet['raw_direct_end_external_logit'];pred=packet['direct_end_prediction_D0_E1_Uminus1']
    need(raw.dtype==np.float32 and raw.shape==pred.shape==(len(rows),)and np.array_equal(decode(raw,packet['original_query_available']),pred),'Stored direct exact raw>=0 and original U parity')
    return packet

def owned_binaries(fit):
    result=[]
    def add(b,kind,arm):
        path=safe(b['path']);need(path.is_relative_to(RUN/arm)and len(b['sha256'])==64,'Owned run/arm binding only')
        need(path.suffix==('.pt'if kind=='OWN_CHECKPOINT'else'.npz')and path.name.startswith('FINAL_'),'Owned final binary basename')
        result.append(dict(kind=kind,arm=arm,relative=str(path.relative_to(V)),path=str(path),sha256=b['sha256'],binding_authority='Exact producer COMPLETE metadata; report does not load/hash checkpoint or unrelated prediction NPZ. Only POINT7040/WEAK44 own4 payloads are read here.'))
    def walk(x,arm):
        if isinstance(x,dict):
            if {'path','sha256'}<=set(x):
                if Path(x['path']).suffix=='.npz':add(x,'OWN_SOURCE_PREDICTION',arm)
            else:
                for v in x.values():walk(v,arm)
        elif isinstance(x,list):
            for v in x:walk(v,arm)
    for arm in ARMS:
        r=fit['arms'][arm];add(r['checkpoint'],'OWN_CHECKPOINT',arm);walk(r['outputs'],arm);walk(r['source_cohorts'],arm)
    by={r['path']:r for r in result};need(len(by)==len(result),'No duplicated owned binary binding')
    need(sum(r['kind']=='OWN_CHECKPOINT'for r in result)==2,'Only two new owned states')
    return result

def source_counts(rows,targets,weak=False):
    keys=('variant','source_frame_index','case_position','component_id')if weak else('base_index','actual_image_scene_index','actual_image_variant','family','declared_CF_to_canonical','factors.material','factors.aspect','factors.geometry_family')
    def value(r,key):
        v=r
        for part in key.split('.'):
            if not isinstance(v,dict)or part not in v:return 'UNKNOWN_NOT_IN_METADATA'
            v=v[part]
        return v
    result=[]
    for key in keys:
        values=[value(r,key)for r in rows]
        counts=Counter(json.dumps(v,sort_keys=True)for v in values)
        for val,count in sorted(counts.items()):result.append(dict(field=key,value=val,all_rows=count,known_rows=sum(int(t>=0)for r,t in zip(rows,targets)if json.dumps(value(r,key),sort_keys=True)==val),scope='Repeated rows/variants are not independent frames'))
    return result

def summarize_log(rows,arm):
    need(len(rows)==4000 and [r['update']for r in rows]==list(range(1,4001)),'Complete fixed4000 log')
    out=[]
    for field in LOSS_FIELDS:
        values=[float(r[field])for r in rows];need(all(math.isfinite(v)for v in values),'Finite saved log scalars')
        out.append(dict(arm=arm,field=field,updates=4000,first=values[0],last=values[-1],mean=math.fsum(values)/4000,minimum=min(values),maximum=max(values),scope='Logged scalars only; not independent neural-loss replay'))
    return out

def complete_alias(review,expected):
    bindings=[review[k]for k in('fit_complete','source_complete')if k in review]
    need(review['completed']and review['PASS']and bindings and all(b==expected for b in bindings),'Independent completed exact-source fit review')

def table(rows):
    if not rows:return 'EMPTY\n'
    keys=list(rows[0]);return '|'+'|'.join(keys)+'|\n|'+'|'.join('---'for _ in keys)+'|\n'+''.join('|'+'|'.join(str(r[k]).replace('|','/')for k in keys)+'|\n'for r in rows)

class Refs(HTMLParser):
    def __init__(self):super().__init__();self.refs=[]
    def handle_starttag(self,tag,attrs):self.refs.extend(v for k,v in attrs if k in('href','src')and v)

def links_exist(base,s,html=False):
    need('/home/'not in s and 'file://'not in s,'Portable rendered text')
    if html:q=Refs();q.feed(s);refs=q.refs
    else:refs=re.findall(r'\]\(([^)]+)\)',s)
    for ref in refs:
        need(not ref.startswith(('/','//'))and not re.match(r'^[A-Za-z]+:',ref)and'v112'not in ref.casefold(),'Local relative links')
        with opened(os.path.normpath(base/ref)):pass
    return len(refs)

def invented():
    tests=[]
    def test(n,ok):need(ok,n);tests.append(n)
    target=np.array([0,0,1,1,-1,-1],np.int8);pred=np.array([0,-1,1,0,1,-1],np.int8);av=np.array([1,1,1,1,1,0],bool)
    m=metric(target,pred,av);test('D_E_by_D_E_U_and_UNKNOWN_separate',m['CM_D_E_by_D_E_U']==[[1,0,1],[1,1,0]]and m['known_N']==4 and m['UNKNOWN_target_rows']==2 and m['original_input_U']==1)
    test('Precision_recall_U_denominators',m['D_precision']==.5 and m['D_recall']==.5 and m['E_precision']==1. and m['E_recall']==.5 and m['U_known_rate']==.25)
    test('Raw_sign_zero_subnormal_nonfinite_and_inputU',decode(np.array([-1e-40,0,np.nan,2,1],np.float32),np.array([1,1,1,1,0],bool)).tolist()==[0,1,-1,1,-1])
    p=paired(target,pred,np.array([0,0,-1,1,0,-1],np.int8));test('Paired_recovery_damage_only_known',p['recovered_by_C63']==2 and p['damaged_by_C63']==1 and p['net_correct_change']==1)
    test('Empty_native_class_remains_null',metric(np.array([-1],np.int8),np.array([0],np.int8),np.ones(1,bool))['E_recall']is None)
    rows=[dict(row_id='a',original_global_index=0,native_slot=7,query_xy=None,L=None,original_query_available=False)]
    packet=dict(row_id=np.array(['a']),original_global_index=np.array([0]),native_slot=np.array([7]),query_xy=np.array([[np.nan,np.nan]]),L=np.array([np.nan]),original_query_available=np.zeros(1,bool));meta_equal(packet,rows);test('Original_inputU_null_NaN_META',True)
    bad=dict(rows[0],L=1.)
    try:meta_equal(packet,[bad]);ok=False
    except AssertionError:ok=True
    test('META_qL_drift_rejected',ok)
    f=dict(arms={a:dict(checkpoint=dict(path=str(RUN/a/'FINAL_CHECKPOINT.pt'),sha256='a'*64),outputs={d:dict(path=str(RUN/a/('FINAL_'+d+'_OUTPUTS.npz')),sha256='b'*64)for d in('POINT7040','WEAK44')},source_cohorts={})for a in ARMS})
    bins=owned_binaries(f);test('Publication_list_new_two_states_four_preds_only',len(bins)==6 and sum(x['kind']=='OWN_CHECKPOINT'for x in bins)==2)
    f['arms'][ARMS[0]]['outputs']['bad']=dict(path=str(V/'inputs.npz'),sha256='a'*64)
    try:owned_binaries(f);ok=False
    except AssertionError:ok=True
    test('Input_or_foreign_binary_outside_owned_run_rejected',ok)
    complete_alias(dict(completed=True,PASS=True,source_complete='same',fit_complete='same'),'same');test('Fit_review_alias_exact',True)
    try:complete_alias(dict(completed=True,PASS=True,source_complete='same',fit_complete='wrong'),'same');ok=False
    except AssertionError:ok=True
    test('Conflicting_source_review_alias_rejected',ok)
    return dict(completed=True,PASS=True,invented_only=True,count=len(tests),checks=tests,actual_source_data_log_score_weight_or_header_reads=0)

def prepare():
    p=read(FIT_PROTOCOL);ready=read(FIT_READY);need(ready['completed']and ready['PASS']and p['trainer_code']['sha256']=='35854077f43180f7cb47504b21a448b08e4008764c98a9af119c1fdd6df345e0','Exact final source implementation metadata')
    write(CHECKS,invented())
    proto=dict(completed=True,prepared_only=True,code=bind(__file__),checks=bind(CHECKS),source_protocol=bind(FIT_PROTOCOL),source_ready=bind(FIT_READY),source_code=p['trainer_code'],teacher_complete=p['teacher_complete'],weak_metadata=p['weak_metadata'],source_complete_path=str(RUN/'COMPLETE.json'),barrier_path=str(RUN/'ALL2_FINAL4000_FROZEN_BEFORE_ANY_SOURCE_OUTPUTS.json'),input_validation_path=str(RUN/'FIT_INPUT_VALIDATION.json'),point_loss_targets_path=str(RUN/'ALL7040_POINT_LOSS_TARGETS_AND_ELIGIBILITY.jsonl'),arms=list(ARMS),parameters_each=118359,updates_each=4000,
      future_payload_whitelist='Only source COMPLETE/barrier/input-validation/terminal/audit, two bound draw-loss JSONLs, source loss-target7040 JSONL, V211 saved7040 mapping+SUMMARY, weak44 metadata, and four COMPLETE.outputs POINT7040/WEAK44 NPZs. NPZ members META6/rawdirect/stored-direct only. No source inputs/target NPZ/checkpoint/GT/RGB/protected evaluation.',
      source_authority=dict(point='Own-image procedural point teacher. All7040 retained; loss-eligible6096 D5853/E243; 944 UNKNOWN-loss preserved. Source TRAIN apparent, not human native gold.',weak='44 assistant local visible-feature weak rows: factual8 +SHAM18 D +COVER18 E; 3 SOURCE public frames. C62 also sees these frames in old auxiliary source pools, so not independent validation.',contrast='C63 primary a priori vs C62 point-only; additional weak branch .25 only. Earlier finite-end327 versus densepoint6096 changes definition and exposure, not label-only causality.',heldout='SOURCE point HOLDOUT18 is a separate V213 lane; never read or merge here.'),
      report_targets=['PORTABLE_REPORT_KO.md','ROOT_REPORT_KO.md','ROOT_RESULTS_KO.json','ROOT_READOUT_KO.html'],output=str(OUT),actual_source_payload_or_score_header_body_reads=0,selected_model=None,adopted=False,fully_solved=False)
    write(PROTOCOL,proto)
    template=dict(approved_completed_source_fit_report=False,root_confirmed_source_terminal_exit0=False,no_private_or_HOLDOUT_metrics_or_model_or_GT_reads=True,code=proto['code'],protocol=bind(PROTOCOL),source_complete=None,source_barrier=None,input_validation=None,root_terminal_evidence=None,independent_fit_review=None,point_loss_targets=None,teacher_mapping=None,teacher_summary=None,weak_metadata=proto['weak_metadata'])
    write(H/'ROOT_ACTUAL_SOURCE_REPORT_ENTRY_TEMPLATE_APPROVED_FALSE.json',template)
    done=dict(completed=True,PASS=True,prepared_only=True,code=proto['code'],protocol=bind(PROTOCOL),checks=bind(CHECKS),entry_template=bind(H/'ROOT_ACTUAL_SOURCE_REPORT_ENTRY_TEMPLATE_APPROVED_FALSE.json'),actual_SOURCE_arrays_logs_scores_headers_weights_GT_or_protected_reads=0,model_forward_or_fit=0,automatic_retry=False)
    write(H/'FINAL_READY.json',done);print(json.dumps(done))

def run(entry):
    e=read(entry);need(e['approved_completed_source_fit_report']is True and e['root_confirmed_source_terminal_exit0']is True and e['no_private_or_HOLDOUT_metrics_or_model_or_GT_reads']is True,'Separate postterminal root entry')
    p=read(check(e['protocol']));need(e['code']==p['code']==bind(__file__)and e['protocol']==bind(PROTOCOL),'Frozen report code/protocol')
    for key,pk in [('source_complete','source_complete_path'),('source_barrier','barrier_path'),('input_validation','input_validation_path'),('point_loss_targets','point_loss_targets_path')]:need(e[key]['path']==p[pk],'Exact current run metadata '+key)
    check(e['root_terminal_evidence']);fit=read(check(e['source_complete']));barrier=read(check(e['source_barrier']));review=read(check(e['independent_fit_review']));complete_alias(review,e['source_complete'])
    need(fit['completed']and fit['PASS']and fit['source_only']and fit['from_scratch']and fit['all2_frozen_before_any_SOURCE_outputs']and fit['parameters_each']==118359 and fit['steps_each']==4000 and fit['total_updates']==8000 and fit['protected_real_COMMON_VAL_access']==0,'Both fixed source finals before all source outputs')
    need(fit['code']==p['source_code']and fit['protocol']==p['source_protocol']and fit['all2_freeze']==e['source_barrier']and barrier['completed']and barrier['source_predictions_emitted']is False,'Exact trainer/protocol/final barrier')
    need(set(fit['arms'])==set(ARMS)and barrier['checkpoints']=={a:fit['arms'][a]['checkpoint']for a in ARMS},'Both final checkpoints exact, no state decode')
    validation=read(check(e['input_validation']));need(validation['completed']and validation['PASS']and validation['point_all7040']==7040 and validation['point_D']==5853 and validation['point_E']==243 and validation['point_excluded_UNKNOWN']==944 and validation['weak_rows']==44 and validation['weak_D']==26 and validation['weak_E']==18,'Fixed all-row target denominators')
    need(validation['original_U_qL_native_q8_unchanged']and validation['original_roles_not_relabelled']and validation['teacher_reference_or_region_or_alpha_not_model_input']and validation['no_old_DIRECT_NPZ_read'],'Input and authority preservation')
    teacher=read(check(fit['point_teacher']));need(teacher['completed']and teacher['PASS']and fit['point_teacher']==p['teacher_complete'],'Exact frozen V211 saved metadata only')
    need(e['teacher_mapping']==teacher['outputs']['ALL7040_INPUT_SOURCE_POINT_MAPPING.jsonl']and e['teacher_summary']==teacher['outputs']['SUMMARY.json'],'Exact saved teacher mapping/summary, no GT')
    pointrows=lines(check(e['teacher_mapping']));lossrows=lines(check(e['point_loss_targets']));need(len(pointrows)==len(lossrows)==7040 and [r['row_id']for r in pointrows]==[r['row_id']for r in lossrows],'All7040 exact row-id order, no cache-index join')
    need(all(r['component_role']=='TRAIN'for r in pointrows),'Only fixed original SOURCE TRAIN point mapping')
    need(all(type(r['loss_target'])is int and r['loss_target']in(-1,0,1)for r in lossrows),'Literal saved D/E/UNKNOWN loss targets, no wrap/coercion')
    pt=np.asarray([r['loss_target']for r in lossrows],np.int8);need([(pt==j).sum()for j in(0,1,-1)]==[5853,243,944],'Saved loss targets6096/944 unchanged')
    weak=read(check(e['weak_metadata']));need(e['weak_metadata']==fit['weak_metadata']==p['weak_metadata']and weak['completed']and weak['PASS'],'Exact44 SOURCE weak metadata')
    weakrows=weak['rows'];need(all(type(r['proposed_weak_target'])is int and r['proposed_weak_target']in(0,1)for r in weakrows),'Literal weak D/E targets only')
    need(all(r['component_role']=='TRAIN'for r in weakrows),'Only original SOURCE TRAIN weak frames')
    wt=np.asarray([r['proposed_weak_target']for r in weakrows],np.int8);need(len(weakrows)==44 and [(wt==j).sum()for j in(0,1)]==[26,18],'All44 weak rows retained')
    summaryteacher=read(check(e['teacher_summary']));need(not OUT.exists(),'Fresh exclusive output, no automatic retry');OUT.mkdir()
    try:
        metrics=[];native=[];packets={};logs=[];logrows=[];ledger=[];counts=source_counts(pointrows,pt)+source_counts(weakrows,wt,True)
        for arm,tag in zip(ARMS,TAGS):
            rec=fit['arms'][arm];bridge,weight=WEAK[arm];need(rec['updates']==4000 and rec['use_sam']is False and rec['spatial_q8']is True and rec['weak_bridge']is bridge and rec['weak_loss_weight']==weight,'Fixed RAW q8 and source role')
            need(rec['initial_state_sha']==fit['same_initial_state_sha']and rec['old_draw_sha']==fit['same_old192_draw_sha']and rec['point_draw_sha']==fit['same_point32_draw_sha'],'Same fresh init/old+point draw hashes')
            packets[tag]={}
            for domain,rows,target in [('POINT7040',pointrows,pt),('WEAK44',weakrows,wt)]:
                packet=output_packet(rec['outputs'][domain],rows);packets[tag][domain]=packet;pred=packet['direct_end_prediction_D0_E1_Uminus1'];av=packet['original_query_available']
                need((target[~av]==-1).all(),'Original inputU never admitted as known source target')
                metrics.append(dict(domain=domain,model=tag,**metric(target,pred,av)))
                for k in range(8):
                    ix=packet['native_slot']==k;native.append(dict(domain=domain,model=tag,native_slot=k,**metric(target[ix],pred[ix],av[ix])))
            lr=lines(check(rec['draws_and_losses']));logs.extend(summarize_log(lr,arm));logrows.extend(dict(arm=arm,update=r['update'],**{k:r[k]for k in LOSS_FIELDS})for r in lr)
        flips=[]
        for domain,rows,target in [('POINT7040',pointrows,pt),('WEAK44',weakrows,wt)]:
            a=packets['C62'][domain];b=packets['C63'][domain]
            need(all(a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and (np.array_equal(a[k],b[k],equal_nan=True)if a[k].dtype.kind=='f'else np.array_equal(a[k],b[k]))for k in META),'Both source outputs identical original META')
            flips.append(dict(domain=domain,**paired(target,a['direct_end_prediction_D0_E1_Uminus1'],b['direct_end_prediction_D0_E1_Uminus1'])))
            for i,r in enumerate(rows):
                row=dict(domain=domain,row_id=r['row_id'],original_global_index=r['original_global_index'],native_slot=r['native_slot'],query_x=None if r['query_xy']is None else r['query_xy'][0],query_y=None if r['query_xy']is None else r['query_xy'][1],L=r['L'],original_query_available=r['original_query_available'],source_loss_target=int(target[i]),original_point0_target_alone_state=r.get('point0_target_alone_state','NOT_APPLICABLE_WEAK_FEATURE'),original_point0_combined_state=r.get('point0_combined_state','NOT_APPLICABLE_WEAK_FEATURE'),original_region_state_context_only=r.get('region_state_context_only','NOT_APPLICABLE_WEAK_FEATURE'),declared_CF_to_canonical=r.get('declared_CF_to_canonical','NOT_APPLICABLE_WEAK_FEATURE'),authority='SOURCE_PROCEDURAL_POINT_NOT_HUMAN_GOLD'if domain=='POINT7040'else'ASSISTANT_SOURCE_LOCAL_FEATURE_WEAK_NOT_NATIVE_GOLD',excluded_reasons=json.dumps(lossrows[i]['excluded_reasons'])if domain=='POINT7040'else'[]')
                for tag in TAGS:
                    packet=packets[tag][domain];raw=float(packet['raw_direct_end_external_logit'][i]);row[tag+'_raw_direct']=raw if math.isfinite(raw)else None;row[tag+'_D0_E1_Uminus1']=int(packet['direct_end_prediction_D0_E1_Uminus1'][i])
                ledger.append(row)
        binaries=owned_binaries(fit)
        csvwrite(OUT/'SOURCE_APPARENT_CM_P_R_U.csv',metrics);csvwrite(OUT/'SOURCE_NATIVE8_CM_P_R_U.csv',native);csvwrite(OUT/'SOURCE_GROUP_COUNTS.csv',counts);csvwrite(OUT/'SOURCE_PAIRED_C62_TO_C63.csv',flips);csvwrite(OUT/'LOSS_SCALAR_SUMMARY.csv',logs);csvwrite(OUT/'ALL8000_LOSS_SCALARS.csv',logrows);csvwrite(OUT/'ALL7084_SOURCE_TARGET_PREDICTION_READOUT.csv',ledger)
        write(OUT/'OWNED_BINARY_PUBLICATION_BINDINGS.json',dict(completed=True,owned_only=True,source_complete=e['source_complete'],entries=binaries,checkpoints=2,prediction_NPZ=sum(x['kind']=='OWN_SOURCE_PREDICTION'for x in binaries),input_or_target_NPZ=0,model_weights_decoded_by_report=False))
        claims=dict(primary_a_priori='C63',ablation_a_priori='C62',source_only=True,source_apparent_not_validation=True,weak44_not_independent_for_C62=True,point_teacher_not_human_native_gold=True,point_HOLDOUT18_not_read_or_merged=True,old_finite_end327_to_point6096_not_label_only_causality=True,both_arms_carry_to_separate_V213=True,zeroU_not_confidence=True,independent_full_neural_training_reproduction=False,selected_model=None,adopted=False,fully_solved=False)
        result=dict(completed=True,PASS=True,phase='V212',code=p['code'],protocol=bind(PROTOCOL),source_complete=e['source_complete'],source_barrier=e['source_barrier'],independent_fit_review=e['independent_fit_review'],root_terminal_evidence=e['root_terminal_evidence'],runtime_seconds=fit['seconds'],parameters_each=118359,updates_each=4000,total_updates=8000,input_validation=validation,metrics=metrics,native8=native,group_counts=counts,paired=flips,loss_summaries=logs,source_teacher_summary=summaryteacher,authority=p['source_authority'],claims=claims,owned_binaries=bind(OUT/'OWNED_BINARY_PUBLICATION_BINDINGS.json'),teacher_mapping=e['teacher_mapping'],point_loss_targets=e['point_loss_targets'],weak_metadata=e['weak_metadata'],actual_prediction_NPZs_opened=4,NPZ_members_decoded=sorted(READ_MEMBERS),model_checkpoint_or_source_input_or_target_NPZ_GT_RGB_or_private_reads=0)
        refs=[('全source CM/P/R/U',OUT/'SOURCE_APPARENT_CM_P_R_U.csv'),('native8',OUT/'SOURCE_NATIVE8_CM_P_R_U.csv'),('원source 그룹 수',OUT/'SOURCE_GROUP_COUNTS.csv'),('paired 변화',OUT/'SOURCE_PAIRED_C62_TO_C63.csv'),('loss 요약',OUT/'LOSS_SCALAR_SUMMARY.csv'),('8000 logged loss',OUT/'ALL8000_LOSS_SCALARS.csv'),('전체7084 source scalar',OUT/'ALL7084_SOURCE_TARGET_PREDICTION_READOUT.csv'),('공개용 owned binary 명세',OUT/'OWNED_BINARY_PUBLICATION_BINDINGS.json'),('SOURCE COMPLETE',e['source_complete']['path']),('두 최종 봉인',e['source_barrier']['path']),('독립 fit 감사',e['independent_fit_review']['path']),('입력 검증',e['input_validation']['path']),('source 감독 protocol',FIT_PROTOCOL),('보고서 코드',Path(__file__))]
        rel=lambda x:Path(os.path.relpath(x,V)).as_posix();doc=['# V212 SOURCE dense-point 학습 결과','', 'C63은 사전 지정 주 모델, C62는 point-only 대조 팔이다. 같은 RAW/native/original q8 구조118,359개 파라미터를 새 초기값으로 각4000회 학습했다. SOURCE apparent fit 결과이며 실사 개선·모델 채택·목표 해결을 주장하지 않는다.','', 'POINT7040 전체를 보존하고 fit-eligible6096(D5853/E243), UNKNOWN-loss944를 분리했다. Weak44는 D26/E18의 source local-feature 약한 감독이다. C62도 old auxiliary에서 같은 공개3frame을 보았으므로 이 weak44 점수는 독립 검증이 아니다.','',table(metrics),'','## 감독과 실험 범위','',p['source_authority']['point'],p['source_authority']['weak'],p['source_authority']['contrast'],p['source_authority']['heldout'],'','CM 행 D/E, 열 D/E/U. Recall은 예측 U를 포함한 원 known 분모를 사용하고 UNKNOWN target은 정확도에서 분리한다. InputU와 known-predictionU도 별개이며 U0은 confidence가 아니다.','', '이전 finite-end327에서 densepoint6096으로의 전환은 감독 정의와 노출량도 달라졌다. C63과C62만 같은 point/old192 draw의 추가 weak branch(.25) 대조다. 두 팔은 SOURCE 점수와 무관하게 모두 별도 V213으로 이어간다.','', '## paired source 변화','',table(flips),'','## 고정 학습·감사 범위','',f'총8000 updates, runtime {fit["seconds"]:.2f}초. AdamW3e-4/wd1e-4/clip5, init1810, old RNG1810·point1811·weak1812. old192+point16D16E, C63만 weak8D8E 가중치.25. 원q/L/native/U·zeroSAM·기존role/nominal fitmask를 유지하고 oldDIRECT input/target NPZ는 제외했다. 두 최종 상태를 먼저 봉인한 뒤 SOURCE 예측을 저장했다.','', '독립 감사는 draw/hash·로그 scalar 산식·opaque checkpoint/봉인을 검토했다. 신경망 loss/gradient/activation 전체를 독립 재학습한 것은 아니다. 원GT·source RGB·input/target NPZ·checkpoint 모델 해석은 이 보고서에서 수행하지 않았다.','', '## 저장 loss 요약','',table(logs),'','## 연결','',*['- [%s](%s)'%(a,rel(b))for a,b in refs],'']
        md='\n'.join(doc);text(V/'PORTABLE_REPORT_KO.md',md);text(V/'ROOT_REPORT_KO.md',md)
        page='<!doctype html><meta charset="utf-8"><style>body{font:17px sans-serif;margin:24px}pre{white-space:pre-wrap}</style><h1>V212 SOURCE apparent fit</h1><pre>'+escape(md)+'</pre><ul>'+''.join('<li><a href="'+rel(b)+'">'+escape(a)+'</a></li>'for a,b in refs)+'</ul>'
        text(V/'ROOT_READOUT_KO.html',page);write(V/'ROOT_RESULTS_KO.json',result)
        n=sum(links_exist(V,(V/x).read_text(),x.endswith('.html'))for x in p['report_targets']if not x.endswith('.json'))
        write(H/'SOURCE_REPORT_BUILD_COMPLETE.json',dict(completed=True,PASS=True,code=p['code'],protocol=bind(PROTOCOL),entry=bind(entry),outputs={x:bind(V/x)for x in p['report_targets']},tables={f.name:bind(f)for f in sorted(OUT.iterdir())if f.is_file()},relative_link_occurrences=n,source_predictions_opened=4,original_source_or_GT_or_model_or_private_or_HOLDOUT_reads=0,browser_rendered=False,selected_model=None,adopted=False,fully_solved=False))
        print(json.dumps(dict(completed=True,PASS=True,complete=bind(H/'SOURCE_REPORT_BUILD_COMPLETE.json'))))
    except BaseException:
        write(OUT/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);a=ap.parse_args();prepare()if a.prepare else run(a.entry)

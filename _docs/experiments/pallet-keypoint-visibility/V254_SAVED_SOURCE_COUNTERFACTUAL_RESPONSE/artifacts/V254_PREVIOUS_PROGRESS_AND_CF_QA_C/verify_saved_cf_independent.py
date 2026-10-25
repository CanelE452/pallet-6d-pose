"""V254 exact six saved SOURCE inputs → 독립 pairing/count/delta QA. 생산자 호출 없음."""
import argparse, collections, csv, hashlib, io, itertools, json, math, os, stat, struct
from pathlib import Path
import numpy as np

COMPLETE_SHA='26b608c88639f4a6056c78905c1cf008f47d7552ced935e43b1540db28a2b54d'
PROTOCOL_SHA='a36c2b65c779b71110261bca43237612d4042a458a733fa01bbbadd991215668'
FAMILIES=('SELF_SCENE15','BEFORE_ANCHOR4')
VARIANTS=('BEFORE','COVER','NEAR_SHAM','FAR_SHAM')
SHAMS=('NEAR_SHAM','FAR_SHAM')
MODELS=('BASE','CONTROL','TREAT')
SAMPLE_KEYS=('row_id','family','target','fit','available','condition')
verified=[]
numeric_fields=0
largest_difference=0.


def safe(value):
    p=Path(value)
    assert p.is_absolute() and '..'not in p.parts and not any(t in str(p).casefold()for t in ('v112','cache','.pyc','current223','publicheld','source_holdout'))
    for a in reversed((p,*p.parents)):
        assert not stat.S_ISLNK(os.lstat(a).st_mode),str(a)
    assert stat.S_ISREG(os.lstat(p).st_mode)
    return p


def read(binding):
    p=safe(binding['path'])
    with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb')as f:data=f.read()
    actual=dict(path=str(p),sha256=hashlib.sha256(data).hexdigest(),size_bytes=len(data))
    assert actual==binding,(actual,binding)
    verified.append(actual)
    return data


def sha(data):return hashlib.sha256(data).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def typed(x):
    if x is None:return b'N'
    if isinstance(x,(list,tuple)):
        return b'L'+struct.pack('<I',len(x))+b''.join(typed(v)for v in x)
    assert not isinstance(x,(bool,np.bool_))and isinstance(x,(int,float,np.number))
    return b'F'+struct.pack('<d',float(x))
def equal64(values):return len(set(typed(x)for x in values))==1
def all3(values):return False if False in values else None if None in values else True
def state(x):return 'UNKNOWN'if x is None else 'PASS'if x else 'FAIL'
def integer_counter(values):return dict(sorted(collections.Counter(values).items()))


def compare(a,b,where=''):
    global numeric_fields,largest_difference
    if isinstance(b,dict):
        assert isinstance(a,dict)and set(a)==set(b),(where,set(a),set(b))
        for k in b:compare(a[k],b[k],where+'/'+str(k))
    elif isinstance(b,(list,tuple)):
        assert isinstance(a,(list,tuple))and len(a)==len(b),where
        for i,v in enumerate(b):compare(a[i],v,where+'/'+str(i))
    elif b is None or isinstance(b,(bool,str)):
        assert a==b and type(a)==type(b),(where,a,b)
    else:
        assert isinstance(a,(float,int))and not isinstance(a,bool),(where,a,b)
        assert math.isfinite(float(a))and math.isfinite(float(b)),where
        delta=abs(float(a)-float(b));largest_difference=max(delta,largest_difference);numeric_fields+=1
        assert delta<=3e-11*max(1.,abs(float(b))),(where,a,b,delta)


def csv_compare(data,rows,name):
    got=list(csv.DictReader(io.StringIO(data.decode(),newline='')))
    assert len(got)==len(rows),(name,len(got),len(rows))
    for i,(a,b)in enumerate(zip(got,rows)):
        assert set(a)==set(b),(name,i,set(a),set(b))
        d={}
        for k,want in b.items():
            s=a[k]
            if want is None:assert s=='';v=None
            elif isinstance(want,bool):assert s in ('True','False');v=s=='True'
            elif isinstance(want,(dict,list,tuple)):v=json.loads(s)
            elif isinstance(want,int):v=int(s)
            elif isinstance(want,float):v=float(s)
            else:v=s
            d[k]=v
        compare(d,b,name+'/'+str(i))
    return len(got)


def stats(values):
    a=sorted(float(v)for v in values)
    if not a:return dict(n=0,mean=None,median=None,min=None,q25=None,q75=None,max=None)
    def quantile(p):
        z=p*(len(a)-1);i=math.floor(z);j=math.ceil(z)
        return a[i]+(a[j]-a[i])*(z-i)
    return dict(n=len(a),mean=math.fsum(a)/len(a),median=quantile(.5),min=a[0],q25=quantile(.25),q75=quantile(.75),max=a[-1])


def compute(payload):
    groups=payload['groups'];rows=payload['all_rows_join'];mapping=payload['point_mapping'];scenes=payload['render_input_scenes'];s=payload['samples']
    assert len(groups)==1767 and len(rows)==15040 and len(mapping)==7040
    assert [g['group_index']for g in groups]==list(range(1767))and len({g['group_id']for g in groups})==1767
    assert [r['full_row_index']for r in rows]==list(range(15040))and len({r['row_id']for r in rows})==15040
    assert set(s)==set(SAMPLE_KEYS)and all(len(x)==7617 for x in s.values())
    assert s['condition'].dtype==np.float32 and s['condition'].shape==(7617,32)and np.isfinite(s['condition']).all()
    assert s['family'].dtype==s['target'].dtype==np.int8 and s['fit'].dtype==s['available'].dtype==bool
    assert np.isin(s['target'],[0,1]).all()and set(s['family'].tolist())=={0,1}and len(set(s['row_id'].tolist()))==7617
    group_hash={}
    for g in groups:
        assert len(g['q8'])==len(g['L8'])==len(g['original_available8'])==8
        assert all(q is None or len(q)==2 for q in g['q8'])and all(type(v)is bool for v in g['original_available8'])
        group_hash[g['group_index']]=sha(typed(g['q8'])+typed(g['L8'])+np.asarray(g['original_available8'],np.bool_).tobytes()+typed(g['box_xyxy'])+typed(g['raw_hw']))
    by_id={};known={};slots=collections.defaultdict(collections.Counter)
    for r in rows:
        g=groups[r['group_index']];k=r['active_slot']
        assert type(k)is int and 0<=k<8 and g['group_id']==r['group_id']and g['family_code']==r['family_code']
        assert equal64([g['q8'][k],r['q']])and equal64([g['L8'][k],r['L']])and g['original_available8'][k]is r['original_query_available']
        slots[r['group_index']][k]+=1;by_id[r['row_id']]=r
        si=r['sample_index']
        if si is not None:
            assert type(si)is int and 0<=si<7617 and si not in known
            assert r['row_id']==str(s['row_id'][si])and r['family_code']==int(s['family'][si])and r['target']==int(s['target'][si])
            assert r['fit']is bool(s['fit'][si])and r['crop_available']is bool(s['available'][si])
            assert r['original_state']==('D'if r['target']==0 else'E')
            if r['family_code']==0:
                assert r['native_slot']==k and r['logical_aux_slot']is None
                one=np.zeros(8,np.float32);one[k]=1
                assert s['condition'][si,:8].tobytes()==one.tobytes()
            else:
                assert r['native_slot']is None and r['logical_aux_slot']==k and not s['condition'][si].any()
            known[si]=r
    assert set(known)==set(range(7617))and all(set(slots[i])==set(range(8))for i in range(1767))
    family_counts=integer_counter(r['family_code']for r in rows)
    assert family_counts=={0:7040,1:8000}and sum(r['sample_index']is None for r in rows)==7423
    assert int(s['fit'].sum())==5876 and int((~s['fit']).sum())==1741
    census=dict(all_original_rows=15040,all_original_groups=1767,known_samples=7617,masked_rows_retained=7423,family_rows=family_counts,original_state_counts=integer_counter(r['original_state']for r in rows),source_role_counts=integer_counter(r['internal_source_role']for r in rows),groups_all_active_slots_0_to_7_complete=True,groups_with_alias_rows=sum(sum(c.values())>8 for c in slots.values()),extra_alias_memberships=sum(sum(c.values())-8 for c in slots.values()),exactly_8_rows_per_group_required=False,known_condition32_finite_and_SYN_native_alignment=True,census_before_any_known_or_score_filter=True,typed_numeric_comparison='Decoded values cast LE FP64, nullable tree preserved; prior F32->F64 promotion preserved; tolerance0, not JSON-text equality')
    assert scenes['completed']is True
    scene_by_id={x['scene_index']:x for x in scenes['rows']};assert len(scene_by_id)==len(scenes['rows'])and len({m['row_id']for m in mapping})==7040
    syn=[]
    for m in mapping:
        r=by_id[m['row_id']];k=m['native_slot'];scene=m['scene_index'];g=groups[r['group_index']];x=scene_by_id.get(scene)
        assert r['family_code']==0 and m['family']in FAMILIES and m['state_name']=='NOMINAL'and m['state_index']==0 and m['variant']in VARIANTS
        assert 0<=k<8 and r['active_slot']==r['native_slot']==k
        checks={}
        checks['actual_scene_variant_order']=scene==4*m['base_index']+VARIANTS.index(m['variant'])
        checks['actual_image_teacher_key']=(m.get('source_teacher_key')==[scene,k]and m.get('actual_image_scene_index')==scene and m.get('actual_image_variant')==m['variant']and m.get('source_image_original_global_index')==8*scene+k and m.get('actual_image_original_global_index')==8*scene+k and m.get('mapping_status')=='EXACT_ORIGINAL_IMAGE_OWN_SCENE')
        checks['row_mapping_active_input']=equal64([r['q'],m['query_xy']])and equal64([r['L'],m['L']])and r['original_query_available']is m['original_query_available']
        anchor=m['anchor_scene_index']
        checks['anchor_identity']=m['anchor_original_global_index']==8*anchor+k and (anchor==scene and m['anchor_variant']==m['variant']if m['family']=='SELF_SCENE15'else anchor==4*m['base_index']and m['anchor_variant']=='BEFORE')
        if x is None:
            checks['render_scene_identity']=None;checks['actual_image_binding']=None;hw=None
        else:
            checks['render_scene_identity']=x['base_index']==m['base_index']and x['variant']==m['variant']
            image=g['image'];inert=m.get('actual_image_source_binding_inert');full=x.get('full_RGB')
            checks['actual_image_binding']=isinstance(inert,dict)and isinstance(full,dict)and image.get('path')==inert.get('path')==full.get('path')and image.get('sha256')==inert.get('sha256')==full.get('sha256')and image.get('format')=='npz'and image.get('member')==x.get('full_RGB_key')and x.get('full_RGB_format')=='NPZ_RGB_UINT8_HWC'
            hw=equal64([g['raw_hw'],x.get('raw_hw')])
        declared=m.get('declared_CF_to_canonical');checks['declared_target_mapping_present']=None if declared is None else isinstance(declared,int)and 0<=declared<8
        si=r['sample_index'];condition_sha=None if si is None else sha(np.asarray(s['condition'][si],dtype='<f4').tobytes())
        syn.append(dict(row_id=r['row_id'],full_row_index=r['full_row_index'],source_index=r['source_index'],sample_index=si,augmentation_family=m['family'],sample_family=0,base_index=m['base_index'],native_slot=k,variant=m['variant'],scene_index=scene,actual_image_scene_index=m.get('actual_image_scene_index'),anchor_scene_index=anchor,declared_CF_to_canonical=declared,fit=r['fit'],source_role='FIT'if r['fit']else'DEV',internal_source_role=r['internal_source_role'],component_id=r['component_id'],original_state=r['original_state'],target=r['target'],original_query_available=r['original_query_available'],crop_available=r['crop_available'],point0_target_alone_state=m.get('point0_target_alone_state','UNKNOWN'),point0_combined_state=m.get('point0_combined_state','UNKNOWN'),group_index=r['group_index'],group_id=r['group_id'],group_input_fingerprint=group_hash[r['group_index']],condition32_sha256=condition_sha,row_lineage=all3(list(checks.values())),row_checks=checks,group_raw_hw=g['raw_hw'],render_raw_hw=None if x is None else x.get('raw_hw'),group_vs_render_raw_hw_nullable_exact=hw,group_vs_render_raw_hw_status='UNKNOWN_GROUP_UNDECLARED'if g['raw_hw']is None else state(hw),image_sha256=g['image']['sha256'],_group=g,_mapping=m,_scene=x))
    syn.sort(key=lambda r:r['source_index'])
    assert {r['row_id']for r in syn}=={r['row_id']for r in rows if r['family_code']==0}
    census.update(SYN7040_preserved_before_saved_score_join=len(syn),SYN_masked_rows_retained=sum(r['sample_index']is None for r in syn),SYN_state_counts=integer_counter(r['original_state']for r in syn),SYN_point0_combined_state_counts=integer_counter(r['point0_combined_state']for r in syn))
    assert census['SYN_masked_rows_retained']==944
    scores=payload['saved_scores'];assert len(scores)==7617;scoremap={}
    for a in scores:
        i=int(a['sample_index']);assert i not in scoremap and 0<=i<7617
        assert a['row_id']==str(s['row_id'][i])and int(a['family'])==int(s['family'][i])and int(a['target'])==int(s['target'][i])
        assert a['role']==('FIT'if s['fit'][i]else'DEV')and a['available']==str(bool(s['available'][i]))
        for model in MODELS:
            assert a[model+'_prediction']in ('D','E','U')
            p={'D':0,'E':1,'U':-1}[a[model+'_prediction']];z=float(a[model+'_E_logit'])
            assert math.isfinite(z)and z==float(a[model+'_logit'+str(int(s['family'][i]))])
            assert p==(int(z>=0)if s['available'][i]else-1)
        scoremap[i]=a
    assert set(scoremap)==set(range(7617))
    for r in syn:
        i=r['sample_index'];r['saved_score_present']=i is not None
        for model in MODELS:
            r[model+'_prediction']=None if i is None else {'D':0,'E':1,'U':-1}[scoremap[i][model+'_prediction']]
            r[model+'_E_logit']=None if i is None else float(scoremap[i][model+'_E_logit'])
    cells=collections.defaultdict(list)
    for r in syn:cells[(r['augmentation_family'],r['base_index'],r['native_slot'],r['variant'])].append(r)
    candidates=[];responses=[]
    for fam,base,k,sham in itertools.product(FAMILIES,sorted({r['base_index']for r in syn}),range(8),SHAMS):
        members=[cells[(fam,base,k,v)]for v in ('BEFORE','COVER',sham)];rr=[a[0]if len(a)==1 else None for a in members]
        if any(len(a)!=1 for a in members):
            check=dict(members_unique=False,lineage=None,input_exact=None,condition32_exact=None,role='UNKNOWN',same_declared_component=None,D_E_D_labels=False,groupIDs_equal_diagnostic_only=None,actual_image_SHAs_different_diagnostic_only=None,saved_reference_nullable_exact_diagnostic_only=None,axes={},reasons=['ABSENT_OR_DUPLICATE_MEMBER'])
        else:
            gs=[r['_group']for r in rr];ss=[r['_scene']for r in rr];ms=[r['_mapping']for r in rr]
            configs=[None if x is None else x.get('source_base_config')for x in ss];declared=[r['declared_CF_to_canonical']for r in rr]
            config_ok=all(isinstance(x,dict)and isinstance(x.get('file'),dict)and x['file'].get('sha256')and x.get('row_index')is not None for x in configs)
            lin=dict(all_member_lineage=all3([r['row_lineage']for r in rr]),common_base=len({r['base_index']for r in rr})==1,common_native_address=len({r['native_slot']for r in rr})==1,common_declared_target_mapping=None if any(v is None for v in declared)else len(set(declared))==1,same_declared_base_config_camera=None if not config_ok else len({canon(x)for x in configs})==1,same_declared_style_factors=None if any(x is None for x in ss)else len({canon([x.get('style'),x.get('factors')])for x in ss})==1)
            geometry=dict(q8_FP64_nullable_exact=equal64([g['q8']for g in gs]),L8_FP64_nullable_exact=equal64([g['L8']for g in gs]),availability8_bool_exact=len({np.asarray(g['original_available8'],np.bool_).tobytes()for g in gs})==1,box_xyxy_FP64_nullable_exact=equal64([g['box_xyxy']for g in gs]),raw_hw_nullable_exact=equal64([g['raw_hw']for g in gs]),active_q_FP64_nullable_exact=equal64([m['query_xy']for m in ms]),active_L_FP64_nullable_exact=equal64([m['L']for m in ms]),active_U_original_exact=len({m['original_query_available']for m in ms})==1,active_native_exact=len({m['native_slot']for m in ms})==1)
            cond=[r['condition32_sha256']for r in rr];cond_ok=None if any(x is None for x in cond)else len(set(cond))==1;geometry['known_condition32_FP32_exact']=cond_ok
            role='FIT'if all(r['fit']for r in rr)else'DEV'if all(not r['fit']for r in rr)else'MIXED'
            component=None if any(r['component_id']is None for r in rr)else len({r['component_id']for r in rr})==1
            axes={**lin,**geometry}
            reasons=[key+':'+state(value)for key,value in axes.items()if value is not True]+([]if role=='FIT'else['NOT_ALL_MEMBER_FIT'])+([]if component is True else['COMPONENT_IDENTITY:'+state(component)])
            check=dict(members_unique=True,lineage=all3(list(lin.values())),input_exact=all3(list(geometry.values())),condition32_exact=cond_ok,role=role,same_declared_component=component,D_E_D_labels=[r['original_state']for r in rr]==['D','E','D']and[r['target']for r in rr]==[0,1,0],groupIDs_equal_diagnostic_only=len({r['group_id']for r in rr})==1,actual_image_SHAs_different_diagnostic_only=len({r['image_sha256']for r in rr})==3,saved_reference_nullable_exact_diagnostic_only=equal64([m.get('saved_projected_reference_xy')for m in ms]),axes=axes,reasons=reasons)
        cid=f'{fam}:{base}:{k}:{sham}';cand=dict(candidate_id=cid,augmentation_family=fam,base_index=base,native_slot=k,sham=sham,**check)
        for prefix,aa in zip(('BEFORE','COVER','SHAM'),members):
            cand[prefix+'_member_count']=len(aa)
            for field,sourcefield in [('row_ids','row_id'),('sample_indices','sample_index'),('states','original_state'),('group_indices','group_index')]:cand[prefix+'_'+field]=[r[sourcefield]for r in aa]
        candidates.append(cand)
        for model in MODELS:
            saved=all(r is not None and r[model+'_prediction']in(0,1)and r['saved_score_present']and r['crop_available']and r['original_query_available']for r in rr)
            exact=bool(saved and check['lineage']is True and check['input_exact']is True)
            primary=exact and check['role']=='FIT'and check['D_E_D_labels'];zs=[None if r is None else r[model+'_E_logit']for r in rr];ps=[None if r is None else r[model+'_prediction']for r in rr]
            response=dict(candidate_id=cid,augmentation_family=fam,base_index=base,native_slot=k,sham=sham,role=check['role'],model=model,members_unique=check['members_unique'],lineage=check['lineage'],input_exact=check['input_exact'],D_E_D_labels=check['D_E_D_labels'],all_saved_scores_available=saved,exact_scored_triplet=exact,primary_FIT_D_E_D=primary,BEFORE_E_logit=zs[0],COVER_E_logit=zs[1],SHAM_E_logit=zs[2],BEFORE_prediction=ps[0],COVER_prediction=ps[1],SHAM_prediction=ps[2],prediction_sequence='/'.join('U'if p is None or p==-1 else'D'if p==0 else'E'for p in ps),predicted_D_E_D=None if not exact else ps==[0,1,0])
            for name,a,b in [('cover_before',1,0),('sham_before',2,0),('cover_sham',1,2)]:response[name]=zs[a]-zs[b]if exact else None
            responses.append(response)
    summary_rows=[]
    for fam,sham,role,k,model in itertools.product(FAMILIES,SHAMS,('ALL','FIT','DEV','MIXED','UNKNOWN'),('ALL',*range(8)),MODELS):
        rr=[r for r in responses if r['augmentation_family']==fam and r['sham']==sham and r['model']==model and(role=='ALL'or r['role']==role)and(k=='ALL'or r['native_slot']==k)]
        pp=[r for r in rr if r['primary_FIT_D_E_D']]
        d=dict(augmentation_family=fam,sham=sham,role=role,native_slot=k,model=model,all_candidates=len(rr),members_unique=sum(r['members_unique']for r in rr),lineage_PASS=sum(r['lineage']is True for r in rr),input_exact_PASS=sum(r['input_exact']is True for r in rr),input_exact_UNKNOWN=sum(r['input_exact']is None for r in rr),original_D_E_D_labels=sum(r['D_E_D_labels']for r in rr),all_saved_scores_available=sum(r['all_saved_scores_available']for r in rr),exact_scored_triplets_any_role=sum(r['exact_scored_triplet']for r in rr),primary_FIT_D_E_D=len(pp),predicted_D_E_D=sum(r['predicted_D_E_D']is True for r in pp),predicted_D_E_D_rate=sum(r['predicted_D_E_D']is True for r in pp)/len(pp)if pp else None)
        for key in ('cover_before','sham_before','cover_sham'):
            vv=[r[key]for r in pp]
            d.update({key+'_'+k:v for k,v in stats(vv).items()})
            d[key+'_positive']=sum(v>0 for v in vv);d[key+'_zero']=sum(v==0 for v in vv);d[key+'_negative']=sum(v<0 for v in vv)
        d.update({'abs_sham_before_'+k:v for k,v in stats([abs(r['sham_before'])for r in pp]).items()})
        d['primary_prediction_sequences']=integer_counter(r['prediction_sequence']for r in pp);summary_rows.append(d)
    for r in syn:
        for key in ('_group','_mapping','_scene'):del r[key]
    summary_checks=dict(census=census,models=list(MODELS),overview=[r for r in summary_rows if r['role']=='ALL'and r['native_slot']=='ALL'],candidate_count_including_missing=len(candidates),candidate_axis_counts={key:integer_counter(state(c[key])for c in candidates)for key in ('lineage','input_exact','condition32_exact')},candidate_role_counts=integer_counter(c['role']for c in candidates),reason_counts=integer_counter(reason for c in candidates for reason in c['reasons']),new_optimizer_updates=0,new_model_forwards=0,SOURCE_derived_teacher_metadata_read=True,current_or_held_payloads=0)
    return census,syn,candidates,responses,summary_rows,summary_checks


def run(path):
    p=safe(path);raw=p.read_bytes();assert sha(raw)==COMPLETE_SHA
    c=json.loads(raw);assert c['completed']is True and c['PASS']is True and c['kind']=='SAVED_VALUE_AUDIT_PIPELINE_NOT_MODEL_PERFORMANCE_PASS'
    assert set(c['inputs6'])=={'saved_scores','samples','groups','all_rows_join','point_mapping','render_input_scenes'}
    source={k:read(v)for k,v in c['inputs6'].items()};outputs={k:read(v)for k,v in c['outputs'].items()}
    payload={k:[json.loads(line)for line in source[k].decode().splitlines()if line]for k in ('groups','all_rows_join','point_mapping')}
    payload['render_input_scenes']=json.loads(source['render_input_scenes'])
    with np.load(io.BytesIO(source['samples']),allow_pickle=False)as z:payload['samples']={k:z[k]for k in SAMPLE_KEYS}
    payload['saved_scores']=list(csv.DictReader(io.StringIO(source['saved_scores'].decode(),newline='')))
    census,syn,candidates,responses,summaries,summary_checks=compute(payload)
    expected_census=json.loads(json.dumps(census));compare(json.loads(outputs['ALL15040_CENSUS.json']),expected_census,'census')
    table_counts={}
    for name,rows in [('SYN7040_ROWS.csv',syn),('CF_TRIPLETS.csv',candidates),('RESPONSE_MODEL_ROWS.csv',responses),('RESPONSE_SUMMARY.csv',summaries)]:table_counts[name]=csv_compare(outputs[name],rows,name)
    saved_summary=json.loads(outputs['SUMMARY.json'])
    for key,want in summary_checks.items():compare(saved_summary[key],json.loads(json.dumps(want)),key)
    assert saved_summary['completed']is True and c['summary']==saved_summary
    assert c['SOURCE_DEV_nonpixel_metadata_and_saved_outputs_loaded']is True and c['SOURCE_DEV_used_for_fit_or_selection']is False
    assert c['SOURCE_derived_teacher_metadata_read']is True and c['current_or_held_payloads']==0 and c['raw_RGB_GT_archive_weights_model_feature_reads']==0
    return dict(completed=True,PASS=True,producer_complete=dict(path=str(p),sha256=COMPLETE_SHA,size_bytes=len(raw)),verified_bindings=verified,table_rows=table_counts,numeric_fields_compared=numeric_fields,max_absolute_numeric_difference=largest_difference,census=expected_census,candidate_axes=summary_checks['candidate_axis_counts'],candidate_roles=summary_checks['candidate_role_counts'],reason_counts=summary_checks['reason_counts'],primary_overview=summary_checks['overview'],independence='생산자 import/call 없이 exact six saved inputs의 census/lineage/nullable-F64 equality/known condition/role/labels/scores→candidate axes→모든 delta/분포/pattern을 별도 구현하여 검산했다.',SOURCE_DEV_nonpixel_metadata_and_saved_outputs_loaded=True,SOURCE_DEV_used_for_fit_or_selection=False,SOURCE_derived_teacher_metadata_read=True,raw_RGB_GT_archives_CP_model_features_or_current_held_reads=0,new_fitting_forward_model_data_architecture_threshold_selection=False,limits=['NEAR/FAR는 BEFORE/COVER를 공유하므로 합쳐 독립 개입 수로 부르면 안 된다.','SYN 입력 주소/point0 expected witness와 물리 native/literal q-owner/wholepart gold를 구분한다.','이것은 반복 노출 SOURCE FIT 저장 반응 검산이며 실제 현재 팔레트 전이나 채택을 입증하지 않는다.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--complete',required=True);parser.add_argument('--out',required=True);a=parser.parse_args()
    out=Path(a.out);assert out.is_absolute()and '..'not in out.parts and not any(t in str(out).casefold()for t in ('v112','cache','.pyc'))
    for p in reversed((out,*out.parents)):
        try:s=os.lstat(p)
        except FileNotFoundError:continue
        assert not stat.S_ISLNK(s.st_mode)
    out.mkdir(exist_ok=False)
    try:
        result=run(a.complete);code=Path(__file__).read_bytes();result['checker']=dict(path=str(Path(__file__)),sha256=sha(code),size_bytes=len(code))
        b=(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode();p=out/'REVIEW.json';p.write_bytes(b)
        print(json.dumps(dict(completed=True,PASS=True,review=dict(path=str(p),sha256=sha(b),size_bytes=len(b)),table_rows=result['table_rows'],candidate_axes=result['candidate_axes'],numeric_fields=numeric_fields,max_numeric_difference=largest_difference,overview=[{k:r[k]for k in ('augmentation_family','sham','model','all_candidates','exact_scored_triplets_any_role','primary_FIT_D_E_D','predicted_D_E_D','cover_sham_positive')}for r in result['primary_overview']]),ensure_ascii=False))
    except BaseException as error:
        (out/'FAILURE.json').write_text(json.dumps(dict(completed=False,PASS=False,error=str(error),error_type=type(error).__name__,automatic_retry=False),ensure_ascii=False,indent=2)+'\n');raise

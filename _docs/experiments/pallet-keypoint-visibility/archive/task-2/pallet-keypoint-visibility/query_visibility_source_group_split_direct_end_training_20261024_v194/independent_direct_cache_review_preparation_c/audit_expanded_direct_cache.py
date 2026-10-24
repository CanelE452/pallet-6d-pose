"""Independent all-source saved cache audit; no producer/model imports."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,math,os,stat,sys,time,traceback
import numpy as np
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
PROTO=H/'PROTOCOL_BEFORE_EXPANDED_DIRECT_CACHE_ARRAY_READS.json'
META=('row_id','original_global_index','native_slot','query_xy','L','original_query_available')
INPUTS={'rgb96','valid96','sam_mask96','sam_available96'}|set(META)
TARGETS={'direct_end_external_target','authority'}|set(META)
SAM={'image_logits','lowres_logits','predicted_quality','mask_logits_gt0','box_xyxy','original_query_xy','original_L','native_slot','original_global_index','original_occurrence_id','original_query_available','prediction_available'}
VARIANTS=('BEFORE','COVER','SHAM')

def need(v,msg):
    if not bool(v):raise AssertionError(msg)
def safe(path):
    p=Path(path);need(p.is_absolute() and '..' not in p.parts and all('v112' not in x.casefold() for x in p.parts),'Safe name before access')
    need(p.is_relative_to(P) and not any(x in str(p).casefold() for x in ('current223','_v169/','_v177/','_v178/','_v184/','_v191/','_v182/','synthetic_val')),'Exact source-only path')
    return p
def opened(path):
    p=safe(path);fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
        end=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    need(stat.S_ISREG(os.fstat(end).st_mode),'Regular exact file only');return os.fdopen(end,'rb')
def raw(path):
    with opened(path) as f:return f.read()
def bind(path):return dict(path=str(path),sha256=hashlib.sha256(raw(path)).hexdigest())
def checked(b):
    z=raw(b['path']);need(hashlib.sha256(z).hexdigest()==b['sha256'],'Exact metadata SHA');return z
def jd(b):return json.loads(checked(b))
def jl(b):return [json.loads(s) for s in checked(b).splitlines() if s.strip()]
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def values(a,b):return np.asarray(a).shape==np.asarray(b).shape and np.array_equal(a,b,equal_nan=True) if np.asarray(a).dtype.kind in 'fc' else np.array_equal(a,b)
def write(path,x):
    with path.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def array(b,keys,allow,exact_keys=False):
    need(b['path'] in allow and allow[b['path']]==b,'Binary exact whitelist before hash/header')
    with opened(b['path']) as f:
        h=hashlib.sha256()
        for z in iter(lambda:f.read(1<<20),b''):h.update(z)
        need(h.hexdigest()==b['sha256'],'Array bound bytes SHA');f.seek(0)
        with np.load(f,allow_pickle=False) as z:
            need((set(z.files)==set(keys)) if exact_keys else set(keys)<=set(z.files),'Explicit packet schema')
            return {k:z[k].copy() for k in keys}

def qarray(q):return np.asarray([np.nan,np.nan] if q is None else q,np.float64)
def box(q8):
    a=np.asarray(q8,np.float64)
    if a.shape!=(8,2) or not np.isfinite(a).all():return None
    out=np.asarray([min(a[:,0]),min(a[:,1]),max(a[:,0]),max(a[:,1])],np.float64)
    return out if out[2]>out[0] and out[3]>out[1] else None

def sample(logits,q,L,available,hw):
    mask=np.zeros((96,96),bool);valid=np.zeros_like(mask)
    qa=qarray(q)
    if not available or qa.shape!=(2,) or not np.isfinite(qa).all() or np.array_equal(qa,[-1.,-1.]) or L is None or not np.isfinite(L) or L<=0:return mask,valid
    need(logits.dtype==np.float32 and logits.shape==tuple(hw) and np.isfinite(logits).all(),'Finite full-image SAM logits')
    off=(np.arange(96,dtype=np.float64)+.5)/96-.5;x,y=np.meshgrid(qa[0]+off*L,qa[1]+off*L)
    ix,iy=np.floor(x).astype(np.int64),np.floor(y).astype(np.int64);fx,fy=x-ix,y-iy;out=np.zeros((96,96),np.float64);h,w=hw
    for ax,ay,weight in ((0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),(0,1,(1-fx)*fy),(1,1,fx*fy)):
        xx,yy=ix+ax,iy+ay;inside=(xx>=0)&(xx<w)&(yy>=0)&(yy<h);out[inside]+=logits[yy[inside],xx[inside]].astype(np.float64)*weight[inside]
    valid=(x>=-.5)&(x<w-.5)&(y>=-.5)&(y<h-.5)
    return (out>0)&valid,valid

def scalar_sample(logits,q,L,available):
    h,w=logits.shape;out=np.zeros((96,96),bool);valid=np.zeros_like(out)
    if not available:return out,valid
    for j in range(96):
        for i in range(96):
            x=q[0]+((i+.5)/96-.5)*L;y=q[1]+((j+.5)/96-.5)*L;ix,iy=math.floor(x),math.floor(y);dx,dy=x-ix,y-iy;s=0.
            for ax,ay,z in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):
                if 0<=ix+ax<w and 0<=iy+ay<h:s+=float(logits[iy+ay,ix+ax])*z
            valid[j,i]=-.5<=x<w-.5 and -.5<=y<h-.5;out[j,i]=s>0 and valid[j,i]
    return out,valid

def selftest():
    z=np.arange(35,dtype=np.float32).reshape(5,7)-17
    for q,L in [([2.25,1.75],8.),([-.25,-.25],2.),([6.25,4.25],9.)]:
        a,b=sample(z,q,L,True,z.shape);c,d=scalar_sample(z,q,L,True);need(exact(a,c) and exact(b,d),'Independent scalar/F64 96 bilinear')
    a,b=sample(None,None,None,False,[5,7]);need(not a.any() and not b.any(),'U before logit read')
    a,b=sample(np.zeros((5,7),np.float32),[2.,2.],2.,True,[5,7]);need(not a.any() and b.all(),'Strict0 SAM threshold')
    need(box([[-2.,1.],[8.,1.],[8.,9.],[-2.,9.]]*2).tolist()==[-2.,1.,8.,9.],'Original unclipped q8 bbox')
    need(ash(np.zeros((2,3,3),np.uint8))!=ash(np.ones((2,3,3),np.uint8)),'ChangedRGB identity differs')
    need(set(INPUTS).isdisjoint({'direct_end_external_target','authority'}),'Targets outside input10')
    return dict(completed=True,PASS=True,invented_only=True,checks=8,source_body_header_or_asset_SHA_reads=0)


RAW4=('image_logits','lowres_logits','predicted_quality','mask_logits_gt0')
PROOF14=('original_q_L_native_flags_and_all8_inputs_unchanged','before_original_RGB96_valid96_target_context_exact','every_candidate_positive_RGB_tap_cover_alpha_one','every_candidate_positive_RGB_tap_sham_alpha_zero','distinct_foreign_material_supported','foreign_in_front_paste_composite_or_source_order','same_transformed_foreign_RGB_alpha_cover_and_sham','equal_positive_alpha_pasted_mass','background_outside_paste_exact','original_crop_cover_RGB_bytes_changed','original_crop_sham_has_donor_RGB_support','end_related_original96_COVER_bytes_changed','end_related_original96_SHAM_bytes_unchanged','certified_surrounding_material_RGB_unchanged_both')
def target8(group):
    out=np.full(8,-1,np.int8)
    if group['admitted']:
        k=group['selected_native_slot'];need(0<=k<8 and group['metadata8'][k]['original_query_available'] is True,'Known direct target requires original available selected slot')
        need(group['admitted_target']==(1 if group['variant']=='COVER' else 0),'Frozen D0/E1/D0 rule');out[k]=group['admitted_target']
    else:need(group['admitted_target']==-1,'Unadmitted target UNKNOWN')
    return out

def join_sidecar(side,keys):
    result={}
    for r in side:
        key=tuple(r['control_join_key']);need(key not in result,'Duplicate sidecar control key');result[key]=r
    need(set(result)==set(keys),'All current control keys, no old ordering alias')
    return result

def reuse_kind(key,seen,old):return 'within' if key in seen else ('old' if key in old else 'fresh')
def more_selftests():
    checks=selftest();n=0
    g=dict(admitted=True,selected_native_slot=4,variant='COVER',admitted_target=1,metadata8=[dict(original_query_available=True)for _ in range(8)])
    need(target8(g).tolist()==[-1,-1,-1,-1,1,-1,-1,-1],'Selected-only direct target');n+=1
    g['metadata8'][4]['original_query_available']=False
    try:target8(g);raise RuntimeError('U guard failed')
    except AssertionError:pass
    n+=1
    key=('domain','attempt','SHAM',7);r=dict(control_join_key=list(key));need(join_sidecar([r],[key])[key] is r,'Keyjoin independent of row order');n+=1
    try:join_sidecar([r,r],[key]);raise RuntimeError('Duplicate guard failed')
    except AssertionError:pass
    n+=1
    try:join_sidecar([], [key]);raise RuntimeError('Missing guard failed')
    except AssertionError:pass
    n+=1
    need([reuse_kind('a',{},{}),reuse_kind('a',{}, {'a':1}),reuse_kind('a',{'a':1},{'a':1})]==['fresh','old','within'],'Explicit old/within/fresh disjoint precedence');n+=1
    z=np.zeros((5,7),np.float32);z[2,3]=1
    a,b=sample(z,[3.,2.],96.,True,z.shape);c,d=scalar_sample(z,[3.,2.],96.,True);need(exact(a,c)and exact(b,d),'Sparse 4tap zero-pad boundary');n+=1
    checks['checks']+=n;return checks

def run(entry_path):
    e=json.loads(raw(entry_path));need(e['approved_actual_independent_cache_audit'] is True and e['source_only'] is True and e['automatic_retry'] is False,'Root exact once entry')
    p=json.loads(raw(PROTO));need(e['code']==p['code']==bind(Path(__file__).absolute())and e['protocol']==bind(PROTO),'Frozen auditor')
    need(e['complete']==p['complete'] and e['split_complete']==p['split_complete']and e['partition_QA']==p['partition_QA'],'Pinned completion/split metadata')
    need(sys.executable==p['runtime']['executable']and np.__version__==p['runtime']['numpy'],'Pinned independent numeric runtime')
    scope=jd(e['root_scope']);need(scope['root_authorized'] is True and scope['source_only']is True and scope['complete']==e['complete'],'Separate root pre-array scope')
    for binding in p['producer_bindings'].values():checked(binding)
    done=jd(e['complete']);need(done['code']==p['producer_bindings']['code']and done['protocol']==p['producer_bindings']['protocol']and done['admission']==p['source_admission']and done['entry']==p['cache_entry'],'Exact frozen producer/entry/admission closure')
    need(done['completed']and done['PASS']and done['source_only']and done['rows']==3144 and done['groups']==393,'Completed expanded source cache')
    need(done['admission']==scope['source_admission']and done['entry']==scope['cache_entry'],'Root admission/cache entry')
    for key,b in scope['exact_cache_input_target_metadata_and_SAM_groups'].items():need(done[key]==b,'Scope exact cache '+key)
    admission=jd(done['admission']);groups=admission['groups'];need(len(groups)==393 and len({g['group_id']for g in groups})==393,'All393 physical groups')
    need(admission['training_admission_direct_end']is True and admission['real_source_replaces_old_whole']is True and admission['no_split_filtering']is True,'Separate direct-only source authority')
    rows=jl(done['metadata']);census=jl(done['census']);sg=jl(done['SAM_groups']);need(len(rows)==3144 and len(census)==3270 and len(sg)==393,'Full denominator')
    split=jd(e['split_complete']);partition=jd(e['partition_QA']);need(split['completed']and split['PASS']and partition['PASS']and partition['same_source_base_frame_query_donor_or_validinput_fingerprint_never_crosses_role']is True,'Frozen source role graph QA')
    side=jl(split['outputs']['DIRECT_CONTROL_SPLIT_ROWS.jsonl']);statusside=jl(split['outputs']['ALL3270_DIRECT_VARIANT_STATUS_AND_ROLES.jsonl']);sideby=join_sidecar(side,[tuple(r['control_join_key'])for r in rows]);statusby={tuple(r[k]for k in ('domain','attempt_id','variant')):r for r in statusside};need(len(statusby)==len(statusside)==3270,'Full source status-role sidecar')
    old=jd(done['old_cache_reference']);oldgroups=jl(old['SAM_groups']);need(old['completed']and old['PASS']and old['SAM_state_unchanged']is True,'Old frozen SAM provenance')
    need(done['SAM_checkpoint']==old['SAM_checkpoint'] and done['SAM_source_commit']==old['SAM_source_commit']and done['SAM_model_state_sha']==old['SAM_model_state_sha'],'Exact pinned checkpoint/source/modelstate metadata only')
    sources={g['source_packet']['path']:g['source_packet']for g in groups};samfiles={g['outputs']['path']:g['outputs']for g in sg if g.get('outputs')};oldfiles={g['outputs']['path']:g['outputs']for g in oldgroups if g.get('outputs')}
    need(len(sources)==199 and len(samfiles)==393,'Exact199source/393fresh SAM packet scope')
    allow={**sources,**samfiles,**oldfiles,done['inputs']['path']:done['inputs'],done['targets']['path']:done['targets']}
    need({(b['path'],b['sha256'])for b in e['array_whitelist']}=={(b['path'],b['sha256'])for b in allow.values()},'Finite exact binary whitelist before any array/header')
    out=V/'independent_direct_cache_review_c/actual_review_v1';need(str(out)==e['output_directory'],'Fresh audit output');out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();stats=Counter();proofs=[];sourcecache={};seen={};oldraw={};targets_count=Counter();physical={};originroles={};rolecounts=Counter();expected={};synthetic_origins={}
    try:
        x=array(done['inputs'],INPUTS,allow,True);y=array(done['targets'],TARGETS,allow,True)
        need(x['rgb96'].dtype==np.uint8 and x['rgb96'].shape==(3144,96,96,3),'RGB96 shape')
        for key in ('valid96','sam_mask96','sam_available96'):need(x[key].dtype==bool and x[key].shape==(3144,96,96),'Bool input '+key)
        for key,dt,sh in [('original_global_index',np.int64,(3144,)),('native_slot',np.int8,(3144,)),('query_xy',np.float64,(3144,2)),('L',np.float64,(3144,)),('original_query_available',bool,(3144,))]:need(x[key].dtype==dt and x[key].shape==sh,'Original META schema '+key)
        need(x['row_id'].dtype.kind=='U'and len(set(x['row_id'].tolist()))==3144,'Unique new occurrences, repeated source indices allowed')
        for key in META:need(exact(x[key],y[key]),'Target META6 byte parity')
        need(y['direct_end_external_target'].dtype==np.int8 and y['direct_end_external_target'].shape==(3144,)and y['authority'].dtype.kind=='U','Separate direct target schema')
        frameplan=jd(admission['original_real_frame_plan']);frames={f['frame_index']:f for f in frameplan['frames']};attempts={r['pair_id']:r for r in frameplan['attempts']};need(len(frames)==21 and len(attempts)==66 and all(f['split']=='SOURCE_DEVELOPMENT'for f in frames.values()),'Original real TRAIN role/21frame authority')
        for source in admission['sources']:
            sd=jd(source['complete']);review=jd(source['independent_review']);need(sd['PASS']and review['PASS'],'Independent original source evidence')
            need(sd['attempts'if source['domain']=='SYNTH_FINITE_END'else'manifest']==source['census'],'Full source census binding')
            rr=jl(source['census']);need(len(rr)==(1024 if source['domain']=='SYNTH_FINITE_END'else 66),'All source attempts')
            if source['domain']=='SYNTH_FINITE_END':synthetic_origins={(r['base_index'],r['native_slot']):r for r in rr};need(len(synthetic_origins)==1024 and all(r['split']=='TRAIN'for r in rr),'All128base/all8 synthetic TRAIN')
            for r in rr:
                domain=source['domain'];aid=r['attempt_id']if domain=='SYNTH_FINITE_END'else r['pair_id'];slot=r['native_slot']if domain=='SYNTH_FINITE_END'else r['original_input']['native_slot']
                supported=r['direct_end_states']==[0,1,0]if domain=='SYNTH_FINITE_END'else(r['PASS']is True and r['direct_end_weak_states']==dict(BEFORE='VISIBLE_WEAK',COVER='EXTERNALLY_BLOCKED_WEAK',SHAM='VISIBLE_WEAK')and r['closed_end_contract']['status']=='CONTROLLED_DIRECT_END_WEAK_PAIR_SUPPORTED'and all(r['closed14_actual_proof'].get(k)is True for k in PROOF14))
                for vi,var in enumerate(VARIANTS):
                    pack=r['outputs']if domain=='SYNTH_FINITE_END'else r['variants'][{'SHAM':'ADJACENT_SHAM'}.get(var,var)]
                    expected[domain,aid,var]=dict(source_packet=pack if pack is not None and 'path'in pack else None,status=r['status'],selected=slot,supported=supported,record=r,variant_index=vi)
        need(len(expected)==3270,'All3270original variant statuses')
        oldindex={}
        for og in oldgroups:
            if og.get('outputs'):
                bx=box([qarray(m['query_xy'])for m in og['metadata8']]);need(bx is not None and og['prompt_available']and og['runtime_available'],'Old valid SAM group')
                oldindex.setdefault((tuple(og['raw_hw']),og['decoded_RGB_sha256'],ash(bx)),og)
        firstfresh=None
        for gi,(g,s)in enumerate(zip(groups,sg)):
            for key,value in g.items():need(s[key]==value,'Fresh group copies admission exactly '+key)
            ck=(g['domain'],g['attempt_id'],g['variant']);er=expected[ck];need(g['source_packet']==er['source_packet']and g['source_status']==er['status']and g['selected_native_slot']==er['selected']and g['admitted']==er['supported'],'Whole supported census rootadmission, no arbitrary promotion')
            mm=g['metadata8'];ii=slice(gi*8,gi*8+8);need([m['native_slot']for m in mm]==list(range(8)),'Native all8')
            binding=g['source_packet'];path=binding['path'];syn=g['layout']=='SYNTH_TRIPLET'
            if path not in sourcecache:
                keys=('full_RGB','rgb96','valid96','query_xy','L','native_slot','original_global_index')+(('original_occurrence_id','original_query_available','new_native_visibility_target','new_quality_target','new_owner_target')if syn else('query_id','query_available','native_visibility_target','point_quality_target','owner_target'))
                sourcecache[path]=array(binding,keys,allow)
            a=sourcecache[path];vi=g['variant_index']if syn else None;rgb=a['full_RGB'][vi]if syn else a['full_RGB'];rgb96=a['rgb96'][vi]if syn else a['rgb96'];valid=a['valid96'][vi]if syn else a['valid96']
            for name in (('new_native_visibility_target','new_quality_target','new_owner_target')if syn else('native_visibility_target','point_quality_target','owner_target')):need(a[name].dtype==np.int8 and a[name].shape==((3,8)if syn else(8,))and(a[name]==-1).all(),'Original other-task labels stayUNKNOWN')
            need(rgb.dtype==np.uint8 and rgb.shape==tuple(g['raw_hw'])+(3,)and rgb96.shape==(8,96,96,3)and rgb96.dtype==np.uint8 and valid.shape==(8,96,96)and valid.dtype==bool,'Saved source schema')
            need(exact(x['rgb96'][ii],rgb96)and exact(x['valid96'][ii],valid),'All3144 raw96/valid source bytes')
            for name in ('query_xy','L','native_slot'):need(exact(x[name][ii],a[name]),'Original metadata bytes '+name)
            need(np.array_equal(x['original_global_index'][ii],a['original_global_index'].astype(np.int64)),'Only explicit lossless real int32 widening')
            av=a['original_query_available']if syn else a['query_available'];oid=a['original_occurrence_id']if syn else a['query_id'];need(exact(x['original_query_available'][ii],av),'Originalavailabilityexact')
            if not syn:
                origin=attempts[g['attempt_id']];fr=frames[origin['frame_index']];need(g['source_frame_index']==origin['frame_index']and g['source_case_id']==origin['case_id']and g['source_donor_id']==origin['donor_id'],'Real exact frame/case/donor authority')
            for k,m in enumerate(mm):
                if syn:
                    origin=synthetic_origins[g['source_base_index'],k];need(all(m[n]==origin[n]for n in ('original_global_index','original_occurrence_id','query_xy','L'))and m['original_input_flags']==origin['original_input_flags'],'Original synthetic all8 metadata')
                else:need(m['original_global_index']==fr['global_index'][k]and m['original_occurrence_id']==fr['query_id'][k]and m['query_xy']==fr['query_xy'][k]and m['L']==fr['L'][k]and m['original_input_flags']==fr['original_query_input_metadata'][k]and m['original_query_available']==fr['native_continuous_valid'][k],'Original real source plan all8 metadata')
            q8=np.stack([qarray(m['query_xy'])for m in mm]);L8=np.asarray([np.nan if m['L']is None else m['L']for m in mm],np.float64);prompt=box(q8);decoded=ash(rgb);need(s['decoded_RGB_sha256']==decoded and s['prompt_available']==(prompt is not None),'OriginalfullRGB SHA/unclippedq8box')
            logits=None
            if prompt is None:need(s['outputs']is None and not s['runtime_available'],'No fabricated promptmissing SAM')
            else:
                need(s['runtime_available']and s['outputs']is not None,'Saved pergroup SAM');z=array(s['outputs'],SAM,allow,True)
                need(z['image_logits'].dtype==np.float32 and z['image_logits'].shape==tuple(g['raw_hw'])and np.isfinite(z['image_logits']).all(),'FiniteSAM FP32 wholelogits')
                need(z['lowres_logits'].dtype==np.float32 and z['lowres_logits'].shape==(256,256)and np.isfinite(z['lowres_logits']).all()and z['predicted_quality'].dtype==np.float32 and z['predicted_quality'].shape==(1,)and np.isfinite(z['predicted_quality']).all(),'SAM raw4 schema')
                need(z['mask_logits_gt0'].dtype==bool and exact(z['mask_logits_gt0'],z['image_logits']>0),'Strictzero fullmask')
                for key,val in [('box_xyxy',prompt),('original_query_xy',q8),('original_L',L8),('native_slot',a['native_slot']),('original_global_index',a['original_global_index'].astype(np.int64)),('original_query_available',av),('prediction_available',av)]:need(exact(z[key],val),'Fresh SAM metadata '+key)
                need(np.array_equal(z['original_occurrence_id'],oid),'Fresh original occurrencevalues')
                rk=(tuple(g['raw_hw']),decoded,ash(prompt));kind=reuse_kind(rk,seen,oldindex);hashes={n:ash(z[n])for n in RAW4}
                need(s['raw4_reused_after_exact_RGB_box_equality']==(kind!='fresh'),'Reusekind exact originalRGB/shape/box')
                if kind=='within':need(s['raw4_old_cache_packet']is None and seen[rk]==hashes,'Withinrunraw4bytes exact')
                elif kind=='old':
                    og=oldindex[rk];need(s['raw4_old_cache_packet']==og['outputs'],'Exact oldcachepacket provenance')
                    if og['outputs']['path']not in oldraw:oldraw[og['outputs']['path']]=array(og['outputs'],SAM,allow,True)
                    oz=oldraw[og['outputs']['path']];om=og['metadata8'];oq=np.stack([qarray(m['query_xy'])for m in om]);ol=np.asarray([np.nan if m['L']is None else m['L']for m in om],np.float64)
                    for name,val in [('box_xyxy',box(oq)),('original_query_xy',oq),('original_L',ol),('native_slot',np.arange(8,dtype=np.int8)),('original_global_index',np.asarray([m['original_global_index']for m in om],np.int64)),('original_query_available',np.asarray([m['original_query_available']for m in om],bool)),('prediction_available',np.asarray([m['original_query_available']for m in om],bool))]:need(exact(oz[name],val),'Oldpacket originalmetadata validatedbefore RAW4onlyreuse')
                    need(np.array_equal(oz['original_occurrence_id'],np.asarray([m['original_occurrence_id']for m in om])),'Old originaloccurrence exact')
                    need(all(exact(oz[name],z[name])for name in RAW4),'Old RAW4 bytes exactly newpacket')
                elif firstfresh is None:firstfresh=g['group_id']
                seen.setdefault(rk,hashes);stats[kind]+=1;stats['SAM_packets']+=1;logits=z['image_logits']
            et=target8(g);need(exact(y['direct_end_external_target'][ii],et)and np.array_equal(y['authority'][ii],np.repeat(g['authority'],8)),'Directselectedslot and authority only')
            need(g['authority'].startswith('CONSTRUCTIVE_SOURCE_FIXED_NEAR_ANCHOR_END')if syn else g['authority'].startswith('SOURCE_RGB_SEMANTIC_END_WEAK'),'Distinct weak authorities')
            for k,m in enumerate(mm):
                i=8*gi+k;r=rows[i];key=(g['domain'],g['attempt_id'],g['variant'],k);sr=sideby[key]
                for name,val in m.items():need(r[name]==val,'All original rowmetadata/flags retained')
                need(r['cache_index']==i and tuple(r['control_join_key'])==key and r['source_packet']==binding and r['raw_RGB96_sha256']==ash(rgb96[k])and r['raw_valid96_sha256']==ash(valid[k]),'Cache inputrow/provenance exact')
                for name in META:
                    expect=qarray(m[name])if name=='query_xy'else(np.nan if m[name]is None else m[name])
                    need(values(x[name][i],expect),'OriginalorderedMETA '+name)
                    need(values(sr[name],m[name]),'Sidecar inputMETA6 '+name)
                need(sr['cache_index']==i and sr['row_id']==m['row_id']and sr['source_packet']==binding and sr['group_id']==g['group_id']and sr['stored_weak_direct_target']==int(et[k])and sr['role']in('TRAIN','HOLDOUT'),'Exact newsidecarjoin noV190indexalias')
                nodes=[json.loads(node)for node in sr['origin_nodes']]
                if m['original_query_available']:need(['AVAILABLE_CONSTRUCTED_RGB96_VALID96',ash(rgb96[k]),ash(valid[k])]in nodes,'Availableinput fingerprintsourcejoin')
                if syn:need(['SYNTH_BASE','V129',g['source_base_index']]in nodes,'Synthetic basecomponent join')
                else:need(sr['original_frame_index']==g['source_frame_index']and ['MR_FRAME','MR6D',sr['source_id']]in nodes and ['YCB_DONOR_CUTOUT_ID',g['source_donor_id']]in nodes,'Realframe/donor componentjoin')
                for node in sr['origin_nodes']:
                    signature=(sr['component_id'],sr['role']);need(node not in originroles or originroles[node]==signature,'Originalorigin nevercrosses componentrole');originroles[node]=signature
                em,ea=sample(logits,m['query_xy'],m['L'],m['original_query_available'],g['raw_hw'])if logits is not None else(np.zeros((96,96),bool),np.zeros((96,96),bool))
                need(exact(x['sam_mask96'][i],em)and exact(x['sam_available96'][i],ea),'All3144 independentF64 SAM96 replay')
                targets_count[int(et[k])]+=1;rolecounts[sr['role']]+=1
            need(ck not in physical,'Unique physicalgroup');physical[ck]=(list(range(8*gi,8*gi+8)),g['admitted']);proofs.append(dict(group_id=g['group_id'],source_packet=binding,PASS=True,SAM_packet=s['outputs'],raw_RGB96_sha256=ash(rgb96),valid96_sha256=ash(valid)))
            if(gi+1)%50==0:print(json.dumps(dict(groups_verified=gi+1,total393=393)),flush=True)
        missing=0
        for r in census:
            ck=(r['domain'],r['attempt_id'],r['variant']);need(ck in expected,'Censusoriginalkey');er=expected.pop(ck);sr=statusby[ck]
            need(r['source_packet']==er['source_packet']and r['source_status']==er['status']and r['selected_native_slot']==er['selected'],'Originalfullcensus status/packet/slot')
            need(sr['source_packet']==r['source_packet']and sr['source_status']==r['source_status']and sr['selected_native_slot']==r['selected_native_slot'],'Frozenstatussidecar originaljoin')
            if ck in physical:need((r['cache_row_indices'],r['direct_training_admitted'])==physical[ck]and sr['cache_row_indices']==r['cache_row_indices'],'Everyphysical cachemapping')
            else:need(r['cache_row_indices']is None and not r['direct_training_admitted']and r['source_packet']is None and not sr['physical'],'MissingNULL neverfake');missing+=1
            need(r['new_native_visibility_target']==r['new_quality_target']==r['new_owner_target']==-1,'No nativequalityownerpromotion')
        need(not expected and missing==2877,'Full3270/2877null')
        need(targets_count=={-1:2772,0:248,1:124}and int((~x['original_query_available']).sum())==112,'Alltargets/U112 exact')
        need((y['direct_end_external_target'][~x['original_query_available']]==-1).all(),'U neverdirectknown')
        need(stats['fresh']==44 and stats['old']==267 and stats['within']==82 and stats['SAM_packets']==393,'Exact44fresh267old82within')
        need(done['SAM_first_repeat']==dict(PASS=True,group_id=firstfresh)and done['SAM_forward_calls_including_repeat']==stats['fresh']+1==45 and done['SAM_state_unchanged']is True,'Firstnewrepeat+counters/state receipt no modelreload')
        need(dict(rolecounts)==partition['all_domains_full_cache_indices_and_native_U_preserved']['DIRECT_CONTROL']['roles'],'Fullsource rolecount unchanged')
        report=dict(completed=True,PASS=True,code=bind(Path(__file__).absolute()),protocol=bind(PROTO),entry=bind(entry_path),root_scope=e['root_scope'],complete=e['complete'],groups393=393,rows3144=3144,source_packets199=len(sourcecache),census3270=3270,missing_null2877=missing,direct_target_counts=dict(targets_count),original_inputU112=112,source_role_counts=dict(rolecounts),all_raw96_valid_qL_META6_sourcebytes_exact=True,all3144_SAM96_mask_and_availability_F64_exact=True,SAM=dict(stats),fullstate_and_firstrepeat_closed_by_receipt=True,no_model_checkpoint_reloaded=True,split_complete=e['split_complete'],partition_QA=e['partition_QA'],split_input_sourceframe_donor_component_join_exact=True,old_native_quality_owner_allUNKNOWN=True,producer_imports=0,source_original_buffers_checkpoint_model_fit_eval_reads=0,seconds=time.monotonic()-start,limits=['Saved source cache/provenance fidelity is not natural/physical native gold or learned performance.','Independent exact sampler is checked scalar/vector; RGB96 is compared to saved original source packets, not recropped.','SAM modelstate/repeat closes sealed producer receipt and old/newraw4/counters, not a new modelcall.','All direct sidecarjoins/originnode roles checked; fullold3-pool componentgraph is inherited from exact bound partitionQA, not independently rebuilt here.'])
        write(out/'ALL393_GROUP_CHECKS.json',proofs);write(out/'REVIEW.json',report);print(json.dumps(dict(completed=True,PASS=True,review=bind(out/'REVIEW.json'),SAM=dict(stats),seconds=report['seconds'])),flush=True)
    except BaseException:
        write(out/'FAILURE.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),groups_completed=len(proofs),automatic_retry=False));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.selftest:print(json.dumps(more_selftests()))
    elif a.entry:run(a.entry)
    else:ap.error('--selftest or root --entry required')

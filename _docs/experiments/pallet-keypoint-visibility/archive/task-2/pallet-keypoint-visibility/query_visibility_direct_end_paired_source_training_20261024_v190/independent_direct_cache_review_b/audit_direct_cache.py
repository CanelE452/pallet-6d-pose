"""Independent all-source saved cache audit; no producer/model imports."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,math,os,stat,sys,time,traceback
import numpy as np
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
PROTO=H/'PROTOCOL_BEFORE_DIRECT_CACHE_ARRAY_READS.json'
META=('row_id','original_global_index','native_slot','query_xy','L','original_query_available')
INPUTS={'rgb96','valid96','sam_mask96','sam_available96'}|set(META)
TARGETS={'direct_end_external_target','authority'}|set(META)
SAM={'image_logits','lowres_logits','predicted_quality','mask_logits_gt0','box_xyxy','original_query_xy','original_L','native_slot','original_global_index','original_occurrence_id','original_query_available','prediction_available'}
VARIANTS=('BEFORE','COVER','SHAM')

def need(v,msg):
    if not bool(v):raise AssertionError(msg)
def safe(path):
    p=Path(path);need(p.is_absolute() and '..' not in p.parts and all('v112' not in x.casefold() for x in p.parts),'Safe name before access')
    need(p.is_relative_to(P) and not any(x in str(p).casefold() for x in ('current223','_v169/','_v177/','_v178/','_v184/')),'Exact source-only path')
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

def run(entry_path):
    e=json.loads(raw(entry_path));need(e['approved_actual_independent_cache_audit'] is True and e['source_only'] is True and e['automatic_retry'] is False,'Explicit root once source audit entry')
    p=json.loads(raw(PROTO));need(e['code']==p['code']==bind(Path(__file__).absolute()) and e['protocol']==bind(PROTO),'Frozen independent code/protocol')
    need(e['root_scope']==p['root_scope'] and e['complete']==p['complete'],'Exact prospective root scope/completion closure')
    need(sys.executable==p['runtime']['executable'] and np.__version__==p['runtime']['numpy'],'Pinned independent arithmetic runtime')
    scope=jd(e['root_scope']);need(scope['root_authorized'] is True and scope['source_only'] is True and scope['complete']==e['complete'],'Root exact pre-array scope')
    done=jd(e['complete']);need(done['completed'] is True and done['PASS'] is True and done['source_only'] is True and done['rows']==2792 and done['groups']==349,'Closed completed direct cache')
    need(done['admission']==scope['source_admission'] and done['entry']==scope['cache_entry'],'Exact root admission/cache entry closure')
    for key,binding in scope['exact_cache_input_target_metadata_and_SAM_groups'].items():need(done[key]==binding,'Exact scope cache field '+key)
    admission=jd(scope['source_admission']);groups=admission['groups'];need(len(groups)==349 and len({g['group_id'] for g in groups})==349,'Full group metadata')
    rows=jl(done['metadata']);census=jl(done['census']);sg=jl(done['SAM_groups']);need(len(rows)==2792 and len(census)==3270 and len(sg)==349,'Full row/group/census denominator')
    sources={g['source_packet']['path']:g['source_packet'] for g in groups};need(len(sources)==155,'Only original155 packets')
    samfiles={g['outputs']['path']:g['outputs'] for g in sg if g['outputs'] is not None};need(len(samfiles)==sum(g['outputs'] is not None for g in sg),'Fresh per-group saved SAM packet path')
    allowed={**sources,**samfiles,done['inputs']['path']:done['inputs'],done['targets']['path']:done['targets']}
    out=H/'actual_review_v1';out.mkdir(exist_ok=False);stats=Counter();proofs=[];start=time.monotonic();sourcecache={};reuse={};outputsha={};targetcounts=Counter();expected_census={}
    try:
        x=array(done['inputs'],INPUTS,allowed,True);y=array(done['targets'],TARGETS,allowed,True)
        need(x['rgb96'].dtype==np.uint8 and x['rgb96'].shape==(2792,96,96,3),'Raw RGB96 dtype/shape')
        for key in ('valid96','sam_mask96','sam_available96'):need(x[key].dtype==bool and x[key].shape==(2792,96,96),'Pixel bool schema '+key)
        for key,dt,sh in [('original_global_index',np.int64,(2792,)),('native_slot',np.int8,(2792,)),('query_xy',np.float64,(2792,2)),('L',np.float64,(2792,)),('original_query_available',bool,(2792,))]:need(x[key].dtype==dt and x[key].shape==sh,'Numeric original metadata schema '+key)
        need(x['row_id'].dtype.kind=='U' and len(set(x['row_id'].tolist()))==2792,'Unique new occurrence row IDs')
        for key in META:need(exact(x[key],y[key]),'Target sidecar META6 byte parity')
        need(y['direct_end_external_target'].dtype==np.int8 and y['direct_end_external_target'].shape==(2792,) and y['authority'].dtype.kind=='U','Separate target and authority schema')
        for source in admission['sources']:
            for r in jl(source['census']):
                domain=source['domain'];aid=r['attempt_id'] if domain=='SYNTH_FINITE_END' else r['pair_id'];slot=r['native_slot'] if domain=='SYNTH_FINITE_END' else r['original_input']['native_slot']
                for vi,var in enumerate(VARIANTS):
                    pack=r['outputs'] if domain=='SYNTH_FINITE_END' else r['variants'][{'SHAM':'ADJACENT_SHAM'}.get(var,var)]
                    expected_census[domain,aid,var]=dict(source_packet=pack if pack is not None and 'path' in pack else None,status=r['status'],selected=slot)
        need(len(expected_census)==3270,'All frozen source census rows')
        physical={};same_input={}
        for gi,(g,s) in enumerate(zip(groups,sg)):
            for key,value in g.items():need(s[key]==value,'SAM group exact root admission '+key)
            mm=g['metadata8'];ii=slice(gi*8,gi*8+8);need([m['native_slot'] for m in mm]==list(range(8)),'Original native8 order')
            binding=g['source_packet'];path=binding['path']
            if path not in sourcecache:
                keys=('full_RGB','rgb96','valid96','query_xy','L','native_slot','original_global_index')+ (('original_occurrence_id','original_query_available') if g['layout']=='SYNTH_TRIPLET' else ('query_id','query_available'))
                sourcecache[path]=array(binding,keys,allowed)
            a=sourcecache[path];vi=g['variant_index'] if g['layout']=='SYNTH_TRIPLET' else None
            rgb=a['full_RGB'][vi] if vi is not None else a['full_RGB'];rgb96=a['rgb96'][vi] if vi is not None else a['rgb96'];valid96=a['valid96'][vi] if vi is not None else a['valid96']
            need(rgb.dtype==np.uint8 and rgb.shape==tuple(g['raw_hw'])+(3,) and rgb96.dtype==np.uint8 and rgb96.shape==(8,96,96,3) and valid96.dtype==bool and valid96.shape==(8,96,96),'Exact source pixel schema')
            need(exact(x['rgb96'][ii],rgb96) and exact(x['valid96'][ii],valid96),'All155sourcepacket raw96/valid bytes exactly cache')
            need(exact(x['query_xy'][ii],a['query_xy']) and exact(x['L'][ii],a['L']) and exact(x['native_slot'][ii],a['native_slot']),'Original q/L/native byte parity')
            need(np.array_equal(x['original_global_index'][ii],a['original_global_index'].astype(np.int64)),'Source original IDs lossless int32 widening only')
            av=a['original_query_available'] if vi is not None else a['query_available'];oid=a['original_occurrence_id'] if vi is not None else a['query_id'];need(exact(x['original_query_available'][ii],av),'Source availability unchanged')
            q8=np.stack([qarray(m['query_xy']) for m in mm]);L8=np.asarray([np.nan if m['L'] is None else m['L'] for m in mm],np.float64);prompt=box(q8);decoded=ash(rgb)
            need(decoded==s['decoded_RGB_sha256'] and s['prompt_available']==(prompt is not None),'Actual original fullRGB SHA/prompt')
            sampler=None
            if prompt is None:
                need(s['outputs'] is None and not s['runtime_available'],'No fake missing prompt SAM')
            else:
                need(s['outputs'] is not None and s['runtime_available'],'Saved source SAM available')
                packet=array(s['outputs'],SAM,allowed,True);need(packet['image_logits'].dtype==np.float32 and packet['image_logits'].shape==tuple(g['raw_hw']) and np.isfinite(packet['image_logits']).all(),'Full image FP32 logits')
                need(packet['lowres_logits'].dtype==np.float32 and packet['lowres_logits'].shape==(256,256) and packet['predicted_quality'].dtype==np.float32 and packet['predicted_quality'].shape==(1,),'Raw4 shape')
                need(packet['mask_logits_gt0'].dtype==bool and exact(packet['mask_logits_gt0'],packet['image_logits']>0),'Unchanged full SAM raw>0 mask')
                for key,expect in [('box_xyxy',prompt),('original_query_xy',q8),('original_L',L8),('native_slot',a['native_slot']),('original_global_index',a['original_global_index'].astype(np.int64)),('original_occurrence_id',oid),('original_query_available',av),('prediction_available',av)]:
                    if key=='original_occurrence_id':need(np.array_equal(packet[key],expect),'Fresh SAM original occurrence values')
                    else:need(exact(packet[key],expect),'Fresh per-group SAM original metadata '+key)
                key=(rgb.shape,decoded,ash(prompt));raw4={name:ash(packet[name]) for name in ('image_logits','lowres_logits','predicted_quality','mask_logits_gt0')}
                reused=key in reuse;need(s['raw4_reused_after_exact_RGB_box_equality']==reused,'Reuse only exact RGB+box+shape; changedRGB cannot alias')
                if reused:need(reuse[key]==raw4,'All reused raw4 bytes exact')
                else:reuse[key]=raw4;stats['fresh_SAM_unique']+=1
                sampler=packet['image_logits'];stats['SAM_packets']+=1;stats['raw4_reused']+=int(reused)
            expected_targets=np.full(8,-1,np.int8)
            if g['admitted']:expected_targets[g['selected_native_slot']]=g['admitted_target']
            need(exact(y['direct_end_external_target'][ii],expected_targets),'Only root-admitted requested slot direct labels')
            need(np.array_equal(y['authority'][ii],np.repeat(g['authority'],8)),'Distinct synthetic/real weak authority exact')
            for k,m in enumerate(mm):
                i=8*gi+k;r=rows[i]
                for name,val in m.items():need(r[name]==val,'Original metadata/flags field unchanged')
                need(r['raw_RGB96_sha256']==ash(rgb96[k]) and r['raw_valid96_sha256']==ash(valid96[k]),'Saved raw row hash bindings')
                for name,expect in [('row_id',m['row_id']),('original_global_index',m['original_global_index']),('native_slot',k),('original_query_available',m['original_query_available'])]:need(x[name][i]==expect,'Cache ordered row metadata '+name)
                em,ea=sample(sampler,m['query_xy'],m['L'],m['original_query_available'],g['raw_hw']) if sampler is not None else (np.zeros((96,96),bool),np.zeros((96,96),bool))
                need(exact(x['sam_mask96'][i],em) and exact(x['sam_available96'][i],ea),'All2792 independent original96 SAM bilinear+availability')
                targetcounts[int(expected_targets[k])]+=1
            ck=(g['domain'],g['attempt_id'],g['variant']);need(ck not in physical,'Physical group unique');physical[ck]=(list(range(8*gi,8*gi+8)),g['admitted'])
            proofs.append(dict(group_id=g['group_id'],source_packet=binding,PASS=True,raw_RGB96_sha256=ash(rgb96),valid96_sha256=ash(valid96),decoded_RGB_sha256=decoded,original_query_xy_sha256=ash(a['query_xy']),SAM_packet=s['outputs'],admitted_target=g['admitted_target']))
            if (gi+1)%50==0:print(json.dumps(dict(groups_verified=gi+1,total349=349)),flush=True)
        missing=0
        for r in census:
            ck=(r['domain'],r['attempt_id'],r['variant']);need(ck in expected_census,'No extra census');er=expected_census.pop(ck)
            need(r['source_packet']==er['source_packet'] and r['source_status']==er['status'] and r['selected_native_slot']==er['selected'],'Original source status/packet/slot census exact')
            if ck in physical:need((r['cache_row_indices'],r['direct_training_admitted'])==physical[ck],'All physical row mappings exact')
            else:need(r['cache_row_indices'] is None and not r['direct_training_admitted'] and r['source_packet'] is None,'Missing null not fake zero');missing+=1
            need(r['new_native_visibility_target']==r['new_quality_target']==r['new_owner_target']==-1,'No oldtask target alias')
        need(not expected_census and missing==2921,'Full3270/census2921missing')
        need(targetcounts=={-1:2486,0:204,1:102} and int((~x['original_query_available']).sum())==60,'All target and originalU counts')
        need((y['direct_end_external_target'][~x['original_query_available']]==-1).all(),'InputU never admitted')
        repeat=done['SAM_first_repeat'];need(repeat['PASS'] is True and repeat['group_id'] in {g['group_id'] for g in groups},'Producer repeat metadata')
        need(done['SAM_forward_calls_including_repeat']==stats['fresh_SAM_unique']+1==268 and done['SAM_state_unchanged'] is True,'Forward counter/state receipt closed')
        report=dict(completed=True,PASS=True,code=bind(Path(__file__).absolute()),protocol=bind(PROTO),root_scope=e['root_scope'],complete=e['complete'],entry=bind(entry_path),groups349=349,rows2792=2792,source_packets155=len(sourcecache),census3270=3270,missing_null2921=missing,direct_target_counts=dict(targetcounts),original_inputU60=60,native_all8_preserved=True,all_raw96_valid_qL_source_bytes_exact=True,all_SAM96_mask_and_availability_exact=True,SAM=dict(stats),SAM_forward_counter268_closed=True,model_state_unchanged_from_producer_receipt_not_weights_reloaded=True,source_authorities=sorted(set(y['authority'].tolist())),no_new_native_quality_owner_or_input_targets=True,producer_imports=0,source_original_buffers_checkpoint_weights_model_fit_or_evaluation_reads=0,seconds=time.monotonic()-start,limits=['This checks cache fidelity/authority joins, not physical end gold or learned performance.','SAM state/repeat are checked via sealed producer receipt/counters; no checkpoint/model is loaded and no SAM forward rerun.'])
        write(out/'ALL349_GROUP_CHECKS.json',proofs);write(out/'REVIEW.json',report);print(json.dumps(dict(completed=True,PASS=True,review=bind(out/'REVIEW.json'),seconds=report['seconds'],SAM=dict(stats))),flush=True)
    except BaseException:
        write(out/'FAILURE.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),groups_completed=len(proofs),automatic_retry=False));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.selftest:print(json.dumps(selftest()))
    elif a.entry:run(a.entry)
    else:ap.error('--selftest or --entry required')

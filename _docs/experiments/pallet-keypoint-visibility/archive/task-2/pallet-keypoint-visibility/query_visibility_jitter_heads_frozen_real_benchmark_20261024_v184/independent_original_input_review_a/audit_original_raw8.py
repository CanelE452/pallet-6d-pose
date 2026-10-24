"""Independent V184 saved-input audit; no producer import, fullRGB or model I/O."""
from pathlib import Path
import argparse,gzip,hashlib,json,math,os,stat,time,traceback
import numpy as np

H=Path(__file__).resolve().parent;V=H.parent
COMPLETE=V/'original_real_raw8_inputs_v1/COMPLETE.json'
COMPLETE_SHA='2d90a7863fe80cfce95e3d54ee1ce529b2b800bcc85373cd7b1f0551c5e68a37'
META=('row_id','original_global_index','native_slot','query_xy','L','original_query_available')
KEYS=set(META)|{'rgb96','valid96'}
ACCESSES=[];BINDINGS={}

def need(ok,message):
    if not bool(ok):raise AssertionError(message)
def path_safe(p):
    s=os.path.abspath(str(p));need('_v112' not in s.lower(),'Forbidden path before access')
    pending=s.split('/')[1:];cur='/';hops=0
    while pending:
        n=os.path.normpath(os.path.join(cur,pending.pop(0)));need('_v112' not in n.lower(),'Forbidden component')
        a=os.lstat(n)
        if stat.S_ISLNK(a.st_mode):
            t=os.readlink(n);need('_v112' not in t.lower(),'Forbidden symlink target')
            t=os.path.normpath(t if os.path.isabs(t) else os.path.join(os.path.dirname(n),t));need('_v112' not in t.lower(),'Forbidden expansion');hops+=1;need(hops<40,'Symlink loop');pending=t.split('/')[1:]+pending;cur='/'
        else:cur=n
    return Path(cur)
def digest(p):
    h=hashlib.sha256()
    with path_safe(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def binding(p):return dict(path=str(Path(p).absolute()),sha256=digest(p))
def verify(b):
    need(digest(b['path'])==b['sha256'],'Input binding '+b['path']);BINDINGS[b['path']]=b['sha256'];return path_safe(b['path'])
def read(p):
    p=path_safe(p)
    with (gzip.open if p.suffix=='.gz' else open)(p,'rt') as f:return json.load(f)
def read_lines(p):return [json.loads(s) for s in path_safe(p).read_text().splitlines() if s]
def projected_json(b,fields):return [{k:r[k] for k in fields} for r in read(verify(b))]
def projected_lines(b,fields):
    result=[]
    with verify(b).open() as f:
        for s in f:
            r=json.loads(s);result.append({k:r[k] for k in fields})
    return result
def write(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def equal(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if a.dtype.kind=='O' and b.dtype.kind in 'fc':a=a.astype(b.dtype)
    if b.dtype.kind=='O' and a.dtype.kind in 'fc':b=b.astype(a.dtype)
    return a.shape==b.shape and (np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a,b))
def scalar_grid(q,L,hw):
    """Independent 1-D scalar coordinate loops then outer conjunction, FP64."""
    h,w=hw
    xx=[float(q[0])+float(L)*((j+0.5)/96.0-0.5) for j in range(96)]
    yy=[float(q[1])+float(L)*((j+0.5)/96.0-0.5) for j in range(96)]
    vx=np.asarray([math.isfinite(x) and -0.5<=x<w-0.5 for x in xx],bool)
    vy=np.asarray([math.isfinite(y) and -0.5<=y<h-0.5 for y in yy],bool)
    return np.logical_and.outer(vy,vx)
def prompt(q,av):
    coords=[]
    for k in range(8):
        x,y=q[k]
        if av[k] and math.isfinite(x) and math.isfinite(y) and not (x==-1 and y==-1):coords.append((float(x),float(y)))
    if not coords:return None,'NO_FINITE_AVAILABLE_INITIAL_Q'
    box=[min(x for x,y in coords),min(y for x,y in coords),max(x for x,y in coords),max(y for x,y in coords)]
    if box[0]>=box[2] or box[1]>=box[3]:return None,'DEGENERATE_INITIAL_Q8_MINMAX'
    return box,'ORIGINAL_FINITE_AVAILABLE_Q8_UNCLIPPED_UNEXPANDED'
def four_arrays(b,keys,npmod=np):
    p=verify(b)
    with npmod.load(p,allow_pickle=False) as z:
        out={}
        for k in keys:out[k]=z[k];ACCESSES.append(dict(path=b['path'],key=k))
    return out
def projected_q(q):return np.asarray([np.nan,np.nan] if q is None else q,np.float64)
def row_meta_check(x,i,r,q,L,av,row_id,index,native):
    expected=dict(row_id=row_id,original_global_index=index,native_slot=native,query_xy=q,L=L,original_query_available=bool(av))
    for k,val in expected.items():need(equal(x[k][i],val),'Saved META6 '+str((index,k)))
    for k,val in expected.items():need(equal(r[k],val),'ROWS META6 '+str((index,k)))
    need(equal(r['actual_crop_grid_query_xy'],q) and equal(r['actual_crop_grid_L'],L) and r['grid_precision']=='ORIGINAL_FLOAT64','F64 actualgrid identity')
def output_packet(b,n,ids,U):
    p=verify(b)
    with np.load(p,allow_pickle=False) as z:
        need(set(z.files)==KEYS,'Exact target-free raw8 keys');x={k:z[k] for k in KEYS}
    for k,dt,shape in [('rgb96',np.uint8,(n,96,96,3)),('valid96',np.bool_,(n,96,96)),('query_xy',np.float64,(n,2)),('L',np.float64,(n,)),('native_slot',np.int8,(n,)),('original_global_index',np.int64,(n,)),('original_query_available',np.bool_,(n,))]:need(x[k].dtype==dt and x[k].shape==shape,'Saved output schema '+k)
    need(x['row_id'].shape==(n,) and x['row_id'].dtype.kind=='U' and len(set(x['row_id'].tolist()))==n,'Domain occurrence IDs')
    need(x['original_global_index'].tolist()==ids and int((~x['original_query_available']).sum())==U,'Domain IDs and original U')
    return x
def check_groups(groups,allrows,authorities):
    need(len(groups)==268 and len({g['group_id'] for g in groups})==268,'All268 groups')
    cursor=0;stats={}
    for g in groups:
        rr=allrows[cursor:cursor+8];cursor+=8;authority=authorities[g['group_id']]
        need(g['original_input_metadata8']==rr,'GROUPS carries exact ROWS')
        need(g['native_slot']==list(range(8)) and all(r['group_id']==g['group_id'] and r['frame_index']==g['frame_index'] and r['id']==g['id'] and r['domain']==g['domain'] for r in rr),'Group domain/frame identity')
        q=np.asarray([r['query_xy'] for r in rr],np.float64);av=[r['original_query_available'] for r in rr];Ls=[r['L'] for r in rr]
        for k in ['original_query_xy8','actual_crop_grid_query_xy8']:need(equal(g[k],q),'Group q8 F64')
        for k in ['original_L8','actual_crop_grid_L8']:need(equal(g[k],Ls),'Group L8 F64')
        need(g['original_query_available8']==av and g['raw_hw']==authority['hw'] and g['decoded_RGB_sha256']==authority['decoded'],'Original group availability/HW/decodedhash metadata')
        need(g['source_fullRGB']==authority['source'],'Original encoded/canonical provenance metadata; not decoded')
        box,why=prompt(q,av);need(g['box_xyxy']==box and g['prompt_reason']==why and g['prompt_available']==(box is not None) and g['prompt_GT_or_clip_or_expansion'] is False,'Independent unclipped finiteavailable prompt')
        d=stats.setdefault(g['domain'],dict(groups=0,prompt_available=0));d['groups']+=1;d['prompt_available']+=box is not None
    need(cursor==2144,'All2144 group rows');return stats
def invented():
    tests={}
    a=scalar_grid([-.5,-.5],96,[2,3]);tests['half_pixel_lower_inclusive_upper_exclusive']=int(a.sum())==6 and a[48,48] and not a[47,48]
    q=[0.99999999,0.99999999];L=96.0
    tests['F64_vs_FP32_alias_boundary_can_change_domain']=not equal(scalar_grid(q,L,[2,2]),scalar_grid(np.asarray(q,np.float32),np.float32(L),[2,2]))
    q8=np.array([[-2.,0.],[1300.,740.],[9999,9999],[np.nan,np.nan],[-1,-1],[0,0],[0,0],[0,0]])
    tests['prompt_available_finite_only_and_unclipped']=prompt(q8,[1,1,0,0,1,0,0,0])[0]==[-2.,0.,1300.,740.]
    tests['missing_and_degenerate_prompt_not_inputU']=prompt(q8,[0]*8)[0] is None and prompt(np.ones((8,2)),[1]*8)[0] is None
    rgb=np.arange(2*3*96*96,dtype=np.uint8).reshape(2,3,96,96);hwc=rgb.transpose(0,2,3,1)
    tests['CHW_HWC_byte_reversible']=hwc.transpose(0,3,1,2).tobytes()==rgb.tobytes()
    keys=[]
    class Zip:
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def __getitem__(self,k):need(k in ('rgb96','crop_valid96','query_xy','side_L'),'Forbidden mixed key');keys.append(k);return np.array([1])
        @property
        def files(self):raise AssertionError('Forbidden mixed inventory')
    class Fake:
        @staticmethod
        def load(*args,**kw):return Zip()
    # Exercise actual whitelist reader, replacing only file binding on toy code.
    global verify
    old=verify
    try:
        verify=lambda b:Path('<invented>');four_arrays(dict(path='<invented>'),('rgb96','crop_valid96','query_xy','side_L'),Fake)
    finally:verify=old
    ACCESSES.clear();tests['actual_mixed_reader_whitelist4_no_inventory']=keys==['rgb96','crop_valid96','query_xy','side_L']
    tests['globalIDs_may_overlap_between_original_domains']=len(set(range(1784))&set(range(584,944)))==360
    tests['JSON_null_numeric_nan_same_original_missing_identity']=equal([None,None],np.array([np.nan,np.nan])) and equal(None,np.nan) and not equal([None,None],[1.,2.])
    toy=H/'INVENTED_SERIALIZED_ALIAS_COUNT.json';write(toy,dict(FP32_alias_nonzero_grid_drift_rows=int(np.int64(7))));tests['actual_writer_accepts_explicit_int_alias_drift_count']=read(toy)==dict(FP32_alias_nonzero_grid_drift_rows=7)
    need(all(tests.values()),'Invented checks failed');return tests
def prepare():
    tests=invented();write(H/'GENERATED_INVENTED_CHECKS.json',dict(completed=True,PASS=True,checks=tests,count=len(tests),real_body_reads=0))
    write(H/'PROTOCOL.json',dict(completed=True,prepared_only=True,code=binding(__file__),checks=binding(H/'GENERATED_INVENTED_CHECKS.json'),input_complete=dict(path=str(COMPLETE),sha256=COMPLETE_SHA),authorization='Root explicit read-only all2144 original input audit after bothheadfreeze; one execution/no automatic retry.',independence='No producer or sampler import/AST. Scalar FP64 96-axis loops and independent finiteavailable prompt loops. Current mixedNPZ exactly4 keys; public45 exactly4 keys.',coverage='All2144 RGB96/valid96 bytes and metadata6/F64 identities, all2144 constructed-domain masks, all268 prompts/provenance metadata; no fullRGB resampling or semantic/annotation/model numerical audit.',forbidden_body_reads=['fullRGB','annotation','reference','old/new scores','feature/P3','SAM/classifier checkpoint'],automatic_retry=False))
    print(json.dumps(dict(PASS=True,prepared_only=True,code=binding(__file__),protocol=binding(H/'PROTOCOL.json'),checks=binding(H/'GENERATED_INVENTED_CHECKS.json'))),flush=True)
def audit():
    need(not (H/'REVIEW.json').exists() and not (H/'ACTUAL_STARTED.json').exists(),'One actual audit; preserve failure')
    protocol=read(H/'PROTOCOL.json');need(protocol['code']==binding(__file__),'Prepared code frozen')
    write(H/'ACTUAL_STARTED.json',dict(code=binding(__file__),protocol=binding(H/'PROTOCOL.json'),pid=os.getpid()));started=time.monotonic()
    try:
        need(digest(COMPLETE)==COMPLETE_SHA,'Actual input COMPLETE');c=read(COMPLETE);need(c['completed'] and c['PASS'],'Completed input adapter')
        p=read(verify(c['protocol']));src=p['sources'];s=src['current'];verify(c['code']);verify(c['scope']);verify(c['authority_clarification'])
        groups=read_lines(verify(c['groups']));allrows=[];authorities={};summary={}
        frames=projected_json(s['prepared_frames'],['id','image','image_sha256','raw_hw','initial_points'])
        sides=projected_json(s['feature_metadata'],['id','side'])
        prior=projected_lines(src['current_prior_input_rows'],['id','native_corner','query_x','query_y','roi_side','valid','network_callable','image_sha256'])
        facts=projected_lines(s['C11_frame_facts'],['frame_index','id','image','raw_hw','decoded_RGB_sha256','RGB96_original_cached_sha256','valid96_original_cached_sha256'])
        need(len(frames)==len(sides)==len(facts)==223 and len(prior)==1784,'Current originalmetadata counts')
        raw=four_arrays(s['original_cache'],('rgb96','crop_valid96','query_xy','side_L'))
        need(raw['rgb96'].dtype==np.uint8 and raw['rgb96'].shape==(1784,3,96,96) and raw['crop_valid96'].dtype==bool and raw['crop_valid96'].shape==(1784,1,96,96),'Current original raw shape')
        need(raw['query_xy'].dtype==raw['side_L'].dtype==np.float32,'Original FP32 aliases dtype')
        d=c['domains']['CURRENT223_1784'];x=output_packet(d['inputs'],1784,list(range(1784)),45);rr=read_lines(verify(d['metadata']));need(len(rr)==1784,'Current ROWS1784')
        grid_rows=placeholder=alias_drift=0;max_drift=0.
        for fi in range(223):
            f,m,proof=frames[fi],sides[fi],facts[fi];need(f['id']==m['id']==proof['id'] and proof['frame_index']==fi and proof['image']==dict(path=f['image'],sha256=f['image_sha256']) and proof['raw_hw']==f['raw_hw'],'Current frame authority')
            q=np.asarray(f['initial_points'],np.float64)[:8];L=float(m['side']);gid='CURRENT_FRAME_%04d'%fi
            need(array_sha(raw['rgb96'][fi*8:fi*8+8])==proof['RGB96_original_cached_sha256'] and array_sha(raw['crop_valid96'][fi*8:fi*8+8])==proof['valid96_original_cached_sha256'],'Original frame8 decoded crop factSHA')
            authorities[gid]=dict(hw=f['raw_hw'],decoded=proof['decoded_RGB_sha256'],source=dict(kind='ORIGINAL_ENCODED_RGB_SAFE_ALIAS',binding=dict(path=f['image'],sha256=f['image_sha256']),safe_alias_loader_source=s['parent_alias_renderer'],decoder=s['fullframe_decoder']))
            for k in range(8):
                i=fi*8+k;old=prior[i];r=rr[i];need(old['id']==f['id'] and old['native_corner']==k and old['roi_side']==L and old['image_sha256']==f['image_sha256'] and equal(q[k],[old['query_x'],old['query_y']]),'Current prior inputauthority')
                need(raw['rgb96'][i].transpose(1,2,0).tobytes()==x['rgb96'][i].tobytes() and raw['crop_valid96'][i,0].tobytes()==x['valid96'][i].tobytes(),'Current original pixel bytes')
                need(equal(raw['query_xy'][i],q[k].astype(np.float32)) and raw['side_L'][i]==np.float32(L),'FP32 aliases are casts after originalF64 grid')
                row_meta_check(x,i,r,q[k],L,old['valid'],f['id']+':native'+str(k),i,k)
                need(r['raw_packet']==s['original_cache'] and r['raw_row']==i and r['crop_constructed']==old['network_callable'] and equal(r['cached_working_query_xy_FP32'],raw['query_xy'][i]) and equal(r['cached_working_L_FP32'],raw['side_L'][i]),'Current cachedalias/source/constructed metadata')
                if old['network_callable']:need(equal(x['valid96'][i],scalar_grid(q[k],L,f['raw_hw'])),'Independent current F64 validgrid');grid_rows+=1
                else:need(not x['valid96'][i].any() and not x['rgb96'][i].any(),'Original current nonconstructed placeholder');placeholder+=1
                if np.isfinite(q[k]).all() and math.isfinite(L):
                    delta=max(abs(float(raw['query_xy'][i,j])-q[k,j])+47.5/96*abs(float(raw['side_L'][i])-L) for j in range(2));max_drift=max(max_drift,delta);alias_drift+=delta>0
        for key in ['rgb96','valid96']:need(array_sha(x[key])==d['actual_raw_array_SHA'][key],'Current saved wholearray hash')
        summary['CURRENT223_1784']=dict(rows=1784,frames=223,original_inputU=45,RGB_valid_byte_equal_rows=1784,metadata6_exact_rows=1784,F64_constructed_grid_checked=grid_rows,nonconstructed_zero_placeholder=placeholder,FP32_alias_nonzero_grid_drift_rows=int(alias_drift),max_abs_axis_alias_drift_bound_px=max_drift)
        allrows.extend(rr);del raw,x
        source=read_lines(verify(src['public_manifest']));need(len(source)==360,'All original public360')
        d=c['domains']['PUBLIC_HELD45_360'];x=output_packet(d['inputs'],360,list(range(584,944)),108);rr=read_lines(verify(d['metadata']));need(len(rr)==360,'Public ROWS360');grid_rows=placeholder=0
        for frame in range(73,118):
            off=(frame-73)*8;eight=source[off:off+8];f=eight[0];b=f['RGB96_packet'];raw=four_arrays(b,('rgb96','valid96','global_index','crop_constructed'))
            need(raw['rgb96'].dtype==np.uint8 and raw['rgb96'].shape==(8,96,96,3) and raw['valid96'].dtype==bool and raw['valid96'].shape==(8,96,96) and raw['global_index'].dtype==np.int32 and raw['crop_constructed'].dtype==bool,'Public original raw schema')
            gid='PUBLIC_HELD_FRAME_%04d'%frame;authorities[gid]=dict(hw=f['raw_hw'],decoded=f['decoded_RGB_sha256'],source=dict(kind='CANONICAL_RGB_NPY',binding=f['canonical_RGB'],expected_dtype='uint8',expected_shape=f['raw_hw']+[3],original_encoded_RGB=f['original_RGB']))
            for k,old in enumerate(eight):
                i=off+k;g=frame*8+k;r=rr[i];q=projected_q(old['query_xy']);L=np.nan if old['L'] is None else float(old['L']);need(old['original_global_index']==g and old['native_slot']==k and old['frame_index']==frame and old['split']=='SOURCE_VALIDATION' and old['packet_native_row']==k and old['RGB96_packet']==b and old['canonical_RGB']==f['canonical_RGB'],'Public original authority')
                need(raw['global_index'][k]==g and raw['crop_constructed'][k]==old['feature_constructed'],'Public original ID/construction')
                need(raw['rgb96'][k].tobytes()==x['rgb96'][i].tobytes() and raw['valid96'][k].tobytes()==x['valid96'][i].tobytes(),'Public original pixel bytes')
                hh=old['original_array_hashes_inherited_not_recomputed'];need(array_sha(raw['rgb96'][k])==hh['RGB96_HWC_sha256'] and array_sha(raw['valid96'][k])==hh['valid96_sha256'],'Public inherited decoded crop hash')
                row_meta_check(x,i,r,q,L,old['original_query_available'],old['query_id'],g,k)
                need(r['split']==old['split'] and r['source_id']==old['source_id'] and r['raw_packet']==b and r['raw_row']==k and r['crop_constructed']==old['feature_constructed'] and r['cached_working_query_xy_FP32'] is None and r['cached_working_L_FP32'] is None,'Public source/split/noFP32alias metadata')
                if old['feature_constructed']:need(equal(x['valid96'][i],scalar_grid(q,L,old['raw_hw'])),'Independent public F64 validgrid');grid_rows+=1
                else:need(not x['rgb96'][i].any() and not x['valid96'][i].any(),'Original public nonconstructed placeholder');placeholder+=1
        for key in ['rgb96','valid96']:need(array_sha(x[key])==d['actual_raw_array_SHA'][key],'Public saved wholearray hash')
        summary['PUBLIC_HELD45_360']=dict(rows=360,frames=45,original_inputU=108,RGB_valid_byte_equal_rows=360,metadata6_exact_rows=360,F64_constructed_grid_checked=grid_rows,nonconstructed_zero_placeholder=placeholder)
        allrows.extend(rr);gstats=check_groups(groups,allrows,authorities)
        for pth,dig in BINDINGS.items():need(digest(pth)==dig,'Source/output unchanged after audit '+pth)
        review=dict(completed=True,PASS=True,code=binding(__file__),protocol=binding(H/'PROTOCOL.json'),checks=binding(H/'GENERATED_INVENTED_CHECKS.json'),input_complete=binding(COMPLETE),domains=summary,groups=gstats,total_original_rows=2144,actual_original_raw_arrays_read=dict(current4keys=4,public45x4keys=180),array_access_whitelist=ACCESSES,all2144_RGB_valid_bytes_equal=True,all2144_metadata6_and_F64_identities_equal=True,all268_prompt_original_metadata_equal=True,source_and_output_bindings_unchanged=True,verified_bindings=[dict(path=p,sha256=s) for p,s in BINDINGS.items()],independence='No producer/sampler imports or calls; scalar FP64 coordinate loops and scalar finiteavailable q8 prompt loops. Sources read first and compared to adapter outputs.',limits=['Raw96 pixels copied from original saved caches and matched byte-for-byte; no fullRGB re-decoding or re-sampling performed.','Frame and prior JSON projected onto explicit input-only fields immediately; no annotation/reference/score fields used.','Validgrid checked for all constructed crops; nonconstructed rows retain original zero placeholders rather than inventing crops.','FullRGB source/decoded hashes checked as metadata correspondence only; fullRGB bytes were not opened or hashed.','No SAM/head outputs, weights, classifier forward, fitting, reference/annotation evaluation or semantic label audit.'],fullRGB_annotation_reference_score_feature_P3_checkpoint_reads=0,model_SAM_classifier_forward_fit_calls=0,new_crop_or_coordinate_changes=0,selected_model=None,fully_solved=False,automatic_retry=False,seconds=time.monotonic()-started)
        write(H/'REVIEW.json',review);print(json.dumps(dict(PASS=True,review=binding(H/'REVIEW.json'),domains=summary,groups=gstats,seconds=review['seconds'])),flush=True)
    except BaseException:
        write(H/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
    ap=argparse.ArgumentParser();m=ap.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true');a=ap.parse_args();prepare() if a.prepare else audit()

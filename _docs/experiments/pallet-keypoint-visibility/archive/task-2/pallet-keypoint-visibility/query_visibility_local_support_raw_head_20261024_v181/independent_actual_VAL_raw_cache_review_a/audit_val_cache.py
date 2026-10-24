"""Independent all2048 VAL cache comparison; no producer/model/helper imports."""
from pathlib import Path
import hashlib,json,time
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;S=V/'synthetic_val_preparation';OUT=V/'synthetic_VAL2048_raw96_cache_v1'
def require(x,m):
    if not bool(x):raise AssertionError(m)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):require(sha(b['path'])==b['sha256'],'Source SHA '+b['path']);return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(t) for t in Path(p).read_text().splitlines() if t]
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def mask96(a,q,L,available):
    if not available:return np.zeros((96,96),bool),np.zeros((96,96),bool)
    u=(np.arange(96,dtype=np.float64)+.5)/96-.5;x,y=np.meshgrid(q[0]+L*u,q[1]+L*u)
    xi=np.floor(x).astype(np.int64);yi=np.floor(y).astype(np.int64);fx=x-xi;fy=y-yi;terms=[]
    for ax,ay in [(0,0),(0,1),(1,0),(1,1)]:
        xx,yy=xi+ax,yi+ay;inside=(xx>=0)&(xx<512)&(yy>=0)&(yy<512);t=np.zeros((96,96),np.float64);w=(fx if ax else 1-fx)*(fy if ay else 1-fy);t[inside]=a[yy[inside],xx[inside]].astype(np.float64)*w[inside];terms.append(t)
    valid=(x>=-.5)&(x<511.5)&(y>=-.5)&(y<511.5);out=np.sum(np.stack(terms),axis=0)>0
    return out&valid,valid
def run():
    start=time.monotonic();complete=read(OUT/'COMPLETE.json');require(complete['completed'] and complete['PASS'] and complete['evaluation_only'] and not complete['training_admission'],'Actual eval-only completion')
    require(set(complete['domains'])=={'SYNTH_VAL2048'},'Only original VAL domain');d=complete['domains']['SYNTH_VAL2048']
    src=lines(S/'VAL2048_ORIGINAL_INPUTS_METADATA_ONLY.jsonl');saved=lines(check(d['metadata']));require(len(src)==len(saved)==2048 and [r['original_global_index'] for r in src]==list(range(4096,6144)),'Original full VAL ordering')
    with np.load(check(d['inputs']),allow_pickle=False) as z:x={k:z[k].copy() for k in z.files}
    wanted={'rgb96','valid96','sam_mask96','sam_available96','original_query_available','row_id','original_global_index','native_slot','query_xy','L'};require(set(x)==wanted,'Input-only exact schema, no GT')
    for k,dt,shape in [('rgb96',np.uint8,(2048,96,96,3)),('valid96',bool,(2048,96,96)),('sam_mask96',bool,(2048,96,96)),('sam_available96',bool,(2048,96,96)),('original_query_available',bool,(2048,)),('original_global_index',np.int64,(2048,)),('native_slot',np.int8,(2048,)),('query_xy',np.float64,(2048,2)),('L',np.float64,(2048,))]:require(x[k].dtype==dt and x[k].shape==shape,'Cache dtype/shape '+k)
    support=read(V/'synthetic_VAL2048_support_v1/COMPLETE.json')
    with np.load(check(support['sidecar']),allow_pickle=False) as z:
        require(z['evaluation_only'].item() and not z['training_admission'].item(),'Original VAL supervision authority')
        owner=z['owner_central4_soft5'].copy();targets=dict(support_target=np.column_stack((z['visible_support_target'],z['foreign_front_support_target'])),owner_soft=owner[:,:3].copy(),owner_target=np.full(2048,-1,np.int8),owner_ambiguous_mass=owner[:,3].copy(),owner_invalid_mass=owner[:,4].copy())
        for key,dst in [('original_global_index','original_global_index'),('original_occurrence_id','row_id'),('native_slot','native_slot'),('query_xy','query_xy'),('L','L'),('query_available','original_query_available')]:require(np.array_equal(z[key],x[dst],equal_nan=True) if z[key].dtype.kind=='f' else np.array_equal(z[key],x[dst]),'Original VAL support metadata '+key)
        for flag in src[0]['original_input_flags']:require(np.array_equal(z[flag],np.array([r['original_input_flags'][flag] for r in src],bool)),'Original source flag '+flag)
    with np.load(check(d['targets']),allow_pickle=False) as z:
        require(set(z.files)==set(targets),'Exact separate target5keys')
        for k,t in targets.items():require(exact(z[k],t),'Original raw support/owner5 target bytes '+k)
    sam=read(check(complete['VAL_SAM_complete']));require(sam['completed'] and sam['PASS'] and sam['evaluation_only'] and sam['scene_count']==256,'Actual same frozen VAL SAM completion')
    groups={s['scene_index']:s for s in sam['scenes']};require(set(groups)==set(range(512,768)),'All original256 VAL scenes')
    used={};raw_path=None;raw=None;packetcount=0;nullgroups=0
    for lo in range(0,2048,8):
        rr=src[lo:lo+8];g=groups[rr[0]['scene_index']];require(g['original_input_metadata8']==rr,'SAM original metadata8')
        enabled=bool(g['prompt_available'] and g['runtime_available']);packet=None
        if g['outputs'] is None:require(not enabled,'Missing group explicitly unavailable');nullgroups+=1
        else:
            sb=g['outputs'];check(sb);used[sb['path']]=sb
            with np.load(sb['path'],allow_pickle=False) as z:packet={k:z[k].copy() for k in z.files}
            require(packet['image_logits'].dtype==np.float32 and packet['image_logits'].shape==(512,512) and np.isfinite(packet['image_logits']).all(),'Saved full logits')
            require(exact(packet['mask_logits_gt0'],packet['image_logits']>0),'Saved SAM threshold0')
            require(np.array_equal(packet['original_query_xy'],np.asarray([[np.nan,np.nan] if r['original_query_xy'] is None else r['original_query_xy'] for r in rr],np.float64),equal_nan=True),'SAM own exact q8')
            require(np.array_equal(packet['original_L'],np.asarray([np.nan if r['original_L'] is None else r['original_L'] for r in rr],np.float64),equal_nan=True),'SAM own exact L8')
            require(np.array_equal(packet['native_slot'],np.arange(8)) and packet['original_occurrence_id'].tolist()==[r['original_occurrence_id'] for r in rr] and packet['original_global_index'].tolist()==[r['original_global_index'] for r in rr],'SAM original native/IDs')
            require(np.array_equal(packet['prediction_available'],[r['original_input_flags']['query_available'] and enabled for r in rr]),'SAM inputU preserved');packetcount+=1
        for j,r in enumerate(rr,lo):
            m=saved[j];q=r['original_query_xy'];L=r['original_L'];av=r['original_input_flags']['query_available']
            require(m['original']==r and m['domain_index']==j and m['split']=='VALIDATION' and m['evaluation_only'] and not m['training_admission'],'Original eval-only row metadata')
            require(str(x['row_id'][j])==r['original_occurrence_id'] and int(x['original_global_index'][j])==r['original_global_index'] and int(x['native_slot'][j])==r['native_slot'] and bool(x['original_query_available'][j])==av,'Input identity/native/U')
            require(np.array_equal(x['query_xy'][j],np.asarray([np.nan,np.nan] if q is None else q,np.float64),equal_nan=True) and ((L is None and np.isnan(x['L'][j])) or L==x['L'][j]),'Original exact input qL')
            path=Path(r['saved_crop_chunk']['path'])
            if path!=raw_path:
                check(r['saved_crop_chunk']);used[str(path)]=r['saved_crop_chunk']
                with np.load(path,allow_pickle=False) as z:raw={k:z[k].copy() for k in ['rgb96','valid96','global_index','crop_constructed']}
                raw_path=path
            k=r['saved_crop_row'];im=raw['rgb96'][k];valid=raw['valid96'][k]
            require(int(raw['global_index'][k])==r['original_global_index'] and bool(raw['crop_constructed'][k])==r['original_input_flags']['crop_constructed'],'Original raw source row')
            require(exact(x['rgb96'][j],im) and exact(x['valid96'][j],valid),'Every original RGB96/valid96 byte')
            require(ash(im)==r['saved_rgb96_sha256_inherited']==m['actual_raw_RGB96_sha256'] and ash(valid)==r['saved_valid96_sha256_inherited']==m['actual_raw_valid96_sha256'],'Original/raw cache fact hashes')
            sm,sa=mask96(None if packet is None else packet['image_logits'],q,L,av and enabled)
            require(exact(sm,x['sam_mask96'][j]) and exact(sa,x['sam_available96'][j]),'All same96 SAM direct4tap independently exact')
            require(m['SAM_output']==g['outputs'],'SAM output lineage')
    for b in used.values():check(b)
    require(int((~x['original_query_available']).sum())==8 and all(int((x['native_slot']==k).sum())==256 for k in range(8)),'All native8/U8 preserved')
    r=dict(completed=True,PASS=True,review_code=bind(__file__),actual_cache_complete=bind(OUT/'COMPLETE.json'),source_VAL_support_complete=bind(V/'synthetic_VAL2048_support_v1/COMPLETE.json'),
        source_support_numeric_review=complete['independent_support_audit'],SAM_complete=complete['VAL_SAM_complete'],coverage=dict(rows2048=2048,scenes256=256,original_U8=8,all8native256each=True,alloriginalflags_preserved=True,allrawRGB96_validbytes_and_hashes_exact=True,allSAMmask96_and_availability_independently_exact=True,allsource_support2_owner3_AMBIG_INVALID_targetbytes_exact=True,SAM_saved_packets=packetcount,explicit_unavailable_groups=nullgroups,source_and_SAM_files_SHA_before_after=len(used)),
        method='No producer/operator/helper imports. Original VAL input manifest/raw crop chunks first, saved SAM logits direct4tap at originalqL96 grid, original new VAL support/owner5 sidecar exactjoined then compared split target5 arrays. Original VALIDATION and all rows retained.',
        evaluation_only=True,training_admission=False,new_model_SAM_fit_or_selection=0,private_publicheld_V112_reads=0,selected_model=None,fully_solved=False,seconds=time.monotonic()-start)
    f=H/'REVIEW.json';f.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(completed=True,PASS=True,review=bind(f),coverage=r['coverage'])))
if __name__=='__main__':run()

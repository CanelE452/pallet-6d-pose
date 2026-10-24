"""Independent, input-only full-table audit. No GT or visibility-score reads."""
from pathlib import Path
import hashlib,json,time,sys
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent; V=H.parent; P=V.parent
START=time.monotonic()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(p):return {'path':str(p),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def checked(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
def eq(a,b):assert np.array_equal(a,b,equal_nan=True)
def close(a,b):assert np.allclose(a,b,rtol=1e-11,atol=1e-13,equal_nan=True)
def pack(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def jl(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def native(q):return q is not None and np.shape(q)==(2,) and np.isfinite(q).all() and not(np.asarray(q)==-1).all()
def extent(q):
    if q is None:return None
    q=np.asarray(q);good=np.array([native(x) for x in q])
    if good.sum()<2:return None
    lo=q[good].min(0);hi=q[good].max(0)
    return np.r_[lo,hi].tolist() if (hi>lo).all() else None
def context_fraction(b,hw):
    if b is None:return np.nan
    h,w=hw;b=np.asarray(b); x=b[0]+(np.arange(224)+.5)*(b[2]-b[0])/224;y=b[1]+(np.arange(224)+.5)*(b[3]-b[1])/224
    xx,yy=np.meshgrid(x,y); ix=np.floor(xx).astype(int);iy=np.floor(yy).astype(int); dx=xx-ix;dy=yy-iy
    valid=np.ones((224,224),bool)
    for ox,oy,ww in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
        valid &= ((ix+ox>=0)&(ix+ox<w)&(iy+oy>=0)&(iy+oy<h))|(ww==0)
    return valid.mean()
def g42(q,k,b,hw):
    h,w=hw;q=np.asarray(q,np.float64); finite=np.isfinite(q).all(1);safe=np.where(finite[:,None],q,0)
    span=np.ptp(safe[finite],axis=0) if finite.any() else np.zeros(2);s=max(float(span.max()),1.)
    rel=(safe-safe[k])/s;rel[~finite]=0
    valid=finite&(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)
    return np.r_[np.eye(8)[k],rel.ravel(),valid,((np.asarray(b).reshape(2,2)-safe[k])/s).ravel(),safe[k]/[max(w,1),max(h,1)],span/s,np.asarray([h,w])/s].astype(np.float32)
GROUPS=(((0,1),(3,2),(4,5),(7,6)),((0,3),(1,2),(4,7),(5,6)),((0,4),(1,5),(2,6),(3,7)))
def null(lines):
    if any(x is None for x in lines):return None,np.nan,np.nan
    a=np.asarray(lines);_,s,vh=np.linalg.svd(a,full_matrices=False);c=float(s[1]/s[0]) if s[0]>1e-10 else np.nan
    if not np.isfinite(c) or c<=1e-10:return None,np.nan,c
    return vh[-1],float(np.sqrt(np.square(a@vh[-1]).mean())),c
def geometry_all(q8):
    out={key:np.zeros((8,3),bool) if key.endswith('valid') else np.full((8,3),np.nan) for key in ('global_group_residual','global_group_condition','global_group_valid','heldout_group_residual','heldout_group_condition','heldout_group_valid')}
    if q8 is None or not all(native(x) for x in q8):return out
    q=np.asarray(q8);s=float(np.ptp(q,axis=0).max())
    if s<=1e-10:return out
    q=(q-q.mean(0))/s
    for gi,edges in enumerate(GROUPS):
        lines=[]
        for a,b in edges:
            l=np.cross(np.r_[q[a],1.],np.r_[q[b],1.]);n=np.linalg.norm(l[:2]);lines.append(l/n if n>1e-10 else None)
        _,res,c=null(lines);out['global_group_condition'][:,gi]=c
        if np.isfinite(res):out['global_group_residual'][:,gi]=res;out['global_group_valid'][:,gi]=True
        for j,(a,b) in enumerate(edges):
            if lines[j] is None:continue
            v,_,c=null([l for ii,l in enumerate(lines) if ii!=j])
            for k,neighbor in ((a,b),(b,a)):
                out['heldout_group_condition'][k,gi]=c
                if v is None:continue
                line=np.cross(np.r_[q[neighbor],1.],v);norm=np.linalg.norm(line[:2])
                if norm>1e-10:out['heldout_group_residual'][k,gi]=abs(line@np.r_[q[k],1.])/norm;out['heldout_group_valid'][k,gi]=True
    return out

sealpath=V/'input_only_reliability_v1/INPUT_SEAL.json';seal=read(sealpath);assert seal['completed'] and seal['PASS'] and seal['GT_read'] is False
for key in ('table','schema','whitelist','result','code','math','protocol','policy','entry'):checked(seal[key])
table=pack(seal['table']['path']);rows=read(seal['whitelist']['path'])['rows'];assert len(rows)==26624
N=len(rows);eq(table['global_index'],np.arange(N));assert list(table['risk_names'])==seal['risk_names']
flags=('finite_non_sentinel_query','native_continuous_valid','native_nearest_valid','context_geometry_valid','crop_constructed','feature_constructed','query_available','factual_query_available','context_available','Q_inside','RGB_central4_available')
for key in ('corpus','source_occurrence_index','occurrence_id','scene_index','base_index','native_slot','split')+flags:
    assert np.array_equal(table[key],np.asarray([r[key] for r in rows]))
eq(table['query_xy'],np.array([r['query_xy'] if r['query_xy'] is not None else [np.nan]*2 for r in rows]));eq(table['L'],np.array([r['L'] if r['L'] is not None else np.nan for r in rows]))
cursor=0;sourcechecks={}
for corpus,src in seal['source_input_bindings'].items():
    for key in ('initializer_result','proposals','queries','crop_facts'):checked(src[key])
    proposals=jl(src['proposals']['path']);queries=jl(src['queries']['path']);facts=read(src['crop_facts']['path'])['rows'];cfg=src['config']
    for si,pr in enumerate(proposals):
        candidates=pr['candidates'];sel=None if not candidates else int(np.argmax([c['score'] for c in candidates]));assert sel==pr['selected_index']
        chosen=None if sel is None else candidates[sel]
        for k in range(8):
            local=si*8+k;r=rows[cursor+local];qr=queries[local];f=facts[local]
            assert r['global_index']==cursor+local and r['source_occurrence_index']==local and r['corpus']==corpus and r['scene_index']==si and r['base_index']==si//4 and r['native_slot']==k
            assert r['occurrence_id']==qr['query_id']==f['occurrence_id']==pr['id']+':native'+str(k)
            assert r['split']==qr['split']==f['split']==pr['split']==('TRAIN' if si//4<cfg['TRAIN_bases'] else 'VALIDATION')
            assert r['raw_hw']==pr['raw_hw'] and r['selected_index']==sel
            assert r['query_xy']==qr['query_xy']==f['query_xy'] and r['L']==qr['roi_side']==f['L'] and r['q8_raw']==f['initial_points']
            assert r['valid96_chunk']==f['rgb96_chunk'] and r['valid96_row']==f['rgb96_row'] and r['valid96_sha256']==f['valid96_sha256']
            for key in flags+('context_box_xyxy',):assert r[key]==f[key]
            assert f['source_emulation'] is False and f['actual_original_initializer'] is True
            if chosen is None:assert r['q8_raw'] is r['native_conf8'] is r['selected_bbox'] is r['selected_detector_confidence'] is None
            else:
                assert r['q8_raw']==chosen['keypoints_xy'][:8] and r['query_xy']==chosen['keypoints_xy'][k]
                assert r['native_conf8']==chosen['keypoints_conf'][:8] and r['selected_detector_confidence']==chosen['score'] and r['selected_bbox']==chosen['box_xyxy']
    assert len(queries)==len(facts)==cfg['slots'] and len(proposals)==cfg['scenes']
    sourcechecks[corpus]={'rows':len(queries),'scenes':len(proposals),'original_selected_argmax_q8_confidence_bbox_L_ID_flags_exact':True};cursor+=len(queries)
assert cursor==N
risks=np.full((N,10),np.nan);G=np.zeros((N,42),np.float32);gc=np.zeros(N,bool);qn=np.full((N,16),np.nan);qnv=np.zeros((N,16),bool);fractions=np.full((N,3),np.nan)
aux={key:np.zeros((N,3),bool) if key.endswith('valid') else np.full((N,3),np.nan) for key in ('global_group_residual','global_group_condition','global_group_valid','heldout_group_residual','heldout_group_condition','heldout_group_valid')}
maskkey=None;maskpack=None;chunks=set();geocache={};contextcache={}
for i,r in enumerate(rows):
    key=r['valid96_chunk']['path']
    if key!=maskkey:
        with np.load(key,allow_pickle=False) as z:maskpack={kk:z[kk] for kk in ('valid96','global_index','crop_constructed')}
        maskkey=key;chunks.add(key)
    j=r['valid96_row'];m=maskpack['valid96'][j];assert int(maskpack['global_index'][j])==r['source_occurrence_index'] and bool(maskpack['crop_constructed'][j])==r['crop_constructed']
    assert hashlib.sha256(np.ascontiguousarray(m).tobytes()).hexdigest()==r['valid96_sha256'] and m.dtype==bool
    q=r['query_xy'];qq=r['q8_raw'];h,w=r['raw_hw'];b=extent(qq);assert b==r['context_box_xyxy']
    if qq is not None:qn[i]=(np.asarray(qq)/[w,h]).ravel();qnv[i]=np.repeat([native(x) for x in qq],2)
    if r['crop_constructed']:
        u=(np.arange(96)+.5)/96-.5;xx,yy=np.meshgrid(q[0]+u*r['L'],q[1]+u*r['L']);expected=(xx>=-.5)&(xx<w-.5)&(yy>=-.5)&(yy<h-.5);eq(m,expected)
        fractions[i,:2]=[m.mean(),m[32:64,32:64].mean()]
    else:assert not m.any()
    frame=(r['corpus'],r['scene_index'])
    if frame not in contextcache:contextcache[frame]=context_fraction(b,[h,w]);geocache[frame]=geometry_all(qq)
    fractions[i,2]=contextcache[frame]
    for key,a in geocache[frame].items():aux[key][i]=a[r['native_slot']]
    if native(q) and b is not None:G[i]=g42(qq,r['native_slot'],b,[h,w]);gc[i]=True
    outside=np.nan;pb=r['selected_bbox']
    if native(q) and pb is not None and (np.asarray(pb[2:])>pb[:2]).all():
        d=np.maximum(np.maximum(np.asarray(pb[:2])-q,np.asarray(q)-pb[2:]),0);outside=np.linalg.norm(d)/max(float((np.asarray(pb[2:])-pb[:2]).max()),1.)
    score=np.nan if r['selected_detector_confidence'] is None else r['selected_detector_confidence'];conf=np.nan if r['native_conf8'] is None else r['native_conf8'][r['native_slot']]
    geom=[]
    for prefix in ('global','heldout'):
        vals=aux[prefix+'_group_residual'][i];valid=aux[prefix+'_group_valid'][i];geom.append(float(np.sqrt(np.square(vals).mean())) if valid.all() else np.nan)
    risks[i]=[1-score,1-conf,*(1-fractions[i]),float(not r['query_available']),float(not(r['native_continuous_valid'] and r['native_nearest_valid'])),outside,*geom]
eq(G,table['G42']);eq(gc,table['G42_constructed']);eq(qn,table['q8_image_normalized']);eq(qnv,table['q8_image_normalized_valid']);eq(fractions,table['crop_valid_fractions_local_central32_context']);close(risks,table['risks']);eq(np.isfinite(risks),table['risk_valid'])
for key,a in aux.items():eq(a,table[key]) if a.dtype==bool else close(a,table[key])
eq(np.c_[G.astype(np.float64),qn],table['descriptors']);eq(np.c_[np.repeat(gc[:,None],42,axis=1),qnv],table['descriptor_valid'])
oldsealpath=P/'query_visibility_native_geometry_control_20261024_v133/input_geometry_v1/INPUT_GEOMETRY_SEAL.json';oldseal=read(oldsealpath);old=pack(checked(oldseal['G42']))
for key in ('G42','query_xy','L','occurrence_id','global_index','source_occurrence_index','scene_index','base_index','native_slot','native_continuous_valid','native_nearest_valid','query_available'):
    a=old[key];b=table[key][:6144];eq(a,b) if a.dtype.kind not in 'US' else None
    if a.dtype.kind in 'US':assert np.array_equal(a,b)
eq(old['geometry_constructed'],gc[:6144]);eq(old['C16_feature_constructed'],table['feature_constructed'][:6144])
noquery=~table['finite_non_sentinel_query'];assert int(noquery.sum())==112 and not G[noquery].any() and not table['descriptor_valid'][noquery].any()
assert not table['risk_valid'][noquery][:,[0,1,2,3,4,7,8,9]].any() and (table['risks'][noquery][:,5:7]==1).all()
assert np.isnan(table['query_xy'][noquery]).all() and np.isnan(table['L'][noquery]).all() and np.isnan(table['q8_raw'][noquery]).all()
counts={corpus:{'rows':int((table['corpus']==corpus).sum()),'noquery':int(((table['corpus']==corpus)&noquery).sum()),'query_available':int(table['query_available'][table['corpus']==corpus].sum())} for corpus in sourcechecks}
receipt=dict(completed=True,PASS=True,input_only_stage=True,input_seal=bind(sealpath),input_table=seal['table'],code=bind(Path(__file__)),all26624_original_input_rows_checked=True,source_checks=sourcechecks,counts=counts,risk_defined_counts=np.isfinite(risks).sum(0).tolist(),all10_fixed_risks_recomputed=True,all58_descriptors_and_validity_recomputed=True,all42_geometry_features_recomputed_exact=True,old6144_original_G42_packet_bitexact=True,old_geometry_seal=bind(oldsealpath),mask_chunks=len(chunks),all_valid96_row_hashes_and_exact_local_crop_geometry_checked=True,context224_positive_bilinear_support_recomputed=True,noquery112_original_rows_missing_geometry_masks_and_risks_preserved=True,maximum_risk_absolute_difference=float(np.nanmax(abs(risks-table['risks']))),GT_reads=0,C29_score_reads=0,RGB_key_or_body_reads=0,visual_feature_checkpoint_model_private_V112_fit_PnP_reads_or_calls=0,seconds=time.monotonic()-START)
out=H/'SAVED_INPUT_TABLE_REVIEW.json';assert not out.exists();out.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps({'completed':True,'PASS':True,'review':bind(out),'counts':counts,'seconds':receipt['seconds']}),flush=True)

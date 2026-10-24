"""All9392 saved TRAIN cache audit from original metadata/raw/target sources; no helper imports."""
from pathlib import Path
from collections import Counter
import hashlib,json,time
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent;OUT=V/'source_raw96_cache_v1';A=V/'raw_cache_preparation'
V180=P/'query_visibility_frozen_sam_source_baseline_20261024_v180';V153=P/'query_visibility_mr6d_full_public_source_20261024_v153';V161=P/'query_visibility_public_train_owner_counterfactual_20261024_v161'
def require(x,msg):
    if not bool(x):raise AssertionError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def check(b):require(sha(b['path'])==b['sha256'],'SHA '+b['path']);return Path(b['path'])
def exact(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def qarray(q):return np.asarray([np.nan,np.nan] if q is None else q,np.float64)

def independent_mask(logit,q,L,enabled,hw):
    if not enabled:return np.zeros((96,96),bool),np.zeros((96,96),bool)
    q=qarray(q);require(np.isfinite(q).all() and np.isfinite(L) and L>0,'Available original q/L')
    offset=(np.arange(96,dtype=np.float64)+.5)/96-.5
    xx,yy=np.meshgrid(q[0]+L*offset,q[1]+L*offset);ix=np.floor(xx).astype(int);iy=np.floor(yy).astype(int);fx=xx-ix;fy=yy-iy
    h,w=hw;terms=[]
    for ax,ay in [(0,0),(0,1),(1,0),(1,1)]:
        px=ix+ax;py=iy+ay;inside=(px>=0)&(px<w)&(py>=0)&(py<h)
        weight=(fx if ax else 1-fx)*(fy if ay else 1-fy);part=np.zeros((96,96),np.float64);part[inside]=weight[inside]*logit[py[inside],px[inside]].astype(np.float64);terms.append(part)
    value=np.sum(np.stack(terms),axis=0,dtype=np.float64);valid=(xx>=-.5)&(xx<w-.5)&(yy>=-.5)&(yy<h-.5)
    return (value>0)&valid,valid

def expected_metadata():
    ss=lines(V180/'ALL4096_TRAIN_INPUTS_METADATA_ONLY.jsonl');facts=lines(V153/'full230_native_DINO_v1/FEATURE_INPUT_FACTS1840_INPUT_ONLY.jsonl')
    pub=[r for r in facts if r['split']=='SOURCE_DEVELOPMENT'];cf=lines(V161/'next_feature_input_metadata/ALL3816_COUNTERFACTUAL_INPUT_OCCURRENCES_NO_FEATURES.jsonl')
    crop={r['frame_index']:r['crop_packet'] for r in read(V153/'full230_native_DINO_v1/CROP_PACKETS230_INPUT_ONLY.json')['sheets']}
    d={}
    for domain,rows in [('SYNTH_TRAIN4096',ss),('PUBLIC_TRAIN1480',pub),('CF3816',cf)]:
        out=[]
        for r in rows:
            if domain=='SYNTH_TRAIN4096':row_id=r['original_occurrence_id'];gi=r['original_global_index'];q=r['original_query_xy'];L=r['original_L'];av=r['original_input_flags']['query_available'];b=r['saved_crop_chunk'];k=r['saved_crop_row'];imsha=r['saved_rgb96_sha256_inherited'];vsha=r['saved_valid96_sha256_inherited'];gid='SYNTH_SCENE_%06d'%r['scene_index'];hw=[512,512]
            elif domain=='PUBLIC_TRAIN1480':row_id=r['query_id'];gi=r['global_index'];q=r['query_xy'];L=r['L'];av=r['native_continuous_valid'];b=crop[r['frame_index']];k=r['native_slot'];imsha=r['RGB96_HWC_sha256'];vsha=r['valid96_sha256'];gid='PUBLIC_FRAME_%04d'%r['frame_index'];hw=[720,1280]
            else:
                row_id=r['input_occurrence_id'];gi=r['original_global_index'];q=r['query_xy'];L=r['L'];av=r['query_available'];b=r['variant_payload'];k=r['native_slot'];imsha=vsha=None;gid=r['pair_id']+'/'+r['variant'];hw=[720,1280]
                f=facts[gi];require(f['query_id']==r['original_query_id'] and f['split']==r['split']=='SOURCE_DEVELOPMENT' and f['native_slot']==r['native_slot'] and f['query_xy']==q and f['L']==L and f['native_continuous_valid']==av,'CF original public qL/slot/availability')
            out.append(dict(original=r,row_id=row_id,gi=gi,q=q,L=L,av=av,binding=b,rawrow=k,imsha=imsha,vsha=vsha,gid=gid,hw=hw))
        d[domain]=out
    return d

def expected_targets(expected,p):
    targets={}
    for domain,rows in expected.items():n=len(rows);targets[domain]=dict(support_target=np.full((n,2),-1,np.int8),owner_soft=np.full((n,3),np.nan,np.float64),owner_target=np.full(n,-1,np.int8),owner_ambiguous_mass=np.full(n,np.nan,np.float64),owner_invalid_mass=np.full(n,np.nan,np.float64))
    b=p['inputs'];ss=expected['SYNTH_TRAIN4096']
    for key in ['end_targets','TRAIN_owner6']:
        with np.load(check(b[key]),allow_pickle=False) as z:
            for src in ['original_global_index','original_occurrence_id','native_slot','query_xy','L','split','query_available']:
                values={'original_global_index':np.array([r['gi'] for r in ss],np.int32),'original_occurrence_id':np.array([r['row_id'] for r in ss]),'native_slot':np.array([r['original']['native_slot'] for r in ss],np.int8),'query_xy':np.array([qarray(r['q']) for r in ss],np.float64),'L':np.array([np.nan if r['L'] is None else r['L'] for r in ss],np.float64),'split':np.array(['TRAIN']*4096),'query_available':np.array([r['av'] for r in ss],bool)}
                require(np.array_equal(z[src],values[src],equal_nan=True) if z[src].dtype.kind=='f' else np.array_equal(z[src],values[src]),'Original SYNTH target exact identity '+src)
            for flag in ss[0]['original']['original_input_flags']:require(np.array_equal(z[flag],np.array([r['original']['original_input_flags'][flag] for r in ss],bool)),'SYNTH target originalflag')
            if key=='end_targets':targets['SYNTH_TRAIN4096']['support_target']=np.column_stack((z['visible_support_target'],z['foreign_front_support_target']))
            else:
                a=z['scene_owner_central4'];require(a.dtype==np.float64 and a.shape==(4096,6),'Original TRAIN owner6')
                t=targets['SYNTH_TRAIN4096'];t['owner_soft']=np.column_stack((a[:,0]+a[:,1],a[:,2],a[:,3]));t['owner_ambiguous_mass']=a[:,4].copy();t['owner_invalid_mass']=a[:,5].copy()
    sourcepub={r['gi']:i for i,r in enumerate(expected['PUBLIC_TRAIN1480'])};publictarget=targets['PUBLIC_TRAIN1480']['owner_target']
    with check(b['public_labels']).open() as f:
        for gi,line in enumerate(f):
            if gi not in sourcepub:continue
            z=json.loads(line);j=sourcepub[gi];r=expected['PUBLIC_TRAIN1480'][j];require(z['global_index']==gi and z['query_id']==r['row_id'] and z['native_slot']==r['original']['native_slot'] and z['split']=='SOURCE_DEVELOPMENT' and z['query_xy']==r['q'] and z['L']==r['L'],'Original PUBLIC owner metadata')
            publictarget[j]=z['owner_aux_target']
    cflabels=lines(check(b['CF_labels']));require(len(cflabels)==3816,'CF all labels')
    for j,(r,z) in enumerate(zip(expected['CF3816'],cflabels)):
        m=r['original'];require(z['cf_global_index']==j,'CF original order')
        for key in ['input_occurrence_id','pair_id','variant','original_query_id','original_global_index','original_frame_index','source_id','split','native_slot','query_xy','L','query_available','feature_constructed','central4_available','pair_pixel_proof_PASS','selected_anchor_native_slot']:require(z[key]==m[key],'CF label original metadata '+key)
        targets['CF3816']['owner_target'][j]=z['owner_aux_target']
    require([r['gi'] for r,y in zip(expected['PUBLIC_TRAIN1480'],publictarget) if y>=0]==p['public_known_original_indices'],'Frozen PUBLIC253 selection')
    require(np.flatnonzero(targets['CF3816']['owner_target']>=0).tolist()==p['CF_known_occurrence_indices'],'Frozen CF314 selection')
    for domain in ['PUBLIC_TRAIN1480','CF3816']:
        t=targets[domain];y=t['owner_target'];known=y>=0;t['owner_soft'][known]=np.eye(3,dtype=np.float64)[y[known]];t['owner_ambiguous_mass'][known]=0;t['owner_invalid_mass'][known]=0
    return targets

def run():
    started=time.monotonic();complete=read(OUT/'COMPLETE.json');require(complete['completed'] and complete['PASS'] and complete['source_only'],'Completed source rawcache barrier')
    p=read(A/'PROTOCOL_BEFORE_RAW_SAM_AND_TARGET_BODIES.json');expected=expected_metadata();targets=expected_targets(expected,p)
    source=read(V180/'frozen_sam_allTRAIN4096_predictions_v2/COMPLETE.json');publiccf=read(V/'frozen_sam_public_cf_inputs_v1/COMPLETE.json')
    groups={'SYNTH_SCENE_%06d'%s['scene_index']:s for s in source['scenes']};groups.update({s['group_id']:s for s in publiccf['groups']})
    used={};checks={};raw_path=None;raw=None;sam_id=None;logits=None
    wanted={'rgb96','valid96','sam_mask96','sam_available96','original_query_available','row_id','original_global_index','native_slot','query_xy','L'}
    for domain,rr in expected.items():
        record=complete['domains'][domain]
        with np.load(check(record['inputs']),allow_pickle=False) as z:x={k:z[k].copy() for k in z.files}
        metadata=lines(check(record['metadata']));require(len(metadata)==len(rr) and set(x)==wanted,'Exact cache rows and input whitelist')
        with np.load(check(record['targets']),allow_pickle=False) as z:
            require(set(z.files)==set(targets[domain]),'Separate target5keys')
            for key,val in targets[domain].items():require(exact(z[key],val),'Actual source target bytes '+domain+'/'+key)
        for j,(r,m) in enumerate(zip(rr,metadata)):
            require(m['original']==r['original'] and m['row_id']==r['row_id'] and m['original_global_index']==r['gi'] and m['native_slot']==r['original']['native_slot'] and m['query_xy']==r['q'] and m['L']==r['L'] and m['original_query_available']==r['av'] and m['split']==r['original']['split'],'All original metadata preserved')
            require(m['raw_packet']==r['binding'] and m['raw_row']==r['rawrow'] and m['domain_index']==j,'Original source packet/row address')
            require(str(x['row_id'][j])==r['row_id'] and int(x['original_global_index'][j])==r['gi'] and int(x['native_slot'][j])==r['original']['native_slot'] and bool(x['original_query_available'][j])==r['av'],'NPZ original scalar metadata')
            require(np.array_equal(x['query_xy'][j],qarray(r['q']),equal_nan=True) and ((r['L'] is None and np.isnan(x['L'][j])) or x['L'][j]==r['L']),'NPZ original exact qL')
            path=Path(r['binding']['path'])
            if raw_path!=path:
                check(r['binding']);used[str(path)]=r['binding']
                with np.load(path,allow_pickle=False) as z:raw={k:z[k].copy() for k in ['rgb96','valid96','crop_constructed']+(['query_xy','L','native_slot','query_id'] if domain=='CF3816' else ['global_index'])}
                raw_path=path
            k=r['rawrow'];im=raw['rgb96'][k];valid=raw['valid96'][k]
            require(im.dtype==np.uint8 and im.shape==(96,96,3) and valid.dtype==bool and valid.shape==(96,96),'Original raw dtype/shape')
            require(exact(x['rgb96'][j],im) and exact(x['valid96'][j],valid),'Every raw RGB96/valid96 byte equals original')
            require(ash(im)==m['actual_raw_RGB96_sha256'] and ash(valid)==m['actual_raw_valid96_sha256'],'Actual cache pixel SHA metadata')
            if r['imsha'] is not None:require(ash(im)==r['imsha'] and ash(valid)==r['vsha'],'Original inherited source pixel hashes')
            if domain=='CF3816':require(str(raw['query_id'][k])==r['original']['original_query_id'] and np.array_equal(raw['query_xy'][k],qarray(r['q']),equal_nan=True) and raw['L'][k]==r['L'],'CF raw q/L identity')
            else:require(int(raw['global_index'][k])==r['gi'],'Original saved raw global ID')
            s=groups[r['gid']];require(m['SAM_output']==s['outputs'],'SAM output lineage exact')
            if sam_id!=r['gid']:
                if s['outputs'] is None:logits=None
                else:
                    sb=s['outputs'];check(sb);used[sb['path']]=sb
                    with np.load(sb['path'],allow_pickle=False) as z:logits=z['image_logits'].copy()
                sam_id=r['gid']
            enabled=bool(r['av'] and s['prompt_available'] and s['runtime_available']);sm,sa=independent_mask(logits,r['q'],r['L'],enabled,r['hw'])
            require(exact(x['sam_mask96'][j],sm) and exact(x['sam_available96'][j],sa),'All same96 SAM mask/availability independent4tap')
        checks[domain]=dict(rows=len(rr),all_original_metadata_exact=True,raw_RGB_valid_bytes_exact=True,all_SAM_mask_availability_exact=True,all_target_source_arrays_byteexact=True,original_inputU=sum(not r['av'] for r in rr),
            hard_owner_counts={str(v):int((targets[domain]['owner_target']==v).sum()) for v in [-1,0,1,2]})
        print(json.dumps(dict(domain=domain,allrows=len(rr),PASS=True,seconds=time.monotonic()-started)),flush=True)
    for b in used.values():check(b)
    for key in ['end_targets','TRAIN_owner6','public_labels','CF_labels']:check(p['inputs'][key])
    require([checks[k]['original_inputU'] for k in checks]==[80,194,360],'All original U counts')
    report=dict(completed=True,PASS=True,review_code=bind(__file__),actual_cache_complete=bind(OUT/'COMPLETE.json'),protocol=bind(A/'PROTOCOL_BEFORE_RAW_SAM_AND_TARGET_BODIES.json'),
        coverage=checks,all9392_original_raw_and_metadata_and_labels_checked=True,unique_original_and_SAM_source_files_SHA_before_after=len(used),
        method='No producer/helper imports. Constructed expected domain rows from original V180/V153/V161 metadata; compared alloriginal rawRGB96/valid96 bytes; independently interpolated savedSAM logits4taps on fixedoriginalq/L grid; source label sidecars directly exactjoined and rawowner6 collapsed without renormalization.',
        limits=['Confirms saved rawcache and named source supervision fidelity. Does not rerun SAM, independently certify source physical native identity, evaluate trained head quality or establish natural-real/private/held transfer.','Original source-end geometry/operator numeric audits remain separate evidence.'],
        new_model_SAM_fit_or_selection=0,private_publicheld_V112_reads=0,selected_model=None,fully_solved=False,seconds=time.monotonic()-started)
    path=H/'REVIEW.json';path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(completed=True,PASS=True,review=bind(path))))
if __name__=='__main__':run()

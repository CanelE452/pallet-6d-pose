"""Independent saved-input audit. No producer/operator import, model or forward."""
from pathlib import Path
import argparse, hashlib, json, math, time
from collections import Counter
import numpy as np
H=Path(__file__).resolve().parent; V=H.parent; P=V.parent
O=V/'common_query_diagnostic_v1'; PRE=V/'common_query_preparation_a'
V180=P/'query_visibility_frozen_sam_source_baseline_20261024_v180'
VARIANTS=['BEFORE','COVER','NEAR_SHAM','FAR_SHAM']
def need(ok,msg):
    if not ok: raise AssertionError(msg)
def sha(p):
    p=Path(p).resolve();need(p.is_relative_to(P),'project only')
    need(not any(s in str(p).lower() for s in ['_v112','_v177','_v178','_v169','current223']),'forbidden path')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def binding(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def load_json(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def check(b):need(sha(b['path'])==b['sha256'],'SHA '+b['path']);return Path(b['path'])
def same(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def qf(q):return np.asarray([np.nan,np.nan] if q is None else q,np.float64)
def raw96(rgb,q,L):
    if q is None or L is None:return np.zeros((96,96,3),np.uint8),np.zeros((96,96),bool)
    # Independent grid and direct four-tap accumulation, preserving original rounding.
    pos=(np.arange(96,dtype=np.float64)+.5)/96-.5
    x,y=np.meshgrid(q[0]+pos*L,q[1]+pos*L);ix=np.floor(x).astype(int);iy=np.floor(y).astype(int)
    fx=x-ix;fy=y-iy;acc=np.zeros((96,96,3),np.float64);h,w=rgb.shape[:2]
    for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
        xx=ix+dx;yy=iy+dy;inside=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
        wt=(fx if dx else 1-fx)*(fy if dy else 1-fy)
        acc[inside]+=rgb[yy[inside],xx[inside]].astype(np.float64)*wt[inside,None]
    valid=(x>=-.5)&(x<w-.5)&(y>=-.5)&(y<h-.5)
    out=np.floor(np.clip(acc,0,255)+.5).astype(np.uint8);out[~valid]=0
    return out,valid
def owner4(packet,q,L):
    if q is None or L is None:return np.full(5,np.nan)
    h,w=packet['part_id'].shape;mass=[0.]*5
    for cy in [47,48]:
        for cx in [47,48]:
            x=q[0]+((cx+.5)/96-.5)*L;y=q[1]+((cy+.5)/96-.5)*L
            if not (-.5<=x<w-.5 and -.5<=y<h-.5):mass[4]+=.25;continue
            i=math.floor(x);j=math.floor(y);fx=x-i;fy=y-j
            for dx in [0,1]:
                for dy in [0,1]:
                    px=i+dx;py=j+dy;wt=(fx if dx else 1-fx)*(fy if dy else 1-fy)*.25
                    if not (0<=px<w and 0<=py<h):c=4
                    elif packet['ambiguity'][py,px]:c=3
                    else:
                        p=int(packet['part_id'][py,px]);c=2 if p<0 else 1 if p==999 else 0
                    mass[c]+=wt
    return np.asarray(mass,np.float64)
def project_patch(g):
    """Independent ray-plane/2x2 solve, integer image centres; no source helper."""
    K=np.array(g['camera']['K']);R=np.array(g['camera']['R']);t=np.array(g['camera']['t']);inv=np.linalg.inv(K)
    dep=np.full((512,512),np.inf);oof=False;good=True
    for quad in g['quads_world']:
        f=np.asarray(quad)@R.T+t
        if np.any(f[:,2]<=g['camera'].get('near',.001)):good=False;oof=True;continue
        uv=f@K.T;uv=uv[:,:2]/uv[:,2,None]
        oof|=bool(np.any(uv<0) or np.any(uv>511))
        e1=f[1]-f[0];e2=f[3]-f[0];n=np.cross(e1,e2)
        gram=np.array([[e1@e1,e1@e2],[e1@e2,e2@e2]])
        lo=np.maximum(np.floor(uv.min(0)).astype(int),0);hi=np.minimum(np.ceil(uv.max(0)).astype(int),511)
        for y in range(lo[1],hi[1]+1):
            for x in range(lo[0],hi[0]+1):
                ray=np.array([x,y,1.])@inv.T;den=float(ray@n)
                if den==0:continue
                hit=ray*(float(f[0]@n)/den)
                if hit[2]<=g['camera'].get('near',.001):continue
                d=hit-f[0];ab=np.linalg.solve(gram,[d@e1,d@e2])
                if np.all(ab>=0) and np.all(ab<=1):dep[y,x]=min(dep[y,x],hit[2])
    return dep,good,oof
def end_classes(before,scene,depth,primary):
    cls=np.zeros((512,512),np.int8)
    for y,x in np.argwhere(np.isfinite(depth)):
        e=float(depth[y,x]);bp=int(before['part_id'][y,x]);sp=int(scene['part_id'][y,x]);bd=float(before['camera_depth'][y,x]);sd=float(scene['camera_depth'][y,x])
        if before['ambiguity'][y,x] or scene['ambiguity'][y,x]:c=4
        else:
            c=5;finite=math.isfinite(bd) and bd>0;tol=64*np.finfo(np.float64).eps*max(1.,abs(bd) if finite else 0.,abs(e))
            if bp>=0 and bp!=999 and finite and bd<e-tol:c=3
            if bp==primary and finite and abs(bd-e)<=tol:
                if sp==bp and before['global_triangle_id'][y,x]==scene['global_triangle_id'][y,x] and bd==sd:c=1
                if sp==999 and math.isfinite(sd) and sd>0 and sd<bd:c=2
        cls[y,x]=c
    return cls
def mass96(cls,q,L):
    # Vectorized independently constructed four-tap one-hot, original domain rule.
    off=((np.arange(96,dtype=np.float64)+.5)/96-.5)*L
    xx,yy=np.meshgrid(q[0]+off,q[1]+off);ix=np.floor(xx).astype(int);iy=np.floor(yy).astype(int);fx=xx-ix;fy=yy-iy
    w=np.zeros((96,96,7));a,b=np.indices((96,96))
    for dx in [0,1]:
        for dy in [0,1]:
            x=ix+dx;y=iy+dy;inside=(x>=0)&(x<512)&(y>=0)&(y<512);c=np.full((96,96),6);c[inside]=cls[y[inside],x[inside]]
            w[a,b,c]+=(fx if dx else 1-fx)*(fy if dy else 1-fy)
    invalid=(xx<-.5)|(xx>=511.5)|(yy<-.5)|(yy>=511.5);w[invalid]=0;w[invalid,6]=1
    return w.reshape(-1,7).mean(0)
def target(m,available,good,oof):
    if not available or not good or oof:return [-1,-1]
    unknown=float(m[3]+m[4]+m[5])
    if m[1]+m[2]+unknown==0:return [0,-1]
    return [1 if m[c]>0 else -1 if unknown>0 else 0 for c in [1,2]]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--complete',type=Path,required=True);args=ap.parse_args();start=time.monotonic()
    c=load_json(args.complete);need(c['PASS'] and c['completed'] and c['fit_admission'] is False,'actual diagnostic complete')
    for k in ['code','protocol','scope','entry','input_groups','input_slots','inputs','targets','metadata','measures']:check(c[k])
    planned=rows(c['input_slots']['path']);actual=rows(c['metadata']['path']);measure=rows(c['measures']['path']);groups=rows(c['input_groups']['path'])
    original=rows(V180/'ALL4096_TRAIN_INPUTS_METADATA_ONLY.jsonl')
    geom=rows(V180/'source_end_supervision_preparation_a/TRAIN1024_FIXED_END_PATCH_GEOMETRY_METADATA.jsonl');gm={(g['base_index'],g['native_slot']):g for g in geom}
    with np.load(c['inputs']['path'],allow_pickle=False) as z:X={k:z[k].copy() for k in z.files}
    with np.load(c['targets']['path'],allow_pickle=False) as z:Y={k:z[k].copy() for k in z.files}
    need(set(X)=={'rgb96','valid96','row_id','original_global_index','native_slot','query_xy','L','original_query_available'},'exact8 input keys/noGTorSAM')
    need(set(Y)=={'support_target','owner_soft','owner_target','owner_ambiguous_mass','owner_invalid_mass'},'separate5 targets')
    need(len(planned)==len(actual)==len(measure)==len(original)==4096 and len(groups)==512,'allcounts')
    expected_dtypes={'rgb96':np.uint8,'valid96':bool,'original_global_index':np.int64,'native_slot':np.int8,'query_xy':np.float64,'L':np.float64,'original_query_available':bool}
    for k,d in expected_dtypes.items():need(X[k].dtype==d,'inputdtype '+k)
    need(X['rgb96'].shape==(4096,96,96,3) and X['valid96'].shape==(4096,96,96),'rawshape')
    need(np.all(Y['owner_target']==-1) and not c['SAM_arrays_present'],'nohardownerGT/SAMsubstitution')
    first={}
    for base in range(128):
        fac=original[base*32]['factors'];key=tuple(fac[k] for k in ['geometry_family','material','aspect']);first.setdefault(key,base)
    selected=set(first.values());need(len(selected)==8,'fixed8factorcells')
    source_bindings={};crop_bindings={};raw_parity=0;cover_parity=0;owner_rows=0;end_rows=0;max_owner=0.;max_end=0.;target_counts=[Counter(),Counter()];rep=[];cached_crops={}
    for base in range(128):
        before=None;patches={}
        if base in selected:
            for k in range(8):patches[k]=project_patch(gm[base,k])
        for vi,var in enumerate(VARIANTS):
            j0=base*32+vi*8;g=groups[base*4+vi];sb=planned[j0]['source_RGB_binding'];path=check(sb);source_bindings[str(path)]=sb
            keys=['RGB_uint8','part_id','ambiguity','camera_depth','global_triangle_id','camera_K','camera_R','camera_t','native8_CF_xy','permCF_to_canonical']
            with np.load(path,allow_pickle=False) as z:packet={k:z[k].copy() for k in keys}
            need(ah(packet['RGB_uint8'])==g['decoded_RGB_SHA'],'decodedRGBsource')
            if vi==0:before=packet
            for k in range(8):
                i=j0+k;r=planned[i];a=actual[i];im=original[i];an=original[base*32+8+k]
                need(all(a[key]==value for key,value in r.items()),'plannedmetadata exact '+str(i))
                need(r['split']=='TRAIN' and r['native_slot']==k and r['variant']==var,'identity')
                need(r['source_image_original_metadata']==im and r['anchor_original_metadata']==an,'dualproductionorigin')
                need(r['anchor_original_global_index']==base*32+8+k and r['source_image_original_global_index']==i,'originindices')
                need(r['query_xy']==an['original_query_xy'] and r['L']==an['original_L'] and r['original_input_flags']==an['original_input_flags'],'COVERqLflags fixed')
                need(X['row_id'][i]==r['row_id'] and X['original_global_index'][i]==i and X['native_slot'][i]==k and X['original_query_available'][i]==r['original_query_available'],'arraymetadata identity')
                need(same(X['query_xy'][i],qf(r['query_xy'])) and same(np.asarray(X['L'][i]),np.asarray(np.nan if r['L'] is None else r['L'],np.float64)),'qLnumericbytes')
                rgb,valid=raw96(packet['RGB_uint8'],r['query_xy'],r['L']);need(same(rgb,X['rgb96'][i]) and same(valid,X['valid96'][i]),'independent originalRGB96/valid '+str(i));raw_parity+=1
                need(a['actual_rgb96_sha256']==ah(rgb) and a['actual_valid96_sha256']==ah(valid),'actualrowfacthash')
                if var=='COVER':
                    cb=r['anchor_saved_crop_packet'];cp=Path(cb['path']);crop_bindings[str(cp)]=cb
                    if str(cp) not in cached_crops:
                        check(cb)
                        with np.load(cp,allow_pickle=False) as z:cached_crops[str(cp)]={key:z[key].copy() for key in ['rgb96','valid96','crop_constructed','global_index']}
                    z=cached_crops[str(cp)];ri=r['anchor_saved_crop_row'];need(same(rgb,z['rgb96'][ri]) and same(valid,z['valid96'][ri]) and z['global_index'][ri]==r['anchor_original_global_index'],'all1024 originalCOVERbytes');cover_parity+=1
                    need(ah(rgb)==r['anchor_saved_rgb96_sha256'] and ah(valid)==r['anchor_saved_valid96_sha256'],'originalcoverfacthash')
                own=owner4(packet,r['query_xy'],r['L']);saved=np.r_[Y['owner_soft'][i],Y['owner_ambiguous_mass'][i],Y['owner_invalid_mass'][i]]
                need(np.allclose(own,saved,atol=2e-15,rtol=0,equal_nan=True),'independent scalar central4 owner '+str(i));owner_rows+=1
                if np.isfinite(own).all():max_owner=max(max_owner,float(np.max(np.abs(own-saved))))
                need(measure[i]['row_id']==r['row_id'] and measure[i]['diagnostic_row_index']==i,'measureidentity')
                em=measure[i]['end'];need(em['query_xy']==r['query_xy'] and em['L']==r['L'] and em['original_input_flags']==r['anchor_original_flags'],'endinput contract')
                need(em['native_visibility_target']==em['point_quality_target']==-1 and em['training_admission'] is False,'separate diagnostic targets')
                need(Y['support_target'][i].tolist()==[em['visible_support_target'],em['foreign_front_support_target']],'serializedtargets')
                for channel in range(2):target_counts[channel][int(Y['support_target'][i,channel])]+=1
                if base in selected:
                    if r['query_xy'] is None or r['L'] is None:
                        need(em['full96_average_mass'] is None and Y['support_target'][i].tolist()==[-1,-1],'missinganchor unknown')
                    else:
                        dep,good,oof=patches[k];cl=end_classes(before,packet,dep,gm[base,k]['primary_part_id']);mass=mass96(cl,r['query_xy'],r['L']);sm=np.asarray(em['full96_average_mass']);diff=float(np.max(np.abs(mass-sm)));max_end=max(max_end,diff)
                        need(diff<=1e-12,'representative independent registered patchmass '+str(i))
                        need(target(mass,r['original_query_available'],good,oof)==Y['support_target'][i].tolist(),'representative UNKNOWN/positive target')
                        rep.append({'diagnostic_row_index':i,'base_index':base,'native_slot':k,'variant':var,'mass_max_abs_difference':diff,'targets':Y['support_target'][i].tolist()})
                    end_rows+=1
        if (base+1)%16==0:print(json.dumps({'audited_bases':base+1,'raw_rows':raw_parity,'COVER_byte_rows':cover_parity,'owner_rows':owner_rows,'representative_end_rows':end_rows}),flush=True)
    need(raw_parity==owner_rows==4096 and cover_parity==1024 and end_rows==256,'actualcoverage')
    need(int((~X['original_query_available']).sum())==96,'U96')
    for r in actual:
        if not r['original_query_available']:need(Y['support_target'][r['diagnostic_row_index']].tolist()==[-1,-1],'inputUtargets')
    for b in list(source_bindings.values())+list(crop_bindings.values()):check(b)
    for key in ['inputs','targets','metadata','measures']:check(c[key])
    out={'completed':True,'PASS':True,'code':binding(__file__),'producer_complete':binding(args.complete),'coverage':{'rows_identity_qL_original_flags':4096,'source_buffers_SHA_before_after':len(source_bindings),'original_crop_chunks_SHA_before_after':len(crop_bindings),'independent_source_RGB96_and_valid96_rows':4096,'saved_original_COVER_full_byte_parity_rows':1024,'independent_scalar_central4_owner_rows':4096,'independent_registered_patch_end_rows':256,'fixed_bases':sorted(selected),'input_U':96,'anchor_U':24},'target_counts':{'visible':dict(target_counts[0]),'foreign':dict(target_counts[1])},'numerical':{'owner5_max_abs_difference':max_owner,'representative_end7_max_abs_difference':max_end},'representative_end_cases':rep,'all_source_and_output_SHA_unchanged':True,'original_inputs_unchanged':True,'SAM_arrays_absent_explicitly':True,'fit_admission':False,'new_fit_or_forward_or_threshold':0,'selected_model':None,'fully_solved':False,'limitations':['Full4096 end masses were checked for serialization/identity; geometry/source-ray end operator independently recomputed only outcome-blind8factorfirstbases256rows.','Literal central4 owner and intended-end support are distinct questions; foreign owner is not native external-E.','Declared source patch identity, not physical real-image corner gold.','SOURCE TRAIN diagnostic only; no VAL/private/current/held access or transfer claim.'],'seconds':time.monotonic()-start}
    with (H/'REVIEW.json').open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'PASS':True,'review':binding(H/'REVIEW.json'),'seconds':out['seconds']}),flush=True)
if __name__=='__main__':main()

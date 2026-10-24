"""Saved alpha/protected-raster metadata only, never RGB/features/models."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent;V=HERE.parent;B=V/'source_pair_intake'
S=B/'image_only_v2_exclusive_alpha'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def rows(p):return [json.loads(s)for s in Path(p).read_text().splitlines()if s]
def array(b,key):
    assert sha(b['path'])==b['sha256']
    with np.load(b['path'],allow_pickle=False)as z:return np.array(z[key],copy=True)
def bilinear(a,x,y):
    i=int(np.floor(x));j=int(np.floor(y));dx=x-i;dy=y-j;value=0.
    for ox,oy,w in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
        xx=i+ox;yy=j+oy
        if 0<=xx<a.shape[1]and 0<=yy<a.shape[0]:value+=float(a[yy,xx])*w
    return float(value)
def overlap(a,q,L,prefix):
    x,y=q;ix,iy=np.floor(q+.5).astype(int);h,w=a.shape
    inside=0<=ix<w and 0<=iy<h
    patch=a[max(0,iy-1):min(h,iy+2),max(0,ix-1):min(w,ix+2)]if inside else np.zeros((0,0),bool)
    vals=[bilinear(a,x+dx*L/192,y+dy*L/192)for dy in [-1,1]for dx in [-1,1]]
    return {prefix+'nearest_q_inside':bool(a[iy,ix])if inside else None,
        prefix+'native3x3_support_pixels':int(patch.sum()),prefix+'native3x3_valid_pixels':int(patch.size),
        prefix+'q_bilinear_alpha':bilinear(a,x,y),prefix+'RGB96_central4_mean_alpha':float(np.mean(vals)),
        prefix+'RGB96_central4_max_alpha':float(max(vals)),prefix+'RGB96_central4_all_zero':not any(v>0 for v in vals)}
def summarize(records):
    if not records:return dict(N=0)
    d=np.array([r['distance_native_cell_centres_px']for r in records]);dn=np.array([r['distance_native_cell_centres_over_L']for r in records])
    out=dict(N=len(records),within_3px=int(np.sum(d<=3)),within_point1L=int(np.sum(dn<=.1)),
        nearest_q_inside_protected=int(sum(r['protected_nearest_q_inside']is True for r in records)),
        protected_native3x3_any=int(sum(r['protected_native3x3_support_pixels']>0 for r in records)),
        protected_RGB96_central4_any=int(sum(not r['protected_RGB96_central4_all_zero']for r in records)),
        distance_px=dict(min=float(d.min()),median=float(np.median(d)),p90=float(np.quantile(d,.9)),max=float(d.max())),
        distance_over_L=dict(min=float(dn.min()),median=float(np.median(dn)),p90=float(np.quantile(dn,.9)),max=float(dn.max())))
    if all(r['accepted_pair']for r in records):
        for variant in ['cover','sham']:
            out[variant+'_nearest_q_inside']=int(sum(r[variant+'_nearest_q_inside']is True for r in records))
            out[variant+'_native3x3_untouched']=int(sum(r[variant+'_native3x3_support_pixels']==0 for r in records))
            out[variant+'_RGB96_central4_untouched']=int(sum(r[variant+'_RGB96_central4_all_zero']for r in records))
    return out

def main():
    assert sha(S/'IMAGE_ONLY_INTAKE_SEAL.json')=='5b1007a76f0f27345db03ad7b559de301cfe16a05d0e0728e1761c8bf0d049c5'
    rr=rows(B/'ALL960_NORMALIZED_INPUT_ONLY.jsonl');gg=rows(B/'CONSTRUCTION960_GT_ONLY.jsonl');ff=rows(S/'FEASIBILITY960_GT_ONLY.jsonl')
    cc={c['id']:c for c in rows(S/'actual/SELECTED_DONOR_RGBA_CERTIFICATES.jsonl')};out=[]
    for i,(r,g,f)in enumerate(zip(rr,gg,ff)):
        if g['original_natural_label_GT_only']!=0:continue
        assert r['slot_index']==g['base_row_index']==f['base_row_index']==i
        q=np.array(r['query_xy'],np.float64);L=float(r['L']);assert np.isfinite(q).all()and L>0
        p=array(f['protected_union_file'],'protected_union');yy,xx=np.nonzero(p);assert len(xx)
        distance=float(np.sqrt((xx-q[0])**2+(yy-q[1])**2).min())
        footprint=float(np.sqrt(np.maximum(np.abs(xx-q[0])-.5,0)**2+np.maximum(np.abs(yy-q[1])-.5,0)**2).min())
        rec=dict(base_row_index=i,input_id=r['input_id'],native_slot_logical_only=r['native_slot_logical_only'],source=g['source'],raw_split=r['split'],q_x=float(q[0]),q_y=float(q[1]),L=L,
            protected_pixels=int(p.sum()),protected_regions=f['protected_region_count'],accepted_pair=f['pair_ready'],pair_reason=f['reason'],
            distance_native_cell_centres_px=distance,distance_native_cell_centres_over_L=distance/L,
            distance_native_cell_footprints_px=footprint,distance_native_cell_footprints_over_L=footprint/L,
            within_3px=distance<=3,within_point1L=distance/L<=.1,**overlap(p,q,L,'protected_'))
        if f['pair_ready']:
            native_alpha=array(cc[f['assigned_donor_id']]['compact_RGBA_cutout'],'alpha')
            h,w=f['donor_transform']['resized_hw']
            alpha=np.array(Image.fromarray(native_alpha).resize((w,h),Image.Resampling.NEAREST),dtype=bool)
            for variant in ['cover','sham']:
                x,y=f[variant+'_xy'];a=np.zeros_like(p);a[y:y+h,x:x+w]=alpha
                rec.update(overlap(a,q,L,variant+'_'))
        out.append(rec)
    assert len(out)==143 and sum(r['accepted_pair']for r in out)==15
    accepted=[r for r in out if r['accepted_pair']]
    keys=sorted({(r['source'],r['raw_split'])for r in out})
    result=dict(completed=True,PASS=True,scope='Saved protected/alpha geometry only; all original q/L/labels preserved.',
        policy=bind(HERE/'QUERY_TO_AUX_SUPPORT_METADATA_POLICY.json'),source_seal=bind(S/'IMAGE_ONLY_INTAKE_SEAL.json'),
        all_before_D=summarize(out),accepted_pairs=summarize(accepted),
        by_source_raw_split={s+'|'+sp:dict(all_before_D=summarize([r for r in out if(r['source'],r['raw_split'])==(s,sp)]),accepted_pairs=summarize([r for r in accepted if(r['source'],r['raw_split'])==(s,sp)]))for s,sp in keys},
        interpretation=['Distances refer to an auxiliary recorded-cue-set support raster, NOT native GT coordinate error or native expected-part correspondence.','Protected-region miss does not imply actual donor miss; accepted-pair cover/sham alpha overlap is reported separately.','An untouched q central neighbourhood can coexist with a covered remote cue in the original crop. This diagnoses an operating-target difference; it does not alone establish the cause of C2 failure.','No images, labels, queries, thresholds or model decisions were changed.'],
        actual_RGB_feature_weight_model_forward_fit_current_PnP=0)
    fields=list(out[0])+[k for r in out for k in r if k not in out[0]];fields=list(dict.fromkeys(fields))
    csvpath=HERE/'QUERY_TO_AUX_SUPPORT_METADATA.csv'
    with csvpath.open('x',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(out)
    result['all143_CSV']=bind(csvpath)
    with (HERE/'QUERY_TO_AUX_SUPPORT_METADATA.json').open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result))

if __name__=='__main__':main()

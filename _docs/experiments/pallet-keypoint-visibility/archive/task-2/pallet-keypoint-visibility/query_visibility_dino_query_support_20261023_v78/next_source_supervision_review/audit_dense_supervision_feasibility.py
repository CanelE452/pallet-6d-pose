"""Bounded source-only feasibility counts, not a training dataset or model run.

All natural area totals come from already sealed native-mask audits.  Before
opening masks/RGB, fix two metadata-SHA-selected rows per source/split.  Only
counts at the unchanged q/L crop coordinates are saved; no label maps/caches.
"""
from pathlib import Path
from collections import defaultdict
import hashlib, json
import numpy as np
from PIL import Image

R=Path(__file__).resolve().parent
P=R.parent.parent
V76=P/'query_visibility_reference_local_support_20261023_v76'
META=V76/'gt_corpus_build/PER_ROW_TEACHER_GT_ONLY.jsonl'
INDEX=V76/'gt_corpus_build/ORDERED_METADATA_INDEX_GT_ONLY.jsonl'
SYNTH=V76/'source_inventory/UNCHANGED_MODEL_INPUT_LINKS.jsonl'
PUBLIC=V76/'public_query_emulation/INPUTS_RGB_ONLY.jsonl'

def b(p):
    return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def lines(p): return [json.loads(x) for x in p.read_text().splitlines()]
def write(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def add(stats, m0, m4, valid):
    return dict(valid=int(valid.sum()),visible=int((valid&m4).sum()),
        covered_support=int((valid&m0&~m4).sum()),outside_expected_support=int((valid&~m0).sum()),
        expected_support=int((valid&m0).sum()))
def crop_counts(m0,m4,q,L,n):
    h,w=m0.shape
    if not np.isfinite(q).all() or (q == -1).all():
        return dict(valid=0,visible=0,covered_support=0,outside_expected_support=0,expected_support=0,unique_native_pixels=0)
    u=((np.arange(n)+.5)/n-.5)*L
    xx,yy=np.meshgrid(q[0]+u,q[1]+u)
    px=np.floor(xx+.5).astype(int);py=np.floor(yy+.5).astype(int)
    valid=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)&(px>=0)&(px<w)&(py>=0)&(py<h)
    a=np.zeros((n,n),bool);v=a.copy()
    a[valid]=m0[py[valid],px[valid]];v[valid]=m4[py[valid],px[valid]]
    result=add(None,a,v,valid)
    result['unique_native_pixels']=len(np.unique(py[valid]*w+px[valid]))
    return result
def native_square_counts(m0,m4,q,L):
    h,w=m0.shape
    if not np.isfinite(q).all() or (q == -1).all():
        return dict(valid=0,visible=0,covered_support=0,outside_expected_support=0,expected_support=0)
    x=np.arange(max(0,int(np.ceil(q[0]-L/2))),min(w,int(np.ceil(q[0]+L/2))))
    y=np.arange(max(0,int(np.ceil(q[1]-L/2))),min(h,int(np.ceil(q[1]+L/2))))
    a=m0[np.ix_(y,x)];v=m4[np.ix_(y,x)]
    return add(None,a,v,np.ones_like(a,bool))

def main():
    meta,index,synth,public=[lines(p) for p in (META,INDEX,SYNTH,PUBLIC)]
    assert len(meta)==len(index)==5203 and len(synth)==1003 and len(public)==4200
    totals=defaultdict(lambda:defaultdict(int))
    for i,(m,ix) in enumerate(zip(meta,index)):
        assert m['row_index']==ix['row_index']==i and m['id']==ix['id']
        assert m['source']==ix['source'] and m['split']==ix['split']
        mm=m['mask_pair'];a=mm['M0_target_area'];v=mm['M4_target_area']
        assert 0<=v<=a
        stats=totals[m['source']+'|'+m['split']]
        for k,val in dict(instance_rows=1,native_canvas_pixel_occurrences=int(np.prod(ix['raw_hw'])),
            visible_target_pixels=v,expected_target_pixels=a,nonvisible_expected_target_pixels=a-v,
            outside_expected_target_pixels=int(np.prod(ix['raw_hw']))-a,
            rows_with_nonvisible_expected_target=int(a>v),rows_full_equals_visible=int(mm['Mfull_equals_Mvisible'])).items():stats[k]+=int(val)
    for stats in totals.values():
        stats['visible_over_expected_ratio']=stats['visible_target_pixels']/stats['expected_target_pixels']
        stats['nonvisible_over_expected_ratio']=stats['nonvisible_expected_target_pixels']/stats['expected_target_pixels']
    salt='V78_DENSE_FEASIBILITY_FIXED_METADATA_TWO_PER_SOURCE_SPLIT_V1'
    chosen=[]
    for source in ('SYNTHETIC','YCB_PUBLIC_REAL'):
        for split in ('train','val'):
            pool=[i for i,m in enumerate(meta) if m['source']==source and m['split']==split]
            chosen.extend(sorted(pool,key=lambda i:hashlib.sha256((salt+'\n'+meta[i]['id']).encode()).hexdigest())[:2])
    lock=dict(scope='Counts only, no dataset/cache/model/fit/current',selection_salt=salt,
        selection='All original rows form totals; exactly first2 SHA(id) per source/split for unchanged q/L crop count audit; no mask/label/quality/score filtering.',
        selected=[dict(row_index=i,id=meta[i]['id'],source=meta[i]['source'],split=meta[i]['split']) for i in chosen],
        bindings=[b(x) for x in (META,INDEX,SYNTH,PUBLIC,Path(__file__).resolve())],
        masks_RGB_opened_before_selection=0)
    write(R/'FIXED_METADATA_SAMPLE_LOCK.json',lock)
    sampled=[];aggregate=defaultdict(lambda:defaultdict(int));RGB_reads=0
    for j,i in enumerate(chosen):
        m,ix=meta[i],index[i]
        if i<1003:
            inp=synth[i];q=np.asarray(inp['unchanged_initial_query9'],float)[:8];L=float(inp['input_L'])
            image=inp['raw_RGB']['path'];imageSHA=inp['raw_RGB']['sha256'];producer='SAVED_N3_SYNTH_INITIALIZER'
        else:
            inp=public[i-1003];q=np.asarray(inp['initial_points'],float);L=float(inp['roi_side'])
            image=inp['image'];imageSHA=inp['image_sha256'];producer='GT_DERIVED_PUBLIC_SOURCE_QUERY_EMULATION'
        assert inp['id']==m['id']
        mm=m['mask_pair'];paths=(Path(mm['M0']['path']),Path(mm['M4']['path']))
        assert b(paths[0])['sha256']==mm['M0']['sha256'] and b(paths[1])['sha256']==mm['M4']['sha256']
        raw0,raw4=[np.asarray(Image.open(p)) for p in paths]
        threshold=127 if m['source']=='SYNTHETIC' else 0
        a,v=raw0>threshold,raw4>threshold
        assert not (v&~a).any() and list(a.shape)==ix['raw_hw'] and a.shape==v.shape
        assert int(a.sum())==mm['M0_target_area'] and int(v.sum())==mm['M4_target_area']
        rgb_checked=False
        if j%2==0:
            assert hashlib.sha256(Path(image).read_bytes()).hexdigest()==imageSHA
            rgb=np.asarray(Image.open(image).convert('RGB'))
            assert list(rgb.shape[:2])==ix['raw_hw'];RGB_reads+=1;rgb_checked=True
        crops=[]
        for k in range(8):
            native=native_square_counts(a,v,q[k],L)
            coarse=crop_counts(a,v,q[k],L,9)
            dense=crop_counts(a,v,q[k],L,96)
            row=dict(native_index=k,q=q[k].tolist(),L=L,native_integer_centers_in_halfopen_square=native,
                fixed_ROI9_nearest_labels=coarse,fixed_RGB96_nearest_labels=dense)
            crops.append(row)
            prefix=m['source']+'|'+m['split']
            for scale,counts in [('native',native),('ROI9',coarse),('RGB96',dense)]:
                for key,value in counts.items():aggregate[prefix][scale+'_'+key]+=value
            aggregate[prefix]['sampled_ROIs']+=1
            aggregate[prefix]['ROIs_with_both_visible_and_nonvisible_expected_support']+=int(native['visible']>0 and native['covered_support']>0)
            aggregate[prefix]['ROIs_with_any_nonvisible_expected_support']+=int(native['covered_support']>0)
        sampled.append(dict(row_index=i,id=m['id'],source=m['source'],split=m['split'],
            RGB_path_lineage_only=image,RGB_sha256=imageSHA,RGB_decoded_for_dimension_and_hash_check=rgb_checked,
            mask_pair=mm,query_producer=producer,crops=crops))
    report=dict(PASS=True,scope='Source-only read-only mask supervision feasibility; not a training corpus or model experiment.',
        exact_prior_native_mask_audit_totals=dict(totals),
        total_note='Instance-pixel occurrences, not unique scene union or independent samples. Native PNG audit areas; NOT BOP px_count_all substitution. M4 subset M0 was sealed-audited, and fixed8 rows replayed.',
        sample_aggregate=dict(aggregate),fixed_sample=sampled,
        sample_note='Native square counts can overlap across eight ROIs. 96-grid labels can repeatedly sample a native pixel; neither denominator is an independent-pixel count. ROI9 coordinates q+(i-4)*L/9; RGB96 coordinates q+((i+.5)/96-.5)*L, masks nearest floor+.5 with continuous+rounded image validity.',
        exact_source_splits_preserved=True,q_L_unchanged=True,GT_query_recenter=0,
        original_mask_pairs_read=8,source_RGB_decodes=RGB_reads,new_dataset_or_feature_cache_created=False,
        weights_backbone_head_forward_fit_current_PnP=0,bindings=lock['bindings'],sample_lock=b(R/'FIXED_METADATA_SAMPLE_LOCK.json'))
    write(R/'DENSE_MASK_DATA_AVAILABILITY_AUDIT.json',report)
    print(json.dumps(dict(PASS=True,report=b(R/'DENSE_MASK_DATA_AVAILABILITY_AUDIT.json'),totals=dict(totals),sample_aggregate=dict(aggregate))),flush=True)

if __name__=='__main__':main()

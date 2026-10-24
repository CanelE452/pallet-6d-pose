"""Own saved source image/mask arithmetic. No feature/model/current reader."""
from pathlib import Path
from collections import Counter
import hashlib
import importlib.util
import json
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent;V=HERE.parent;BASE=V/'source_pair_intake'
SRC=BASE/'image_only_v2_exclusive_alpha'
SEAL_SHA='5b1007a76f0f27345db03ad7b559de301cfe16a05d0e0728e1761c8bf0d049c5'
checked={};mask_cache={};json_cache={};rgb_cache={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x)for x in Path(p).read_text().splitlines()if x]
def check(b):
    p=b['path']
    if p not in checked:assert sha(p)==b['sha256'];checked[p]=b['sha256']
    assert checked[p]==b['sha256'];return Path(p)
def pixelsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def erode(a):
    p=np.pad(a,1);out=np.ones_like(a)
    for y in range(3):
        for x in range(3):out &= p[y:y+a.shape[0],x:x+a.shape[1]]
    return out
def mask(b):
    p=str(check(b))
    if p not in mask_cache:
        a=np.array(Image.open(p));assert a.dtype==np.uint8 and np.isin(a,[0,255]).all();mask_cache[p]=a>0
    return mask_cache[p]
def rgb(b):
    p=str(check(b))
    if p not in rgb_cache:rgb_cache[p]=np.array(Image.open(p).convert('RGB'),copy=True)
    return rgb_cache[p]
def load_npz(b):
    with np.load(check(b),allow_pickle=False)as z:return {k:np.array(z[k],copy=True)for k in z.files}
def partition(s):return {'TRAIN':'TRAIN','TRAIN_LAB':'TRAIN','VALIDATION':'VALIDATION','VAL_HALL':'VALIDATION'}[s]
def protected(row,g):
    out=np.zeros(row['raw_hw'],bool)
    boxes=([g['support_bbox_xyxy'],g['uncertainty_bbox_xyxy']] if g['source']=='LOCO'
           else [r['bbox']for r in g['protected_regions_GT_only']])
    for box in boxes:
        lo=np.floor(np.array(box[:2])+.5).astype(int)
        hi=(np.ceil(np.array(box[2:])+.5) if g['source']=='LOCO' else np.floor(np.array(box[2:])+.5)+1).astype(int)
        assert (lo>=0).all() and hi[0]<=out.shape[1] and hi[1]<=out.shape[0]
        out[lo[1]:hi[1],lo[0]:hi[0]]=True
    return out
def mass96(a,q,L):
    axis=(np.arange(96,dtype=np.float64)+.5)/96-.5
    xx,yy=np.meshgrid(q[0]+axis*L,q[1]+axis*L);ix=np.floor(xx).astype(int);iy=np.floor(yy).astype(int)
    dx=xx-ix;dy=yy-iy;out=np.zeros((96,96),np.float64)
    for ax,ay,w in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
        x=ix+ax;y=iy+ay;ok=(x>=0)&(y>=0)&(x<a.shape[1])&(y<a.shape[0]);out[ok]+=a[y[ok],x[ok]]*w[ok]
    valid=(xx>=-.5)&(xx<a.shape[1]-.5)&(yy>=-.5)&(yy<a.shape[0]-.5)
    return float(np.where(valid,out,0).sum())

def main():
    assert sha(SRC/'IMAGE_ONLY_INTAKE_SEAL.json')==SEAL_SHA
    seal=read(SRC/'IMAGE_ONLY_INTAKE_SEAL.json');assert seal['completed']and seal['PASS']
    check(seal['protocol']);check(seal['metadata_lock'])
    for b in seal['files']:check(b)
    locked=read(seal['metadata_lock']['path'])
    rows=lines(BASE/'ALL960_NORMALIZED_INPUT_ONLY.jsonl');gt=lines(BASE/'CONSTRUCTION960_GT_ONLY.jsonl')
    source=lines(BASE/'SOURCE960_DRIVER_LINEAGE.jsonl');assign=lines(BASE/'ONE_DONOR_ASSIGNMENT960_LOCK_BEFORE_PIXELS.jsonl')
    feats=lines(SRC/'FEASIBILITY960_GT_ONLY.jsonl');occs=lines(SRC/'OCCURRENCES2880_LINEAGE.jsonl')
    certs=lines(SRC/'actual/SELECTED_DONOR_RGBA_CERTIFICATES.jsonl');summary=read(SRC/'IMAGE_ONLY_FEASIBILITY_SUMMARY_KO.json')
    assert len(rows)==len(gt)==len(source)==len(assign)==len(feats)==960 and len(occs)==2880 and len(certs)==130
    pool=lines(check(locked['inputs']['donor_metadata']))
    nat_gt={}
    for name in ['loco_gt','pallox_gt']:nat_gt[name]=lines(check(locked['inputs'][name]))
    all_labels=[];by_split=Counter();why=Counter();accepted=[];nullD=0
    for i,(r,g,s,a,f)in enumerate(zip(rows,gt,source,assign,feats)):
        assert r['slot_index']==g['base_row_index']==s['slot_index']==a['base_row_index']==f['base_row_index']==i
        assert r['input_id']==g['input_id']==s['input_id'] and r['native_slot_logical_only']==g['native_slot']==s['native_slot_logical_only']
        assert r['query_xy']==g['original_q']==s['query_xy']==f['original_q'] and r['L']==g['original_L']==s['L']==f['original_L']
        assert r['split']==g['split']==s['split']==f['split'] and r['image_sha256']==g['image_sha256']==s['image_sha256']
        natural=g['original_natural_label_GT_only']
        if i<160:assert natural==(0 if nat_gt['loco_gt'][i]['before_aux_state']=='D' else -1)
        else:
            old=nat_gt['pallox_gt'][i-160];assert natural==old['scalar_target_D0_E1_Uminus1']
            assert g['retained_FULL_GT_cues']==old['retained_FULL_GT_cues']
            if natural==0:
                expected=[dict(kind='PALLOX_CLOSED_NATIVE_UNCERTAINTY',bbox=x['cue_uncertainty_bbox_unchanged'],cue_id=x['cue_id'],cue_view_index=x['cue_view_index'],r_unchanged=x['cue_reference_xy_unchanged'],certified_duplicate_group=x['certified_duplicate_group'])for x in old['retained_FULL_GT_cues']]
                assert g['protected_regions_GT_only']==expected
        candidates=[d for d in pool if d['split']==partition(r['split']) and d['image_sha256']!=r['image_sha256'] and d['object_id']%2==(1 if partition(r['split'])=='TRAIN' else 0)]
        key=lambda d:hashlib.sha256('|'.join(map(str,(20261023,r['input_id'],r['native_slot_logical_only'],'one_donor',d['id'],d['image_sha256']))).encode()).hexdigest()
        chosen=min(candidates,key=key)if candidates else None
        assert a['assigned_donor']==chosen and f['assigned_donor_id']==(chosen['id']if chosen else None)
        assert a['replacement_allowed']is False and a['one_donor_only']is True
        if natural==0:
            p=protected(r,g);saved=load_npz(f['protected_union_file'])['protected_union'];assert np.array_equal(p,saved)
            assert int(p.sum())==f['protected_pixels'] and pixelsha(p)==f['protected_sha256']
            nullD+=int((g.get('null_unlocated_limit_count')or 0)>0);why[f['reason']]+=1
        if f['pair_ready']:assert natural==0;accepted.append(i)
        for j,v in enumerate(['BEFORE','OPAQUE_COVER','CUE_PRESERVING_SHAM']):
            o=occs[3*i+j];expected=natural if j==0 else ([1,0][j-1]if f['pair_ready']else -1)
            assert o['occurrence_index']==3*i+j and o['base_row_index']==o['slot_index']==i and o['variant']==v
            assert o['occurrence_id']==f"{r['input_id']}::slot{r['native_slot_logical_only']}::{v}"
            assert o['candidate_aux_hidden_label_GT_only']==expected and o['known_candidate_GT_only']==(expected>=0)
            assert o['original_natural_label_GT_only']==natural and o['pair_ready']==f['pair_ready']
            assert o['human_gold']is False and o['native_expected_part_state']=='UNKNOWN'
            assert o['full_rgb_changed']==bool(j and f['pair_ready'])
            if not o['full_rgb_changed']:assert o['edited_fullRGB_file']is None and o['full_rgb_decoded_sha256']==s['decoded_RGB_sha256']
            all_labels.append(expected);by_split[(partition(r['split']),v,expected)]+=1
    assert Counter(all_labels)=={0:158,1:35,-1:2687} and len(accepted)==15
    assert why==summary['eligible_reasons']

    # Independent rule fixtures before actual mask replay.
    spec=importlib.util.spec_from_file_location('exclusive_rule_only',SRC/'exclusive_alpha_rule.py')
    rule=importlib.util.module_from_spec(spec);spec.loader.exec_module(rule)
    rng=np.random.default_rng(194)
    for n in range(100):
        a=rng.integers(0,2,(9,11),dtype=np.uint8).astype(bool);b=rng.integers(0,2,(9,11),dtype=np.uint8).astype(bool)
        out,c=rule.exclusive_alpha(a,b,True);assert np.array_equal(out,a&~b)and c['admitted']==bool(erode(a&~b).any())
        missing,c=rule.exclusive_alpha(a,b,False);assert not missing.any()and not c['admitted']
    cert_by_id={c['id']:c for c in certs};conflicts=0;exclusive_pixels=0;frames=set();admitted=0
    for c in certs:
        owner=c['scene_GT_owner_completeness'];scene=read(check(owner['scene_gt']))
        expected=scene[str(c['frame_id'])];assert owner['expected_instance_ids']==list(range(len(expected)))
        assert owner['available_metadata_instance_ids']==owner['expected_instance_ids'] and owner['complete']
        assert [x['instance_id']for x in owner['instance_masks']]==owner['expected_instance_ids']
        others=np.zeros((480,640),bool);selected=None
        for x in owner['instance_masks']:
            assert x['object_id']==expected[x['instance_id']]['obj_id']
            mm=mask(x['mask']);assert mm.shape==(480,640)
            if x['instance_id']==c['instance_id']:selected=mm;assert x['object_id']==c['object_id']
            else:others|=mm
        assert selected is not None and np.array_equal(selected,mask(c['mask']))
        ex=selected&~others;core=erode(ex);saved=load_npz(c['native_alpha_proof'])
        assert set(saved)=={'selected_visible','other_visible_union','exclusive_alpha','erode1_exclusive_alpha'}
        for k,v in [('selected_visible',selected),('other_visible_union',others),('exclusive_alpha',ex),('erode1_exclusive_alpha',core)]:assert np.array_equal(v,saved[k])
        assert not (core&others).any() and not (core&~selected).any()
        cc=c['exclusive_alpha_certificate'];assert cc['original_conflicting_pixels']==int((selected&others).sum())
        assert cc['retained_exclusive_alpha_pixels']==int(ex.sum())and cc['retained_erode1_pixels']==int(core.sum())
        assert c['foreign_opaque_cutout_admitted']==bool(core.any())
        conflicts+=int((selected&others).sum());exclusive_pixels+=int(ex.sum());admitted+=int(core.any());frames.add((c['scene_id'],c['frame_id']))
        if core.any():
            yy,xx=np.nonzero(core);bbox=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
            assert c['native_bbox_xyxy']==bbox
            cut=load_npz(c['compact_RGBA_cutout']);assert np.array_equal(cut['alpha'],core[bbox[1]:bbox[3],bbox[0]:bbox[2]])
    assert len(frames)==121 and admitted==129 and conflicts==79226 and exclusive_pixels==1834236

    # Reconstruct all15 accepted cover/sham rasters; do not re-search placements.
    mass_max_error=0.;relative_diffs=[];accepted_counts=Counter()
    for i in accepted:
        r,g,s,f=rows[i],gt[i],source[i],feats[i];c=cert_by_id[f['assigned_donor_id']]
        donor=rgb(c['image']);assert pixelsha(donor)==c['image_decoded_RGB_sha256']
        cut=load_npz(c['compact_RGBA_cutout']);x0,y0,x1,y1=c['native_bbox_xyxy']
        assert np.array_equal(cut['rgb'],donor[y0:y1,x0:x1])
        h,w=f['donor_transform']['resized_hw'];rgba=np.array(Image.fromarray(cut['rgb']).resize((w,h),Image.Resampling.BILINEAR))
        alpha=np.array(Image.fromarray(cut['alpha']).resize((w,h),Image.Resampling.NEAREST),dtype=bool)
        target=rgb(s['fullRGB_file_binding']);assert pixelsha(target)==s['decoded_RGB_sha256']
        p=protected(r,g);alpha_full=[]
        for j,name in enumerate(['cover_xy','sham_xy']):
            x,y=f[name];assert x>=0 and y>=0 and x+w<=target.shape[1] and y+h<=target.shape[0]
            af=np.zeros(target.shape[:2],bool);af[y:y+h,x:x+w]=alpha;alpha_full.append(af)
            image=target.copy();view=image[y:y+h,x:x+w];view[alpha]=rgba[alpha]
            o=occs[3*i+j+1];actual=rgb(o['edited_fullRGB_file'])
            assert np.array_equal(image,actual) and pixelsha(actual)==o['full_rgb_decoded_sha256']
        ca,sa=alpha_full;assert ca[p].all()and not sa[p].any()
        assert ca.sum()==sa.sum()==f['same_global_alpha_cells']==int(alpha.sum())
        q=r['query_xy'];L=r['L'];yy,xx=np.indices(ca.shape);extent=L*47.5/96
        inside=(xx>=q[0]-extent)&(xx<=q[0]+extent)&(yy>=q[1]-extent)&(yy<=q[1]+extent)
        assert (ca&~inside).sum()==(sa&~inside).sum()==0
        masses=[mass96(a,q,L)for a in alpha_full];stored=f['mass_recheck']['bilinear_RGB96_alpha_pixel_equivalents']
        err=max(abs(a-b)for a,b in zip(masses,stored));mass_max_error=max(mass_max_error,err)
        assert err<=1e-10 and min(masses)>0 and abs(masses[0]-masses[1])<=1e-9+.01*np.mean(masses)
        relative_diffs.append(float(abs(masses[0]-masses[1])/np.mean(masses)));accepted_counts[(g['source'],partition(r['split']))]+=1
    result=dict(completed=True,PASS=True,blockers=[],intake_seal=bind(SRC/'IMAGE_ONLY_INTAKE_SEAL.json'),policy=bind(HERE/'AUDIT_POLICY_BEFORE_SAVED_PACKETS.json'),
        all_bound_files_SHA_verified=len(seal['files']),base_rows=960,occurrences=2880,canonical_before={'D':143,'E':20,'U':797},canonical_all={'D':158,'E':35,'U':2687},
        split_variant_counts={'|'.join(map(str,k)):v for k,v in sorted(by_split.items())},
        all960_minSHA_one_donor_assignments_exact=True,all137_Pallox_D_retained_FULL_cues_and_closed_union_exact=True,
        all143_protected_native_masks_exact=True,canonical_natural_E20_preserved_before_after_U=True,
        selected_donors=130,unique_donor_frames=121,all_sceneGT_instances_and_original_native_masks_replayed=True,
        original_conflicting_pixels_removed=conflicts,retained_exclusive_alpha_pixels=exclusive_pixels,admitted_donors=admitted,
        known_other_overlap_in_exclusive_eroded_native_alpha=0,accepted_pairs=15,accepted_pair_counts={'|'.join(k):v for k,v in accepted_counts.items()},
        failures_retained=128,failure_reason_counts=dict(why),all30_changed_fullRGB_rasters_exact=True,
        same_donor_shape_scale_global_alpha_and_local_mass_verified=True,mass96_max_absolute_difference_from_saved=mass_max_error,
        relative_mass_difference_max=max(relative_diffs),before_D_rows_with_null_unlocated_limit=nullD,
        practical_training_assessment='Nonzero new counterfactual supervision exists in TRAIN9 and VAL6 accepted pairs, all from Pallox. The controlled known-set labels are consistent with full protected-union cover/sham certificates. LOCO contributes natural D6 only; failed LOCO edits remain U. Tiny controlled pair counts, shared physical-bin environment, recorded-only cue sets/null unlocated limits, and donor/RGB-edge shortcuts limit generalization. No new admission gate or fit authorization is issued.',
        limits=['Native alpha ownership is exact against supplied visible instance masks; it is not error-free optical or semantic ownership gold.','RGB bilinear edge colors can mix across the alpha boundary; alpha ownership does not prove pure donor appearance.','The unchanged fixed support labels are conditional AI cue-set augmentation, not native expected-part/human DES gold.','No P3/feature/model/current data were read.'],activity=dict(actual_source_masks_read=len(mask_cache),accepted_source_RGB_images_read=len(rgb_cache),source_feature_weight_model_forward_fit_current_PnP=0))
    with (HERE/'REVIEW.json').open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    with (HERE/'SEAL.json').open('x')as f:json.dump(dict(completed=True,PASS=True,review=bind(HERE/'REVIEW.json'),script=bind(__file__),policy=bind(HERE/'AUDIT_POLICY_BEFORE_SAVED_PACKETS.json')),f,indent=2);f.write('\n')
    print(json.dumps(result))

if __name__=='__main__':main()

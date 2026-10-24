"""Bounded source-only inventory: paths, array headers, annotations and byte SHA.

No RGB/mask pixel decoding, model import/forward, teacher labeling or training.
Existing source input coordinates/features are never recomputed or rewritten.
"""
from pathlib import Path
import csv,gzip,hashlib,json,time,zipfile
from collections import Counter
import numpy as np

HERE=Path(__file__).resolve().parent
V76=HERE.parent
P=V76.parent
V60=P.parent/'external_ray_proxy_head_20261022_v60'
V61=P.parent/'target_presence_classifier_20261022_v61'
V74=P/'query_visibility_loco_native_context_20261023_v74'
V75=P/'query_visibility_native_context_pretrain_20261023_v75'
REPO=Path('/home/minjae/Documents/github/pallet-pose')
CONTRACT=V76/'TARGET_CONTRACT_BEFORE_CORPUS.json'
CONTRACT_SHA='83e07b7f8ca241ab57f994db86ad2d9012d84499d5820ccb2f3400b47117c1ff'
START=time.monotonic()

def sha_stream(f):
    h=hashlib.sha256()
    for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def sha(p):
    with open(p,'rb')as f:return sha_stream(f)
def bind(p):return dict(path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size)
def load(p):
    with (gzip.open(p,'rt')if str(p).endswith('.gz')else open(p))as f:return json.load(f)
def readlines(p):return [json.loads(x)for x in p.read_text().splitlines()]
def write(p,v):
    assert not p.exists(),p
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def jsonlines(p,rows):
    assert not p.exists(),p
    p.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n'for r in rows))
def csvrows(p,rows):
    assert not p.exists(),p
    with p.open('w')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def npzheaders(p):
    result={}
    with zipfile.ZipFile(p)as z:
        for n in z.namelist():
            with z.open(n)as f:
                ver=np.lib.format.read_magic(f)
                shape,fortran,dtype=(np.lib.format.read_array_header_1_0(f)if ver==(1,0)else np.lib.format.read_array_header_2_0(f))
            result[n.removesuffix('.npy')]=dict(shape=list(shape),dtype=str(dtype),fortran_order=fortran,uncompressed_npy_bytes=z.getinfo(n).file_size)
    return result
ziphandles={}
def locator_sha(loc):
    if '::'in loc:
        archive,member=loc.split('::',1)
        if archive not in ziphandles:ziphandles[archive]=zipfile.ZipFile(archive)
        with ziphandles[archive].open(member)as f:return sha_stream(f)
    return sha(loc)

assert sha(CONTRACT)==CONTRACT_SHA
contract=load(CONTRACT)
for b in contract['bindings']:assert sha(b['path'])==b['sha256']
u=load(V60/'PREPARED_INPUTS_SYNTH.json.gz');meta=load(V60/'SYNTH_FEATURE_METADATA.json')
assert len(u)==len(meta)==1003 and len({r['id']for r in u})==1003
mask_audit={r['id']:r for r in csv.DictReader(open(V61/'MASK_CONTRACT_AUDIT.csv'))}
prior_frames={r['id']:r for r in readlines(P/'query_visibility_visible_preservation_20261023_v67/source_audit/SYNTH_REFERENCE_FRAME_MANIFEST.jsonl')}
anchors={}
for r in csv.DictReader(open(P/'query_visibility_visible_preservation_20261023_v67/source_audit/SYNTH_REFERENCE_ANCHOR_INVENTORY.csv')):
    anchors.setdefault(r['id'],[]).append(r)
selection=load(V74/'native_context_cache/SOURCE_ROW_SELECTION_LOCK.json')['selected']
rowmap={r['id']:r for r in readlines(V74/'source_contract/TYPED_SOURCE_ROW_MAP.jsonl')}
assert len(selection)==5120 and all(r['id']in rowmap for r in selection)
input_rows=[];teacher_rows=[];counts=Counter();errors=[];filechecks=[]
for i,(r,m)in enumerate(zip(u,meta)):
    assert r['id']==m['id'] and r['split']in ('train','validation')
    image=Path(r['image']);ann=image.parent.parent/'labels'/image.name.replace('_rgb.png','_label.json')
    ma=mask_audit[r['id']];frame=prior_frames[r['id']]
    assert image.exists()and ann.exists()
    image_digest=sha(image);ann_digest=sha(ann)
    assert image_digest==r['image_sha256']==frame['image_sha256']
    assert ann_digest==ma['annotation_sha256']==frame['annotation_sha256']
    d=load(ann);assert len(d['objects'])==1;o=d['objects'][0];lab=o['v2_labels']
    ref=np.asarray(o['projected_cuboid'],float);q=np.asarray(r['initial_points'],float)
    assert ref.shape==(8,2)and q.shape==(9,2)and np.isfinite(ref).all()and np.isfinite(q).all()
    assert o['keypoint_convention']=='camera_dynamic_0123_v4'and sorted(o['perm_v4'])==list(range(8))
    assert [d['camera_data']['height'],d['camera_data']['width']]==r['raw_hw']
    ar=sorted(anchors[r['id']],key=lambda x:int(x['native_corner']))
    assert np.array_equal(ref,np.array([[float(x['reference_x']),float(x['reference_y'])]for x in ar]))
    assert np.array_equal(q[:8],np.array([[float(x['initial_query_x']),float(x['initial_query_y'])]for x in ar]))
    masks={}
    for kind,alias in [('amodal','M0'),('visible','M4')]:
        path=Path(ma[kind+'_path']);assert path.exists();digest=sha(path);assert digest==ma[kind+'_sha256']
        values=json.loads(ma[kind+'_values']);areas=json.loads(ma[kind+'_counts'])
        assert set(values)<={0,1,2,254,255}and sum(areas)==np.prod(r['raw_hw'])
        target_area=sum(v for key,v in zip(values,areas)if key>127)
        assert target_area==int(float(ma[kind+'_area']))==int(lab['mask_area_'+('amodal'if kind=='amodal'else'visible')])
        masks[alias]=dict(path=str(path),sha256=digest,raw_dtype='uint8',native_hw=r['raw_hw'],target_rule='raw >127',raw_values=values,raw_counts=areas,target_area=target_area)
        counts[alias+'_present']+=1
    assert ma['PASS']=='True'and masks['M4']['target_area']<=masks['M0']['target_area']
    counts['frames']+=1;counts[r['split']+'_frames']+=1;counts['query9_finite']+=int(np.isfinite(q).all(1).sum());counts['reference8_finite']+=8
    counts['mask_profile_'+str(lab.get('mask_profile'))]+=1;counts['native_convention_'+o['keypoint_convention']]+=1
    input_rows.append(dict(id=r['id'],split=r['split'],raw_RGB=dict(path=str(image),sha256=image_digest,raw_hw=r['raw_hw']),
        unchanged_initial_query9=q.tolist(),model_native_query_indices=list(range(8)),centroid_query_index_excluded_from8ROI=8,
        frozen_ROI=dict(path=str(V60/'SYNTH_ROI_CACHE.npz'),frame_index=i,x_field='x',pixel_valid_field='pixel_valid',shape=[8,65,9,9]),
        input_L=m['side'],gain=m['gain'],offset=m['offset'],backbone_input_shape=m['input_shape'],
        producer=r['upstream_provenance'],GT_reference_mask_pose_or_state_forward=False))
    teacher_rows.append(dict(id=r['id'],split=r['split'],annotation_GT_only=dict(path=str(ann),sha256=ann_digest),
        reference_projected_dynamic8_GT_only=ref.tolist(),reference_convention=o['keypoint_convention'],stored_dynamic_to_writer_fixed_permutation_GT_only=o['perm_v4'],
        M0_GT_only=masks['M0'],M4_GT_only=masks['M4'],raw_hw=r['raw_hw'],teacher_pair_evidence='sameimagebasename/nativeHW/singleobject annotation, v61 audited masks and annotation area invariants; M4 subsetM0 previously pixel-audited',
        original_mask_profile=lab.get('mask_profile'),occlusion_decomposition_available=lab.get('occlusion_decomposition_available'),
        occlusion_decomposition_order=lab.get('occlusion_decomposition_order'),semantic_human_D_E_S='UNKNOWN',
        physical_corner_on_mesh='NOT_CERTIFIED',self_geometry_proxy='SEPARATE,notM0/M4label',new_teacher_labels_generated=False))
    if (i+1)%250==0:print(json.dumps(dict(stage='existing1003_path_and_bytehash',frames=i+1,seconds=time.monotonic()-START)),flush=True)
assert counts['train_frames']==768 and counts['validation_frames']==235
roi=V60/'SYNTH_ROI_CACHE.npz';headers=npzheaders(roi)
assert headers['x']['shape']==[1003,8,65,9,9]and headers['x']['dtype']=='float16'and headers['pixel_valid']['shape']==[1003,8]
assert sha(roi)==load(V60/'SYNTH_FEATURES_COMPLETE.json')['cache_sha256']==load(V61/'LABEL_PREPARATION.json')['ROI_cache_sha256']
inputpath=HERE/'UNCHANGED_MODEL_INPUT_LINKS.jsonl';teacherpath=HERE/'REFERENCE_AND_M0_M4_TEACHER_LINKS_GT_ONLY.jsonl'
jsonlines(inputpath,input_rows);jsonlines(teacherpath,teacher_rows)

# Pretrain TRAIN byte hashes are raw RGB, never compared against padded PNG
# hashes as if those were the same representation. ZIP source members streamed.
pretrain=[]
for j,r in enumerate(selection):
    m=rowmap[r['id']];assert m['partition']==r['partition']and m['cache_row']==r['cache_row']
    a=m['annotation_locator_GT_only'];a=a if '::'in a or Path(a).is_absolute()else str(REPO/a)
    rawsha=locator_sha(m['raw_rgb_locator'])if r['partition']=='train'else None
    pretrain.append(dict(id=r['id'],source=r['source'],partition=r['partition'],cache_row=r['cache_row'],raw_RGB_locator=m['raw_rgb_locator'],raw_RGB_sha256=rawsha,
        prepared_RGB_locator=m['prepared_image'],prepared_RGB_sha256=m['prepared_image_sha256'],annotation_locator_GT_only=a,annotation_sha256=m['annotation_sha256'],scenario_id=m['scenario_id']))
    if (j+1)%1024==0:print(json.dumps(dict(stage='pretrain_selection_metadata_and_TRAIN_rawbyteSHA',rows=j+1,seconds=time.monotonic()-START)),flush=True)
for z in ziphandles.values():z.close()
overlaps=[];overlapstats={}
for oldsplit in ('train','validation','ALL'):
    old=[r for r in teacher_rows if oldsplit=='ALL'or r['split']==oldsplit]
    oldids={r['id']for r in old};oldann={r['annotation_GT_only']['path']for r in old};oldannsha={r['annotation_GT_only']['sha256']for r in old}
    oldinput=[r for r in input_rows if oldsplit=='ALL'or r['split']==oldsplit]
    oldraw={r['raw_RGB']['path']for r in oldinput};oldrawsha={r['raw_RGB']['sha256']for r in oldinput}
    by={}
    for prepart in ('train','heldout'):
        selected=[r for r in pretrain if r['partition']==prepart]
        methods=dict(id=lambda x:x['id']in oldids,raw_locator=lambda x:x['raw_RGB_locator']in oldraw,annotation_locator=lambda x:x['annotation_locator_GT_only']in oldann,
            annotation_SHA=lambda x:x['annotation_sha256']in oldannsha)
        if prepart=='train':methods['raw_RGB_byte_SHA']=lambda x:x['raw_RGB_sha256']in oldrawsha
        by[prepart]={k:sum(fn(r)for r in selected)for k,fn in methods.items()}
        for r in selected:
            if r['id']in oldids:overlaps.append(dict(old1003_split=oldsplit,V75_partition=prepart,id=r['id'],source=r['source'],cache_row=r['cache_row'],annotation_sha256=r['annotation_sha256']))
    overlapstats[oldsplit]=dict(old_frames=len(old),V75=by)
csvrows(HERE/'OVERLAPPING_IDENTITIES.csv',overlaps)
jsonlines(HERE/'V75_SELECTED_INPUT_LINEAGE_AND_TRAIN_RAW_SHA.jsonl',pretrain)
producers=[V60/'prepare_inputs.py',V60/'extract_features.py',V60/'head_and_roi.py',V61/'build_presence_labels.py',V61/'fit_presence.py']
inputs=[CONTRACT,V60/'PREPARED_INPUTS_SYNTH.json.gz',V60/'SYNTH_FEATURE_METADATA.json',V60/'SYNTH_FEATURES_COMPLETE.json',V60/'PREPARED.json',roi,
    V61/'MASK_CONTRACT_AUDIT.csv',V61/'LABEL_PREPARATION.json',V61/'PRESENCE_LABELS.npz',
    P/'query_visibility_visible_preservation_20261023_v67/source_audit/SYNTH_REFERENCE_FRAME_MANIFEST.jsonl',
    P/'query_visibility_visible_preservation_20261023_v67/source_audit/SYNTH_REFERENCE_ANCHOR_INVENTORY.csv',
    V74/'native_context_cache/SOURCE_ROW_SELECTION_LOCK.json',V74/'source_contract/TYPED_SOURCE_ROW_MAP.jsonl',V75/'PRETRAIN_PROTOCOL.json',V75/'runs/SOURCE_PRETRAIN_V1/MODEL_FREEZE.json']
summary=dict(completed=True,PASS=True,contract=bind(CONTRACT),code=bind(Path(__file__)),available_counts=dict(counts),missing=[],
    array_headers_inventory_only={'SYNTH_ROI_CACHE':headers,'V61_presence_labels':npzheaders(V61/'PRESENCE_LABELS.npz')},
    original_split=dict(TRAIN_frames=768,VAL_frames=235,TRAIN_slots8=6144,VAL_slots8=1880,V61_M4_valid_TRAIN=5673,V61_M4_valid_VAL=1742),
    model_input_schema='RawRGB + unchanged savedquery9 (first8ROI,9thcentroid) + frozenP3ROI/valid +inputL/affine only. No r/mask/GTpose/GTperm/state forwarded.',
    teacher_schema='GT-onlydynamicprojected_cuboid8 + originalstoredperm annotation +M0targetisolated/M4visible target >127; samecamera/nativeHW; semanticgoldUNKNOWN. No teacher labels generated ininventory.',
    producer_notes=['Initialq9 provenance SEALED_ACTUAL_N3_SUBPIX_F; rawcoordinatesalreadyunpadded; no perframeGTpose/box/visibility/maskinputs declared upstream.',
        'v60 native_grid takes first8 savedq; L=.25maxq8spanclipped[16,maxHW],9x9centerednativegrid; P3stride8bilinear sampling align_cornersFalse,FP16ROI+validchannel.',
        'Mask rawvalues0,1,2,254,255: only>127 target; mask0 at virtualr is not automatic semanticE.',
        'Storedr is camera_dynamic_0123_v4; use stored8 directly; no nearestGT/repermutation selection.',
        'Inventory does not recalculate q/L/RGB/P3, decode masks or produce V76targetlabels.'],
    overlap=overlapstats,pretrain_TRAIN_raw_RGB_SHA_computed=sum(x['raw_RGB_sha256']is not None for x in pretrain),
    independence_limitations=['Only exact IDs/locators/annotation byteSHA and TRAINrawRGBbyteSHA checked; no proof of distinct assets/backgrounds or allrenderfamily independence.',
        'V75sourceheldout was used for its750stepselector. Any overlapping existing1003VAL is reusedselectiondata, notnewuntouchedvalidation evenifnoV75TRAINoverlap.',
        'Existing1003 had repeatedheadvalidation and upstreamdetectorvalidation exposure; original768/235split is retained, not declarednewindependenttest.'],
    bindings=[bind(x)for x in inputs],producer_code=[bind(x)for x in producers],
    outputs=[bind(inputpath),bind(teacherpath),bind(HERE/'OVERLAPPING_IDENTITIES.csv'),bind(HERE/'V75_SELECTED_INPUT_LINEAGE_AND_TRAIN_RAW_SHA.jsonl')],
    RGB_or_mask_pixel_decodes=0,newteacherlabels=0,new_model_forward=0,optimizer_updates=0,new_backbone=0,current_reads=0,PnP=0,seconds=time.monotonic()-START)
write(HERE/'SYNTHETIC_SOURCE_INVENTORY.json',summary)
print(json.dumps(dict(inventory=bind(HERE/'SYNTHETIC_SOURCE_INVENTORY.json'),counts=dict(counts),overlap=overlapstats,seconds=time.monotonic()-START)),flush=True)

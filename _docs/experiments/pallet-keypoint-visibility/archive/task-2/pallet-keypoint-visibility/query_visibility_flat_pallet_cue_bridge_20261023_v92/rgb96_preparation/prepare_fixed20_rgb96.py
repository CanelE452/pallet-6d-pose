"""RGB-only review of immutable LOCO20 saved native queries, with no model/GT.

generated/freeze are pixel-free. run needs a separately root-approved entry.
Canonical decoded-RGB NPYs are the exact arrays used by the old initializer;
original JPEG bytes/SHAs are checked without introducing a different decoder.
"""
from pathlib import Path
from collections import Counter
import argparse
import ast
import hashlib
import json
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
P=ROOT.parent
POLICY=ROOT/'FIXED20_SOURCE_QUERY_RGB_POLICY_BEFORE_PIXELS.json'
POLICY_SHA='d94f9d4281c6b0e4c817920fbad8097e4a70c390a763e6c32b72c4b330126a75'
ROI=P.parent/'external_ray_proxy_head_20261022_v60/head_and_roi.py'
CROP=P/'query_visibility_loco_native_context_20261023_v74/native_context_cache/compile_input_pack.py'
GENERATED=HERE/'GENERATED_RGB96_PRODUCER_TESTS.json'
PROTOCOL=ROOT/'RGB96_PRODUCER_PROTOCOL_BEFORE_PIXELS.json'
INPUT_KEYS={'id','image','image_sha256','RGB_pixels_sha256','raw_hw','source_subset','officialpath_lineage','decoded_RGB_input'}
PROPOSAL_KEYS={'input_id','image_sha256','candidates','selected_index','no_detection','input_shape','canvas_shape','added_border','forward_hooks'}
CANDIDATE_KEYS={'candidate_index','score','box_xyxy','keypoints_xy','keypoints_conf'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def bind(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))


def check(b):assert sha(b['path'])==b['sha256'],b['path']


def read(path):return json.loads(Path(path).read_text())


def lines(path):return[json.loads(s)for s in Path(path).read_text().splitlines()if s.strip()]


def write(path,d):
    with Path(path).open('x')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def pure(path,name):
    nodes=[n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name]
    assert len(nodes)==1;env={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env);return env[name]


def helpers():return pure(ROI,'native_grid'),pure(CROP,'crop')


def validate_selected(inputrow,p):
    """Original first argmax only. No quality threshold, GT target choice, or clip."""
    assert set(inputrow)==INPUT_KEYS and set(p)==PROPOSAL_KEYS
    assert p['input_id']==inputrow['id']and p['image_sha256']==inputrow['image_sha256']
    candidates=p['candidates'];assert all(set(c)==CANDIDATE_KEYS for c in candidates)
    if not candidates:
        assert p['selected_index']is None and p['no_detection']is True
        return None
    assert p['no_detection']is False
    scores=np.asarray([c['score']for c in candidates],np.float64);assert np.isfinite(scores).all()
    selection=int(np.argmax(scores));assert selection==p['selected_index']
    c=candidates[selection];q9=np.asarray(c['keypoints_xy'],np.float64);box=np.asarray(c['box_xyxy'],np.float64)
    assert q9.shape==(9,2)and np.isfinite(q9).all()and box.shape==(4,)and np.isfinite(box).all()
    assert np.asarray(c['keypoints_conf']).shape==(9,)
    return dict(candidate_index=c['candidate_index'],selected_index=selection,score=float(c['score']),
                predictor_box_xyxy=box,query9_xy=q9,query8_xy=q9[:8].copy())


def metadata():
    assert sha(POLICY)==POLICY_SHA;p=read(POLICY)
    for b in p['bindings'].values():check(b)
    inputs=lines(p['bindings']['saved_input_metadata']['path']);proposals=lines(p['bindings']['saved_proposals']['path'])
    result=read(p['bindings']['saved_initializer_result']['path'])
    assert result['completed']and result['frames']==20 and result['no_detection_frames']==2 and result['checkpoint_unchanged']and result['P3_first_repeat_bit_exact']
    assert any(b['sha256']==p['bindings']['saved_proposals']['sha256']for b in result['outputs'])
    assert len(inputs)==len(proposals)==20 and len({r['id']for r in inputs})==20
    native,_=helpers();frames=[]
    for i,(r,pr)in enumerate(zip(inputs,proposals)):
        selected=validate_selected(r,pr);subset=int(r['source_subset']);assert subset in(1,2,3,4,5)
        split='VALIDATION'if subset in(1,4)else'TRAIN'
        if selected is None:q=None;L=None;box=None;selection=None;score=None
        else:
            q=selected['query8_xy'];_,_,L=native(q,r['raw_hw']);box=selected['predictor_box_xyxy'];selection=selected['selected_index'];score=selected['score']
        frames.append(dict(frame_index=i,input_id=r['id'],image=r['image'],image_sha256=r['image_sha256'],
            RGB_pixels_sha256=r['RGB_pixels_sha256'],decoded_RGB_input=r['decoded_RGB_input'],raw_hw=r['raw_hw'],source_subset=subset,split=split,
            officialpath_lineage=r['officialpath_lineage'],candidate_count=len(pr['candidates']),selected_index=selection,selected_score=score,
            selected_predictor_box_xyxy=None if box is None else box.tolist(),query8_xy=None if q is None else q.tolist(),L=L,
            old_initializer_group_state='UNASSIGNED_GROUP_HOLD',native8_semantic_mapping='UNKNOWN',GT_label_created=False))
    assert Counter(f['split']for f in frames)=={'TRAIN':12,'VALIDATION':8}
    assert sum(f['selected_index']is None for f in frames)==2
    return p,frames


def make_frame_arrays(rgb,frame,crop):
    """Pure RGB/query conversion; missing and OOB rows remain factual U."""
    assert rgb.dtype==np.uint8 and list(rgb.shape[:2])==frame['raw_hw']and rgb.shape[-1]==3
    out=dict(rgb96=np.zeros((8,3,96,96),np.uint8),crop_valid96=np.zeros((8,1,96,96),bool),
        query_xy=np.full((8,2),np.nan,np.float64),L=np.full(8,np.nan,np.float64),
        query_exists=np.zeros(8,bool),query_continuous_valid=np.zeros(8,bool),query_nearest_valid=np.zeros(8,bool),query_center_valid=np.zeros(8,bool),
        crop_anyvalid=np.zeros(8,bool),native_slot=np.arange(8,dtype=np.int8))
    if frame['selected_index']is None:return out
    q=np.asarray(frame['query8_xy'],np.float64);L=float(frame['L']);h,w=frame['raw_hw']
    assert q.shape==(8,2)and np.isfinite(q).all()and np.isfinite(L)and L>0
    out['query_xy']=q.copy();out['L'][:]=L;out['query_exists'][:]=True
    pixel=np.floor(q+.5)
    out['query_continuous_valid']=(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)
    out['query_nearest_valid']=(pixel[:,0]>=0)&(pixel[:,0]<w)&(pixel[:,1]>=0)&(pixel[:,1]<h)
    out['query_center_valid']=out['query_continuous_valid']&out['query_nearest_valid']
    for k in range(8):out['rgb96'][k],out['crop_valid96'][k]=crop(rgb,q[k],L,96)
    out['crop_anyvalid']=out['crop_valid96'].reshape(8,-1).any(1)
    return out


def scalar_crop_pixel(rgb,q,L,iy,ix):
    x=q[0]+((ix+.5)/96-.5)*L;y=q[1]+((iy+.5)/96-.5)*L;h,w=rgb.shape[:2]
    valid=-.5<=x<w-.5 and-.5<=y<h-.5
    acc=np.zeros(3,np.float64);x0=int(np.floor(x));y0=int(np.floor(y));dx=x-x0;dy=y-y0
    for yy,xx,weight in[(y0,x0,(1-dx)*(1-dy)),(y0,x0+1,dx*(1-dy)),(y0+1,x0,(1-dx)*dy),(y0+1,x0+1,dx*dy)]:
        if 0<=xx<w and 0<=yy<h:acc+=rgb[yy,xx].astype(np.float64)*weight
    return np.floor(np.clip(acc,0,255)+.5).astype(np.uint8)if valid else np.zeros(3,np.uint8),valid


def generated():
    native,crop=helpers();rng=np.random.default_rng(20261023);checks=0
    for hw in[(31,47),(480,640)]:
        rgb=rng.integers(0,256,(*hw,3),np.uint8)
        q=np.array([[0,0],[hw[1]-1,hw[0]-1],[-3,12],[hw[1]+2,12],[10.25,12.75],[hw[1]-.5,12],[4,8],[8,4]],np.float64)
        _,_,L=native(q,hw);frame=dict(selected_index=0,query8_xy=q.tolist(),L=L,raw_hw=list(hw))
        arr=make_frame_arrays(rgb,frame,crop);assert np.array_equal(arr['query_xy'],q)and np.all(arr['L']==L)
        for k in range(8):
            for iy,ix in[(0,0),(95,95),(47,47),(48,48),(13,17),(23,37)]:
                pv,v=scalar_crop_pixel(rgb,q[k],L,iy,ix);assert np.array_equal(pv,arr['rgb96'][k,:,iy,ix])and v==arr['crop_valid96'][k,0,iy,ix];checks+=1
        missing=make_frame_arrays(rgb,dict(selected_index=None,raw_hw=list(hw)),crop)
        assert np.isnan(missing['query_xy']).all()and np.isnan(missing['L']).all()and not missing['rgb96'].any()and not missing['crop_valid96'].any()and not missing['query_exists'].any()
        assert not arr['query_center_valid'][2:4].any()and not arr['query_center_valid'][5]
    def candidate(score,index):return dict(candidate_index=index,score=score,box_xyxy=[-1,2,30,40],keypoints_xy=np.arange(18).reshape(9,2).tolist(),keypoints_conf=[.7]*9)
    r=dict(id='GENERATED',image='NEVER_OPEN.jpg',image_sha256='a'*64,RGB_pixels_sha256='b'*64,raw_hw=[480,640],source_subset=1,officialpath_lineage='generated',decoded_RGB_input=dict(path='NEVER_OPEN.npy',sha256='c'*64))
    p=dict(input_id='GENERATED',image_sha256='a'*64,candidates=[candidate(.2,0),candidate(.2,1)],selected_index=0,no_detection=False,input_shape=[544,640],canvas_shape=[680,840],added_border=100,forward_hooks=1)
    s=validate_selected(r,p);assert s['selected_index']==0 and np.array_equal(s['query8_xy'],np.arange(18).reshape(9,2)[:8])and s['predictor_box_xyxy'][0]==-1
    bad=dict(p,selected_index=1)
    try:validate_selected(r,bad)
    except AssertionError:pass
    else:raise AssertionError('GT/alternate candidate substitution accepted')
    none=dict(p,candidates=[],selected_index=None,no_detection=True);assert validate_selected(r,none)is None
    write(GENERATED,dict(completed=True,PASS=True,code=bind(__file__),crop_code=bind(CROP),native_grid_code=bind(ROI),
        exact_crop_AST_reused=True,independent_scalar_RGB_pixels=checks,generated_edges_ramp_masks_rounding=True,
        selected_first_argmax_tie_first=True,all_native9_slots_order_preserved=True,native8_centroid_excluded=True,
        q_box_negative_coordinates_not_clipped=True,missing8_U_placeholder_no_query_or_L_invented=True,
        OOB_continuous_NN_factual_flags_separate=True,no_GT_or_semanticlabel_rule=True,
        source_RGB_NPY_JPEG_P3_weights_current_or_GT_reads=0,new_model_forward_training_PnP=0))
    print(json.dumps(bind(GENERATED)),flush=True)


def freeze():
    policy,frames=metadata();proof=read(GENERATED);assert proof['PASS']and proof['code']==bind(__file__)
    minimal=HERE/'FIXED20_SELECTED_QUERY_INPUTS_ONLY.jsonl'
    with minimal.open('x')as f:
        for r in frames:f.write(json.dumps(r,allow_nan=False,separators=(',',':'))+'\n')
    p=dict(schema='V92_FIXED20_RGB96_ONLY_PRODUCER_V1',frozen_before_source_pixels=True,policy=bind(POLICY),code=bind(__file__),generated=bind(GENERATED),
        source_metadata_bindings=policy['bindings'],crop_code=bind(CROP),native_grid_code=bind(ROI),input_manifest=bind(minimal),
        frames=20,slots=160,selected_frames=18,predicted_query_slots=144,no_detection_frames=2,no_detection_slots=16,
        split_counts={'TRAIN':12,'VALIDATION':8},selection='All original20 order; unchanged firstargmax/selectedq9; use0..7 only, no score filter or GT choice.',
        RGB='Read existing canonical V74 decoded_RGB_input NPY only after entry, verify NPYbytes/nativeHW/pixelSHA and originalJPEGbytesSHA. No new JPEG decoder, resize, border subtraction or rawRGB copy.',
        crop='Exact V74 crop AST RGB96 bilinear nativepixel centres q+((j+.5)/96-.5)*L; constant0/floor(clip+.5); valid[-.5,W-.5). q at96 index47.5.',
        L='Exact V60 native_grid initialq8 span clip(.25*maxspan,16,maxHW); metadata-only; no q recenter/scale enlargement.',
        missing='Missing detection16slots use NaNq/L pluszero RGB/valid placeholders and typedinputU; OOBq remains unchanged, flags separate. No physical D/E/S labels.',
        forward_or_label_contract='NONE: only inputpixelreview producer; no P3/head/backbone/GTbbox/cue labels/pose/depth/masks/current data.',
        runtime=dict(executable='/home/minjae/anaconda3/bin/python',numpy='2.3.5',Pillow='12.0.0'),
        output_directory=str(ROOT/'rgb96_review'),output_budget_bytes=100000000,
        render='All20 full canonicalRGB with selected predictorbox/q8 overlay plus8 actualRGB96 andvalidity thumbnails; noGT overlays; native slots unlabeledsemantic.',
        native8_semantic_mapping='UNKNOWN',model_training_admission=False,current_allowed=False)
    write(PROTOCOL,p)
    write(ROOT/'RGB96_ROOT_ENTRYLOCK_TEMPLATE.json',dict(schema='V92_ROOT_APPROVED_RGB96_REVIEW_ENTRY_V1',approved=False,policy=bind(POLICY),
        producer_protocol=bind(PROTOCOL),code=bind(__file__),generated=bind(GENERATED),
        independent_review=dict(path='ROOT_BIND_REVIEW',sha256='0'*64),source_only=True,model_forward_allowed=False,training_allowed=False,current_allowed=False,PnP_allowed=False))
    print(json.dumps(dict(protocol=bind(PROTOCOL),manifest=bind(minimal)),indent=2),flush=True)


def authorize(path):
    e=read(path);assert e['schema']=='V92_ROOT_APPROVED_RGB96_REVIEW_ENTRY_V1'and e['approved']is True
    assert e['source_only']and not any(e[k]for k in('model_forward_allowed','training_allowed','current_allowed','PnP_allowed'))
    for k in('policy','producer_protocol','code','generated','independent_review'):check(e[k])
    assert e['policy']==bind(POLICY)and sha(POLICY)==POLICY_SHA and e['producer_protocol']==bind(PROTOCOL)and e['code']==bind(__file__)
    assert read(e['independent_review']['path'])['PASS']and read(e['generated']['path'])['PASS']
    p=read(PROTOCOL)
    for k in('crop_code','native_grid_code','input_manifest'):check(p[k])
    _,frames=metadata();assert frames==lines(p['input_manifest']['path'])
    return e,p,frames


def render(rgb,frame,arr,path):
    from PIL import Image,ImageDraw,ImageFont
    font=ImageFont.load_default(size=15);small=ImageFont.load_default(size=12)
    full=Image.fromarray(rgb.copy());draw=ImageDraw.Draw(full)
    if frame['selected_index']is not None:
        box=frame['selected_predictor_box_xyxy'];draw.rectangle(box,outline='cyan',width=3)
        for k,(x,y)in enumerate(frame['query8_xy']):
            draw.line((x-6,y,x+6,y),fill='yellow',width=3);draw.line((x,y-6,x,y+6),fill='yellow',width=3);draw.text((x+7,y+3),str(k),fill='yellow',font=font)
    full.thumbnail((1280,650));canvas=Image.new('RGB',(1280,650+70+2*360),'#14212b');d=ImageDraw.Draw(canvas)
    d.text((12,8),f"{frame['frame_index']:02d} {frame['input_id']} / {frame['split']} / selected={frame['selected_index']} / candidates={frame['candidate_count']} / L={frame['L']}",fill='white',font=font)
    d.text((12,30),'Input-only: cyan predictor box; yellow saved native slot tokens. Native8 semantic mapping UNKNOWN; no GT bbox/cue labels.',fill='white',font=small)
    canvas.paste(full,(0,60))
    for k in range(8):
        x=(k%4)*320;y=720+(k//4)*350
        cropimg=Image.fromarray(arr['rgb96'][k].transpose(1,2,0)).resize((256,256),Image.Resampling.NEAREST)
        canvas.paste(cropimg,(x,y+35))
        mask=Image.fromarray(arr['crop_valid96'][k,0].astype(np.uint8)*255).convert('RGB').resize((64,64),Image.Resampling.NEAREST)
        canvas.paste(mask,(x+256,y+35))
        state='NO_QUERY U'if not arr['query_exists'][k]else('qvalid'if arr['query_center_valid'][k]else'INPUT_INVALID U')
        d.text((x+6,y+7),f"slot{k} {state} / valid96={arr['crop_valid96'][k].mean():.3f}",fill='white',font=small)
        if arr['query_exists'][k]:d.text((x+6,y+300),f"q={arr['query_xy'][k].tolist()} unchanged",fill='white',font=small)
        else:d.text((x+6,y+300),'Zero placeholder; no q/L or label invented',fill='white',font=small)
    canvas.save(path)


def run(entry_path):
    e,p,frames=authorize(entry_path)
    from PIL import Image
    import PIL
    assert Path(sys.executable).resolve()==Path(p['runtime']['executable']).resolve()and np.__version__==p['runtime']['numpy']and PIL.__version__==p['runtime']['Pillow']
    _,crop=helpers();out=Path(p['output_directory']);out.mkdir(exist_ok=False);(out/'figures').mkdir();start=time.monotonic()
    arrays=[];records=[];pixels=[]
    for frame in frames:
        # First source pixel accesses occur only after authorize checks above.
        assert sha(frame['image'])==frame['image_sha256'];check(frame['decoded_RGB_input'])
        rgb=np.load(frame['decoded_RGB_input']['path'],allow_pickle=False)
        assert rgb.dtype==np.uint8 and rgb.shape==tuple(frame['raw_hw'])+(3,)and hashlib.sha256(rgb.tobytes()).hexdigest()==frame['RGB_pixels_sha256']
        arr=make_frame_arrays(rgb,frame,crop);arrays.append(arr)
        fig=out/'figures'/f"frame_{frame['frame_index']:02d}.png";render(rgb,frame,arr,fig)
        pixels.append(dict(frame_index=frame['frame_index'],input_id=frame['input_id'],source_RGB=frame['decoded_RGB_input'],
            original_JPEG_SHA256=frame['image_sha256'],decoded_pixel_SHA256=frame['RGB_pixels_sha256'],figure=bind(fig),
            selected_index=frame['selected_index'],q_L_unchanged=True))
        for k in range(8):
            missing=not bool(arr['query_exists'][k]);valid=bool(arr['query_center_valid'][k])
            records.append(dict(frame_index=frame['frame_index'],input_id=frame['input_id'],native_slot=k,split=frame['split'],image_sha256=frame['image_sha256'],
                query_xy=None if missing else arr['query_xy'][k].tolist(),L=None if missing else float(arr['L'][k]),query_exists=not missing,
                query_center_valid=valid,continuous_valid=bool(arr['query_continuous_valid'][k]),nearest_valid=bool(arr['query_nearest_valid'][k]),
                crop_anyvalid=bool(arr['crop_anyvalid'][k]),crop_valid_fraction=float(arr['crop_valid96'][k].mean()),
                input_status='NO_DETECTION_QUERY_U'if missing else'SAVED_UNLABELED_QUERY'if valid else'OUTSIDE_OR_INVALID_QUERY_U',
                semantic_state='UNLABELED_UNKNOWN',native8_semantic_mapping='UNKNOWN',new_training_label=False))
        assert sum(f.stat().st_size for f in out.rglob('*')if f.is_file())<p['output_budget_bytes']
    packed={k:np.concatenate([r[k]for r in arrays])for k in arrays[0]}
    packed.update(frame_index=np.repeat(np.arange(20,dtype=np.int32),8),input_id=np.repeat([f['input_id']for f in frames],8),split=np.repeat([f['split']for f in frames],8))
    assert packed['rgb96'].shape==(160,3,96,96)and packed['query_exists'].sum()==144 and packed['query_center_valid'].sum()==122
    np.savez_compressed(out/'INPUT_RGB96_ONLY.npz',**packed)
    line=out/'ROWS_INPUT_ONLY.jsonl';line.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n'for r in records))
    write(out/'RGB_PROVENANCE_AND_FIGURES.json',dict(completed=True,frames=pixels))
    contact=Image.new('RGB',(1280,5*360),'#14212b')
    for i,r in enumerate(pixels):
        im=Image.open(r['figure']['path']);im.thumbnail((320,360));contact.paste(im,((i%4)*320,(i//4)*360))
    contact.save(out/'ALL20_INPUT_CONTACT.png')
    total=sum(f.stat().st_size for f in out.rglob('*')if f.is_file());assert total<p['output_budget_bytes']
    write(out/'RGB96_REVIEW_SEAL.json',dict(completed=True,PASS=True,entry=bind(entry_path),policy=bind(POLICY),protocol=bind(PROTOCOL),code=bind(__file__),
        frames=20,slots=160,selected_frames=18,no_detection_frames=2,query_slots=144,no_detection_U_slots=16,factual_query_valid=122,
        image_order_query_L_selected_index_and_split_unchanged=True,old_groupHOLD_preserved=True,native8_mapping='UNKNOWN',GT_or_new_semantic_labels_read_or_created=0,
        source_canonical_RGB_NPY_pixel_reads=20,new_JPEG_decodes=0,new_P3_backbone_head_training_current_PnP=0,
        files=[bind(f)for f in sorted(out.rglob('*'))if f.is_file()],bytes_before_seal=total,seconds=time.monotonic()-start))
    print(json.dumps(bind(out/'RGB96_REVIEW_SEAL.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['generated','freeze','run']);p.add_argument('--entry',type=Path);a=p.parse_args()
    if a.stage=='generated':generated()
    elif a.stage=='freeze':freeze()
    else:
        if a.entry is None:p.error('--entry required')
        run(a.entry)

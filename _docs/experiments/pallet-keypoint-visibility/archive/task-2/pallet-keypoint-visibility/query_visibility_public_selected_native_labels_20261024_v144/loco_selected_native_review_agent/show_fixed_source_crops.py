"""Source-only original-q/L96 crops. No model, GT reference or crop correction."""
from pathlib import Path
import ast,hashlib,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V=P/'query_visibility_public_selected_native_labels_20261024_v144';H=Path(__file__).resolve().parent
V141=P/'query_visibility_public_native_initializer_association_20261024_v141'
POLICY=V/'ROOT_SOURCE_NATIVE_ALIGNMENT_SCOPE.json'
CROP=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
def font(n):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)
assert sha(POLICY)=='d2d356ca7325aff8918c0839b6ca3657bb4dd8c4a1adf96d119a52661af161ca'
assert sha(CROP)=='6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b'
node=next(n for n in ast.parse(CROP.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='crop')
ns=dict(np=np);exec(compile(ast.Module(body=[node],type_ignores=[]),str(CROP)+'::crop_only','exec'),ns);crop=ns['crop']
seal=read(V141/'actual_public_initializer_canonical_transport_v2/PREDICTION_SEAL.json');assert seal['completed']and seal['PASS']
assert sha(V141/'actual_public_initializer_canonical_transport_v2/PREDICTION_SEAL.json')=='b8e26a9b76d6e1c9084902cbd63e692ab4a7606a97033a93088d7d1d021d2dce'
outputs={Path(b['path']).name:b for b in seal['outputs']}
props=[json.loads(l)for l in check(outputs['FROZEN_PROPOSALS_ALL16_INPUT_ONLY.jsonl']).read_text().splitlines()]
with np.load(check(outputs['ORIGINAL_NATIVE_QUERY128_AND_VALID9X9_INPUT_ONLY.npz']),allow_pickle=False)as f:a={k:f[k].copy()for k in f.files}
result=read(check(outputs['RESULT.json']));transport=read(check(result['canonical_transport_seal']));tr=read(check(transport['manifest']))['rows']
assocseal=read(V141/'saved_public_association_v1/ASSOCIATION_COMPLETE.json');assert assocseal['completed']and assocseal['PASS']
facts=read(check(assocseal['results']))['frame_diagnostics']
out=H/'fixed_source96_contact_v1';assert not out.exists();out.mkdir()
rows=[];sheets=[];totalanns=0
for i in range(12):
    r=props[i];t=tr[i];ff=facts[i];assert r['frame_index']==i and r['source']=='LOCO'and r['id']==t['id']==ff['id']
    rgb=np.load(check(t['canonical_RGB_NPY']),allow_pickle=False);assert rgb.dtype==np.uint8 and list(rgb.shape)==r['raw_hw']+[3]and hashlib.sha256(rgb.tobytes()).hexdigest()==r['decoded_RGB_sha256']
    raw=Image.fromarray(rgb);overlay=raw.copy();d=ImageDraw.Draw(overlay);hh,ww=rgb.shape[:2]
    for ann in ff['official_instance_diagnostics']:d.rectangle(ann['official_bbox_xyxy'],outline=(60,160,80),width=max(1,ww//1000))
    totalanns+=len(ff['official_instance_diagnostics'])
    if ff['selected_bbox_xyxy']is not None:d.rectangle(ff['selected_bbox_xyxy'],outline=(255,145,0),width=max(3,ww//350))
    crs=[];vs=[]
    for k in range(8):
        j=i*8+k;q=a['query_xy'][j];L=a['L'][j];constructed=bool(a['finite_non_sentinel'][j]and np.isfinite(L)and L>0)
        if constructed:cr,va=crop(rgb,q,L,96);cr=cr.transpose(1,2,0);va=va[0]
        else:cr=np.zeros((96,96,3),np.uint8);va=np.zeros((96,96),bool)
        crs.append(cr);vs.append(va)
        if np.isfinite(q).all()and 0<=q[0]<ww and 0<=q[1]<hh:
            x,y=q;rad=max(3,ww//250);d.ellipse((x-rad,y-rad,x+rad,y+rad),outline=(255,0,210),width=max(2,ww//650));d.text((min(max(0,x+rad),ww-70),min(max(0,y),hh-30)),str(k),font=font(max(20,ww//65)),fill=(255,0,210),stroke_width=1,stroke_fill='black')
        rows.append(dict(global_index=j,frame_index=i,id=r['id'],source_id=r['source_id'],native_slot=k,query_xy=[float(v)if np.isfinite(v)else None for v in q],
            L=float(L)if np.isfinite(L)else None,no_detection=bool(a['no_detection'][j]),original_input_valid=bool(a['continuous_valid'][j]),crop_constructed=constructed,
            rgb96_SHA256=hashlib.sha256(cr.tobytes()).hexdigest(),valid96_SHA256=hashlib.sha256(va.tobytes()).hexdigest(),crop_valid_fraction=float(va.mean())))
    packet=out/f'FRAME_{i:02d}_FIXED96.npz';np.savez_compressed(packet,rgb96=np.stack(crs),valid96=np.stack(vs),global_index=np.arange(i*8,i*8+8,dtype=np.int32))
    width=960;height=round(hh*width/ww);canvas=Image.new('RGB',(1920,110+height+285),'white');cd=ImageDraw.Draw(canvas)
    cd.text((12,8),r['id']+f' | original firstargmax={r["selected_index"]} | candidate count={len(r["candidates"])}',font=font(24),fill='black')
    cd.text((12,42),'Public source only. Left raw; right input selectedbox/original q8 and ALL official boxes. Native visibility is not mask presence.',font=font(20),fill='black')
    cd.text((12,72),'Below: untouched factual96 crops at exact original q/L, display NEAREST x2. Noquery=zero placeholder, not E.',font=font(20),fill='black')
    canvas.paste(raw.resize((width,height)),(0,110));canvas.paste(overlay.resize((width,height)),(960,110))
    for k in range(8):
        x=k*240;top=120+height;rr=rows[i*8+k];cd.text((x+5,top),f'native{k} | valid={rr["original_input_valid"]}',font=font(18),fill='black')
        cd.text((x+5,top+24),'L='+str(None if rr['L']is None else round(rr['L'],2)),font=font(17),fill='black')
        canvas.paste(Image.fromarray(crs[k]).resize((192,192),Image.Resampling.NEAREST),(x+24,top+52))
    fig=out/f'FRAME_{i:02d}_WHOLESCENE_AND_FIXED96.png';canvas.save(fig);sheets.append(dict(frame_index=i,id=r['id'],sheet=bind(fig),crop_packet=bind(packet)))
assert len(rows)==96 and totalanns==424
with(out/'CROP_FACTS96.json').open('x')as f:json.dump(dict(completed=True,rows=rows,sheets=sheets,all_official_instances=424,policy=bind(POLICY),crop_operator=bind(CROP),initializer_seal=bind(V141/'actual_public_initializer_canonical_transport_v2/PREDICTION_SEAL.json'),
    originalqL_firstargmax_unchanged=True,GTcrop_recenter=False,model_encoder_initializer_fit_private_V112_PnP=0),f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(dict(completed=True,sheets=len(sheets),slots=len(rows),output=str(out))))

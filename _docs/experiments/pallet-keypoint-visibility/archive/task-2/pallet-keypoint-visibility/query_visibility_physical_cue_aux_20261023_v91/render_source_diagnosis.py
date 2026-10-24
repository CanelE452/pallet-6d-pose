"""Saved-output/source-RGB diagnostic only; never constructs or runs a model."""
from pathlib import Path
import hashlib
import json
import html
import numpy as np
from PIL import Image, ImageDraw, ImageFont

R=Path(__file__).resolve().parent
P=R.parent
RUN=R/'runs/PHYSICAL_CUE_AUXILIARY_V1'
OUT=R/'source_visual_diagnosis_v1'

def load(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines()]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(s): return hashlib.sha256(s.encode()).hexdigest()
def bind(p): return dict(path=str(Path(p).resolve()),sha256=sha(p),bytes=Path(p).stat().st_size)
def write(p,v):
    with p.open('x') as f: json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

assert load(RUN/'FIT_COMPLETE.json')['completed']
policy=load(R/'VISUAL_DIAGNOSIS_POLICY_BEFORE_RESULTS.json')
assert not OUT.exists();OUT.mkdir()
GT=R/'teacher_preparation/corpus_build_v1'
viewrows=rows(GT/'ALL604_CUE_VIEW_TEACHERS_GT_ONLY.jsonl')
frames=rows(P/'query_visibility_public_cue_grid_20261023_v89/inputs_metadata/FRAMES_INPUTS_ONLY.jsonl')
tiles=rows(P/'query_visibility_public_cue_grid_20261023_v89/inputs_metadata/INPUTS_GRID_ONLY.jsonl')
frame_by_id={f['frame_input_id']:f for f in frames}
pairviews={(r['cue_id'],r['view']):r for r in viewrows}
with np.load(RUN/'SOURCE_SCORES.npz',allow_pickle=False) as z: scores={k:z[k] for k in z.files}
assert np.array_equal(scores['cue_view_index'],np.arange(604))
pred=np.where(scores['region_available'],scores['region_logit']>=0,2).astype(int)
rpred=np.where(scores['reference_available'],scores['reference_logit']>=0,2).astype(int)
selected=[]
for truth,guess in [(0,0),(0,1),(1,0),(1,1)]:
    ix=[i for i,r in enumerate(viewrows) if r['split']=='VAL_HALL' and r['teacher_label']==truth and pred[i]==guess]
    if ix:
        i=min(ix,key=lambda i:(key(viewrows[i]['cue_id']),i));selected.append(dict(stratum=f'true{truth}_pred{guess}',index=i))
ix=[i for i,r in enumerate(viewrows) if r['split']=='VAL_HALL' and r['teacher_label'] in (0,1) and pred[i]!=rpred[i] and i not in [x['index'] for x in selected]]
if ix: selected.append(dict(stratum='region_reference_disagreement',index=min(ix,key=lambda i:(key(viewrows[i]['cue_id']),i))))
with np.load(GT/'ALL77916_TILE_CUE_TRAINING_UNITS_GT_ONLY.npz',allow_pickle=False) as z:
    unit_tile=z['tile_index'];unit_view=z['cue_view_index'];full=z['full']
manifest=load(P/'query_visibility_public_cue_inputs_20261023_v90/input_pack/PACK_MANIFEST.json')
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
LABEL={0:'D (cue observed)',1:'E (external)',-1:'U',2:'U'}

def description(row):
    i=row['cue_view_index'];z=float(scores['region_logit'][i]);rz=float(scores['reference_logit'][i])
    return f"{row['view']} GT={LABEL[row['teacher_label']]} | region={LABEL[pred[i]]} z={z:.3f} | r={LABEL[rpred[i]]} z={rz:.3f}"

def annotated_original(row):
    f=frame_by_id[row['frame_input_id']];assert sha(f['image'])==f['image_sha256']
    im=Image.open(f['image']).convert('RGB');d=ImageDraw.Draw(im)
    x,y=row['reference_xy_unchanged'];box=row['uncertainty_bbox_xyxy_half_open_unchanged']
    d.rectangle(box,outline='yellow',width=2);d.line((x-7,y,x+7,y),fill='red',width=2);d.line((x,y-7,x,y+7),fill='red',width=2)
    return im

def saved_crop(ti):
    chunk=next(c for c in manifest['chunks'] if c['start']<=ti<c['end'])
    with np.load(chunk['file']['path'],allow_pickle=False) as z:
        k=ti-chunk['start'];assert int(z['tile_index'][k])==ti
        im=Image.fromarray(z['rgb96'][k].transpose(1,2,0))
    return im

records=[]
for n,sel in enumerate(selected):
    row=viewrows[sel['index']];before=pairviews[(row['cue_id'],'before')];after=pairviews[(row['cue_id'],'after')]
    actual_tiles=[]
    # Fixed policy displays model inputs from the after frame, even when the
    # outcome stratum selected the before view. Both views remain shown above.
    ai=after['cue_view_index']
    fulltiles=unit_tile[(unit_view==ai)&full]
    for L in (128,256):
        candidates=[int(t) for t in fulltiles if tiles[int(t)]['roi_side']==L]
        assert candidates
        actual_tiles.append(min(candidates,key=lambda t:key(tiles[t]['tile_id'])))
    record=dict(**sel,cue_id=row['cue_id'],pair_id=row['pair_id'],selected_view=row['view'],
        selected_cue_view=row,displayed_input_tile_indices=actual_tiles,
        tile_selection='minSHA tile_id among pre-existing FULL after-frame tiles, each L; diagnostic only, no forward rerun')
    canvas=Image.new('RGB',(1280,850),'#fafafa');d=ImageDraw.Draw(canvas)
    d.text((16,10),f"{n+1}. {sel['stratum']} | {row['cue_id']}",fill='black',font=font)
    for j,rr in enumerate((before,after)):
        d.text((j*640+8,42),description(rr),fill='black',font=small)
        canvas.paste(annotated_original(rr),(j*640,70))
    d.text((16,563),'After-view actual saved RGB96 inputs (two of all129 fixed tiles; diagnostic selection only)',fill='black',font=font)
    for j,ti in enumerate(actual_tiles):
        tile=tiles[ti];im=saved_crop(ti).resize((240,240),Image.Resampling.NEAREST)
        cropdraw=ImageDraw.Draw(im);q=np.array(tile['query_xy']);L=tile['roi_side'];r=np.array(after['reference_xy_unchanged'])
        px,py=(96*(r-q)/L+47.5)*2.5
        cropdraw.line((px-7,py,px+7,py),fill='red',width=2);cropdraw.line((px,py-7,px,py+7),fill='red',width=2)
        canvas.paste(im,(16+j*620,596))
        d.text((270+j*620,613),f"L={L}; q={tile['query_xy']}",fill='black',font=font)
        d.text((270+j*620,646),f"tile index={ti}",fill='black',font=font)
        d.text((270+j*620,680),'Yellow: uncertainty on full RGB',fill='black',font=small)
        d.text((270+j*620,708),'Red: original cue r (report overlay)',fill='black',font=small)
        d.text((270+j*620,736),'No GT coordinates entered forward',fill='black',font=small)
    path=OUT/f'case_{n:02d}.png';canvas.save(path)
    record['figure']=bind(path);records.append(record)
write(OUT/'FIXED_CASE_SELECTION_AND_FIGURES.json',dict(completed=True,policy=bind(R/'VISUAL_DIAGNOSIS_POLICY_BEFORE_RESULTS.json'),
    scores=bind(RUN/'SOURCE_SCORES.npz'),selection_before_RGB_reads=True,cases=records,new_model_forward_fit_current_PnP=0))
print(json.dumps(dict(completed=True,cases=len(records),manifest=bind(OUT/'FIXED_CASE_SELECTION_AND_FIGURES.json'))))

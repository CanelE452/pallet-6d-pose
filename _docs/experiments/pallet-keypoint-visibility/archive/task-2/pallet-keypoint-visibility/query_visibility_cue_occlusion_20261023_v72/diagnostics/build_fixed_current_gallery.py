"""Post-seal fixed same six cases. Saved RGB96/attention/scores only; no model."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'diagnostics'
RUN=ROOT/'private_evaluation/CUE_ATTENTION_V3'
OLD=ROOT.parent/'query_visibility_controlled_real_20261022_v66/diagnostics/FIXED_REAL_CROPS_V66.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bind(path):return dict(path=str(path),sha256=sha(path))
def check(r):assert sha(r['path'])==r['sha256'],r['path']
def write(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def font(n,bold=False):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold' if bold else '')+'.ttf',n)


def marker(im,xy,color,r=8,cross=False):
    d=ImageDraw.Draw(im);x,y=xy
    if cross:d.line((x-r,y,x+r,y),fill=color,width=2);d.line((x,y-r,x,y+r),fill=color,width=2)
    else:d.ellipse((x-r,y-r,x+r,y+r),outline=color,width=3)


def main():
    assert not (OUT/'FIXED_CURRENT_CROP_METADATA.json').exists(),'No overwrite'
    seal=json.loads((RUN/'INFERENCE_SEAL.json').read_text());assert seal['completed'] and seal['frames']==223 and seal['slots']==1784
    for b in seal['outputs']:check(b)
    prep=json.loads((RUN/'PREPARATION_SEAL.json').read_text())
    for b in prep['outputs']:check(b)
    cases=json.loads(OLD.read_text())
    actual=list(csv.DictReader((RUN/'REAL_QUERY_SCORES.csv').open()))
    index={(r['id'],int(r['native_corner'])):(i,r) for i,r in enumerate(actual)}
    with np.load(RUN/'CURRENT_MODEL_INPUT.npz',allow_pickle=False) as z:
        rgb96=z['rgb96'];qdata=z['query_xy'];Ldata=z['side_L'];valid=z['crop_valid96']
    with np.load(RUN/'RAW_NETWORK_OUTPUTS.npz',allow_pickle=False) as z:attention=z['cue_attention_weights'];logits=z['logit_D_given_DE'];heat=z['cue_heatmap_logits']
    rawdir=OUT/'fixed_saved_input96';rawdir.mkdir(parents=True,exist_ok=False)
    all_rows=[];figs=[]
    for state,selected in [('D',cases['fixed_semantic_D_cases']),('E',cases['fixed_semantic_E_cases'])]:
        canvas=Image.new('RGB',(1640,155+430*len(selected)),(248,249,252));draw=ImageDraw.Draw(canvas)
        draw.text((30,18),f'V72 FIXED HUMAN {state} | same preselected cases; saved outputs only',font=font(28,True),fill=(15,23,42))
        draw.text((30,60),'Cyan +: initial q   Magenta ring: existing semantic reference   Green ring: predicted attention maximum',font=font(19),fill=(35,45,65))
        draw.text((30,90),'pD is D/E-conditional. Native pixel GT is unavailable. Attention is a diagnostic, not verified cue localization.',font=font(18),fill=(60,65,80))
        xs=[30,565,1100]
        for x,t in zip(xs,['ACTUAL MODEL RGB96: entire input L','SAME RGB96 + saved 9x9 attention','RAW NATIVE96: context only']):draw.text((x,122),t,font=font(17,True),fill=(25,35,50))
        for n,c in enumerate(selected):
            i,r=index[(c['id'],c['native_corner'])];assert r['valid']=='True'
            assert np.array_equal(qdata[i],np.asarray(c['query'],np.float32))
            assert Ldata[i]==np.float32(c['roi_side'])
            assert float(r['query_x'])==c['query'][0] and float(r['query_y'])==c['query'][1]
            check(c['raw_native96'])
            raw=np.ascontiguousarray(rgb96[i].transpose(1,2,0));raw_image=Image.fromarray(raw)
            rawpath=rawdir/(c['id'].replace(':','_')+f"_k{c['native_corner']}_MODEL_RGB96.png");raw_image.save(rawpath)
            q=np.asarray(c['query'],float);reference=np.asarray(c['reference'],float);L=float(c['roi_side'])
            ref96=(reference-q)*96/L+47.5
            peak=np.unravel_index(attention[i].argmax(),(9,9));py,px=map(int,peak)
            peak_native=q+(np.array([px,py])-4)*L/9
            peak96=(np.array([px,py])+.5)*96/9-.5
            scale=3;left=raw_image.resize((288,288),Image.Resampling.NEAREST)
            marker(left,((47.5+.5)*scale-.5,(47.5+.5)*scale-.5),(0,230,255),cross=True)
            marker(left,tuple((ref96+.5)*scale-.5),(255,0,220),r=10)
            at=np.asarray(Image.fromarray(attention[i]).resize((96,96),Image.Resampling.BILINEAR),np.float32)
            strength=at/max(float(at.max()),1e-12)
            overlay=raw.astype(float)*(1-.48*strength[...,None])+np.array([255,60,0])[None,None,:]*(.48*strength[...,None])
            mid=Image.fromarray(np.clip(overlay,0,255).astype(np.uint8)).resize((288,288),Image.Resampling.NEAREST)
            marker(mid,tuple((peak96+.5)*scale-.5),(80,255,20),r=11)
            marker(mid,tuple((ref96+.5)*scale-.5),(255,0,220),r=7)
            context=Image.open(c['raw_native96']['path']).convert('RGB').resize((288,288),Image.Resampling.NEAREST)
            bounds=np.asarray(c['diagnostic96_bounds_xyxy'],float)
            contextq=(q-bounds[:2]+.5)*scale-.5;contextr=(reference-bounds[:2]+.5)*scale-.5
            marker(context,tuple(contextq),(0,230,255),cross=True);marker(context,tuple(contextr),(255,0,220),r=10)
            y=155+n*430
            for x,image in zip(xs,[left,mid,context]):canvas.paste(image,(x,y))
            draw.text((xs[0]+302,y+5),f"{c['id']}\nk{c['native_corner']} human {state}\nq-r={c['error_px']:.3f}px\nL={L:.2f}px",font=font(17),fill=(15,30,45))
            draw.text((xs[0],y+300),f"v61 saved pV={c['v61_p_visible']:.4f}",font=font(20),fill=(45,55,75))
            pd=float(r['p_D_given_DE']);pred=r['predicted_D_or_E']
            draw.text((xs[0],y+330),f"V72 pD={pd:.4f}, pE={1-pd:.4f} -> {pred}",font=font(21,True),fill=(20,80,30) if pred==state else (180,35,35))
            draw.text((xs[1],y+300),f'peak [row,col]=[{py},{px}], mass={attention[i,py,px]:.3f}',font=font(19),fill=(25,55,35))
            draw.text((xs[1],y+330),f'peak-to-ref={np.linalg.norm(peak_native-reference):.2f} native px',font=font(19),fill=(35,45,65))
            draw.text((xs[2],y+300),'96 native pixels around rounded q.',font=font(18),fill=(45,55,75))
            draw.text((xs[2],y+330),'Not resized to the model span L.',font=font(18),fill=(45,55,75))
            all_rows.append(dict(id=c['id'],native_corner=c['native_corner'],human_semantic_state=state,case_selection='Unchanged V65/V66 fixed same cases',query=c['query'],reference=c['reference'],reference_provenance=c['reference_coordinate_origin'],reference_overlay_exposure=c['reference_overlay_exposure'],native8_semantic_association='Existing evaluation only; no new coordinate label',error_px=c['error_px'],roi_side=L,same_nearest_pixel=c['same_nearest_pixel'],v61_saved_p_visible=c['v61_p_visible'],v72_saved_p_D_given_DE=pd,v72_saved_p_E_given_DE=1-pd,v72_prediction=pred,v72_raw_logit=float(logits[i]),valid=True,saved_array_index=i,model_RGB96=bind(rawpath),native96=c['raw_native96'],attention_peak_row_col=[py,px],attention_peak_native_xy=peak_native.tolist(),attention_peak_mass=float(attention[i,py,px]),attention_peak_to_reference_native_px=float(np.linalg.norm(peak_native-reference)),model_full_L_preserved=True,GT_markers_display_only=True,pixel_GT='UNAVAILABLE'))
        fig=OUT/f'FIXED_{state}_MODEL_INPUT_AND_ATTENTION.png';canvas.save(fig);figs.append(bind(fig))
    write(OUT/'FIXED_CURRENT_CROP_METADATA.json',dict(schema='V72_POSTSEAL_FIXED_SAME_REAL_INPUT_ATTENTION_DIAGNOSTIC',figures_ready=True,visual_findings_in_separate_addendum=True,
        selection='Exact ordered V66 fixed4D/2E; no score-based selection.',bindings=[bind(OLD),bind(RUN/'INFERENCE_SEAL.json'),bind(RUN/'PREPARATION_SEAL.json'),bind(RUN/'CURRENT_MODEL_INPUT.npz'),bind(RUN/'RAW_NETWORK_OUTPUTS.npz'),bind(RUN/'REAL_QUERY_SCORES.csv')],code=bind(__file__),figures=figs,cases=all_rows,
        notes=['Model image96 represents entire unchanged native span L; native96 context is a different diagnostic view.','Reference marker is existing semantic evaluation coordinate, not native pixel visible/full GT.','Attention maximum and distance are descriptive; not validated semantic localization or causal failure proof.','No inferred E is accepted as calibrated human external-occlusion probability.','Six exposed development cases are not a holdout performance estimate.'],new_model_forward_training_PnP=0,old_files_modified=False))
    print(json.dumps(dict(figures=figs,metadata=bind(OUT/'FIXED_CURRENT_CROP_METADATA.json'))))


if __name__=='__main__':main()

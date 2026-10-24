"""Render fixed public saved RGB96 only; no raw recrop, P3 or model reader."""
from pathlib import Path
import json,hashlib,math,zipfile,itertools
from collections import defaultdict
import numpy as np
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
V=HERE.parent;P=V.parent;QA=HERE/'PUBLIC_SAVED_CROP_QA'
def j(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def bind(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return dict(path=str(p),sha256=h.hexdigest(),bytes=p.stat().st_size)
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def read_member_row(ref):
    # Read only the requested NPY row; never open the P3 member.
    with zipfile.ZipFile(ref['file']['path']) as z:
        with z.open(ref['key']+'.npy') as f:
            version=np.lib.format.read_magic(f)
            assert version in [(1,0),(2,0)],version
            reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
            shape,fortran,dt=reader(f)
            assert not fortran and dt in [np.dtype('uint8'),np.dtype('bool')]
            n=math.prod(shape[1:])*dt.itemsize;f.seek(ref['index']*n,1)
            raw=f.read(n);assert len(raw)==n
            a=np.frombuffer(raw,dtype=dt).reshape(shape[1:]).copy()
    return a,dict(member=ref['key']+'.npy',index=ref['index'],shape=list(shape),dtype=str(dt),row_pixels_sha256=hashlib.sha256(a.tobytes()).hexdigest())
def font(n):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)
def main():
    assert j(HERE/'REVIEW.json')['PASS']
    assert not QA.exists(),'No overwrite'
    QA.mkdir()
    ledger=rows(V/'ALL960_QUERY_ANCHORED_AUX_INVENTORY_GT_ONLY.jsonl')
    reasons=defaultdict(list)
    for x in ledger:
        i=x['row_index'];old=x['original_C1_V95_label_D0_E1_Uminus1'];new=x['query_anchored_AUX_label_D0_E1_Uminus1']
        if new==1:reasons[i].append('ALL_NEW_E')
        if old==-1 and new!=-1:reasons[i].append('ALL_OLD_U_TO_NEW_KNOWN')
    for source,split in sorted({(x['source'],x['raw_split']) for x in ledger}):
        i=next(x['row_index'] for x in ledger if x['source']==source and x['raw_split']==split and x['query_anchored_AUX_label_D0_E1_Uminus1']==0)
        reasons[i].append('FIRST_ORIGINAL_ROW_D_PER_SOURCE_RAW_SPLIT')
    driverpath=P/'query_visibility_same_q_counterfactual_20261023_v95/source_pair_intake/SOURCE960_DRIVER_LINEAGE.jsonl'
    driver=rows(driverpath)
    proto=j(V/'INVENTORY_PROTOCOL_BEFORE_QUERY_ANCHOR_LABELS.json')
    teachers={x['cue_view_index']:x for x in rows(proto['inputs']['original_cue_views604']['path'])}
    loco=rows(proto['inputs']['loco_root160']['path'])
    selected=[]
    for i,why in sorted(reasons.items()):
        x=ledger[i];r=driver[i]
        assert r['slot_index']==i and r['input_id']==x['input_id'] and r['native_slot_logical_only']==x['native_slot_logical_only']
        assert r['query_xy']==x['original_q'] and r['L']==x['original_L'] and r['split']==x['raw_split'] and r['image_sha256']==x['image_sha256']
        refs={k:r['before_input_refs'][k] for k in ['rgb96','crop_valid96']}
        selected.append(dict(row_index=i,reasons=why,source=x['source'],raw_split=x['raw_split'],input_id=x['input_id'],native_slot_logical_only=x['native_slot_logical_only'],q=x['original_q'],L=x['original_L'],old_label=x['original_C1_V95_label_D0_E1_Uminus1'],new_label=x['query_anchored_AUX_label_D0_E1_Uminus1'],matched_group=x['matched_certified_group_GT_only'],input_refs=refs))
    assert len(selected)==17
    lock=dict(completed=True,selection='ALL4 newE +ALL11 oldU-to-known +first D per4 source/rawfold, dedup originalrow order',selected_count=17,selected_rows=selected,
        code=bind(__file__),metadata_audit=bind(HERE/'REVIEW.json'),ledger=bind(V/'ALL960_QUERY_ANCHORED_AUX_INVENTORY_GT_ONLY.jsonl'),driver_lineage=bind(driverpath),
        timing='Written before RGB96/crop-valid member reads in this script. Aggregate C1 baseline counts were exposed by root earlier; no score bodies or scores used in selection.',
        scope='Saved source RGB96 only, enlarged without recrop/recentering. Raw and marked separated. No full-RGB context, P3/model/current/PnP. Labels unchanged. Auxiliary AI, nativeUNKNOWN/humanfalse/trainingfalse.',
        overlays='Original q at47.5,47.5; orange original cue uncertainty intervals, magenta original r, blue native positive-bilinear cells. LOCO root support and uncertainty shown without inventing r. Uncertainty is not a material/ownership mask.')
    write(QA/'SELECTION_LOCK_BEFORE_PUBLIC_CROP_ARRAYS.json',lock)
    bindings={ref['file']['path']:ref['file'] for s in selected for ref in s['input_refs'].values()}
    for path,b in bindings.items():assert bind(path)['sha256']==b['sha256']
    panels=[];receipts=[]
    for s in selected:
        i=s['row_index'];x=ledger[i];rgb,rgbreceipt=read_member_row(s['input_refs']['rgb96']);valid,vreceipt=read_member_row(s['input_refs']['crop_valid96'])
        assert rgb.shape==(3,96,96) and rgb.dtype==np.uint8 and valid.shape==(1,96,96) and valid.dtype==bool
        raw=Image.fromarray(np.transpose(rgb,(1,2,0)));marked=raw.resize((384,384),Image.Resampling.NEAREST);draw=ImageDraw.Draw(marked)
        def xy(p):return tuple(((float(p[a])-s['q'][a])/s['L']*96+47.5)*4+2 for a in [0,1])
        qx=qy=47.5*4+2
        draw.line((qx-16,qy,qx+16,qy),fill='cyan',width=2);draw.line((qx,qy-16,qx,qy+16),fill='cyan',width=2)
        used=[]
        if s['source']=='PALLOX':
            for k in x['source_cue_view_indices']:
                t=teachers[k]
                if t['duplicate_group_link_only']!=s['matched_group']:continue
                box=t['uncertainty_bbox_xyxy_half_open_unchanged'];r=t['reference_xy_unchanged']
                if box is not None:
                    a=xy(box[:2]);b=xy(box[2:]);draw.rectangle((*a,*b),outline=(255,175,0),width=3)
                if r is not None:
                    rx,ry=xy(r);draw.ellipse((rx-4,ry-4,rx+4,ry+4),outline='magenta',width=2)
                used.append(dict(cue_view_index=k,cue_id=t['cue_id'],teacher_label=t['teacher_label'],original_r=r,original_uncertainty=box))
        else:
            t=loco[i]
            for key,col in [('support_bbox_xyxy',(255,175,0)),('uncertainty_bbox_xyxy',(255,235,0))]:
                b=t[key];a=xy(b[:2]);z=xy(b[2:]);draw.rectangle((*a,*z),outline=col,width=3)
            used.append(dict(cue_id='LOCO_SAME_QUERY_ROOT_SINGLE_CUE',original_support=t['support_bbox_xyxy'],original_uncertainty=t['uncertainty_bbox_xyxy'],original_r=None))
        for cell in x['central4_positive_bilinear_native_cells']:
            cx,cy=xy(cell);draw.rectangle((cx-2,cy-2,cx+2,cy+2),outline=(64,160,255),width=1)
        rawpath=QA/f'ROW_{i:03d}_RAW_RGB96.png';raw.save(rawpath)
        markedpath=QA/f'ROW_{i:03d}_MARKED.png';marked.save(markedpath)
        panel=Image.new('RGB',(1080,470),'#111827');d=ImageDraw.Draw(panel)
        d.text((12,7),f"row{i:03d} | {s['source']} {s['raw_split']} | old{['D','E','U'][s['old_label'] if s['old_label']>=0 else 2]} -> new{['D','E'][s['new_label']]} | slot{s['native_slot_logical_only']}",font=font(20),fill='white')
        d.text((12,34),f"q=({s['q'][0]:.3f},{s['q'][1]:.3f}) L={s['L']:.3f} | valid96={valid.mean():.1%} | AI auxiliary, native UNKNOWN",font=font(16),fill='#d1d5db')
        panel.paste(raw.resize((384,384),Image.Resampling.NEAREST),(12,72));panel.paste(marked,(410,72))
        mask=Image.fromarray(valid[0].astype('uint8')*255).convert('RGB').resize((192,192),Image.Resampling.NEAREST);panel.paste(mask,(815,80))
        d.text((815,280),'crop-valid mask',font=font(17),fill='white');d.text((815,310),'cyan: original q',font=font(16),fill='cyan');d.text((815,335),'orange: cue range',font=font(16),fill='#ffaf00');d.text((815,360),'magenta: saved r',font=font(16),fill='magenta');d.text((815,385),'blue: native stencil',font=font(16),fill='#40a0ff')
        panels.append(panel)
        receipts.append(dict(**s,rgb_row=rgbreceipt,valid_row=vreceipt,crop_valid_fraction=float(valid.mean()),overlay_cue_records=used,raw_file=bind(rawpath),marked_file=bind(markedpath),no_new_crop_or_recenter=True,labels_unchanged=True,native_expected_part_state='UNKNOWN',human_gold=False,training_admission=False))
    sheets=[]
    for start in range(0,len(panels),6):
        batch=panels[start:start+6];im=Image.new('RGB',(1080,470*len(batch)),'#111827')
        for z,panel in enumerate(batch):im.paste(panel,(0,470*z))
        file=QA/f'CONTACT_{start//6+1:02d}_FIXED_ROWS.png';im.save(file);sheets.append(bind(file))
    write(QA/'SAVED_PUBLIC_CROP_RENDER_RECEIPTS.json',dict(completed=True,PASS=True,selection_lock=bind(QA/'SELECTION_LOCK_BEFORE_PUBLIC_CROP_ARRAYS.json'),selected_count=17,new_E_count=4,old_U_to_new_known_count=11,rows=receipts,sheets=sheets,opened_NPZ_members=['rgb96.npy','crop_valid96.npy'],P3_deserialization=0,full_RGB_decode=0,newcrop=0,models_fit_current_PnP=0))
    print(json.dumps(dict(completed=True,rows=17,sheets=sheets)))
if __name__=='__main__':main()

"""Display only frozen current crops/heatmaps after metric census CASE_IDS freeze."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
sys.dont_write_bytecode=True

def need(v,m):
    if not bool(v):raise ValueError(m)

def safe(value,fresh=False):
    p=Path(value);need(p.is_absolute()and'..'not in p.parts,'Absolute nontraversal')
    need(not any(s in str(p).casefold()for s in('v112','cache','.pyc','source_holdout','private_evaluation_held')),'Forbidden before filesystem')
    for a in[*reversed(p.parents),p]:
        try:mode=os.lstat(a).st_mode
        except FileNotFoundError:continue
        need(not stat.S_ISLNK(mode),'No symlink artifact/ancestor')
    if fresh:need(not p.exists(),'Fresh immutable output')
    return p

def ident(b):return(b['path'],b['sha256'],b['size_bytes'])
def binding(p):
    p=safe(p);raw=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))
def read(b,white):
    need(ident(b)in{ident(x)for x in white},'Explicit closed whitelist')
    raw=safe(b['path']).read_bytes();need(len(raw)==b['size_bytes']and hashlib.sha256(raw).hexdigest()==b['sha256'],'Exact saved artifact bytes');return raw
def put(p,d):
    with safe(p,True).open('x',encoding='utf8')as f:json.dump(d,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    return binding(p)
def font(n):
    try:return ImageFont.truetype('DejaVuSans.ttf',n)
    except OSError:return ImageFont.load_default()
def scalar(s):
    try:x=float(s)
    except(TypeError,ValueError):return None
    return x if np.isfinite(x)else None
def xy(s):
    if isinstance(s,str):
        try:s=json.loads(s)
        except json.JSONDecodeError:return None
    if not isinstance(s,(list,tuple))or len(s)!=2:return None
    p=[scalar(x)for x in s];return p if all(x is not None for x in p)else None
def rgbsha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def arrays(spec,white):
    raw=read(spec['binding'],white)
    if spec['format']=='npy':return np.load(io.BytesIO(raw),allow_pickle=False)
    need(spec['format']=='npz'and isinstance(spec['member'],str),'Explicit NPY or NPZ member')
    with np.load(io.BytesIO(raw),allow_pickle=False)as z:return z[spec['member']]

def row_array(data,spec,i):
    # Optional fixed first-axis arm/view; numerical values are never altered.
    a=data[spec['fixed_axis0'],i]if spec.get('fixed_axis0')is not None else data[i]
    if spec.get('layout')=='CHW':a=a.transpose(1,2,0)
    return np.asarray(a)

def mark(d,p,color,r=6):
    if p is None:return
    x,y=map(float,p);d.line((x-r,y,x+r,y),fill=color,width=2);d.line((x,y-r,x,y+r),fill=color,width=2)

def run(entry_path):
    ep=safe(entry_path);e=json.loads(ep.read_bytes());need(e.get('approved_saved_current_gallery')is True,'Root authorization after all freezes')
    scope=json.loads(read(e['root_scope'],[e['root_scope']]));need(scope.get('approved_saved_current_gallery')is True and scope['entry_contract']=={k:v for k,v in e.items()if k!='root_scope'},'Exact Root scope')
    need(e['code']==binding(Path(__file__).absolute()),'Executing code selfpin')
    out=safe(e['output_directory'],True);out.mkdir()
    try:
        mw=e['metadata_whitelist'];complete=json.loads(read(e['evaluation_complete'],mw));selection=json.loads(read(e['selection'],mw))
        need(complete.get('PASS')is True and complete['outputs']['selection']==e['selection'],'Exact completed evaluation/selection')
        need(selection['pixel_reads_before_selection']==0 and selection['no_empty_cell_replacement']is True and len(selection['cells'])==12,'Pre-pixel12 census selection')
        ledger=list(csv.DictReader(io.StringIO(read(e['joined_ledger'],mw).decode('utf8'))));need(len(ledger)==1784,'Exact frozen all1784 joined rows')
        need(complete['outputs']['joined_rows']==e['joined_ledger'],'Evaluation-owned joined ledger')
        # This is the first selected-pixel numerical access in this program.
        pw=e['pixel_tensor_whitelist'];loaded={};specs=e['crop_tensors'];heat_spec=e['JOINT_heatmaps']
        for kind,spec in specs.items():loaded[kind]=arrays(spec,pw)
        heat=arrays(heat_spec,pw);cards=[];made=[];case_data=[]
        for n,cell in enumerate(selection['cells']):
            card=Image.new('RGB',(1080,540),'#15191d');d=ImageDraw.Draw(card)
            title=f"{cell['group']} | {cell['cell']} | census n={cell['census_N']}"
            d.text((12,10),title,font=font(18),fill='white')
            if cell['empty']:
                need(cell['selected_global_index']is None and cell['census_N']==0,'Empty cell not replaced')
                d.text((36,160),'EMPTY n=0 — no substitute example',font=font(28),fill='white')
                item=dict(cell=cell,display='EMPTY_NO_REPLACEMENT')
            else:
                i=cell['selected_global_index'];r=ledger[i];need(int(r['global_index'])==i,'Saved original index')
                q=xy([r['query_x'],r['query_y']]);ref=xy([r['human_reference_x'],r['human_reference_y']]);need(q is not None,'Selected finite original q')
                details={};panels={}
                for arm in('VIS_ONLY_SELECTED','JOINT_SELECTED'):
                    spec=specs[e['crop_for_arm'][arm]];crop=row_array(loaded[e['crop_for_arm'][arm]],spec,i)
                    need(crop.dtype==np.uint8 and crop.shape==(96,96,3),'Exact frozen RGB96')
                    L=scalar(r[arm+'__crop_L']);need(L is not None and L>0,'Saved frozen selected crop side')
                    clean=out/(f'CELL{n:02d}_G{i:04d}_{arm}_CLEAN96.png');Image.fromarray(crop).save(clean);made.append(binding(clean))
                    panels[arm]=Image.fromarray(crop).resize((288,288),Image.Resampling.NEAREST)
                    def local(p):return 47.5+96*(np.asarray(p)-q)/L if p is not None else None
                    details[arm]=dict(crop_L=L,clean=binding(clean),RGB96_HWC_sha256=rgbsha(crop),q_crop=[47.5,47.5],reference_crop=None if ref is None else local(ref).tolist(),
                        proposed_crop=None if xy(r[arm+'__LOC_proposed_xy'])is None else local(xy(r[arm+'__LOC_proposed_xy'])).tolist(),
                        final_crop=None if xy(r[arm+'__final_xy'])is None else local(xy(r[arm+'__final_xy'])).tolist(),
                        E_logit=scalar(r[arm+'__visibility_E_logit']),semantic_VIS_prediction=r[arm+'__semantic_VIS_prediction'],accepted=r[arm+'__accepted'])
                hm=row_array(heat,heat_spec,i);need(hm.ndim==2 and np.isfinite(hm).all(),'Saved finite trained JOINT heatmap')
                low=float(hm.min());high=float(hm.max());z=(hm-low)/(high-low)if high>low else np.zeros_like(hm)
                colors=np.stack((255*z,180*z,255*(1-z)),-1).astype(np.uint8)
                cmap=Image.fromarray(colors).resize((288,288),Image.Resampling.BILINEAR)
                mixed=Image.blend(panels['JOINT_SELECTED'],cmap,.45);od=ImageDraw.Draw(mixed);j=details['JOINT_SELECTED']
                def display(p):return tuple((np.asarray(p)+.5)*3-.5)if p is not None else None
                mark(od,display(j['q_crop']),'cyan',8);mark(od,display(j['reference_crop']),'magenta',8);mark(od,display(j['proposed_crop']),'orange',6);mark(od,display(j['final_crop']),'lime',4)
                card.paste(panels['VIS_ONLY_SELECTED'],(12,104));card.paste(panels['JOINT_SELECTED'],(354,104));card.paste(mixed,(696,104))
                d.text((12,45),f"row {i} | {r['row_id']} | human {r['human_state']} | manual source {r['annotation_coordinate_source']}",font=font(14),fill='white')
                d.text((12,72),f"VIS {details['VIS_ONLY_SELECTED']['semantic_VIS_prediction']} Elogit={details['VIS_ONLY_SELECTED']['E_logit']} | JOINT {j['semantic_VIS_prediction']} Elogit={j['E_logit']} | accepted={j['accepted']}",font=font(14),fill='white')
                d.text((12,400),'VIS clean frozen crop96',font=font(16),fill='white');d.text((354,400),'JOINT clean frozen crop96',font=font(16),fill='white');d.text((696,400),'JOINT saved heatmap: min/max display',font=font(14),fill='white')
                d.text((12,434),'DISPLAY markers: original q cyan; proposal orange; final lime; inherited manual/reference r magenta.',font=font(16),fill='white')
                d.text((12,462),'No GT recrop. Labels/masks/crop/acceptance rules unchanged. Heatmap color is not calibrated confidence.',font=font(16),fill='white')
                item=dict(cell=cell,row_id=r['row_id'],global_index=i,human_state=r['human_state'],reference_authority=r['annotation_coordinate_source'],details=details,heatmap_shape=list(hm.shape),heatmap_display_min=low,heatmap_display_max=high)
            d.text((12,510),'Result-based diagnostic examples. Full cell census shown; not model, threshold, or training/configuration selection.',font=font(14),fill='white')
            path=out/('CELL%02d_CARD.png'%n);card.save(path);cb=binding(path);made.append(cb);item['card']=cb;case_data.append(item);cards.append(card)
        overview=Image.new('RGB',(1080,1620),'#15191d')
        for i,c in enumerate(cards):overview.paste(c.resize((540,270),Image.Resampling.LANCZOS),((i%2)*540,(i//2)*270))
        overview.save(out/'OVERVIEW.png');made.append(binding(out/'OVERVIEW.png'))
        made.append(put(out/'CASE_DATA.json',dict(cases=case_data,selection=e['selection'],all_crops_original_saved_bytes=True,reference_is_inherited_evaluation_coordinate_not_new_native_gold=True,markers_in_display_only=True)))
        html='<!doctype html><meta charset="utf-8"><title>V263 current 12-cell diagnostic</title><style>body{background:#15191d;color:white;font-family:sans-serif}img{max-width:100%}</style><h1>고정 규칙에 따른 12개 진단 셀</h1><p>현재 평가셋은 이미 노출되어 반복 사용되었습니다. 결과 기반 진단 사례이며 각 셀 전체 건수와 최소 원본 인덱스를 표시합니다. 빈 셀은 대체하지 않았습니다. 모델·문턱·학습 설정 선택에 사용하지 않습니다. r는 기존 평가 좌표이며 새로운 실물 native 정답을 만들지 않습니다.</p>'
        for item in case_data:html+=f'<h2>{item["cell"]["group"]} / {item["cell"]["cell"]}: n={item["cell"]["census_N"]}</h2><img src="{Path(item["card"]["path"]).name}">'
        with safe(out/'INDEX_KO.html',True).open('x',encoding='utf8')as f:f.write(html+'\n')
        made.append(binding(out/'INDEX_KO.html'))
        cb=put(out/'COMPLETE.json',dict(completed=True,PASS=True,code=e['code'],entry=binding(ep),evaluation_complete=e['evaluation_complete'],selection=e['selection'],outputs=made,cells=12,nonempty_cells=sum(not x['cell']['empty']for x in case_data),original_fullRGB_model_encoder_training_reads=0,GT_matching_recrop_or_reference_changes=0))
        print(json.dumps(cb),flush=True)
    except Exception as ex:
        put(out/'FAILURE.json',dict(completed=False,PASS=False,error_type=type(ex).__name__,error=str(ex),partial_preserved=True,automatic_retry=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',required=True);run(p.parse_args().entry)

"""Saved original-M4 SYN DEV diagnostics, no fits/model forwards/current data.

All three full-frame saved logit maps are restored to original image dimensions
before evaluation. Fixed boundary tolerances1px/3px are both reported.
"""
from pathlib import Path
import csv
import hashlib
import io
import json
import time

import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).absolute().parent
PREV=HERE.parent/'V267_FULL_IMAGE_CLIP_VISIBLE_MASK'
ARMS=('V267_CS32','HIGHRES_REFINE','BOUNDARY_REFINE')
FIXED=(776,768,770,769,792,900)
TOLERANCES=(1,3)


def safe(value):
    p=Path(value)
    assert p.is_absolute() and not any(x in str(p).lower() for x in ('v112','cache','.pyc','..')),str(p)
    for a in reversed((p,*p.parents)):assert not a.is_symlink(),str(a)
    return p


def binding(path):
    p=safe(path);raw=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))


def checked(bound):
    assert binding(bound['path'])==bound
    return safe(bound['path'])


def write_json(path,value):
    p=safe(path);assert not p.exists()
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n')


def prediction(logits,hw):
    x=torch.from_numpy(np.asarray(logits,np.float32)).reshape(1,1,*logits.shape[-2:])
    return F.interpolate(x,size=tuple(hw),mode='bilinear',align_corners=False)[0,0].numpy()>=0


def boundary(mask):
    return mask & ~binary_erosion(mask,structure=np.ones((3,3),bool),border_value=0)


def score(pred,truth):
    tp=int((pred&truth).sum());fp=int((pred&~truth).sum());fn=int((~pred&truth).sum());tn=int((~pred&~truth).sum())
    union=tp+fp+fn;den=2*tp+fp+fn
    result=dict(TP=tp,FP=fp,FN=fn,TN=tn,IoU=tp/union if union else 1.,Dice=2*tp/den if den else 1.)
    pb,gb=boundary(pred),boundary(truth);npred,ngold=int(pb.sum()),int(gb.sum())
    dg=distance_transform_edt(~gb) if ngold else None
    dp=distance_transform_edt(~pb) if npred else None
    for tol in TOLERANCES:
        pm=int((pb&(dg<=tol)).sum()) if dg is not None else 0
        gm=int((gb&(dp<=tol)).sum()) if dp is not None else 0
        precision=pm/npred if npred else 0.
        recall=gm/ngold if ngold else 0.
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.
        result[f'boundary{tol}px']=dict(predicted_boundary_pixels=npred,GT_boundary_pixels=ngold,
             matched_predicted_boundary_pixels=pm,matched_GT_boundary_pixels=gm,
             precision=precision,recall=recall,F1=f1)
    return result


def aggregate(rows):
    tp=sum(r['TP'] for r in rows);fp=sum(r['FP'] for r in rows);fn=sum(r['FN'] for r in rows);tn=sum(r['TN'] for r in rows)
    result=dict(frames=len(rows),CM_TP_FP_FN_TN=[tp,fp,fn,tn],micro_IoU=tp/max(tp+fp+fn,1),micro_Dice=2*tp/max(2*tp+fp+fn,1),
         mean_frame_IoU=float(np.mean([r['IoU'] for r in rows])),mean_frame_Dice=float(np.mean([r['Dice'] for r in rows])))
    for tol in TOLERANCES:
        rr=[r[f'boundary{tol}px'] for r in rows]
        n=sum(r['predicted_boundary_pixels'] for r in rr);g=sum(r['GT_boundary_pixels'] for r in rr)
        nm=sum(r['matched_predicted_boundary_pixels'] for r in rr);gm=sum(r['matched_GT_boundary_pixels'] for r in rr)
        p=nm/max(n,1);re=gm/max(g,1)
        result[f'boundary{tol}px']=dict(precision=p,recall=re,F1=2*p*re/(p+re) if p+re else 0.,
             predicted_boundary_pixels=n,GT_boundary_pixels=g,matched_predicted_boundary_pixels=nm,matched_GT_boundary_pixels=gm)
    return result


def overlay(rgb,mask):
    out=rgb.astype(np.float32).copy();out[mask]=.55*out[mask]+.45*np.asarray([30,235,100],np.float32)
    return Image.fromarray(np.clip(out,0,255).astype(np.uint8))


def main():
    start=time.time();torch.set_num_threads(1)
    out=safe(HERE/'syn_mask_diagnostic_v1');assert not out.exists()
    oldp=safe(PREV/'syn_training_v1/COMPLETE.json');newp=safe(HERE/'syn_training_v1/COMPLETE.json')
    old=json.loads(oldp.read_text());new=json.loads(newp.read_text());assert old['PASS'] and new['PASS']
    sealed=new['outputs']['BOTH_CHECKPOINTS_FROZEN_BEFORE_SYN_DEV.json'];seal=json.loads(checked(sealed).read_text());assert seal['completed'] and seal['SYN_DEV_refiner_forwards']==0
    data=json.loads(safe(PREV/'data_audit/COMPLETE.json').read_text())
    inputs=[json.loads(l) for l in checked(data['outputs']['FULL_FRAME_RGB_INPUTS.jsonl']).read_text().splitlines()]
    targets=[json.loads(l) for l in checked(data['outputs']['VISIBLE_MASK_TARGETS.jsonl']).read_text().splitlines()]
    dev=np.asarray([r['frame_index'] for r in inputs if r['split']=='validation'],np.int32);assert len(dev)==235 and set(FIXED)<=set(dev)
    bound={'V267_CS32':old['outputs']['CLIP_SURGERY_ARCH_SYN_DEV_MASKS.npz']}
    bound.update({a:new['outputs'][a+'_SYN_DEV_MASKS.npz'] for a in ARMS[1:]})
    maps={}
    for arm,b in bound.items():
        with np.load(checked(b),allow_pickle=False) as z:
            assert np.array_equal(z['frame_ids'],dev)
            maps[arm]=np.array(z['mask_logits'],copy=True)
        side=32 if arm=='V267_CS32' else 256
        assert maps[arm].shape==(235,1,side,side) and np.isfinite(maps[arm]).all()
    out.mkdir()
    write_json(out/'METRIC_AND_VISUAL_PLAN_BEFORE_M4_OR_RGB.json',dict(completed=True,
        plan='All235originalSYNDEVframes; fixedlogit>=0; originalGT=M4uint8>127; restoreeachlogitmaptooriginalHWbilinearalignFalse.',
        boundaries='Inner boundary =binarymask minus3x3all-neighborerosion, outside-frame0; includesholes. Nearest-boundarymatchtolerances1and3originalpx bothreported.',
        fixed_visual_frame_indices=list(FIXED),visual_selection='PreviouslyfixedV267sourcecases; no newmodelselection bymetricsorRGB.',
        new_checkpoints_frozen_before_DEV=sealed,prediction_bindings=bound,
        no_current_or_real_data_reads=True,no_new_model_forwards=True))
    records=[];metrics={a:[] for a in ARMS};selected={};mask_inputs=[]
    for position,index in enumerate(dev):
        t=targets[int(index)];r=inputs[int(index)];assert t['frame_index']==r['frame_index']==int(index) and r['id']==t['id']
        p=checked(t['M4'])
        with Image.open(io.BytesIO(p.read_bytes())) as im:truth=np.array(im)>127
        assert truth.shape==tuple(r['raw_hw']) and int(truth.sum())==t['M4_area']
        mask_inputs.append(t['M4']);pp={a:prediction(maps[a][position,0],truth.shape) for a in ARMS}
        row=dict(frame_index=int(index),id=r['id'],raw_hw=r['raw_hw'],M4=t['M4'],GT_target_pixels=int(truth.sum()))
        row['models']={a:score(pp[a],truth) for a in ARMS}
        for a in ARMS:metrics[a].append(row['models'][a])
        records.append(row)
        if int(index) in FIXED:selected[int(index)]=(truth,pp,row)
    results={a:aggregate(rr) for a,rr in metrics.items()}
    write_json(out/'ORIGINAL_PIXEL_RESULTS.json',dict(completed=True,PASS=True,scope='SYN235DEVoriginalM4only',results=results,
        GT='OriginalresolutionbinaryM4>127; not256softtarget and not32upsampledGT.',
        threshold='SavedrawlogitsupsampledoriginalHWbilinearalignFalse, targetif>=0',
        matching='Two-sidedboundaryP/R via nearestboundarypoint; tolerances1and3rawpxbothreported.',
        empty_boundary_policy='Empty predicted boundary yields0perframeP/R/F1 whenGTnonempty; both-emptyIoU/Dice1.',
        no_recipe_or_checkpoint_or_threshold_selection=True))
    write_json(out/'ALL235_ORIGINAL_PIXEL_DETAILS.json',dict(completed=True,frames=records))
    with safe(out/'ORIGINAL_PIXEL_COMPARISON.csv').open('x') as f:
        names=['model','micro_IoU','micro_Dice','mean_frame_IoU','mean_frame_Dice','boundary1px_precision','boundary1px_recall','boundary1px_F1','boundary3px_precision','boundary3px_recall','boundary3px_F1'];wr=csv.DictWriter(f,fieldnames=names);wr.writeheader()
        for a,r in results.items():
            row={'model':a,**{k:r[k] for k in names[1:5]}}
            for tol in TOLERANCES:
                for k in ('precision','recall','F1'):row[f'boundary{tol}px_{k}']=r[f'boundary{tol}px'][k]
            wr.writerow(row)
    fontpath=safe(Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'));font=ImageFont.truetype(str(fontpath),18);small=ImageFont.truetype(str(fontpath),15)
    cards=[]
    for index in FIXED:
        truth,pp,row=selected[index];r=inputs[index]
        rgbpath=checked(r['RGB'])
        with Image.open(io.BytesIO(rgbpath.read_bytes())) as im:rgb=np.array(im.convert('RGB'),copy=True)
        assert rgb.shape[:2]==truth.shape
        yy,xx=np.where(truth);margin=24;h,w=truth.shape;bbox=(max(int(xx.min())-margin,0),max(int(yy.min())-margin,0),min(int(xx.max())+margin+1,w),min(int(yy.max())+margin+1,h))
        gt_image=np.zeros_like(rgb);gt_image[truth]=[30,235,100]
        views=[Image.fromarray(rgb),Image.fromarray(gt_image)]+[overlay(rgb,pp[a]) for a in ARMS]
        labels=['원본 전체RGB','원본 M4 정답','V267 CS32','HIGHRES_REFINE','BOUNDARY_REFINE']
        card=Image.new('RGB',(1472,720),'white');d=ImageDraw.Draw(card)
        d.text((16,8),f'SYN frame{index} · {r["id"]} · 전체프레임 모델 입력, 원본GT로 평가',font=font,fill='#123')
        for c,(view,label) in enumerate(zip(views,labels)):
            x=16+290*c;d.text((x,42),label,font=small,fill='#123');card.paste(view.resize((272,272),Image.Resampling.NEAREST),(x,70));card.paste(view.crop(bbox).resize((272,272),Image.Resampling.NEAREST),(x,388))
            if c>=2:
                rr=row['models'][ARMS[c-2]];d.text((x,666),f'IoU {rr["IoU"]:.3f} · BF1@3px {rr["boundary3px"]["F1"]:.3f}',font=small,fill='#123')
        d.text((16,352),'아래: 같은 원본 M4 bbox+24px의 표시 전용 확대. 학습·추론에 이 crop을 넣지 않았습니다.',font=small,fill='#9d3516')
        p=safe(out/f'SYN_FRAME{index:04d}.png');card.save(p);cards.append(dict(frame_index=index,source_RGB=r['RGB'],source_M4=row['M4'],display_only_bbox=list(bbox),card=binding(p),metrics=row['models']))
    write_json(out/'FIXED6_VISUAL_CASES.json',dict(completed=True,cases=cards,fixed_frame_order=list(FIXED),RGB_frames_decoded=6,current_or_real_reads=0))
    html='<!doctype html><meta charset="utf-8"><title>V268 합성 원본 픽셀 진단</title><style>body{font:18px sans-serif;max-width:1500px;margin:30px auto;color:#123;background:#f5f7f9}img{max-width:100%}table{border-collapse:collapse}th,td{padding:10px;border:1px solid #ccd}</style><h1>합성 DEV235장: 원본 M4와의 픽셀·경계 비교</h1><p>원본 픽셀 GT로 평가했습니다. 전체 이미지가 모델 입력이며, 아래 확대는 표시 전용입니다. 고정1px/3px 경계 허용 오차를 모두 보고하며 모델 선택에 사용하지 않습니다. 실사 분할 정확도를 뜻하지 않습니다.</p><table><tr><th>모델</th><th>원본 IoU</th><th>원본 Dice</th><th>경계P/R@1px</th><th>경계P/R@3px</th></tr>'
    for a,r in results.items():html+=f'<tr><td>{a}</td><td>{r["micro_IoU"]:.4f}</td><td>{r["micro_Dice"]:.4f}</td><td>{r["boundary1px"]["precision"]:.4f}/{r["boundary1px"]["recall"]:.4f}</td><td>{r["boundary3px"]["precision"]:.4f}/{r["boundary3px"]["recall"]:.4f}</td></tr>'
    html+='</table>'+''.join(f'<img src="SYN_FRAME{index:04d}.png">' for index in FIXED);safe(out/'INDEX_KO.html').write_text(html)
    names=['METRIC_AND_VISUAL_PLAN_BEFORE_M4_OR_RGB.json','ORIGINAL_PIXEL_RESULTS.json','ALL235_ORIGINAL_PIXEL_DETAILS.json','ORIGINAL_PIXEL_COMPARISON.csv','FIXED6_VISUAL_CASES.json','INDEX_KO.html']+[f'SYN_FRAME{index:04d}.png' for index in FIXED]
    write_json(out/'COMPLETE.json',dict(completed=True,PASS=True,code=binding(Path(__file__).absolute()),old_training=binding(oldp),new_training=binding(newp),outputs={n:binding(out/n) for n in names},original_M4_files_read=235,full_RGB_frames_read=6,new_model_forwards=0,optimizer_updates=0,current_or_real_reads=0,elapsed_seconds=time.time()-start))
    print(json.dumps(dict(completed=True,PASS=True,complete=binding(out/'COMPLETE.json'),results=results),ensure_ascii=False),flush=True)


if __name__=='__main__':main()

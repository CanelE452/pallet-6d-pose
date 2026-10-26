"""Portable saved original-M4 five-arm SYN DEV replay, no model/current data.

All five full-frame saved logit maps are restored to original image dimensions
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
ARMS=('OPENAI_FULL_CLIP','V267_CS','EVA_STANDARD','EVA_CSA','EVA_DECLIP')
FIXED=(768,769,770,776,792,900)
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


def relative_checked(bound):
    rel=bound['relative_path'];assert not Path(rel).is_absolute()
    p=safe(HERE/rel);b=binding(p)
    assert b['sha256']==bound['sha256']and b['size_bytes']==bound['size_bytes']
    return p


def rebased_checked(relative,bound):
    p=safe(HERE/relative);b=binding(p)
    assert b['sha256']==bound['sha256']and b['size_bytes']==bound['size_bytes']
    return p


def main():
    start=time.time();torch.set_num_threads(1)
    fixture=safe(HERE/'syn_pixel_replay_v1');manifest=json.loads(safe(fixture/'REPLAY_INPUTS.json').read_bytes())
    assert manifest['PASS']and manifest['fixed_visual_frame_indices']==list(FIXED)
    newp=safe(HERE/'syn_training_v1/COMPLETE.json');new=json.loads(newp.read_bytes())
    assert new['PASS']and new['new_optimizer_updates']==3000
    sealb=new['outputs']['ALL_CHECKPOINTS_FROZEN_BEFORE_SYN_DEV.json']
    seal=json.loads(rebased_checked('syn_training_v1/ALL_CHECKPOINTS_FROZEN_BEFORE_SYN_DEV.json',sealb).read_bytes())
    assert seal['SYN_DEV_head_forwards']==0 and set(seal['arms'])==set(ARMS[2:])
    with np.load(relative_checked(manifest['packed_M4']),allow_pickle=False)as z:
        packed=np.array(z['packed'],copy=True);offsets=z['byte_offsets'];hw=z['raw_hw'];dev=z['frame_ids'];source_shas=z['source_M4_sha256']
    assert np.array_equal(dev,np.arange(768,1003))and len(hw)==235
    maps={};bounds={}
    for a in ARMS:
        if a in manifest['old_logits']:
            b=manifest['old_logits'][a];p=relative_checked(b)
        else:
            b=new['outputs'][a+'_SYN_DEV_MASKS.npz'];p=rebased_checked('syn_training_v1/'+a+'_SYN_DEV_MASKS.npz',b)
        bounds[a]=dict(relative_path=str(p.relative_to(HERE)),sha256=b['sha256'],size_bytes=b['size_bytes'])
        with np.load(p,allow_pickle=False)as z:
            assert np.array_equal(z['frame_ids'],dev);maps[a]=np.array(z['mask_logits'],copy=True)
        assert maps[a].shape==(235,1,32,32)and maps[a].dtype==np.float32 and np.isfinite(maps[a]).all()
    out=safe(HERE/'syn_mask_diagnostic_v1');prior=out.exists()
    if not prior:
        out.mkdir()
        write_json(out/'PLAN_BEFORE_PIXEL_METRICS.json',dict(completed=True,arms=list(ARMS),frames=235,
            packed_original_M4=manifest['packed_M4'],prediction_bindings=bounds,new_final_checkpoints_sealed_before_DEV=sealb,
            fixed_visual_frame_indices=list(FIXED),metric='Original M4 binary>127, raw32 logit bilinear to originalHW alignFalse, threshold>=0, inner3x3 boundary nearest distance1/3px',
            new_model_forwards=0,new_fits=0,current_reads=0))
    records=[];metrics={a:[]for a in ARMS};selected={}
    for position,index in enumerate(dev):
        h,w=map(int,hw[position]);truth=np.unpackbits(packed[int(offsets[position]):int(offsets[position+1])],bitorder='little',count=h*w).reshape(h,w).astype(bool)
        pp={a:prediction(maps[a][position,0],(h,w))for a in ARMS}
        row=dict(frame_index=int(index),raw_hw=[h,w],original_M4_SHA256=source_shas[position].decode(),GT_target_pixels=int(truth.sum()),models={a:score(pp[a],truth)for a in ARMS})
        records.append(row)
        for a in ARMS:metrics[a].append(row['models'][a])
        if int(index)in FIXED:selected[int(index)]=(truth,pp,row)
    results={a:aggregate(rr)for a,rr in metrics.items()}
    if prior:
        expected=json.loads(safe(out/'ORIGINAL_PIXEL_RESULTS.json').read_bytes())['results']
        assert results==expected
        receipt=dict(completed=True,PASS=True,all235_five_arm_counts_and_metrics_exact_match=True,
            raw_RGB_M4_source_files_read=0,new_model_forwards=0,new_fits=0,current_reads=0)
        p=safe(fixture/'REPLAY_VERIFICATION.json')
        if p.exists():assert json.loads(p.read_bytes())==receipt
        else:write_json(p,receipt)
        print(json.dumps(dict(**receipt,results=results)),flush=True);return
    write_json(out/'ORIGINAL_PIXEL_RESULTS.json',dict(completed=True,PASS=True,scope='SYN235DEV original-resolution M4 binary only',results=results,
        GT='Lossless packbits of original uint8 M4>127, not32/256-upsampled GT',
        threshold='Raw32logit bilinear to originalHW align_cornersFalse then>=0',
        boundary='Inner boundary mask minus3x3 erosion/outside0, nearest-boundary matching at both fixed1px/3px',
        no_selection=True,no_real_pixel_GT_or_real_segmentation_IoU_claim=True))
    write_json(out/'ALL235_ORIGINAL_PIXEL_DETAILS.json',dict(completed=True,frames=records))
    fields=['model','micro_IoU','micro_Dice','mean_frame_IoU','mean_frame_Dice']+[f'boundary{t}px_{k}'for t in TOLERANCES for k in ('precision','recall','F1')]
    with safe(out/'ORIGINAL_PIXEL_COMPARISON.csv').open('x')as f:
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader()
        for a,r in results.items():
            row=dict(model=a,**{k:r[k]for k in fields[1:5]})
            for t in TOLERANCES:
                for k in ('precision','recall','F1'):row[f'boundary{t}px_{k}']=r[f'boundary{t}px'][k]
            wr.writerow(row)
    fontpath=safe(Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'));font=ImageFont.truetype(str(fontpath),18);small=ImageFont.truetype(str(fontpath),15)
    cards=[]
    for index in FIXED:
        truth,pp,row=selected[index];raw=manifest['raw6'][str(index)]
        with Image.open(io.BytesIO(relative_checked(raw['RGB']).read_bytes()))as im:rgb=np.array(im.convert('RGB'),copy=True)
        with Image.open(io.BytesIO(relative_checked(raw['M4']).read_bytes()))as im:gt=np.array(im)>127
        assert np.array_equal(gt,truth)and rgb.shape[:2]==truth.shape
        yy,xx=np.where(truth);h,w=truth.shape;bbox=(max(int(xx.min())-24,0),max(int(yy.min())-24,0),min(int(xx.max())+25,w),min(int(yy.max())+25,h))
        gt_image=np.zeros_like(rgb);gt_image[truth]=[30,235,100]
        views=[Image.fromarray(rgb),Image.fromarray(gt_image)]+[overlay(rgb,pp[a])for a in ARMS]
        labels=['전체 RGB','원본 M4 GT',*ARMS]
        card=Image.new('RGB',(2052,720),'white');d=ImageDraw.Draw(card)
        d.text((16,8),f'SYN frame{index} · 전체영상 입력 / 원본 픽셀 GT 평가 / 녹색=모델 대상 마스크',font=font,fill='#123')
        for c,(view,label)in enumerate(zip(views,labels)):
            x=16+290*c;d.text((x,42),label,font=small,fill='#123')
            card.paste(view.resize((272,272),Image.Resampling.NEAREST),(x,70));card.paste(view.crop(bbox).resize((272,272),Image.Resampling.NEAREST),(x,388))
            if c>=2:
                rr=row['models'][ARMS[c-2]];d.text((x,666),f'IoU {rr["IoU"]:.3f} · BF1@3 {rr["boundary3px"]["F1"]:.3f}',font=small,fill='#123')
        d.text((16,352),'아래는 같은 GT bbox+24px 표시 전용 확대. 모델은 이 crop을 입력받지 않았습니다.',font=small,fill='#9d3516')
        p=safe(out/f'SYN_FRAME{index:04d}.png');card.save(p)
        cards.append(dict(frame_index=index,source_RGB=raw['RGB'],source_M4=raw['M4'],display_only_bbox=list(bbox),card=binding(p),metrics=row['models']))
    write_json(out/'FIXED6_VISUAL_CASES.json',dict(completed=True,cases=cards,fixed_frame_order=list(FIXED),RGB_frames_decoded=6,current_reads=0))
    html='<!doctype html><meta charset="utf-8"><title>V269 합성 원본 픽셀 5모델 비교</title><style>body{font:18px sans-serif;max-width:2052px;margin:30px auto;color:#123;background:#f5f7f9}img{max-width:100%}th,td{padding:8px;border:1px solid #ccd}table{border-collapse:collapse}</style><h1>합성 DEV235장: 원본 M4로 5모델 비교</h1><p>전체 영상이 모델 입력이며 아래 확대는 표시 전용입니다. 원본 픽셀의 IoU·Dice와 경계1/3px precision/recall/F1을 같은 조건으로 계산했습니다. 실사 키포인트 보임/가림 성능을 대신하지 않습니다. 공개 DeCLIP 백본은 COCO 이미지 무라벨 증류를 거쳤고 새 팔레트 mask head 학습은 합성뿐입니다.</p><table><tr><th>모델</th><th>IoU</th><th>Dice</th><th>경계F1@1px</th><th>경계F1@3px</th></tr>'
    for a,r in results.items():html+=f'<tr><td>{a}</td><td>{r["micro_IoU"]:.4f}</td><td>{r["micro_Dice"]:.4f}</td><td>{r["boundary1px"]["F1"]:.4f}</td><td>{r["boundary3px"]["F1"]:.4f}</td></tr>'
    html+='</table>'+''.join(f'<img src="SYN_FRAME{x:04d}.png">'for x in FIXED);safe(out/'INDEX_KO.html').write_text(html)
    write_json(fixture/'PORTABLE_SCORED_BINDINGS.json',dict(completed=True,PASS=True,prediction_bindings=bounds,
        expected_results=results,code=binding(Path(__file__).absolute()),replay='python -B analyze_syn_pixels_v269.py; rerun verifies existing results without raw RGB/M4'))
    names=['PLAN_BEFORE_PIXEL_METRICS.json','ORIGINAL_PIXEL_RESULTS.json','ALL235_ORIGINAL_PIXEL_DETAILS.json','ORIGINAL_PIXEL_COMPARISON.csv','FIXED6_VISUAL_CASES.json','INDEX_KO.html']+[f'SYN_FRAME{x:04d}.png'for x in FIXED]
    write_json(out/'COMPLETE.json',dict(completed=True,PASS=True,code=binding(Path(__file__).absolute()),new_training=binding(newp),
        packed_M4_input=manifest['packed_M4'],prediction_bindings=bounds,outputs={n:binding(out/n)for n in names},
        original_M4_files_read=0,packed_original_masks_decoded=235,local_fixed_RGB_M4_pairs_decoded=6,
        new_model_forwards=0,optimizer_updates=0,current_reads=0,elapsed_seconds=time.time()-start))
    print(json.dumps(dict(completed=True,PASS=True,complete=binding(out/'COMPLETE.json'),results=results),ensure_ascii=False),flush=True)


if __name__=='__main__':main()



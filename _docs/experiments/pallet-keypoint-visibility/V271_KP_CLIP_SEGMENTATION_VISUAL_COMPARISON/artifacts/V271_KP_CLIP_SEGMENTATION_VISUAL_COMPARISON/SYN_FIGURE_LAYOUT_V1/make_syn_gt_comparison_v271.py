"""Saved SYN original RGB/M4 versus full-frame CLIP predicted masks.

--prepare copies exact two original RGB/M4 pairs plus saved FP32 logits.
Default/replay requires only the tiny compact input folder, no original corpus,
model, GPU, fitting, current223 or real segmentation annotation.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import shutil

import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from torch.nn import functional as F

HERE=Path(__file__).absolute().parent
PREV=HERE.parent/'V269_FULL_IMAGE_DECLIP_MATCHED_BACKBONE'
FIXED=(768,770)
ARMS=('OPENAI_FULL_CLIP','V267_CS','EVA_DECLIP')


def safe(value):
    p=Path(value)
    if not p.is_absolute() or any(t in str(p).lower() for t in ('v112','cache','.pyc','..')):raise ValueError(str(p))
    for a in reversed((p,*p.parents)):
        if a.is_symlink():raise ValueError(str(a))
    return p


def binding(path):
    p=safe(path);b=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}


def checked(bound):
    assert binding(bound['path'])==bound,bound
    return safe(bound['path'])


def relative_checked(root,bound):
    r=Path(bound['relative_path']);assert not r.is_absolute()
    p=safe(root/r);b=binding(p)
    assert b['sha256']==bound['sha256'] and b['size_bytes']==bound['size_bytes'];return p


def savej(path,x):
    p=safe(path);assert not p.exists(),p
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n')


def prepare(directory):
    directory=safe(directory);assert not directory.exists(),directory;directory.mkdir()
    src=safe(PREV/'syn_pixel_replay_v1/REPLAY_INPUTS.json');m=json.loads(src.read_text());assert m['PASS']
    training_path=safe(PREV/'syn_training_v1/COMPLETE.json');training=json.loads(training_path.read_text());assert training['PASS']
    sources={};maps={};positions=None
    for arm in ARMS:
        if arm in m['old_logits']:
            b=m['old_logits'][arm];p=relative_checked(PREV,b)
        else:
            b=training['outputs'][arm+'_SYN_DEV_MASKS.npz'];p=checked(b)
        with np.load(p,allow_pickle=False)as z:
            ix=np.asarray([np.flatnonzero(z['frame_ids']==i)[0] for i in FIXED]);v=z['mask_logits'][ix,0].copy()
            assert v.shape==(2,32,32) and v.dtype==np.float32 and np.isfinite(v).all();maps[arm]=v
        sources[arm]={'path':str(p),'sha256':binding(p)['sha256'],'size_bytes':p.stat().st_size}
    np.savez_compressed(safe(directory/'SAVED_MASK_LOGITS2.npz'),frame_index=np.asarray(FIXED,np.int32),**maps)
    files={};pairs={}
    for i in FIXED:
        pairs[str(i)]={}
        for kind in ('RGB','M4'):
            b=m['raw6'][str(i)][kind];p=relative_checked(PREV,b);name=f'RAW_{kind}_{i:04d}.png'
            q=safe(directory/name);shutil.copyfile(p,q);actual=binding(q)
            assert actual['sha256']==b['sha256'] and actual['size_bytes']==b['size_bytes']
            files[name]={'file':name,'sha256':actual['sha256'],'size_bytes':actual['size_bytes']}
            pairs[str(i)][kind]=name
    b=binding(directory/'SAVED_MASK_LOGITS2.npz');files['SAVED_MASK_LOGITS2.npz']={'file':'SAVED_MASK_LOGITS2.npz','sha256':b['sha256'],'size_bytes':b['size_bytes']}
    savej(directory/'COMPACT_INPUTS.json',{'completed':True,'PASS':True,'fixed_frames':list(FIXED),'arms':list(ARMS),
        'source_replay_inputs':binding(src),'source_training_complete':binding(training_path),'source_logits':sources,
        'pairs':pairs,'files':files,'original_RGB_M4_encoded_bytes_preserved':True,
        'scope':'Original SYN DEV examples only; no actual real segmentation GT is supplied.',
        'model_forwards':0,'optimizer_steps':0,'current223_reads':0})


def panel(image,size):
    out=Image.new('RGB',size,'black');im=image.copy();im.thumbnail(size,Image.Resampling.NEAREST)
    out.paste(im,((size[0]-im.width)//2,(size[1]-im.height)//2));return out


def mask_image(mask):
    out=np.zeros((*mask.shape,3),np.uint8);out[mask]=[25,235,100];return Image.fromarray(out)


def render(directory,outpath):
    directory=safe(directory);m=json.loads(safe(directory/'COMPACT_INPUTS.json').read_text())
    assert m['PASS'] and m['fixed_frames']==list(FIXED) and m['arms']==list(ARMS)
    paths={}
    for n,b in m['files'].items():
        assert Path(b['file']).name==b['file'];p=safe(directory/b['file']);a=binding(p)
        assert a['sha256']==b['sha256'] and a['size_bytes']==b['size_bytes'];paths[n]=p
    with np.load(paths['SAVED_MASK_LOGITS2.npz'],allow_pickle=False)as z:maps={k:z[k] for k in z.files}
    assert maps['frame_index'].tolist()==list(FIXED)
    torch.set_num_threads(1)
    fontpath=safe(Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
    titlefont=ImageFont.truetype(str(fontpath),24);font=ImageFont.truetype(str(fontpath),18);small=ImageFont.truetype(str(fontpath),16)
    canvas=Image.new('RGB',(1660,1400),'white');d=ImageDraw.Draw(canvas)
    d.text((20,10),'합성 실제 segmentation GT와 저장 예측 · 전체 RGB 입력 · 새 모델 실행 없음',font=titlefont,fill='#123')
    d.text((20,48),'GT=M4 원본 픽셀>127. 예측=raw32 logit를 원본 HW로 bilinear 복원 후≥0. 녹색=대상 팔레트.',font=font,fill='#123')
    records=[]
    for row,i in enumerate(FIXED):
        y=95+row*645;pair=m['pairs'][str(i)]
        rgb=np.asarray(Image.open(paths[pair['RGB']]).convert('RGB'));raw=np.asarray(Image.open(paths[pair['M4']]))
        truth=raw>127;assert truth.ndim==2 and rgb.shape[:2]==truth.shape and raw.dtype==np.uint8
        h,w=truth.shape;preds={};metrics={}
        for arm in ARMS:
            rawlogit=torch.from_numpy(np.asarray(maps[arm][row],np.float32))[None,None]
            restored=F.interpolate(rawlogit,size=(h,w),mode='bilinear',align_corners=False)[0,0].numpy()
            pred=restored>=0;preds[arm]=pred
            tp=int((pred&truth).sum());fp=int((pred&~truth).sum());fn=int((~pred&truth).sum())
            metrics[arm]={'TP':tp,'FP':fp,'FN':fn,'IoU':tp/max(1,tp+fp+fn),
                'Dice':2*tp/max(1,2*tp+fp+fn),'hard_mask_SHA256':hashlib.sha256(pred.astype(np.uint8).tobytes()).hexdigest()}
        yy,xx=np.where(truth);bbox=(max(0,int(xx.min())-24),max(0,int(yy.min())-24),min(w,int(xx.max())+25),min(h,int(yy.max())+25))
        rgb_im=Image.fromarray(rgb);fullrgb=rgb_im.copy();draw=ImageDraw.Draw(fullrgb);draw.rectangle(bbox,outline=(255,210,25),width=2)
        fullviews=[fullrgb,mask_image(truth),*[mask_image(preds[a]) for a in ARMS]]
        zoomviews=[rgb_im,mask_image(truth),*[mask_image(preds[a]) for a in ARMS]]
        names=['원본 SYN RGB','실제 M4 GT','OpenAI CLIP 예측','V267 Surgery 예측','V269 DeCLIP 예측']
        d.text((20,y),f'SYN frame {i} · 원본 {w}×{h} · 상단=전체, 하단=동일 bbox+24px 표시 전용 확대',font=font,fill='#123')
        for col,(name,full,zoom) in enumerate(zip(names,fullviews,zoomviews)):
            x=20+328*col;d.text((x,y+34),name,font=font,fill='#123')
            canvas.paste(panel(full,(308,220)),(x,y+66));canvas.paste(panel(zoom.crop(bbox),(308,270)),(x,y+310))
            if col>=2:d.text((x,y+588),f'원본 픽셀 IoU {metrics[ARMS[col-2]]["IoU"]:.3f}',font=small,fill='#123')
        records.append({'frame_index':i,'raw_hw':[h,w],'display_only_bbox':list(bbox),
            'GT_binary_definition':'Original encoded M4 uint8>127; not32grid GT','metrics':metrics})
    d.text((20,1380),'이 그림의 GT·IoU는 합성에만 존재합니다. 실사 mask는 예측이며 pixel GT/IoU가 아닙니다.',font=small,fill='#9b331b')
    outpath=safe(outpath);assert not outpath.exists(),outpath;canvas.save(outpath)
    metadata={'completed':True,'PASS':True,'code':binding(Path(__file__).absolute()),'compact_inputs':binding(directory/'COMPACT_INPUTS.json'),
        'output':binding(outpath),'cases':records,'transform':'Saved float32 raw32logit ->torch bilinear original(H,W),align_corners=False ->hard>=0; no per-image normalization.',
        'model_input':'Uncropped whole RGB512square; displayed bbox is GT-only visualization, never input.',
        'original_M4_GT_mask_count':2,'real_pixel_GT_mask_count':0,'RGB_images_decoded':2,
        'model_forwards':0,'new_training':0,'current223_reads':0}
    savej(outpath.with_suffix('.json'),metadata)
    print(json.dumps({'figure':binding(outpath),'metadata':binding(outpath.with_suffix('.json'))},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--inputs',default='SYN_GT_COMPACT_INPUTS_V1');parser.add_argument('--out',default='SYN_GT_VS_PRED.png')
    args=parser.parse_args();directory=safe(HERE/args.inputs)
    if args.prepare:prepare(directory)
    render(directory,safe(HERE/args.out))

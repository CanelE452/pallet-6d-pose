"""Obtain actual frozen YOLO point observations for the line-attention pilot."""
from pathlib import Path
import json
import sys
import torch
KIT=Path(__file__).resolve().parent
ROOT=KIT.parents[2]
OLD=ROOT/'scripts/research/pallet_point_line_decisive_v6'
sys.path.insert(0,str(OLD))
from cli_train import Split,sha
from stock_geometry_baseline import R0,EXPECTED

RUN=ROOT/'data/pallet/results/pallet_point_attention_v9'
EXPORT=ROOT/'data/pallet/results/pallet_point_line_v5_local/export'


def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    from ultralytics import YOLO
    torch.set_num_threads(2);RUN.mkdir(parents=True,exist_ok=True)
    model=None
    for split in ('train','calibration','synth_val'):
        out=RUN/f'points_{split}.json';data=Split(EXPORT/f'{split}.json')
        if out.exists():
            j=json.loads(out.read_text());assert j['checkpoint_sha256']==EXPECTED
            assert [r['id'] for r in j['rows']]==[r['id'] for r in data.records]
            continue
        reuse={'calibration':ROOT/'data/pallet/results/pallet_point_line_causal_v7/STOCK_PREDICTIONS.json',
               'synth_val':ROOT/'data/pallet/results/pallet_point_line_decisive_v6/STOCK_GEOMETRY_BASELINE.json'}
        if split in reuse:
            j=json.loads(reuse[split].read_text());assert j['checkpoint_sha256']==EXPECTED
            rows=[dict(id=r['id'],points=r['points']) for r in j['rows']]
        else:
            assert sha(R0)==EXPECTED
            if model is None:model=YOLO(str(R0),task='pose')
            rows=[]
            for i,rec in enumerate(data.records):
                O,_,_=data.batch([i],'cpu',targets=False)
                result=model.predict(str(ROOT/rec['original_image']),imgsz=640,conf=.001,iou=.7,max_det=300,
                                     device='cuda:0',half=False,verbose=False)[0]
                boxes=result.boxes.xyxy.cpu()-100.;assert len(boxes)
                dist=(boxes-O['box'][0]).abs().amax(-1);best=int(dist.argmin());assert float(dist[best])<.1
                points=result.keypoints.xy[best,:8].cpu()-100.
                rows.append(dict(id=rec['id'],points=points.tolist(),box_delta=float(dist[best])))
                if (i+1)%200==0:print('YOLO point export',split,i+1,'/',len(data.records),flush=True)
        assert [r['id'] for r in rows]==[r['id'] for r in data.records]
        write(out,dict(rows=rows,checkpoint_sha256=EXPECTED,manifest_sha256=sha(data.path),
                       source=str(reuse[split]) if split in reuse else str(R0),
                       scope='Actual frozen YOLO predictions. No GT point replacement; existing detector-match population is GT-conditioned.'))
        print('POINTS COMPLETE',split,len(rows),flush=True)


if __name__=='__main__':main()

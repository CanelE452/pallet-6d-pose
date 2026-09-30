"""Recreate the audit contact sheet from frozen evidence and existing RGB files.

No model, inference, training, target selection or metric recomputation. Uses
all eight pre-recorded cases. Original RGB files are private local inputs; their
SHA-256 and byte counts must match the published audit JSON. The output path
must be new so a reviewed figure can never be silently overwritten.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / '_docs/experiments/pallet_pose_joint_recovery_20261001_v1'


def render(evidence: Path, root: Path, font_path: Path, output: Path) -> None:
    payload = json.loads(evidence.read_text())
    odd = payload['mismatched8']
    assert len(odd) == len({row['id'] for row in odd}) == 8
    assert [row['id'] for row in odd] == sorted(row['id'] for row in odd)
    assert not output.exists(), f'Refusing to overwrite: {output}'
    for row in odd:
        image_path = (root / row['image']['path']).resolve()
        assert image_path.is_relative_to(root.resolve())
        data = image_path.read_bytes()
        assert len(data) == row['image']['bytes']
        assert hashlib.sha256(data).hexdigest() == row['image']['sha256']
        assert row['category'] != 'MATCHED_OBJECT'
        assert row['candidate_count'] == len(row['candidates'])
    fontpath=str(font_path)
    def font(n):return ImageFont.truetype(fontpath,n)
    W,H=900,555; margin=16; header=100
    canvas=Image.new('RGB',(2*W+3*margin,4*H+5*margin+header),'white');draw=ImageDraw.Draw(canvas)
    draw.text((margin,15),'Natural99: 박스 미매칭 8장 전체 / 기존 R0 출력 그대로',font=font(26),fill='#172536')
    draw.text((margin,53),'주황=원래 선택(#0), 파랑=나머지 모든 저장 후보, 초록 점선=사후 참조 박스(추론에 미사용)',font=font(21),fill='#172536')
    draw.text((margin,80),'원본 640×480 전체 표시. IoU와 T/R은 기존 진단 재인용. 후보 교체·추론·학습 0회.',font=font(17),fill='#354352')
    for j,r in enumerate(odd):
     x=margin+(j%2)*(W+margin);y=header+margin+(j//2)*(H+margin)
     draw.text((x,y),r['id'],font=font(19),fill='#101010')
     draw.text((x,y+26),f"{r['recording']} | T {r['T_cm']:.2f} cm / R {r['R_deg']:.2f}° | {r['candidate_count']} candidates",font=font(17),fill='#333333')
     im=Image.open(root/r['image']['path']).convert('RGB');assert im.size==(640,480);di=ImageDraw.Draw(im)
     for c in reversed(r['candidates']):
      box=c['box_xyxy'];color='#ff861b' if c['index']==r['selected_index'] else '#328bff';di.rectangle(box,outline=color,width=4 if c['index']==r['selected_index'] else 2)
      tx=max(0,min(610,int(box[0])));ty=max(0,min(452,int(box[1])-20));label=str(c['index'])
      di.rectangle((tx,ty,tx+22,ty+21),fill=color);di.text((tx+3,ty),label,font=font(16),fill='black')
     rx1,ry1,rx2,ry2=r['reference_box_xyxy']
     for xa,ya,xb,yb in ((rx1,ry1,rx2,ry1),(rx2,ry1,rx2,ry2),(rx2,ry2,rx1,ry2),(rx1,ry2,rx1,ry1)):
      length=float(np.hypot(xb-xa,yb-ya))
      for pos in np.arange(0,length,16):
       t1=pos/max(length,1);t2=min(pos+9,length)/max(length,1)
       di.line((xa+t1*(xb-xa),ya+t1*(yb-ya),xa+t2*(xb-xa),ya+t2*(yb-ya)),fill='#18da84',width=3)
     canvas.paste(im,(x,y+57));sx=x+652
     draw.text((sx,y+58),'idx  score       IoU(ref)',font=font(15),fill='#333333')
     for n,c in enumerate(r['candidates']):draw.text((sx,y+88+25*n),f"{c['index']:>2}   {c['score']:.6f}   {c['reference_IoU']:.3f}",font=font(16),fill='#b34d00' if c['index']==r['selected_index'] else '#235b96')
     text=['후보에 참조 매칭 있음','원래 선택은 미매칭'] if r['category']=='CORRECT_BOX_IN_POOL_WRONG_SELECTION' else ['저장 후보 전부','참조 IoU < 0.5']
     for n,line in enumerate(text):draw.text((sx,y+320+24*n),line,font=font(18),fill='#192b36')
     visual=['선택: 전경 콘 받침'] if r['recording']=='REC_027' else ['선택: 뒤집힌 콘의','사각 스티커/패치']
     for n,line in enumerate(visual):draw.text((sx,y+393+24*n),line,font=font(17),fill='#333333')
     draw.text((sx,y+477),'관찰용 / routing 없음',font=font(16),fill='#555555')
    with output.open('xb') as handle:
     canvas.save(handle,format='JPEG',quality=94,subsampling=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=DOC / 'ASSOCIATION_AUDIT.json')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--font', type=Path, default=Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf'))
    parser.add_argument('--output', type=Path, required=True, help='New JPG path; existing files are refused.')
    args = parser.parse_args()
    render(args.evidence, args.root, args.font, args.output)


if __name__ == '__main__':
    main()

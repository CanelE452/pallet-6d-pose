"""Human role review: no model files read; randomized mapping remains private."""
import argparse
import random
import tkinter as tk
from tkinter import messagebox
from datetime import datetime,timezone
import numpy as np
from PIL import Image,ImageTk,ImageDraw,ImageFont
from . import common as C

def prepare():
    candidates=C.read(C.DOC/'C4_CANDIDATES.json')['candidates']
    shuffled=candidates.copy();random.Random(20260924).shuffle(shuffled)
    mapping={letter:c['c4'] for letter,c in zip('ABCD',shuffled)}
    C.put(C.RAW/'BLIND_MAPPING_PRIVATE.json',dict(seed=20260924,mapping=mapping,
        annotation=C.bind(C.ANN),geometry=C.bind(C.DOC/'GEOMETRY_ONLY_CANDIDATES.json')))
    q=C.points();direct=C.direct_mask();raw=Image.open(C.RGB).convert('RGB')
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    images=[]
    for letter,c in zip('ABCD',shuffled):
        # Same canvas includes off-image coordinates. No marker is drawn for PnP-generated points.
        panel=Image.new('RGB',(780,480),(40,40,40));panel.paste(raw,(0,0));draw=ImageDraw.Draw(panel)
        p=c['perm_new_to_stored'];v=q[p];m=direct[p]
        for a,b in [(0,1),(1,2),(2,3),(3,0)]:
            start,end=v[a],v[b]
            if m[a] and m[b]:draw.line([tuple(start),tuple(end)],fill='#ffcc44',width=3)
            else:
                for t in np.arange(0,1,.07):
                    draw.line([tuple(start+(end-start)*t),tuple(start+(end-start)*min(t+.035,1))],fill='#ffcc44',width=2)
        for j in range(8):
            x,y=v[j];label=f'P{j}'+('' if m[j] else '*')
            tx=min(max(x+7,4),725);ty=min(max(y-24,4),450)
            box=draw.textbbox((tx,ty),label,font=font);draw.rectangle(box,fill='#101010');draw.text((tx,ty),label,fill='white',font=font)
            if m[j]:draw.ellipse((x-5,y-5,x+5,y+5),outline='white',width=2)
        draw.rectangle((0,0,780,30),fill='#151515')
        draw.text((8,4),f'{letter}   circles: direct clicks | * and dashed edges: inferred, not clicked',fill='white',font=font)
        # Zoom same target ROI, do not change image coordinates or fitted geometry.
        panel=panel.crop((320,170,780,380)).resize((690,315))
        header=Image.new('RGB',(690,342),'#151515');header.paste(panel,(0,27))
        ImageDraw.Draw(header).text((10,4),f'Candidate {letter}   (* = inferred role location)',fill='white',font=font)
        C.OUT.mkdir(parents=True,exist_ok=True);header.save(C.OUT/f'candidate_{letter}.png');images.append(header)
    sheet=Image.new('RGB',(1380,684),'#151515')
    for i,im in enumerate(images):sheet.paste(im,((i%2)*690,(i//2)*342))
    C.FIG.mkdir(parents=True,exist_ok=True);sheet.save(C.FIG/'03_c4_candidates_blind.jpg',quality=93)
    neutral=raw.copy();draw=ImageDraw.Draw(neutral)
    for x,y in q[direct]:draw.ellipse((x-5,y-5,x+5,y+5),outline='white',width=2)
    neutral.save(C.FIG/'02_011067_raw_manual_points.jpg',quality=93)
    C.put(C.RAW/'HUMAN_UI_LOCK.json',dict(mapping_sha256=C.sha(C.RAW/'BLIND_MAPPING_PRIVATE.json'),
        blind_figure_sha256=C.sha(C.FIG/'03_c4_candidates_blind.jpg'),
        code_sha256=C.sha(C.ROOT/'scripts/research/pallet_011067_corner_contract_v1/review_front_role.py'),
        models_displayed=False,legacy_current_label_displayed=False,
        direct_circles_only=True,inferred_role_text_and_dashed_outline_disclosed=True))

class Review:
    def __init__(self,root):
        self.root=root;self.choice=tk.StringVar(value='');self.confidence=tk.StringVar(value='')
        root.title('011067 앞면 역할 확인 — 예측 숨김 / 점 입력 없음')
        width=min(root.winfo_screenwidth()-80,1450);height=min(root.winfo_screenheight()-100,930)
        root.geometry(f'{width}x{height}')
        tk.Label(root,text='0–3은 카메라 쪽 앞면: 0=위 왼쪽, 1=위 오른쪽, 3=아래 왼쪽, 2=아래 오른쪽\n흰 원만 직접 클릭입니다. *번호·점선은 PnP 보완 위치라 독립 근거가 아닙니다. 애매하면 모호함을 선택하세요.',font=('Sans',12),justify='left').pack()
        source=Image.open(C.FIG/'03_c4_candidates_blind.jpg');source.thumbnail((width-20,height-230))
        self.image=ImageTk.PhotoImage(source);tk.Label(root,image=self.image).pack()
        row=tk.Frame(root);row.pack(pady=8)
        for text,value in [('A','A'),('B','B'),('C','C'),('D','D'),('모호함 / 구분 불가','AMBIGUOUS')]:
            tk.Radiobutton(row,text=text,variable=self.choice,value=value,font=('Sans',13)).pack(side='left',padx=10)
        row=tk.Frame(root);row.pack()
        for text,value in [('확신 있음','CONFIDENT'),('확신 없음','UNCERTAIN')]:
            tk.Radiobutton(row,text=text,variable=self.confidence,value=value,font=('Sans',12)).pack(side='left',padx=12)
        tk.Button(root,text='선택 저장',command=self.save,font=('Sans',13)).pack(pady=7)
        self.footer=tk.Label(root,text='자동 선택 없음. 배치와 확신 정도를 선택한 뒤 저장하세요.');self.footer.pack()
    def save(self):
        if not self.choice.get() or not self.confidence.get():
            messagebox.showinfo('선택 필요','배치(또는 모호함)와 확신 정도를 모두 선택하세요.');return
        if (C.RAW/'HUMAN_DECISION_PRIVATE.json').exists():
            messagebox.showinfo('저장됨','이미 저장된 결정을 덮어쓰지 않습니다. 수정이 필요하면 CLI에 알려주세요.');return
        C.put(C.RAW/'HUMAN_DECISION_PRIVATE.json',dict(choice=self.choice.get(),confidence=self.confidence.get(),
            utc=datetime.now(timezone.utc).isoformat(),input='human GUI selection',
            ui_lock_sha256=C.sha(C.RAW/'HUMAN_UI_LOCK.json')))
        self.footer.config(text='저장 완료. CLI에 완료했다고 알려주세요. GT나 모델은 변경하지 않았습니다.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if not (C.RAW/'HUMAN_UI_LOCK.json').exists():prepare()
    if a.prepare:return
    assert C.sha(C.FIG/'03_c4_candidates_blind.jpg')==C.read(C.RAW/'HUMAN_UI_LOCK.json')['blind_figure_sha256']
    root=tk.Tk();app=Review(root);root.update()
    if a.smoke:
        assert app.choice.get()=='' and app.confidence.get()=='';root.destroy();print('PASS UI no automatic human decision')
    else:root.mainloop()

if __name__=='__main__':main()

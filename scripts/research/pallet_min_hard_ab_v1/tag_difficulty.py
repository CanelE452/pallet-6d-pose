"""Raw-only human tagging. Saving a keyboard answer advances immediately."""
import argparse
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from . import common as C

GUIDE = '''C  깨끗함 / CLEAN
외부 가림·잘림이 없거나 매우 경미.
주요 모서리와 앞면 구조가 명확함.

M  중간 / MODERATE
가림·잘림·강한 시점 때문에 일부 모서리가
애매하지만 대부분의 앞면/윗면 구조는 보임.
일부 보이는 꼭짓점은 확신 있게 찍을 수 있음.

S  심함 / SEVERE
여러 주요 모서리가 안 보이거나 잘려 있고
구조 복원이 어려움. 일부 점은 직접 보임.

U  판단 불가 / UNCERTAIN
중간·심함 또는 앞면 역할을 확신 못함.

X  사용 불가 / INVALID
팔레트가 없거나 흐림·손상으로 주석 불가.

모델이 틀릴 것 같은지가 아니라
지금 보이는 원본 영상만 판단하세요.
점/박스는 찍지 않습니다.

키 한 번 → 자동 저장 → 다음 이미지
Z: 이번 라운드의 직전 응답 취소
창 닫기: 저장됨, 다음 실행에서 이어하기'''
KEYS = dict(c='CLEAN',m='MODERATE',s='SEVERE',u='UNCERTAIN',x='INVALID')
PHYSICAL = {54:'c',58:'m',39:'s',30:'u',53:'x',52:'z'}

class TagStore:
    def __init__(self, rows, round_no, path=None):
        self.rows = rows
        self.round = round_no
        self.path = path or C.RAW/'DIFFICULTY_TAGS_PRIVATE.json'
        self.queue_sha = C.sha(C.RAW/'DIFFICULTY_QUEUE_PRIVATE.json')
        self.data = C.read(self.path) if self.path.exists() else dict(queue_sha256=self.queue_sha,responses={},events=[])
        assert self.data['queue_sha256'] == self.queue_sha
        valid = {r['frame_id'] for r in C.queue()}
        assert set(self.data['responses']) <= valid
        assert all(r['tag'] in C.TAGS for r in self.data['responses'].values())
    def next_index(self):
        return next((i for i,r in enumerate(self.rows) if r['frame_id'] not in self.data['responses']),None)
    def put(self, index, tag):
        assert tag in C.TAGS
        r = self.rows[index]; fid = r['frame_id']
        assert fid not in self.data['responses']
        self.data['responses'][fid] = dict(tag=tag,round=self.round,at=C.now(),image_sha256=r['image']['sha256'])
        self.data['events'].append(dict(action='TAG',frame_id=fid,tag=tag,at=C.now()))
        C.save(self.path,self.data)
    def undo(self):
        permitted = {r['frame_id'] for r in self.rows}
        chosen = [k for k in self.data['responses'] if k in permitted]
        if not chosen:
            return None
        fid = chosen[-1]; old = self.data['responses'].pop(fid)
        self.data['events'].append(dict(action='UNDO',frame_id=fid,previous=old,at=C.now()))
        C.save(self.path,self.data)
        return next(i for i,r in enumerate(self.rows) if r['frame_id']==fid)

class App:
    def __init__(self, root, store, smoke=False):
        self.root=root;self.store=store;self.smoke=smoke;self.index=store.next_index()
        self.down=set();self.release_jobs={};self.photo=None;self.image=None
        root.title('Hard 난도 태깅 — 원본 RGB만 · C/M/S/U/X 입력 즉시 다음')
        root.geometry('1450x900');root.minsize(950,650)
        self.header=tk.Label(root,font=('Sans',15,'bold'),anchor='w');self.header.pack(fill='x',padx=12,pady=10)
        panel=tk.Frame(root);panel.pack(fill='both',expand=True)
        self.canvas=tk.Canvas(panel,bg='#17212a',highlightthickness=0);self.canvas.pack(side='left',fill='both',expand=True)
        right=tk.Frame(panel,width=330);right.pack(side='right',fill='y',padx=10)
        tk.Label(right,text=GUIDE,justify='left',anchor='nw',font=('Sans',11)).pack(fill='both',expand=True)
        for k,label in [('c','C 깨끗함'),('m','M 중간'),('s','S 심함'),('u','U 모름'),('x','X 사용 불가'),('z','Z 되돌리기')]:
            tk.Button(right,text=label,command=lambda k=k:self.answer(k),font=('Sans',12)).pack(fill='x',pady=2)
        self.footer=tk.Label(root,text='모델·GT·PnP·기존 난도 태그를 표시하지 않습니다.',font=('Sans',11));self.footer.pack(fill='x',pady=8)
        self.canvas.bind('<Configure>',lambda e:self.draw())
        root.bind('<KeyPress>',self.keypress);root.bind('<KeyRelease>',self.keyrelease)
        root.protocol('WM_DELETE_WINDOW',root.destroy)
        self.load()
        if not smoke:
            root.after(150,lambda:(root.lift(),root.focus_force()))
    def keypress(self,event):
        code=event.keycode
        if code in self.release_jobs:
            self.root.after_cancel(self.release_jobs.pop(code))
        if code in self.down:
            return
        self.down.add(code)
        key=event.keysym.lower()
        if key not in KEYS and key!='z':
            key=PHYSICAL.get(code,'') # Korean keyboard input mode uses the same physical keys.
        self.answer(key)
    def keyrelease(self,event):
        code=event.keycode
        self.release_jobs[code]=self.root.after_idle(lambda:self.down.discard(code))
    def answer(self,key):
        if key=='z':
            i=self.store.undo()
            if i is not None:self.index=i;self.load()
        elif key in KEYS and self.index is not None:
            self.store.put(self.index,KEYS[key]);self.index=self.store.next_index();self.load()
    def load(self):
        if self.index is None:
            self.header.config(text=f'라운드 {self.store.round} 완료 — {len(self.store.rows)}장 저장됨')
            self.canvas.delete('all');self.image=None
            self.footer.config(text='완료했습니다. 창을 닫고 cli resume을 실행하거나 대화에 “다 했어”라고 알려주세요. Z로 마지막 응답 수정 가능.')
            return
        row=self.store.rows[self.index];C.verify(row['image'])
        with Image.open(C.ROOT/row['image']['path']) as im:self.image=im.convert('RGB')
        recs=sorted({r['recording'] for r in C.queue()})
        n=sum(r['frame_id'] in self.store.data['responses'] for r in self.store.rows)
        self.header.config(text=f'라운드 {self.store.round} · {self.index+1}/{len(self.store.rows)} · 저장 {n}장 · 촬영 {recs.index(row["recording"])+1:02d}')
        self.draw()
    def draw(self):
        if self.image is None:return
        w=max(1,self.canvas.winfo_width());h=max(1,self.canvas.winfo_height())
        scale=min(w/self.image.width,h/self.image.height)
        self.photo=ImageTk.PhotoImage(self.image.resize((max(1,int(self.image.width*scale)),max(1,int(self.image.height*scale))),Image.Resampling.LANCZOS))
        self.canvas.delete('all');self.canvas.create_image(w/2,h/2,image=self.photo)

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
    state=C.state()
    if state['status']!='WAITING_FOR_HUMAN_DIFFICULTY_TAGS':
        raise SystemExit('No open difficulty round: '+state['status'])
    rows=[r for r in C.queue() if r['round']==state['round']]
    with C.exclusive('tagging'):
        root=tk.Tk()
        if a.smoke:
            C.OUT.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='hard_tag_smoke_',dir=C.OUT) as tmp:
                store=TagStore(rows,state['round'],Path(tmp)/'tags.json');app=App(root,store,smoke=True);root.update()
                first=app.index;event=SimpleNamespace(keysym='m',keycode=58)
                app.keypress(event);assert app.index!=first
                after=app.index;app.keypress(event);assert app.index==after,'Held key must not tag multiple images'
                app.keyrelease(event);root.update()
                assert store.data['responses'][rows[first]['frame_id']]['tag']=='MODERATE'
                app.answer('z');assert app.index==first and not store.data['responses']
                app.answer('x');assert store.next_index()!=first
                resumed=TagStore(rows,state['round'],store.path);assert resumed.next_index()==app.index
                root.destroy();print('PASS GUI raw image / keypress without click / autorepeat guard / immediate advance / undo / resume; no real tags changed')
        else:
            App(root,TagStore(rows,state['round']));root.mainloop()

if __name__=='__main__':main()

"""Local model/reference-blind Tk UI. Only raw RGB, own clicks and canonical guide."""
import copy
import json
import argparse
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from . import common as C

LABEL_NAMES = ('D 직접 보임 (클릭 필수)', 'V 직선 연장 추정 (클릭 선택)',
               'S 자체 가림', 'E 외부 물체 가림', 'O 화면 밖', 'U 판단 불가')

class App:
    def __init__(self, root):
        self.root=root
        self.selection=C.read(C.RAW/'ANCHOR_SELECTION.json')
        binding=C.read(C.DOC/'INPUT_BINDINGS.json')
        assert C.sha(C.RAW/'ANCHOR_SELECTION.json')==binding['selection_sha256']
        if (C.RAW/'FIRST_PASS_LOCK.json').exists():
            raise RuntimeError('First pass locked. Do not edit: resume coverage/QA in a separate pass.')
        path=C.RAW/'LABELS.json'
        self.labels=C.read(path if path.exists() else C.RAW/'LABEL_TEMPLATE.json')
        assert self.labels['selection_sha256']==binding['selection_sha256']
        self.index=0; self.corner=0; self.undo_stack=[]; self.scale=1.; self.offset=[0.,0.]
        root.title('Pallet verified anchor — 모델/정답 숨김 · 학습 없음')
        root.geometry('1450x950')
        toolbar=tk.Frame(root);toolbar.pack(fill='x')
        tk.Label(toolbar,text='이름 (선택):').pack(side='left')
        self.name=tk.StringVar(value=self.labels['annotator'])
        tk.Entry(toolbar,textvariable=self.name,width=18).pack(side='left')
        tk.Button(toolbar,text='이전',command=lambda:self.navigate(-1)).pack(side='left')
        tk.Button(toolbar,text='다음',command=lambda:self.navigate(1)).pack(side='left')
        tk.Button(toolbar,text='전체 보기 (F)',command=self.fit).pack(side='left')
        tk.Button(toolbar,text='되돌리기 Ctrl+Z',command=self.undo).pack(side='left')
        tk.Button(toolbar,text='완료 확인',command=self.finish).pack(side='left')
        self.header=tk.Label(root,anchor='w');self.header.pack(fill='x')
        main=tk.Frame(root);main.pack(fill='both',expand=True)
        self.canvas=tk.Canvas(main,bg='#20252b',highlightthickness=0);self.canvas.pack(side='left',fill='both',expand=True)
        panel=tk.Frame(main,width=360);panel.pack(side='right',fill='y');panel.pack_propagate(False)
        ref=Image.open(C.DOC/'figures/corner_index_reference.png');ref.thumbnail((355,245))
        self.ref_photo=ImageTk.PhotoImage(ref)
        tk.Label(panel,image=self.ref_photo).pack()
        tk.Label(panel,text='가까운 면 P0~P3 / 반대편 P4~P7\n윗면 P0·P1·P4·P5\n번호가 애매하면 U: 판단 불가',justify='left').pack()
        self.buttons=[]
        for i in range(8):
            b=tk.Button(panel,anchor='w',command=lambda i=i:self.choose(i));b.pack(fill='x');self.buttons.append(b)
        for j,title in enumerate(LABEL_NAMES):
            tk.Button(panel,text=title,command=lambda j=j:self.status(j)).pack(fill='x',pady=1)
        tk.Label(panel,text='D·V·S·E·O·U 입력 → 다음 번호 이동\n상태를 먼저 연속 입력할 수 있습니다\n좌표: 0~7로 해당 점을 다시 선택 후 클릭\nD는 좌표 필수 / V는 좌표 선택\n휠: 확대/축소 · 오른쪽 드래그: 이동\n숨은 점 추측 클릭 금지 · P8 제외\n모든 변경 자동 저장',justify='left').pack(pady=8)
        self.footer=tk.Label(root,anchor='w');self.footer.pack(fill='x')
        self.canvas.bind('<Button-1>',self.click)
        self.canvas.bind('<Button-3>',self.pan_start)
        self.canvas.bind('<B3-Motion>',self.pan_move)
        self.canvas.bind('<Button-4>',lambda e:self.zoom(e,1.2))
        self.canvas.bind('<Button-5>',lambda e:self.zoom(e,1/1.2))
        self.canvas.bind('<MouseWheel>',lambda e:self.zoom(e,1.2 if e.delta>0 else 1/1.2))
        root.bind('<Key>',self.key)
        root.bind('<Control-z>',lambda e:self.undo())
        root.protocol('WM_DELETE_WINDOW',self.close)
        root.update_idletasks();self.load()

    def record(self): return self.labels['frames'][self.index]
    def item(self): return self.record()['corners'][self.corner]

    def load(self):
        r=self.selection['frames'][self.index];path=C.ROOT/r['image']['path']
        assert C.sha(path)==r['image']['sha256'], 'Original image hash changed'
        assert self.record()['frame_id']==r['frame_id']
        self.original=Image.open(path).convert('RGB');self.fit()

    def fit(self):
        if not hasattr(self,'original'): return
        w,h=self.original.size;cw,ch=self.canvas.winfo_width(),self.canvas.winfo_height()
        self.scale=min(cw/w,ch/h)*.96;self.offset=[(cw-w*self.scale)/2,(ch-h*self.scale)/2];self.draw()

    def draw(self):
        self.canvas.delete('all')
        cw,ch=self.canvas.winfo_width(),self.canvas.winfo_height()
        # Render the viewport instead of allocating enormous images at high zoom.
        im=self.original.transform((max(1,cw),max(1,ch)),Image.Transform.AFFINE,
            (1/self.scale,0,-self.offset[0]/self.scale,0,1/self.scale,-self.offset[1]/self.scale),
            resample=Image.Resampling.BILINEAR,fillcolor='#20252b')
        self.photo=ImageTk.PhotoImage(im);self.canvas.create_image(0,0,image=self.photo,anchor='nw')
        for c in self.record()['corners']:
            if c['xy'] is None:continue
            x,y=c['xy'];x=x*self.scale+self.offset[0];y=y*self.scale+self.offset[1]
            col='#44ff88' if c['status']=='DIRECT_VISIBLE' else '#ffb344'
            self.canvas.create_oval(x-5,y-5,x+5,y+5,outline=col,width=2)
            self.canvas.create_text(x+12,y-12,text=f'P{c["id"]}',fill=col,font=('Sans',14,'bold'))
        for i,b in enumerate(self.buttons):
            c=self.record()['corners'][i];s=c['status'] or '미확인'
            if s=='DIRECT_VISIBLE' and c['xy'] is None:s+=' — 클릭 필요'
            b.config(text=f'{"▶" if i==self.corner else "  "} P{i}: {s}',bg='#bbddff' if i==self.corner else '#eeeeee')
        r=self.selection['frames'][self.index];p=C.progress(self.labels,self.selection)
        self.header.config(text=f'{self.index+1}/18 · {r["severity"]} · {r["recording"]} · 현재 P{self.corner} · 원본 해상도 {self.original.size}')
        self.footer.config(text=f'완료 {p["completed_images"]}/18장 · 상태 {p["completed_statuses"]}/144 · 내 입력만 표시 · 변경 자동 저장')

    def checkpoint(self):
        self.undo_stack.append((self.index,self.corner,copy.deepcopy(self.record())))
        return True

    def save(self,action):
        self.labels['annotator']=self.name.get().strip() or 'anonymous_local'
        self.record()['annotator']=self.labels['annotator']
        # Append-only human action log preserves clicks, status changes, undo and timestamps.
        with (C.RAW/'HUMAN_HISTORY.jsonl').open('a') as f:
            f.write(json.dumps(dict(time=C.now(),action=action,frame=copy.deepcopy(self.record())),ensure_ascii=False)+'\n')
        C.atomic(C.RAW/'LABELS.json',self.labels)
        C.atomic(C.RAW/'ANNOTATION_PROGRESS.json',C.progress(self.labels,self.selection))
        self.draw()

    def choose(self,i):
        self.canvas.focus_set();self.corner=i;self.draw()
    def advance(self):
        # Stay on P7 for review; never silently jump to another image.
        self.corner=min(7,self.corner+1);self.draw()
    def status(self,j):
        if not self.checkpoint():return
        self.canvas.focus_set()
        c=self.item();c['status']=C.STATUSES[j]
        if j>=2:c['xy']=None
        self.save('status')
        self.advance()

    def click(self,e):
        self.canvas.focus_set()
        if self.item()['status'] not in (None,'DIRECT_VISIBLE','VIRTUAL_INFERABLE'):
            self.footer.config(text='먼저 D(직접 보임) 또는 V(추정 가능)를 선택하세요. 가려진 점은 클릭하지 않습니다.');return
        x=(e.x-self.offset[0])/self.scale;y=(e.y-self.offset[1])/self.scale
        w,h=self.original.size
        if not (0<=x<w and 0<=y<h):return
        if not self.checkpoint():return
        if self.item()['status'] is None:self.item()['status']='DIRECT_VISIBLE'
        self.item()['xy']=[round(x,3),round(y,3)];self.save('click');self.advance()

    def navigate(self,delta):
        self.index=max(0,min(len(self.labels['frames'])-1,self.index+delta));self.corner=0;self.load()
    def undo(self):
        if not self.undo_stack:return
        self.index,self.corner,old=self.undo_stack.pop();self.labels['frames'][self.index]=old;self.load();self.save('undo')
    def pan_start(self,e):self.drag=(e.x,e.y,*self.offset)
    def pan_move(self,e):
        x,y,ox,oy=self.drag;self.offset=[ox+e.x-x,oy+e.y-y];self.draw()
    def zoom(self,e,factor):
        new=max(.1,min(15.,self.scale*factor));ratio=new/self.scale
        self.offset=[e.x-(e.x-self.offset[0])*ratio,e.y-(e.y-self.offset[1])*ratio];self.scale=new;self.draw()
    def key(self,e):
        if isinstance(self.root.focus_get(),tk.Entry):return
        k=e.keysym.lower()
        if k in list('01234567'):self.choose(int(k))
        elif k in ('d','v','s','e','o','u'):self.status(('d','v','s','e','o','u').index(k))
        elif k=='f':self.fit()
        elif k=='right':self.navigate(1)
        elif k=='left':self.navigate(-1)
    def finish(self):
        p=C.progress(self.labels,self.selection)
        if p['completed_images']<18:
            messagebox.showinfo('아직 확인할 점이 있습니다',f'{p["completed_images"]}/18장 완료. 각 이미지의 8개 상태를 확인하세요.');return
        messagebox.showinfo('첫 확인 입력 완료','입력은 자동 저장됐습니다. CLI에 완료했다고 알려주세요.\n다음 단계: 첫 입력 잠금 → coverage 확인 → 필요한 QA.\n아직 평가하거나 학습하지 않습니다.')
    def close(self):self.root.destroy()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    root=tk.Tk();app=App(root)
    if args.smoke:
        root.update();print('UI_SMOKE_OK',root.winfo_width(),root.winfo_height(),flush=True);root.destroy()
    else:root.mainloop()

if __name__=='__main__':main()

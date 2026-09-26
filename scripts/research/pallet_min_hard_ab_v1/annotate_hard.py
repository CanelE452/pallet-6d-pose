"""Manual bbox + visible corners only, no predictions, legacy labels or PnP."""
import copy
import argparse
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tkinter as tk
from tkinter import messagebox
from PIL import Image,ImageTk
from . import common as C
from .labels import validate_frame

class App:
    def __init__(self,root):
        self.root=root;self.state=C.state();C.verify(C.read(C.DOC/'HARD_SELECTION_LOCK.json')['private_selection'])
        self.rows=C.read(C.RAW/'HARD_SELECTION_PRIVATE.json')['rows'][:self.state['active_frames']]
        self.path=C.annotation_labels_path();self.locksha=C.sha(C.DOC/'HARD_SELECTION_LOCK.json')
        self.data=C.read(self.path) if self.path.exists() else dict(selection_sha256=self.locksha,frames={},history=[])
        assert self.data['selection_sha256']==self.locksha
        self.index=next((i for i,r in enumerate(self.rows) if not self.data['frames'].get(r['frame_id'],{}).get('complete')),0)
        self.corner=0;self.mode='bbox';self.drag=None;self.undo_stack=[];self.scale=1.;self.offset=(0,0)
        self.zoom=1.;self.pan=(0.,0.);self.pan_start=None
        root.title('Hard visible 수동 GT — 직접 보이는 점만 · PnP/예측 없음');root.geometry('1480x920')
        self.header=tk.Label(root,font=('Sans',14,'bold'));self.header.pack(fill='x')
        body=tk.Frame(root);body.pack(fill='both',expand=True)
        self.canvas=tk.Canvas(body,bg='#15222c');self.canvas.pack(side='left',fill='both',expand=True)
        right=tk.Frame(body,width=335);right.pack(side='right',fill='y')
        # Static 2D index schematic, never projected from the current image.
        guide=tk.Canvas(right,width=310,height=180,bg='white');guide.pack()
        uv=[(35,95),(210,115),(210,155),(35,135),(110,25),(285,45),(285,85),(110,65)]
        for a,b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:guide.create_line(*uv[a],*uv[b],fill='#789')
        for i,(x,y) in enumerate(uv):guide.create_text(x,y-9,text=f'P{i}',fill='#b64c00' if i<4 else '#1675b0')
        tk.Label(right,text='번호 안내 그림일 뿐 정답 자세가 아닙니다.\n가까운 면 P0~3 / 먼 면 P4~7\nC2 180° 동치 ≠ 90° 역할 재배치\n앞면이 모호하면 역할 불확실로 두세요.',justify='left').pack()
        for text,func in [('역할 확실 (C)',lambda:self.role('ROLE_CONFIDENT')),('역할 불확실 — 학습 제외',lambda:self.role('ROLE_UNCERTAIN')),('박스 다시 드래그 (B)',lambda:self.set_mode('bbox'))]:tk.Button(right,text=text,command=func).pack(fill='x')
        tk.Label(right,text='박스: 관찰되는 팔레트의 타이트한 범위.\n화면 밖/가려진 외곽을 추정하지 마세요.\n점은 클릭 즉시 다음 번호로 이동합니다.\n연장선 추정점은 찍지 마세요.\n휠 확대 · 우클릭 드래그 이동 · F 전체 보기',justify='left').pack(pady=6)
        corner_panel=tk.Frame(right);corner_panel.pack(fill='x')
        for i in range(8):tk.Button(corner_panel,text=f'P{i}',command=lambda i=i:self.choose(i)).grid(row=i//4,column=i%4,sticky='ew')
        for i in range(4):corner_panel.columnconfigure(i,weight=1)
        for text,status in [('O 가려짐 / 자체 가림','OCCLUDED'),('X 화면 밖','OUT_OF_FRAME'),('U 점 위치 모호','UNCERTAIN')]:tk.Button(right,text=text,command=lambda s=status:self.mark(s)).pack(fill='x')
        tk.Button(right,text='Z 되돌리기',command=self.undo).pack(fill='x')
        tk.Label(right,text='역할 관련 메모 (선택)').pack();self.note=tk.Entry(right);self.note.pack(fill='x')
        tk.Button(right,text='이 이미지 입력 완료 (Enter)',command=self.finish).pack(fill='x',pady=5)
        self.qa_button=tk.Button(right,text='2차 QA: 내 입력 재확인 완료',command=self.qa_confirm)
        if self.state['status']=='WAITING_FOR_HUMAN_HARD_QA':self.qa_button.pack(fill='x')
        nav=tk.Frame(root);nav.pack(fill='x')
        tk.Button(nav,text='이전 이미지',command=lambda:self.move(-1)).pack(side='left');tk.Button(nav,text='다음 이미지',command=lambda:self.move(1)).pack(side='left')
        self.footer=tk.Label(nav,anchor='w');self.footer.pack(side='left',fill='x',expand=True)
        self.canvas.bind('<Configure>',lambda e:self.draw());self.canvas.bind('<ButtonPress-1>',self.click)
        self.canvas.bind('<ButtonRelease-1>',self.release);root.bind('<KeyPress>',self.key)
        self.canvas.bind('<Button-4>',lambda e:self.zoom_at(e,1.25));self.canvas.bind('<Button-5>',lambda e:self.zoom_at(e,.8))
        self.canvas.bind('<MouseWheel>',lambda e:self.zoom_at(e,1.25 if e.delta>0 else .8))
        self.canvas.bind('<ButtonPress-3>',lambda e:setattr(self,'pan_start',(e.x,e.y,self.pan)))
        self.canvas.bind('<B3-Motion>',self.pan_move)
        self.load()
    def frame(self):return self.data['frames'][self.rows[self.index]['frame_id']]
    def checkpoint(self):self.undo_stack.append(copy.deepcopy(self.data));self.frame()['qa_confirmed']=False;self.frame()['complete']=False
    def save(self,action):
        self.data['history'].append(dict(frame_id=self.rows[self.index]['frame_id'],action=action,at=C.now()))
        C.save(self.path,self.data)
    def load(self):
        self.zoom=1.;self.pan=(0.,0.)
        r=self.rows[self.index];C.verify(r['image'])
        with Image.open(C.ROOT/r['image']['path']) as im:self.im=im.convert('RGB')
        self.data['frames'].setdefault(r['frame_id'],dict(image_sha256=r['image']['sha256'],size=list(self.im.size),role=None,bbox=None,
                corners=[dict(status=None,xy=None) for _ in range(8)],complete=False,qa_confirmed=False))
        self.corner=next((i for i,p in enumerate(self.frame()['corners']) if p['status'] is None),0)
        self.mode='bbox' if self.frame()['bbox'] is None else 'point'
        self.note.delete(0,'end');self.note.insert(0,self.frame().get('role_note',''));self.draw()
    def draw(self):
        if not hasattr(self,'im'):return
        w=max(1,self.canvas.winfo_width());h=max(1,self.canvas.winfo_height());self.scale=min(w/self.im.width,h/self.im.height)*self.zoom
        self.offset=((w-self.im.width*self.scale)/2+self.pan[0],(h-self.im.height*self.scale)/2+self.pan[1])
        viewport=self.im.transform((w,h),Image.Transform.AFFINE,(1/self.scale,0,-self.offset[0]/self.scale,0,1/self.scale,-self.offset[1]/self.scale),resample=Image.Resampling.BICUBIC,fillcolor='#15222c')
        self.photo=ImageTk.PhotoImage(viewport);self.canvas.delete('all');self.canvas.create_image(0,0,anchor='nw',image=self.photo)
        def pt(x,y):return self.offset[0]+x*self.scale,self.offset[1]+y*self.scale
        f=self.frame()
        if f['bbox']:self.canvas.create_rectangle(*pt(*f['bbox'][:2]),*pt(*f['bbox'][2:]),outline='#f5c542',width=2)
        for i,p in enumerate(f['corners']):
            if p['xy'] is not None:
                x,y=pt(*p['xy']);self.canvas.create_oval(x-4,y-4,x+4,y+4,outline='#00ddff',width=2);self.canvas.create_text(x+10,y-10,text=f'P{i}',fill='#00ddff')
        self.header.config(text=f'{self.index+1}/{len(self.rows)} · {f["role"] or "먼저 역할을 선택하세요"} · '+('박스 드래그' if self.mode=='bbox' else f'P{self.corner}: 직접 보이면 클릭 / O·X·U는 입력 즉시 다음'))
        self.footer.config(text=' | '.join(f'P{i}:{p["status"] or "미입력"}' for i,p in enumerate(f['corners'])))
    def role(self,value):
        self.checkpoint();self.frame()['role']=value
        if value=='ROLE_UNCERTAIN':self.frame()['bbox']=None;self.frame()['corners']=[dict(status='UNCERTAIN',xy=None) for _ in range(8)]
        self.save('ROLE');self.draw()
    def set_mode(self,mode):self.mode=mode;self.draw()
    def choose(self,k):self.corner=k;self.mode='point';self.draw()
    def coords(self,e):return ((e.x-self.offset[0])/self.scale,(e.y-self.offset[1])/self.scale)
    def zoom_at(self,e,factor):
        x,y=self.coords(e);self.zoom=max(1.,min(12.,self.zoom*factor))
        w=self.canvas.winfo_width();h=self.canvas.winfo_height();scale=min(w/self.im.width,h/self.im.height)*self.zoom
        self.pan=(e.x-x*scale-(w-self.im.width*scale)/2,e.y-y*scale-(h-self.im.height*scale)/2);self.draw()
    def pan_move(self,e):
        if self.pan_start:
            x,y,old=self.pan_start;self.pan=(old[0]+e.x-x,old[1]+e.y-y);self.draw()
    def click(self,e):
        self.canvas.focus_set()
        if self.frame()['role']!='ROLE_CONFIDENT':return
        x,y=self.coords(e)
        if not(0<=x<self.im.width and 0<=y<self.im.height):return
        if self.mode=='bbox':self.drag=(x,y)
        else:
            self.checkpoint();self.frame()['corners'][self.corner]=dict(status='DIRECT_VISIBLE',xy=[round(x,3),round(y,3)])
            self.save('CLICK');self.corner=min(7,self.corner+1);self.draw()
    def release(self,e):
        if self.mode!='bbox' or self.drag is None:return
        x,y=self.coords(e);x=min(self.im.width,max(0,x));y=min(self.im.height,max(0,y));a,b=self.drag;self.drag=None
        if abs(x-a)<1 or abs(y-b)<1:return
        self.checkpoint();self.frame()['bbox']=[min(a,x),min(b,y),max(a,x),max(b,y)];self.save('BBOX');self.mode='point';self.draw()
    def mark(self,status):
        if self.frame()['role']!='ROLE_CONFIDENT':return
        self.checkpoint();self.frame()['corners'][self.corner]=dict(status=status,xy=None);self.save('STATUS');self.corner=min(7,self.corner+1);self.mode='point';self.draw()
    def undo(self):
        if self.undo_stack:self.data=self.undo_stack.pop();self.save('UNDO');self.load()
    def finish(self):
        self.frame()['role_note']=self.note.get().strip();errors,warnings=validate_frame(self.frame())
        if errors:messagebox.showwarning('미완료','\n'.join(errors));return
        self.frame()['complete']=True;self.save('FRAME_COMPLETE')
        if self.index+1<len(self.rows):self.move(1)
        else:messagebox.showinfo('저장 완료','입력이 저장되었습니다. 창을 닫고 cli resume 또는 대화에 “다 했어”라고 알려주세요. 모델 추론/학습은 아직 하지 않았습니다.')
    def qa_confirm(self):
        errors,warnings=validate_frame(self.frame())
        if errors:messagebox.showwarning('수정 필요','\n'.join(errors));return
        self.frame()['qa_confirmed']=True;self.frame()['complete']=True;self.save('SECOND_PASS_QA');self.move(1)
    def move(self,step):self.index=max(0,min(len(self.rows)-1,self.index+step));self.load()
    def key(self,e):
        if isinstance(self.root.focus_get(),tk.Entry):return
        k=e.keysym.lower()
        if k in '01234567' and len(k)==1:self.choose(int(k))
        elif k=='c':self.role('ROLE_CONFIDENT')
        elif k=='b':self.set_mode('bbox')
        elif k in ('o','x','u'):self.mark({'o':'OCCLUDED','x':'OUT_OF_FRAME','u':'UNCERTAIN'}[k])
        elif k=='z':self.undo()
        elif k=='f':self.zoom=1.;self.pan=(0.,0.);self.draw()
        elif k=='return':self.finish()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    if (C.DOC/'HARD_LABEL_LOCK.json').exists():raise SystemExit('Labels are locked; refusing edits.')
    if C.state()['status'] not in ('WAITING_FOR_HUMAN_HARD_ANNOTATION','WAITING_FOR_HUMAN_HARD_QA'):
        raise SystemExit('Complete human difficulty tagging and selection first.')
    with C.exclusive('annotating'):
        root=tk.Tk()
        if args.smoke:
            C.OUT.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='hard_annotation_smoke_',dir=C.OUT) as tmp:
                with patch.object(C,'annotation_labels_path',return_value=Path(tmp)/'test_labels.json'):
                    app=App(root);root.update();app.role('ROLE_CONFIDENT')
                    def event(x,y):return SimpleNamespace(x=app.offset[0]+x*app.scale,y=app.offset[1]+y*app.scale)
                    pivot=event(100,100);app.zoom_at(pivot,2.);assert all(abs(a-b)<1e-6 for a,b in zip(app.coords(pivot),(100,100)))
                    app.click(event(10,10));app.release(event(200,200));assert app.frame()['bbox'] and app.mode=='point'
                    app.click(event(25,25));assert app.corner==1 and app.frame()['corners'][0]['status']=='DIRECT_VISIBLE'
                    app.mark('OCCLUDED');assert app.corner==2 and app.frame()['corners'][1]['xy'] is None
                    app.undo();assert app.frame()['corners'][1]['status'] is None
                    app.role('ROLE_UNCERTAIN');assert all(p['xy'] is None for p in app.frame()['corners'])
                    assert not validate_frame(app.frame())[0]
                    assert C.read(app.path)['frames']
            root.destroy();print('PASS annotation GUI: manual bbox, visible click auto-advance, hidden ignore, undo, role exclusion; actual labels untouched')
        else:
            App(root);root.after(150,lambda:(root.lift(),root.focus_force()));root.mainloop()

if __name__=='__main__':main()

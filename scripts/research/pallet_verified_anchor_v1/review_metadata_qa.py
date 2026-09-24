"""Two-corner second pass: raw RGB + own first click, no predictions/legacy overlay."""
import argparse
import copy
from pathlib import Path
import tempfile
import tkinter as tk
from tkinter import messagebox
from . import common as C
from .complete_directive import QA
from .review_saved_keypoints import Review

class MetadataQA(Review):
    def __init__(self,root):
        super().__init__(root)
        todo=[p for p in self.queue if self.labels['frames'][p[0]]['corners'][p[1]].get('qa_decision') is None]
        self.index,self.corner=(todo or self.queue)[0]
        root.title('2점만 재확인 — K/Enter 유지 · 상태 키 변경 · 기존 정답/모델 숨김')
        def replace_help(widget):
            for child in widget.winfo_children():
                if isinstance(child,tk.Label) and str(child.cget('text')).startswith('노란 원 ='):
                    child.config(text='노란 원 = 본인이 입력한 위치\nK 또는 Enter: 현재 판정 유지\nD/V/S/E/O/U: 상태 변경 후 다음\n좌표 수정: 점 클릭 (D/V만 허용)\nCtrl+Z 취소 · 변경 자동 저장\n기존 정답/모델 예측 표시 안 함')
                replace_help(child)
        replace_help(root);self.load()
    def draw(self):
        super().draw()
        if not self.queue:return
        done=sum(self.labels['frames'][f]['corners'][c].get('qa_decision') is not None for f,c in self.queue)
        self.header.config(text=f'재확인 {self.queue.index([self.index,self.corner])+1}/2 · P{self.corner} · 현재 {self.item()["status"]} · 맞으면 K/Enter, 바꿀 때만 상태 키')
        self.footer.config(text=f'확인 완료 {done}/2 · 새 이미지/새 점 입력 불필요 · 예전 기록이 맞다고 가정하지 마세요')
    def keep(self):
        self.checkpoint();self.item()['qa_decision']='KEEP';self.save('QA_KEEP');self.advance()
    def status(self,j):
        self.checkpoint();c=self.item();c['status']=C.STATUSES[j]
        c['xy']=copy.deepcopy(c['input_xy']) if j<2 else None
        c['qa_decision']='CHANGE_STATUS';self.save('QA_CHANGE_STATUS');self.advance()
    def click(self,e):
        self.canvas.focus_set()
        if self.item()['status'] not in ('DIRECT_VISIBLE','VIRTUAL_INFERABLE'):return
        x=(e.x-self.offset[0])/self.scale;y=(e.y-self.offset[1])/self.scale
        w,h=self.original.size
        if not(0<=x<w and 0<=y<h):return
        self.checkpoint();self.item()['xy']=[round(x,3),round(y,3)]
        self.item()['qa_decision']='RECLICK';self.save('QA_RECLICK');self.advance()
    def key(self,e):
        if isinstance(self.root.focus_get(),tk.Entry):return
        if e.keysym.lower() in ('k','return','kp_enter'):self.keep()
        else:super().key(e)
    def finish(self):
        done=sum(self.labels['frames'][f]['corners'][c].get('qa_decision') is not None for f,c in self.queue)
        messagebox.showinfo('재확인', '완료했습니다. CLI에 알려주시면 teacher 보조 비교와 최종 보고서를 마무리합니다.' if done==2 else f'{done}/2 완료. 맞으면 K/Enter로 확인하세요.')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    C.RAW=QA
    root=tk.Tk();app=MetadataQA(root);root.update()
    if args.smoke:
        with tempfile.TemporaryDirectory(prefix='anchor_metadata_qa_test_') as tmp:
            C.RAW=Path(tmp)
            start=[app.index,app.corner];old=copy.deepcopy(app.item())
            app.key(type('Event',(),dict(keysym='Return'))())
            assert app.labels['frames'][start[0]]['corners'][start[1]]['qa_decision']=='KEEP'
            assert [app.index,app.corner]!=start
            app.undo();assert [app.index,app.corner]==start and app.item()==old
            app.status(5);assert app.labels['frames'][start[0]]['corners'][start[1]]['xy'] is None
            assert app.labels['frames'][start[0]]['corners'][start[1]]['qa_decision']=='CHANGE_STATUS'
        root.destroy();print('PASS metadata QA UI: keep/advance/undo/status; actual labels untouched')
    else:root.mainloop()

if __name__=='__main__':main()

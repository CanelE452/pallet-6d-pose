"""Status-only human review of manual clicks; never promote PnP coordinates."""
import argparse
import copy
from collections import Counter
import tkinter as tk
import tempfile
from pathlib import Path
from tkinter import messagebox
from . import common as C
from .label_anchor import App

ORIGINAL_RAW=C.RAW
REVIEW=ORIGINAL_RAW/'keypoints_status_review'
WORK=C.OUT/'keypoints_first/WORKSPACE.json'

def prepare():
    selection=C.read(ORIGINAL_RAW/'ANCHOR_SELECTION.json');workspace=C.read(WORK)
    by_id={r['frame_id']:r for r in workspace['frames']}
    labels=copy.deepcopy(C.read(ORIGINAL_RAW/'LABEL_TEMPLATE.json'))
    bindings=[];counts=Counter();severity={s:Counter() for s in C.SEVERITIES};missing=[];queue=[]
    for fi,(frame,sel) in enumerate(zip(labels['frames'],selection['frames'])):
        assert frame['frame_id']==sel['frame_id']
        r=by_id[frame['frame_id']];path=C.ROOT/r['output_annotation']
        assert C.sha(C.ROOT/r['image']['path'])==r['image']['sha256']
        assert C.sha(C.ROOT/r['source_annotation']['path'])==r['source_annotation']['sha256'], 'Original GT changed'
        if not path.exists():
            missing.append(frame['frame_id']);severity[sel['severity']]['unsaved']+=1
            frame['keypoints_saved']=False
            continue
        bindings.append(dict(path=r['output_annotation'],sha256=C.sha(path)))
        entries=C.read(path)['objects'][0]['keypoint_annotations'];assert len(entries)>=8
        frame['keypoints_saved']=True;severity[sel['severity']]['saved']+=1
        for ci,e in enumerate(entries[:8]):
            src=e['source'];counts[src]+=1;severity[sel['severity']][src]+=1
            corner=frame['corners'][ci];corner['coordinate_source']=src
            # Existing visibility=2 alone is NOT a verified DIRECT_VISIBLE judgment.
            if src=='manual_click':
                xy=e['xy']
                if C.valid_corner(dict(status='DIRECT_VISIBLE',xy=xy),sel['hw']):
                    corner['xy']=xy;corner['input_xy']=xy;queue.append([fi,ci])
                else:counts['manual_outside_or_invalid']+=1
            # No PnP or extrapolated coordinate is imported into the status review.
    labels['protocol']='KEYPOINTS_FIRST_PNP_ASSISTED_THEN_MANUAL_STATUS'
    labels['review_queue']=queue
    labels['unreviewed_nonmanual_corners']='Excluded, not silently classified as hidden or uncertain'
    audit=dict(selected_images=18,saved_images=len(bindings),unsaved_images=len(missing),sources=dict(counts),
        severity={s:dict(v) for s,v in severity.items()},manual_status_queue=len(queue),
        confirmed_DIRECT_VISIBLE=0,model_evaluation_performed=False,
        missing_frames_replaced=False,original_GT_unchanged=True,protocol=labels['protocol'])
    lock=dict(annotations=bindings,missing_frame_ids=missing,queue=queue,
              selection_sha256=C.sha(ORIGINAL_RAW/'ANCHOR_SELECTION.json'))
    if (REVIEW/'INPUT_LOCK.json').exists():
        assert C.read(REVIEW/'INPUT_LOCK.json')==lock, 'Annotations changed since snapshot: preserve review, reconcile explicitly'
    else:
        C.save_new(REVIEW/'INPUT_LOCK.json',lock)
        C.save_new(REVIEW/'ANCHOR_SELECTION.json',selection)
        assert C.sha(REVIEW/'ANCHOR_SELECTION.json')==lock['selection_sha256']
        C.save_new(REVIEW/'LABEL_TEMPLATE.json',labels)
        C.save_new(REVIEW/'IMPORT_AUDIT.json',audit)
        C.save_new(C.DOC/'KEYPOINTS_SAVED_SUMMARY.json',audit)
    return audit

class Review(App):
    def __init__(self,root):
        self.queue=[]
        super().__init__(root)
        self.queue=self.labels['review_queue']
        assert self.queue
        todo=[p for p in self.queue if self.labels['frames'][p[0]]['corners'][p[1]]['status'] is None]
        self.index,self.corner=(todo or self.queue)[0]
        root.title('찍은 점 상태만 확인 — D 직접 / V 추정 / U 애매 · 좌표 재입력 없음')
        def help_labels(widget):
            for child in widget.winfo_children():
                if isinstance(child,tk.Label) and str(child.cget('text')).startswith('D·V·S·E·O·U 입력'):
                    child.config(text='노란 원 = 지금 분류할 점\nD 직접 보임 / V 추정 / U 애매\n상태 키 → 다음 직접 입력점\n좌표 재입력 없음 · 클릭 비활성\nPnP 보완점은 표시·평가하지 않음\nCtrl+Z 취소 · 자동 저장')
                help_labels(child)
        help_labels(root)
        self.load()
        # Base UI help is superseded by the explicit review header/footer below.
    def draw(self):
        super().draw()
        if not self.queue:return
        done=sum(self.labels['frames'][f]['corners'][c]['status'] is not None for f,c in self.queue)
        c=self.item();xy=c.get('input_xy')
        if xy is not None:
            x,y=xy;x=x*self.scale+self.offset[0];y=y*self.scale+self.offset[1]
            self.canvas.create_oval(x-12,y-12,x+12,y+12,outline='#ffff00',width=3)
        self.header.config(text=f'현재 P{self.corner} — 노란 원으로 강조된 직접 입력점의 상태만 선택 · D 직접 보임 / V 추정 / S 자체 가림 / E 외부 가림 / O 화면 밖 / U 애매')
        self.footer.config(text=f'상태 확인 {done}/{len(self.queue)}개 · 상태 키 입력 → 다음 직접 입력점 · 좌표 클릭 비활성 · PnP 점 제외')
    def click(self,event):
        self.canvas.focus_set()
    def choose(self,i):
        if [self.index,i] in self.queue:super().choose(i)
    def advance(self):
        p=self.queue.index([self.index,self.corner])
        self.index,self.corner=self.queue[min(p+1,len(self.queue)-1)];self.load()
    def navigate(self,delta):
        p=self.queue.index([self.index,self.corner])
        self.index,self.corner=self.queue[max(0,min(len(self.queue)-1,p+delta))];self.load()
    def status(self,j):
        # D/V restore the already-entered coordinate; no new coordinate guesses.
        self.checkpoint();self.canvas.focus_set();c=self.item();c['status']=C.STATUSES[j]
        c['xy']=copy.deepcopy(c['input_xy']) if j<2 else None
        self.save('status_only');self.advance()
    def finish(self):
        todo=sum(self.labels['frames'][f]['corners'][c]['status'] is None for f,c in self.queue)
        messagebox.showinfo('상태 확인',f'아직 {todo}개 상태가 남았습니다.' if todo else
            '직접 입력점 상태 확인을 저장했습니다. CLI에 완료했다고 알려주세요.\n미입력/PnP 점은 평가에서 제외하며 아직 재채점하지 않았습니다.')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    audit=prepare();print(audit,flush=True)
    if args.prepare_only:return
    C.RAW=REVIEW
    root=tk.Tk();app=Review(root)
    if args.smoke:
        root.update()
        with tempfile.TemporaryDirectory(prefix='anchor_status_smoke_') as tmp:
            C.RAW=Path(tmp)
            start=[app.index,app.corner];point=copy.deepcopy(app.item()['input_xy'])
            app.status(0)
            assert app.labels['frames'][start[0]]['corners'][start[1]]['xy']==point
            assert [app.index,app.corner]!=start
            app.undo();assert [app.index,app.corner]==start
            app.status(1)
            assert app.labels['frames'][start[0]]['corners'][start[1]]['status']=='VIRTUAL_INFERABLE'
            app.undo();app.status(5)
            assert app.labels['frames'][start[0]]['corners'][start[1]]['xy'] is None
        root.destroy();print('STATUS_REVIEW_SMOKE_OK; no real labels changed',flush=True)
    else:root.mainloop()

if __name__=='__main__':main()

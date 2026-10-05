"""Local launcher for the remaining human actions; no models or control imports."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LIFTER = ROOT/'scripts/research/pallet_lifter_case_review_20261003_v1'
STATIC = ROOT/'scripts/research/pallet_static_registry_review_20261003_v1'
REVIEW = ROOT/'data/pallet/results/pallet_lifter_case_review_20261003_v1/review'
STATIC_OUT = ROOT/'data/pallet/results/pallet_static_registry_review_20261003_v1'
OUT = ROOT/'data/pallet/results/pallet_combined_closeout_20261003_v1/human_review'
BATCH = REVIEW/'SMALL_BATCH_12_V1.json'
TASKS = {
    'static': STATIC/'open_corner_visibility.py',
    'lifter': LIFTER/'open_existing_annotation.py',
    'match': LIFTER/'open_object_match_annotation.py',
    'stationary': LIFTER/'open_stationary_review.py',
}
TITLES = dict(static='Annotate - Static corner visibility', lifter='Annotate',
              match='Annotate - Selected pallet match 1 2 3',
              stationary='Pallet raw video - stationary interval review')


def focus_running(task):
    """Raise this task's existing window; never send annotation key presses."""
    running = False
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit():
            continue
        try:
            args = (folder/'cmdline').read_bytes().split(b'\0')
            if len(args)>1 and Path(args[0].decode(errors='replace')).name.startswith('python'):
                running |= any(arg.decode(errors='replace') in (str(TASKS[task]),
                    str(TASKS[task].relative_to(ROOT))) for arg in args[1:])
        except (OSError,ValueError):
            pass
    if not running:
        return False
    result = subprocess.run(['xwininfo','-name',TITLES[task]],capture_output=True,text=True)
    if result.returncode == 0:
        import ctypes
        import re
        match = re.search(r'Window id: (0x[0-9a-fA-F]+)',result.stdout)
        if match:
            lib=ctypes.CDLL('libX11.so.6')
            lib.XOpenDisplay.argtypes=[ctypes.c_char_p];lib.XOpenDisplay.restype=ctypes.c_void_p
            lib.XRaiseWindow.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
            lib.XSetInputFocus.argtypes=[ctypes.c_void_p,ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong]
            lib.XSync.argtypes=[ctypes.c_void_p,ctypes.c_int];lib.XCloseDisplay.argtypes=[ctypes.c_void_p]
            display=lib.XOpenDisplay(None)
            if display:
                wid=int(match[1],16)
                lib.XRaiseWindow(display,wid);lib.XSetInputFocus(display,wid,2,0)
                lib.XSync(display,0);lib.XCloseDisplay(display)
    return True


def read(path, default=None):
    if not Path(path).is_file():
        return {} if default is None else default
    return json.loads(Path(path).read_text(encoding='utf-8'))


def progress():
    manifest = read(REVIEW/'MANIFEST.json')
    excluded = set(read(REVIEW/'USER_EXCLUSIONS.json').get('excluded_frame_ids', []))
    ids = {f['frame_id'] for f in manifest.get('frames', [])} - excluded
    repeat = {f['frame_id'] for f in manifest.get('frames', []) if f['repeat_review']} - excluded
    records = read(REVIEW/'annotations_in_progress.json').get('records', [])
    primary_done = {r['frame_id'] for r in records
                    if r['review_pass']=='primary' and r['status'] in ('reviewed','skipped')} & ids
    repeat_done = {r['frame_id'] for r in records
                   if r['review_pass']=='repeat' and r['status'] in ('reviewed','skipped')} & repeat
    visibility = read(STATIC_OUT/'native_corner_visibility/STATIC_CORNER_VISIBILITY_INPUTS.json')
    entered = sum(len(r['corners']) for r in visibility.get('records', {}).values())
    unknown = sum(p['status']=='UNKNOWN' for r in visibility.get('records', {}).values()
                  for p in r['corners'].values())
    active = read(REVIEW/'native_object_match/ACTIVE_QUEUE.json')
    match_required = match_done = 0
    if active.get('folder'):
        folder = Path(active['folder'])
        match_required = len(read(folder/'OBJECT_MATCH_QUEUE.json').get('records', []))
        match_done = len(read(folder/'object_match_in_progress.json').get('records', []))
    stationary = read(REVIEW/'STATIONARY_INTERVALS_REVIEWED.json')
    pnp = read(REVIEW/'NATIVE_PNP_PROGRESS.json')
    pnp_records = pnp.get('records', {}).values()
    pnp_primary = {r['frame_id'] for r in pnp_records if r.get('review_pass')=='primary'
                   and r.get('frame_id') in ids and Path(r.get('assisted_file','')).is_file()}
    pnp_repeat = {r['frame_id'] for r in pnp_records if r.get('review_pass')=='repeat'
                  and r.get('frame_id') in repeat and Path(r.get('assisted_file','')).is_file()}
    semantics = read(OUT/'SEVERITY_CRITERION_CONFIRMATION.json')
    batch = read(BATCH)
    batch_ids = set(batch.get('frame_ids', []))
    if batch:
        if (batch.get('schema') != 'lifter_human_review_batch_v1'
                or batch.get('source_kind') != 'partial_human_review_queue'
                or batch.get('frozen_plan_modified') is not False
                or batch.get('bindings', {}).get('manifest_sha256') != hashlib.sha256((REVIEW/'MANIFEST.json').read_bytes()).hexdigest()
                or len(batch_ids) != len(batch.get('frame_ids', []))
                or not batch_ids or not batch_ids <= ids):
            raise ValueError('Partial review queue mismatch')
    label_complete = label_categories = manual_coordinates = declared_coordinates_absent = 0
    if batch_ids:
        sys.path.insert(0,str(LIFTER))
        from open_existing_annotation import Context, NativeReview
        context = Context(REVIEW/'MANIFEST.json', REVIEW.parent/'LIFTER_EVALUATION_PLAN.json',
                          REVIEW/'CORNER_CONTRACT.json', REVIEW/'annotations_in_progress.json')
        native = NativeReview(context,None,REVIEW/'native_annotations',passes=('primary',),
                              native_pnp=False,batch_plan=BATCH)
        for fid in batch_ids:
            status = native.visibility_only_status(fid,'primary')
            label_complete += status['complete']
            label_categories += status['confirmed_category_count']
            manual_coordinates += status['manual_coordinate_count']
            declared_coordinates_absent += len(status['declared_point_ids'])
    return dict(static_pending=max(0,3030-entered), static_entered=entered,
        static_unknown=unknown, preserved_points=71,
        primary_total=len(ids), primary_done=len(primary_done),
        pnp_primary_saved=len(pnp_primary),pnp_repeat_saved=len(pnp_repeat),
        batch_active=bool(batch),batch_primary_total=len(batch_ids),
        batch_pnp_saved=len(batch_ids & pnp_primary),batch_reference_done=len(batch_ids & primary_done),
        batch_visibility_label_complete=label_complete,batch_visibility_categories=label_categories,
        batch_manual_coordinates=manual_coordinates,batch_visible_without_coordinates=declared_coordinates_absent,
        batch_uncertain_states=sum(c.get('visibility')=='uncertain' for record in records
            if record.get('frame_id') in batch_ids and record.get('review_pass')=='primary'
            and record.get('status')=='reviewed' for c in record.get('corners', [])),
        batch_match_ready=bool(batch_ids) and batch_ids <= primary_done,
        repeat_total=len(repeat), repeat_done=len(repeat_done), excluded=len(excluded),
        match_ready=bool(ids) and ids==primary_done, match_required=match_required, match_done=match_done,
        stationary_sessions=len(stationary.get('session_reviews', {})),
        stationary_intervals=len(stationary.get('intervals', [])),
        stationary_strict_status=stationary.get('strict_evaluator_status','WAITING_HUMAN'),
        criterion=semantics.get('criterion','NOT_CONFIRMED'))


def confirm_criterion(value):
    """Called only by an actual button confirmation; existing labels are unchanged."""
    if value not in ('external_occlusion','overall_visual_difficulty'):
        raise ValueError('Invalid human criterion')
    sources = [STATIC_OUT/'native_severity_review/STATIC_SEVERITY_INPUTS.json',
               STATIC_OUT/'existing_square150_severity_review/GREEN150_SEVERITY_INPUTS.json',
               REVIEW/'SEVERITY_REVIEW_IN_PROGRESS.json']
    missing = [str(p) for p in sources if not p.is_file()]
    if missing:
        raise ValueError('Existing classification sidecar missing: '+', '.join(missing))
    at = datetime.now(timezone.utc).isoformat()
    bindings = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    entry = dict(schema='human_severity_criterion_confirmation_v1', criterion=value,
                 source_kind='human_direct_confirmation', input_action='explicit_launcher_button_confirmed',
                 confirmed_at=at, human_identity_confirmed=False,
                 actor_id='local-reviewer', source_bindings=bindings,
                 original_labels_changed=False, counts_matched_by_editing=False)
    path = OUT/'SEVERITY_CRITERION_CONFIRMATION.json'
    OUT.mkdir(parents=True,exist_ok=True)
    if path.is_file():
        with (OUT/'SEVERITY_CRITERION_CONFIRMATION.history.jsonl').open('a',encoding='utf-8') as handle:
            handle.write(json.dumps(dict(superseded_at=at,previous=read(path)),ensure_ascii=False)+'\n')
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(entry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tmp.replace(path)


def launch():
    sys.path.insert(0,str(ROOT))
    from scripts.annotate.korean_tk import install
    install()
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    root.title('팔레트 · 남은 사람 작업')
    root.geometry(f'620x{min(960,root.winfo_screenheight()-90)}+35+45')
    root.configure(bg='#17202b')
    root.option_add('*Font', ('Noto Sans CJK KR',11))
    children = {}
    labels = {}
    canvas = tk.Canvas(root,bg='#17202b',highlightthickness=0)
    scroll = tk.Scrollbar(root,orient='vertical',command=canvas.yview)
    canvas.configure(yscrollcommand=scroll.set)
    scroll.pack(side='right',fill='y');canvas.pack(side='left',fill='both',expand=True)
    frame = tk.Frame(canvas,bg='#17202b',padx=22,pady=18)
    item = canvas.create_window((0,0),window=frame,anchor='nw')
    frame.bind('<Configure>',lambda event:canvas.configure(scrollregion=canvas.bbox('all')))
    canvas.bind('<Configure>',lambda event:canvas.itemconfigure(item,width=event.width))
    root.bind('<Button-4>',lambda event:canvas.yview_scroll(-2,'units'))
    root.bind('<Button-5>',lambda event:canvas.yview_scroll(2,'units'))
    tk.Label(frame,text='완료한 입력은 그대로 사용합니다',font=('Noto Sans CJK KR',17,'bold'),
             bg='#17202b',fg='white').pack(anchor='w')
    tk.Label(frame,text='완료한 가림 분류는 다시 하지 않습니다.\n식별자 입력 없이 저장하고, 닫았다 열면 이어집니다.',
             justify='left',bg='#17202b',fg='#cfe0ed').pack(anchor='w',pady=(7,14))
    def start(task, extra=()):
        if focus_running(task):
            return
        old = children.get(task)
        if old and old.poll() is None:
            messagebox.showinfo('이미 열려 있습니다','작업 표시줄에서 해당 Annotate 창을 선택하세요.',parent=root)
            return
        # All point and object tasks invoke the original native annotate.py loop.
        OUT.mkdir(parents=True,exist_ok=True)
        log = (OUT/(task+'.log')).open('a',encoding='utf-8')
        children[task] = subprocess.Popen([sys.executable,str(TASKS[task]),*extra],
                            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        log.close()
        root.after(350,refresh)
    def section(name,help_text):
        box=tk.Frame(frame,bg='#223140',padx=12,pady=10)
        box.pack(fill='x',pady=5)
        tk.Label(box,text=name,font=('Noto Sans CJK KR',12,'bold'),bg='#223140',fg='white').pack(anchor='w')
        label=tk.Label(box,text='',bg='#223140',fg='#a5e4c3',justify='left')
        label.pack(anchor='w',pady=4)
        tk.Label(box,text=help_text,bg='#223140',fg='#cedce8',justify='left').pack(anchor='w')
        return box,label
    box,labels['static']=section('1. 정적 코너 상태','완료한 상태는 다시 입력할 필요가 없습니다.')
    static_button=tk.Button(box,text='남은 정적 상태 열기',command=lambda:start('static'),
              bg='#176b45',fg='white')
    static_button.pack(fill='x',pady=(7,0))
    small_args=('--batch-plan',str(BATCH)) if BATCH.is_file() else ()
    box,labels['lifter']=section('2. 리프터 키포인트 → PnP',
            '이미 입력한 키포인트를 재사용합니다.\n가시성 분류와 평가용 참조 승인은 별도로 셉니다.' if small_args else
        '먼저 점 입력 → G/F 자동 채움 → 필요하면 M 자세 조작.\n가시성 검수는 PnP 입력 이후에 합니다.')
    buttons=tk.Frame(box,bg='#223140');buttons.pack(fill='x',pady=(7,0))
    pnp_button=tk.Button(buttons,text='남은 키포인트 열기' if small_args else '주 검수 열기',
              command=lambda:start('lifter',('--pass','primary',*small_args)),
              bg='#176b45',fg='white')
    pnp_button.pack(side='left',fill='x',expand=True,padx=(0,5))
    if small_args:
        visibility_button=tk.Button(buttons,text='가시성 분류 확인',
            command=lambda:start('lifter',('--pass','primary','--visibility-only',*small_args)),
            bg='#315371',fg='white')
        visibility_button.pack(side='left',fill='x',expand=True)
    else:
        tk.Button(buttons,text='반복 검수 열기',command=lambda:start('lifter',('--pass','repeat')),
                  bg='#315371',fg='white').pack(side='left',fill='x',expand=True)
    box,labels['match']=section('3. 추가 리프터 코너 정확도 평가',
        '참조 승인과 대상 확인이 있어야 정확도를 계산합니다.\n이 작업 없이도 정적 결과와 8,910장 연속 출력은 사용합니다.')
    match_button=tk.Button(box,text='선택 박스 대상 확인 열기',command=lambda:start('match',small_args),bg='#315371',fg='white')
    match_button.pack(fill='x',pady=(7,0))
    box,labels['stationary']=section('4. 정지 구간 (정지 잡음 표를 채울 때)','원영상에서 정지 시작·끝만 선택. 확인 불가능하면 x로 유지 가능.')
    tk.Button(box,text='원영상 정지 구간 확인 열기',command=lambda:start('stationary'),
              bg='#315371',fg='white').pack(fill='x',pady=(7,0))
    box,labels['criterion']=section('분류 기준 한 번 확인','앞에서 나눈 없음·중간·어려움이 어떤 기준인지 선택하세요.')
    def criterion(value):
        label = '다른 물체에 가린 정도' if value=='external_occlusion' else '거리·시점·조명까지 포함한 전체 난이도'
        if messagebox.askokcancel('기존 분류 기준 확인',f'기존 분류는 「{label}」 기준인가요?\n분류 숫자와 원본 주석은 바꾸지 않습니다.',parent=root):
            try:confirm_criterion(value)
            except (ValueError,OSError) as err:messagebox.showerror('저장 오류',str(err),parent=root)
            refresh()
    buttons=tk.Frame(box,bg='#223140');buttons.pack(fill='x',pady=(7,0))
    for text,value in [('외부 가림 정도','external_occlusion'),('전체 난이도','overall_visual_difficulty')]:
        tk.Button(buttons,text=text,command=lambda v=value:criterion(v),bg='#315371',fg='white').pack(side='left',fill='x',expand=True,padx=3)
    tk.Label(frame,text='원본 보존 · 자동 추정은 사람 검수로 기록하지 않음',bg='#17202b',fg='#9cb3c7').pack(anchor='w',pady=8)
    def refresh():
        try:
            p=progress()
            labels['static'].configure(text=f'남은 상태 {p["static_pending"]:,}개 · 기존 확정 71개 유지 · 보류 {p["static_unknown"]}개')
            static_button.configure(text='정적 상태 입력 완료' if p['static_pending']==0 else '남은 정적 상태 열기',
                                    state='disabled' if p['static_pending']==0 else 'normal')
            if p['batch_active']:
                labels['lifter'].configure(text=f'키포인트·PnP {p["batch_pnp_saved"]}/{p["batch_primary_total"]}장 · 가시성 분류 {p["batch_visibility_label_complete"]}/{p["batch_primary_total"]}장\n상태 {p["batch_visibility_categories"]}/96개 · 실제 수동 좌표 {p["batch_manual_coordinates"]}점\n보임 판단만 기록 {p["batch_visible_without_coordinates"]}점 · 평가용 참조 승인 {p["batch_reference_done"]}장')
                visibility_button.configure(text='가시성 분류 완료' if p['batch_visibility_label_complete']==p['batch_primary_total'] else '가시성 분류 확인',
                    state='normal' if p['batch_pnp_saved']==p['batch_primary_total'] and p['batch_visibility_label_complete']<p['batch_primary_total'] else 'disabled')
                pnp_button.configure(text='키포인트·PnP 입력 완료' if p['batch_pnp_saved']==p['batch_primary_total'] else '남은 키포인트 열기',
                    state='disabled' if p['batch_pnp_saved']==p['batch_primary_total'] else 'normal')
                match_ready=p['batch_match_ready']
                pending_text='가시성 분류는 완료됐습니다. 정확도 참조는 아직 미승인입니다.\n좌표 없는 보임 5점은 자동 좌표로 채우지 않으며 정확도는 x입니다.'
            else:
                labels['lifter'].configure(text=f'키포인트·PnP 주 {p["pnp_primary_saved"]}/{p["primary_total"]}장 · 반복 {p["pnp_repeat_saved"]}/{p["repeat_total"]}장\n가시성 검수 주 {p["primary_done"]}/{p["primary_total"]}장 · 반복 {p["repeat_done"]}/{p["repeat_total"]}장 · 제외 5장')
                match_ready=p['match_ready']
                pending_text='주 검수 115장 완료 후 열립니다.'
            labels['match'].configure(text=(f'완료 {p["match_done"]}/{p["match_required"]}장 · 열 수 있음' if match_ready else pending_text))
            match_button.configure(state='normal' if match_ready else 'disabled')
            labels['stationary'].configure(text=f'영상 확인 {p["stationary_sessions"]}/4 · 저장 구간 {p["stationary_intervals"]}개')
            labels['criterion'].configure(text={'NOT_CONFIRMED':'아직 기준 미확인 (분류 재작업 없음)',
                'external_occlusion':'외부 가림 기준 확인됨','overall_visual_difficulty':'전체 난이도 기준 확인됨'}.get(p['criterion'],p['criterion']))
        except (ValueError,OSError,KeyError,TypeError) as err:
            labels['static'].configure(text='진행 상태 읽기 오류: '+str(err)[:75])
    def tick():
        refresh();root.after(2000,tick)
    tick()
    root.mainloop()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.prepare_only:
        print(json.dumps(dict(status='READY_HUMAN_REVIEW_LAUNCHER',progress=progress(),
            tasks={key:str(path.relative_to(ROOT)) for key,path in TASKS.items()},
            all_task_scripts_present=all(p.is_file() for p in TASKS.values()),new_training=0),ensure_ascii=False),flush=True)
        return
    launch()


if __name__=='__main__':
    main()

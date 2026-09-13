"""Strict new-root outputs and immutable historical experiment bindings."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
DOC=ROOT/'_docs/experiments/pallet_final_ml_contribution_test_v1'
RAW=ROOT/'data/pallet/results/pallet_final_ml_contribution_test_v1'
A=DOC/'A_task_risk_robustness';B=DOC/'B_line_vs_point'
ARAW=RAW/'A_task_risk_robustness';BRAW=RAW/'B_line_vs_point'
LINE=ROOT/'data/pallet/results/pallet_line_pose_v1'
LINE_CODE=ROOT/'scripts/research/pallet_line_pose_v1'
TASK=ROOT/'data/pallet/results/pallet_task_risk_active_v1'
TASK_DOC=ROOT/'_docs/experiments/pallet_task_risk_active_v1'
AL=ROOT/'data/pallet/results/pallet_active_learning_v1/retrospective_v1'
POSE=ROOT/'data/pallet/results/paper_pose_metric_closure_v1'
R0=ROOT/'challenge/yolo_pose_one_model/spatial_concat_scratch/runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt'
R0_SHA='970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7'
START='a9640cbf9a1ca2a7aab323258577a49b15bbf3d0'
BOOT=10000;STAT_SEED=20260913

def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def vsha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
def write(path,value):
    path=Path(path);assert any(path.resolve().is_relative_to(p) for p in (DOC,RAW))
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert path.read_text()==text,('Frozen artifact differs',path)
    else:
        with path.open('x') as f:f.write(text)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
def old(name):
    key='final_ml_old_line_'+name
    if key not in sys.modules:
        before=sys.path[:];sys.path.insert(0,str(LINE_CODE));sys.path.insert(1,str(ROOT))
        try:return module(key,LINE_CODE/(name+'.py'))
        finally:sys.path[:]=before
    return sys.modules[key]
def gpu():
    status=subprocess.check_output(['nvidia-smi','--query-gpu=name,temperature.gpu,memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True).strip()
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True).strip()
    foreign=[r for r in processes.splitlines() if r.split(',')[0].strip()!=str(os.getpid())]
    result=dict(timestamp=datetime.now(timezone.utc).isoformat(),gpu=status,compute=processes,foreign_compute=foreign,system_changes=False)
    if foreign:
        write(DOC/('GPU_BUSY_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),result)
        raise RuntimeError('Foreign GPU compute: clean stop, no wait/kill')
    assert float(status.split(',')[1])<80,'GPU thermal guard'
    return result

def initialize():
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='main'
    subprocess.check_call(['git','merge-base','--is-ancestor',START,'HEAD'])
    assert sha(R0)==R0_SHA
    assert read(TASK_DOC/'VERDICT.json')['scientific_verdict']=='TASK_RISK_AL_NO_SIGNAL'
    assert read(LINE/'VERDICT.json')['overall_accuracy_improved'] is False
    write(DOC/'MASTER_PROTOCOL_LOCK.json',dict(timestamp=datetime.now(timezone.utc).isoformat(),start_sha=START,
        branch='main',old_task_verdict='TASK_RISK_AL_NO_SIGNAL',old_line_overall_accuracy_improved=False,
        order='A no-training robustness -> B preflight/design locks/tests -> P1/P2/P3 training -> synthetic selection freeze -> DEV319 -> statistics/runtime/audit',
        A_QA_rule='Flag iff preexisting migration_status MANUAL_REVIEW_REQUIRED or nonempty manual_review_reasons or archived explicit review list. QA_CLEAN requires explicit existing clean/verified metadata; otherwise QA_UNKNOWN. No outcome-dependent classification.',
        A_S1='Fixed QA_CLEAN membership only; empty clean population -> NOT_ESTIMABLE, never remove just the catastrophic frame as new primary.',
        A_empty_QA_verdict='TASK_RISK_RESULT_QA_SENSITIVE_REQUIRES_CAUTION; original failure unchanged; no claim QA explains method-specific catastrophe',
        A_robustness_rule='On nonempty clean set, proposed translation CVaR seed mean worse than both mandatory controls -> ROBUST; neither strictly better than both -> SENSITIVE_BUT_NO_POSITIVE_SIGNAL; otherwise QA_SENSITIVE_REQUIRES_CAUTION. Diagnostic only.',
        B_primary='seed-mean same-seed L-P pooled9kp median on fixed DEV319; paired13-session bootstrap95% high<0',
        B_safety='P90 and gross20 L<=P; equal pose coverage; >=3/4 downstream mean directions L better, and no downstream session CI excludes0 toward L harm',
        bootstrap=dict(draws=BOOT,seed=STAT_SEED,frame_and_session=True,seed_mean_within_each_paired_draw=True),
        mechanism=dict(easy_R0_error_px_max=5,mid_max=10,hard_min_exclusive=10,GT_assisted_diagnostic_only=True),
        B_fits=3,student_seeds=[1,2,3],L_retraining=0,R0_retraining=0,real_training=0,
        B_budget='Exact existing TRAIN_PROTOCOL; no result-based changes; source/parameter/evidence/order locks before first update',
        forbidden=['task-risk retraining','new risk scores','line/DHT/Hough rescue','real-based hyperparameter selection','paper/final edits','additional seeds'],
        final_architecture_rule='LINE_BIAS if all gates. Otherwise GENERIC_POINT_REFINER_DOMINATES if session primary low>0 and reverse P90/gross/coverage/downstream safety; else LOCAL_REFINEMENT_ONLY if both L/P seed-mean median<R0 with >=2/3 seed improvements each; otherwise NO_REFINEMENT_SIGNAL. No independent or novelty claim.'))
    # Full byte bindings for core protected outputs; huge read-only feature arrays
    # are included, not rewritten or copied. Other historical areas use tracked
    # tree identities and file metadata to detect any touched file.
    roots=[TASK,TASK_DOC,ROOT/'scripts/research/pallet_task_risk_active_v1',
        ROOT/'scripts/research/pallet_active_learning_v1',ROOT/'_docs/experiments/pallet_active_learning_v1',
        ROOT/'data/pallet/results/pallet_active_learning_v1',LINE,LINE_CODE,ROOT/'_docs/paper/final']
    paths=sorted({p for r in roots for p in r.rglob('*') if p.is_file() and '__pycache__' not in p.parts})
    paths+=[ROOT/'_docs/notes/pallet_line_pose.md',R0]
    bindings={}
    for i,p in enumerate(paths):
        bindings[str(p.relative_to(ROOT))]=sha(p)
        if p.stat().st_size>1e9 or (i+1)%150==0:print('SOURCE_BINDING',i+1,'/',len(paths),str(p.relative_to(ROOT)),flush=True)
    write(DOC/'PRESERVED_SOURCES.json',dict(paths=bindings,count=len(bindings),full_byte_sha=True))
    print('MASTER_LOCK_AND_PRESERVATION_READY',flush=True)

if __name__=='__main__':initialize()

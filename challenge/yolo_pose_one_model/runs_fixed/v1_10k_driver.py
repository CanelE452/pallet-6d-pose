"""V1_FIXED_MATCHED10K 60ep seed42 — train -> 검증 -> 진단 -> 알림. 한 파일에서."""
from __future__ import annotations
import csv, json, math, os, subprocess, sys, time
ROOT="/home/minjae/Documents/github/pallet-pose"
D=os.path.join(ROOT,"challenge/yolo_pose_one_model"); RF=os.path.join(D,"runs_fixed")
RUN="V1_FIXED_MATCHED10K_60EP_SEED42_UBUNTU"; OUT=os.path.join(RF,RUN)
LOCK=json.load(open(os.path.join(RF,"V1_MATCHED10K_CONFIG_LOCK.json")))["locked"]
YOLO="/home/minjae/anaconda3/envs/pallet-yolo26/bin/yolo"
NOTIFY=os.path.expanduser("~/.claude/hooks/discord-notify.sh")
sys.path.insert(0,RF)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)
def notify(t):
    try: subprocess.run([NOTIFY,t],timeout=90)
    except Exception as e: log(f"notify {e}")
def rows_of(p): return list(csv.DictReader(open(p)))
def done():
    p=os.path.join(OUT,"results.csv")
    if not os.path.exists(p): return False,None
    r=rows_of(p); return (bool(r) and int(float(r[-1]["epoch"]))>=60), r
def bad(v):
    try: f=float(v); return math.isnan(f) or math.isinf(f)
    except Exception: return False
def main():
    ok,_=done(); restarts=0
    if not ok:
        notify("**V1_FIXED_MATCHED10K 60ep seed42 시작**\n\n"
               "handoff manifest 가 지정한 정확히 10,000 프레임 (임의 first-10K 아님).\n"
               "recipe = paper60 그대로, data 만 다름. clean pretrained, resume 없음.\n"
               "★ Windows V2 와 machine 이 달라 이 비교는 ENGINEERING_EARLY_SCREEN.")
        while True:
            cmd=[YOLO,"pose","train"]+[f"{k}={v}" for k,v in LOCK.items()]
            if restarts: cmd=[YOLO,"pose","train",f"model={os.path.join(OUT,'weights/last.pt')}","resume=True"]
            log(" ".join(cmd)[:200])
            with open(os.path.join(RF,"train_v1_10k.log"),"a") as fh:
                fh.write(f"\n===== {time.strftime('%F %T')} resume={bool(restarts)} =====\n"); fh.flush()
                subprocess.run(cmd,stdout=fh,stderr=subprocess.STDOUT)
            ok,_=done()
            if ok: break
            tail=open(os.path.join(RF,"train_v1_10k.log"),errors="ignore").read()[-40000:]
            if "CUDA out of memory" in tail or "OutOfMemoryError" in tail:
                notify("❌ **V1 10K TRAIN_BLOCKED_OOM** — batch/imgsz 축소 금지. 중단."); return
            if restarts>=1:
                notify("❌ **V1 10K TRAIN_FAILED** — 2회째 실패. 자동 반복 안 함."); return
            restarts+=1; log("비-OOM 실패 — 1회 resume")
    ok,tr=done()
    nan=sum(1 for r in tr for v in r.values() if bad(v))
    log(f"완주. NaN {nan}. 진단 시작")
    import overnight_60ep as OV
    OV.OUT=OUT; OV.LOCK=LOCK; OV.RUN=RUN
    rowsout=OV.evaluate()
    with open(os.path.join(OUT,"CONVERGENCE.csv"),"w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=list(rowsout[0])); w.writeheader()
        for r in rowsout: w.writerow(r)
    try: OV.plots(rowsout,tr)
    except Exception as e: log(f"plot {e}")
    last=rowsout[-1]
    ref={}
    for n,p in (("seed42_full40k","FIXED_OBJECT_BROAD40K_60EP_SEED42_ADAPTIVE_CONFIRM/ADAPTIVE_60EP_VERDICT.json"),
                ("seed43_full40k","FIXED_OBJECT_BROAD40K_60EP_SEED43_CONFIRM/SEED43_VERDICT.json")):
        f=os.path.join(RF,p)
        if os.path.exists(f):
            j=json.load(open(f))
            o=j.get("observations") or j.get("seed43")
            ref[n]={"pose_mAP50":o.get("pose_mAP50_60ep",o.get("pose_mAP50")),
                    "pose_mAP50_95":o.get("pose_mAP50_95_60ep",o.get("pose_mAP50_95")),
                    "identity_best":o.get("identity_best_60ep",o.get("identity_best")),
                    "yaw180_best":o.get("yaw180_best_60ep",o.get("yaw180_best"))}
    res={"run":RUN,"dataset":"V1_FIXED_MATCHED10K (handoff manifest, N=10,000)",
     "n_train":9867,"n_val":133,
     "★val_133_note":"BROAD40K 의 원 val 소속 133장. 임의 재분할하지 않았다. "
                     "checkpoint 선택은 last.pt 라 val 이 선택에 영향 없음.",
     "★comparison_scope":"ENGINEERING_EARLY_SCREEN (machine 이 Windows V2 와 다름)",
     "epochs":len(tr),"nan_inf":nan,"restarts":restarts,
     "v1_10k":{"pose_mAP50":last["pose_mAP50"],"pose_mAP50_95":last["pose_mAP50_95"],
               "box_mAP50":last["box_mAP50"],"identity_best":last["identity_best_fraction"],
               "yaw180_best":last["yaw180_best_fraction"],
               "collapsed_channels":last["collapsed_channels"]},
     "full40k_reference":ref,"curve":rowsout}
    json.dump(res,open(os.path.join(OUT,"V1_MATCHED10K_VERDICT.json"),"w"),indent=1,ensure_ascii=False)
    t="\n".join(f"{k:16} {v['pose_mAP50']:.4f}  {v['pose_mAP50_95']:.4f}  "
                f"{v['identity_best']:.3f}  {v['yaw180_best']:.3f}" for k,v in ref.items())
    notify(f"**V1_FIXED_MATCHED10K 60ep seed42 완료** (NaN {nan}, 재시도 {restarts})\n\n"
           f"```\n{'run':16} {'mAP50':>6} {'50-95':>7} {'ident':>6} {'yaw180':>6}\n"
           f"{'V1_10K seed42':16} {last['pose_mAP50']:.4f}  {last['pose_mAP50_95']:.4f}  "
           f"{last['identity_best_fraction']:.3f}  {last['yaw180_best_fraction']:.3f}\n{t}\n```\n"
           f"★ ENGINEERING_EARLY_SCREEN — machine 이 Windows V2 와 다릅니다.\n"
           f"★ real 평가는 reviewed GT 도착 후. V2 결과 오면 나란히 보고.")
    log("done")
if __name__=="__main__": main()

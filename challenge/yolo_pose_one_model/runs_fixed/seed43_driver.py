"""seed43 — train -> 완주검증 -> 수렴/semantic 진단 -> 알림. 한 파일에서 끝낸다."""
from __future__ import annotations
import csv, json, math, os, subprocess, sys, time
ROOT="/home/minjae/Documents/github/pallet-pose"
D=os.path.join(ROOT,"challenge/yolo_pose_one_model"); RF=os.path.join(D,"runs_fixed")
RUN="FIXED_OBJECT_BROAD40K_60EP_SEED43_CONFIRM"; OUT=os.path.join(RF,RUN)
LOCK=json.load(open(os.path.join(RF,"SEED43_CONFIG_LOCK.json")))["locked"]
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
        notify(f"**FIXED seed43 학습 시작**\n\nseed42 와 seed 만 다름(43). "
               f"pretrained clean start, resume 없음. 예상 4.3시간.")
        while True:
            cmd=[YOLO,"pose","train"]+[f"{k}={v}" for k,v in LOCK.items()]
            if restarts: cmd=[YOLO,"pose","train",f"model={os.path.join(OUT,'weights/last.pt')}","resume=True"]
            log(" ".join(cmd))
            with open(os.path.join(RF,"train_seed43.log"),"a") as fh:
                fh.write(f"\n===== {time.strftime('%F %T')} resume={bool(restarts)} =====\n"); fh.flush()
                subprocess.run(cmd,stdout=fh,stderr=subprocess.STDOUT)
            ok,_=done()
            if ok: break
            tail=open(os.path.join(RF,"train_seed43.log"),errors="ignore").read()[-40000:]
            if "CUDA out of memory" in tail or "OutOfMemoryError" in tail:
                notify("❌ **seed43 TRAIN_BLOCKED_OOM** — batch/imgsz 축소 금지. 중단."); return
            if restarts>=1:
                notify("❌ **seed43 TRAIN_FAILED** — 2회째 실패. 자동 반복 안 함."); return
            restarts+=1; log(f"비-OOM 실패 — 1회 resume 재시도")
    ok,tr=done()
    nan=sum(1 for r in tr for v in r.values() if bad(v))
    log(f"seed43 완주. NaN {nan}. 진단 시작")
    import overnight_60ep as OV
    OV.OUT=OUT; OV.LOCK=LOCK; OV.RUN=RUN
    rowsout=OV.evaluate()
    with open(os.path.join(OUT,"CONVERGENCE.csv"),"w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=list(rowsout[0])); w.writeheader()
        for r in rowsout: w.writerow(r)
    try: OV.plots(rowsout,tr)
    except Exception as e: log(f"plot {e}")
    last=rowsout[-1]
    s42=json.load(open(os.path.join(RF,"FIXED_OBJECT_BROAD40K_60EP_SEED42_ADAPTIVE_CONFIRM",
                                    "ADAPTIVE_60EP_VERDICT.json")))["observations"]
    res={"run":RUN,"seed":43,"restarts":restarts,"nan_inf":nan,
         "epochs":len(tr),"paper_main_checkpoint":"last.pt (사전 규칙)",
         "seed43":{"pose_mAP50":last["pose_mAP50"],"pose_mAP50_95":last["pose_mAP50_95"],
                   "box_mAP50":last["box_mAP50"],
                   "identity_best":last["identity_best_fraction"],
                   "yaw180_best":last["yaw180_best_fraction"],
                   "collapsed_channels":last["collapsed_channels"]},
         "seed42_reference":{"pose_mAP50":s42["pose_mAP50_60ep"],
                             "pose_mAP50_95":s42["pose_mAP50_95_60ep"],
                             "identity_best":s42["identity_best_60ep"],
                             "yaw180_best":s42["yaw180_best_60ep"]},
         "★note":"seed 평균 하나로 숨기지 않는다. 개별 보고.",
         "curve":rowsout}
    json.dump(res,open(os.path.join(OUT,"SEED43_VERDICT.json"),"w"),indent=1,ensure_ascii=False)
    notify(f"**FIXED seed43 완료** (재시도 {restarts}, NaN {nan})\n\n"
           f"```\n         poseMAP50  50-95   identity  yaw180\n"
           f"seed42    {s42['pose_mAP50_60ep']:.4f}  {s42['pose_mAP50_95_60ep']:.4f}   "
           f"{s42['identity_best_60ep']:.3f}    {s42['yaw180_best_60ep']:.3f}\n"
           f"seed43    {last['pose_mAP50']:.4f}  {last['pose_mAP50_95']:.4f}   "
           f"{last['identity_best_fraction']:.3f}    {last['yaw180_best_fraction']:.3f}\n```\n"
           f"★ seed 평균으로 합치지 않음. real 평가는 reviewed fixed GT 도착 후.")
    log("done")


if __name__=="__main__":
    main()

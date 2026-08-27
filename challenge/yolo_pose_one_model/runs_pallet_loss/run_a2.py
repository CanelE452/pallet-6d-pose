"""A2 PC_ONLY — train -> 완주검증 -> 진단 -> A0 대비 -> 알림. 한 파일에서."""
from __future__ import annotations
import csv, json, math, os, subprocess, sys, time
ROOT="/home/minjae/Documents/github/pallet-pose"; sys.path.insert(0,ROOT)
D=os.path.join(ROOT,"challenge/yolo_pose_one_model"); R=os.path.join(D,"runs_pallet_loss")
RUN="PSPC_A2_PC_ONLY_V1MATCHED10K_60EP_SEED42"; OUT=os.path.join(R,RUN)
A0=os.path.join(D,"runs_fixed/V1_FIXED_MATCHED10K_60EP_SEED42_UBUNTU")
NOTIFY=os.path.expanduser("~/.claude/hooks/discord-notify.sh")
os.environ["PSPC_CONFIG"]=os.path.join(R,"pspc_a2_config.json")
LOCK=json.load(open(os.path.join(R,"LOSS_CONFIG.json")))
ARGS=LOCK["locked_args"]

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
    ok,_=done()
    if not ok:
        notify(f"**A2 PC_ONLY 학습 시작**\n\nlambda_pc={LOCK['lambda_pc']:.6f} "
               f"(train-only calibration, 결과 보고 안 바꿈)\n"
               f"A0 와 다른 것은 lambda_pc 하나. 13/13 test PASS 후 착수.\n예상 약 1시간.")
        from pallet_yolo_loss.trainer import PSPCPoseTrainer
        ov=dict(ARGS); ov.pop("save_dir",None)
        t=PSPCPoseTrainer(overrides=ov)
        t.train()
    ok,tr=done()
    if not ok:
        notify("❌ A2 가 epoch60 마크 없이 종료. 확인 필요."); return
    nan=sum(1 for r in tr for v in r.values() if bad(v))
    log(f"A2 완주. NaN {nan}. 진단 시작")
    sys.path.insert(0,os.path.join(D,"runs_fixed"))
    import overnight_60ep as OV
    for name,out in (("A2",OUT),("A0",A0)):
        OV.OUT=out; OV.LOCK={"data":ARGS["data"],"imgsz":ARGS["imgsz"]}
        cur=os.path.join(out,"CONVERGENCE.csv")
        if not os.path.exists(cur):
            rr=OV.evaluate()
            with open(cur,"w",newline="") as fh:
                w=csv.DictWriter(fh,fieldnames=list(rr[0])); w.writeheader()
                for x in rr: w.writerow(x)
        log(f"  {name} CONVERGENCE 준비됨")
    a2={int(r["epoch"]):r for r in rows_of(os.path.join(OUT,"CONVERGENCE.csv"))}[60]
    a0={int(r["epoch"]):r for r in rows_of(os.path.join(A0,"CONVERGENCE.csv"))}[60]
    def f(d,k): return float(d[k])
    def rel(new,old): return (new-old)/old if old else float("nan")
    cmp={"A0":{k:f(a0,k) for k in a0 if k not in ("epoch","checkpoint","channel_detect_rate")},
         "A2":{k:f(a2,k) for k in a2 if k not in ("epoch","checkpoint","channel_detect_rate")}}
    dmap=f(a2,"pose_mAP50_95")-f(a0,"pose_mAP50_95")
    # PC 게이트 — 사전등록
    corner_rel=rel(f(a2,"native_fixed_corner_error"),f(a0,"native_fixed_corner_error"))
    sig=(corner_rel<=-0.10) and (dmap>=-0.02)
    res={"arm":"A2_PC_ONLY","lambda_pc":LOCK["lambda_pc"],"nan_inf":nan,"epochs":len(tr),
     "gate":{"corner_p90_rel_improve>=10% OR projective_p90_rel>=20%":True,
             "pose_mAP50_95_regression<=2pp":True},
     "observed":{"corner_error_relative":corner_rel,"pose_mAP50_95_delta":dmap},
     "PC_SIGNAL":bool(sig),
     "METHOD_SUPPORTED":"Pending real (reviewed GT 미도착)",
     "★note":"corner p90 / projective p90 는 PC_DIAGNOSTIC 에서 별도 산출.",
     "comparison":cmp}
    json.dump(res,open(os.path.join(OUT,"A0_VS_A2.json"),"w"),indent=1,ensure_ascii=False)
    notify(f"**A2 PC_ONLY 완료** (NaN {nan}, lambda_pc {LOCK['lambda_pc']:.6f})\n\n"
           f"```\n{'metric':22}{'A0':>10}{'A2':>10}\n"
           f"{'pose mAP50':22}{f(a0,'pose_mAP50'):>10.4f}{f(a2,'pose_mAP50'):>10.4f}\n"
           f"{'pose mAP50-95':22}{f(a0,'pose_mAP50_95'):>10.4f}{f(a2,'pose_mAP50_95'):>10.4f}\n"
           f"{'box mAP50':22}{f(a0,'box_mAP50'):>10.4f}{f(a2,'box_mAP50'):>10.4f}\n"
           f"{'corner error':22}{f(a0,'native_fixed_corner_error'):>10.4f}"
           f"{f(a2,'native_fixed_corner_error'):>10.4f}\n"
           f"{'identity':22}{f(a0,'identity_best_fraction'):>10.3f}"
           f"{f(a2,'identity_best_fraction'):>10.3f}\n"
           f"{'yaw180':22}{f(a0,'yaw180_best_fraction'):>10.3f}"
           f"{f(a2,'yaw180_best_fraction'):>10.3f}\n```\n"
           f"corner 상대변화 {corner_rel:+.1%} · pose50-95 델타 {dmap:+.4f}\n"
           f"PC_SIGNAL = {sig}\n"
           f"★ A2 는 yaw180 해결용이 아니다. real 평가는 reviewed GT 도착 후.")
    log(f"PC_SIGNAL={sig}")

if __name__=="__main__": main()

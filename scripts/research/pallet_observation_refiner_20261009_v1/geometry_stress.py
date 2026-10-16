"""Fixed seed, render-free mathematical mask/outlier diagnostic (not real data)."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from solver import HypothesisBank, POLICY, cuboid, project

SEED = 20261009
CONDITIONS = ["VALID_ONLY", "DROP_ONE_CORRECT", "DROP_TWO_CORRECT",
              "KEEP_ONE_WRONG", "KEEP_TWO_WRONG", "DROP_KEEP_ONE_EACH",
              "DROP_KEEP_TWO_EACH", "TWO_COHERENT_WRONG", "FOUR_VALID",
              "FIVE_VALID", "NEAR_COLLINEAR"]


def stats(values):
    a=np.array([v for v in values if v is not None],float)
    return dict(n=len(a),mean=float(a.mean()),variance=float(a.var(ddof=1)) if len(a)>1 else None,
                std=float(a.std(ddof=1)) if len(a)>1 else None,median=float(np.median(a)),
                P90=float(np.quantile(a,.9)),maximum=float(a.max())) if len(a) else dict(n=0)


def run(scenes,output):
    if scenes < 1 or scenes > 128:
        raise ValueError("Fixed analytic budget permits 1..128 scenes")
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    destinations=[output/"GEOMETRY_STRESS.jsonl.gz",output/"GEOMETRY_STRESS.csv",output/"GEOMETRY_STRESS_SUMMARY.json"]
    if any(p.exists() for p in destinations):
        raise FileExistsError("Refusing to overwrite geometry stress outputs")
    cv2.setNumThreads(1)
    rng=np.random.default_rng(SEED)
    K=np.array([[570.,0.,320.],[0.,573.,240.],[0.,0.,1.]])
    xyz=np.array([1.3,.11,1.1]);X=cuboid(*xyz)
    rows=[];ledger=None;started=time.perf_counter()
    for scene in range(scenes):
        # Analytical settings are fixed independently of real images or GT.
        rv=np.array([rng.uniform(-.6,.6),rng.uniform(-.6,.6),rng.uniform(-.2,.2)])
        R=cv2.Rodrigues(rv)[0]
        t=np.array([rng.uniform(-.15,.15),rng.uniform(-.1,.1),rng.uniform(3.,4.5)])
        true=project(X,R,t,K)
        noisy=true+rng.normal(0.,1.,true.shape)
        center=project(np.zeros((1,3)),R,t,K)[0]
        order=rng.permutation(8).tolist()
        drop_ids=order[:2];bad_ids=order[2:4]
        # Reuse banks when q is unchanged; masked/full scoring still differs.
        banks={}
        for condition in CONDITIONS:
            q=np.vstack([noisy.copy(),center]);excluded=[];bad=[];geometry_control=False
            if condition=="DROP_ONE_CORRECT":excluded=drop_ids[:1]
            elif condition=="DROP_TWO_CORRECT":excluded=drop_ids
            elif condition=="KEEP_ONE_WRONG":bad=bad_ids[:1]
            elif condition=="KEEP_TWO_WRONG":bad=bad_ids
            elif condition=="DROP_KEEP_ONE_EACH":excluded=drop_ids[:1];bad=bad_ids[:1]
            elif condition=="DROP_KEEP_TWO_EACH":excluded=drop_ids;bad=bad_ids
            elif condition=="TWO_COHERENT_WRONG":bad=bad_ids
            elif condition=="FOUR_VALID":excluded=[3,5,6,7]
            elif condition=="FIVE_VALID":excluded=[3,6,7]
            elif condition=="NEAR_COLLINEAR":
                # 2D rank approaches one; these remain invalid 3D-2D correspondences.
                q[:8]=np.column_stack([np.linspace(150.,450.,8),240.+1e-6*np.linspace(-1.,1.,8)**2])
                bad=list(range(8));geometry_control=True
            if condition=="TWO_COHERENT_WRONG":
                alternate_t=t+np.array([24.*t[2]/K[0,0],0.,0.])
                alternative_projection=project(X,R,alternate_t,K)
                q[bad]=alternative_projection[bad]+(noisy-true)[bad]
            elif bad and not geometry_control:
                for i in bad:
                    angle=float(rng.uniform(0.,2*np.pi))
                    q[i]+=24.*np.array([np.cos(angle),np.sin(angle)])
            digest=hashlib.sha256(q.tobytes()).hexdigest()
            if digest not in banks:
                banks[digest]=HypothesisBank(q,K,xyz)
            bank=banks[digest]
            for robust in [False,True]:
                result=bank.solve(excluded,robust=robust,hidden=excluded)
                U=result["used"]
                true_correct=[i for i in U if i not in bad]
                fi=result["inliers"]
                eT=eR=add=None
                if result["available"]:
                    estimate_R=np.array(result["R_physical"]);estimate_t=np.array(result["centroid"])
                    eT=float(np.linalg.norm(estimate_t-t)*100.)
                    rel=R.T@estimate_R
                    eR=float(np.degrees(np.arccos(np.clip((np.trace(rel)-1.)/2.,-1.,1.))))
                    add=float(np.linalg.norm((estimate_R@X.T).T+estimate_t-((R@X.T).T+t),axis=1).mean())
                actual_bad_errors=np.linalg.norm(q[:8]-true,axis=1)
                row=dict(scene_id=f"ANALYTIC_{scene:03d}",condition=condition,
                         solver="ROBUST" if robust else "STANDARD",analytical_only=True,
                         K=K.tolist(),dimensions_m=xyz.tolist(),truth_R=R.tolist(),truth_t=t.tolist(),
                         points_input=q.tolist(),excluded=excluded,bad_correspondence_ids=bad,
                         correct_remaining_ids=true_correct,correct_remaining_count=len(true_correct),
                         remaining_count=len(U),wrong_retained_count=len(set(bad)&set(U)),
                         false_removed_count=len(excluded) if not geometry_control else 0,
                         injected_wrong_under_8px_ids=[i for i in bad if actual_bad_errors[i]<=8.],
                         bad_displacement_from_truth_px={str(i):float(actual_bad_errors[i]) for i in bad},
                         inlier_correct_count=len(set(fi)&set(true_correct)),
                         inlier_wrong_count=len(set(fi)&set(bad)),
                         correct_remaining_layout=X[true_correct].tolist(),
                         pose_available=result["available"],translation_cm=eT,rotation_deg=eR,ADD_m=add,
                         result=result)
                rows.append(row)
        for bank in banks.values():
            if ledger is None:ledger={k:0 for k in bank.ledger}
            for k,v in bank.ledger.items():ledger[k]+=v
        if (scene+1)%16==0:
            print(json.dumps(dict(completed_scenes=scene+1,logical_paths=len(rows),wall_seconds=time.perf_counter()-started)),flush=True)
    with gzip.open(destinations[0],"wt",encoding="utf-8") as fp:
        for row in rows:fp.write(json.dumps(row,separators=(",",":"),allow_nan=False)+"\n")
    columns=["scene_id","condition","solver","pose_available","translation_cm","rotation_deg","ADD_m",
             "false_removed_count","wrong_retained_count","correct_remaining_count","remaining_count",
             "inlier_correct_count","inlier_wrong_count","inliers","condition_number","reason",
             "candidate_count","eligible_candidate_count","refit_count","generic_calls","lm_calls"]
    with destinations[1].open("w",newline="") as fp:
        writer=csv.DictWriter(fp,fieldnames=columns);writer.writeheader()
        for r in rows:
            flat={k:r.get(k) for k in columns};z=r["result"]
            flat.update(inliers=json.dumps(z["inliers"]),
                        condition_number=z["geometry"].get("jacobian",{}).get("condition_number"),
                        reason=z["reason"],candidate_count=z["candidate_count"],
                        eligible_candidate_count=z["eligible_candidate_count"],refit_count=z["refit_count"],
                        generic_calls=z["operation_counts"]["generic_calls"],lm_calls=z["operation_counts"]["lm_calls"])
            writer.writerow(flat)
    summary={}
    for condition in CONDITIONS:
        summary[condition]={}
        for solver in ["STANDARD","ROBUST"]:
            subset=[r for r in rows if r["condition"]==condition and r["solver"]==solver]
            summary[condition][solver]=dict(total=len(subset),available=sum(r["pose_available"] for r in subset),
                     translation_cm=stats([r["translation_cm"] for r in subset]),
                     rotation_deg=stats([r["rotation_deg"] for r in subset]),
                     ADD_m=stats([r["ADD_m"] for r in subset]),
                     inlier_correct_count=stats([r["inlier_correct_count"] for r in subset]),
                     inlier_wrong_count=stats([r["inlier_wrong_count"] for r in subset]))
    doc=dict(complete=True,seed=SEED,scenes=scenes,conditions=CONDITIONS,logical_paths=len(rows),
             prior_used=False,policy=POLICY,noise_sigma_px=1.,outlier_move_px=24.,
             synthetic_RGB_rendered=0,generation="fixed-seed analytical settings; no physical visibility claim",
             ledger=ledger,wall_seconds=time.perf_counter()-started,summary=summary,
             interpretation="Rendered RGB=0. Controlled mathematical diagnostic, not real-world evidence. Four-consensus is weak; wrong coherent correspondence can be pose-consistent.")
    destinations[2].write_text(json.dumps(doc,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(complete=True,logical_paths=len(rows),wall_seconds=doc["wall_seconds"],ledger=ledger)))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--scenes",type=int,default=128);p.add_argument("--output",required=True)
    a=p.parse_args();run(a.scenes,a.output)

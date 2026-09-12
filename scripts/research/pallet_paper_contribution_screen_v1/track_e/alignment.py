"""Normal-direction RGB/depth boundary audit excluding ground-contact edges."""
import csv
import json
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.contracts import ROOT,RAW,DOC,sha,write

OUT=DOC/'E_privileged_depth_teacher'
PRIOR=ROOT/'data/pallet/results/paper_depth_selftrain_v1/sensor_validation_v1'

def summarize(rows):
    values=np.array([r['signed_px'] for r in rows if r['signed_px'] is not None])
    sessions={s:[r for r in rows if r['session']==s] for s in sorted({r['session'] for r in rows})}
    per={}
    for name,rr in sessions.items():
        v=np.array([r['signed_px'] for r in rr if r['signed_px'] is not None])
        per[name]=dict(attempted=len(rr),usable=len(v),coverage=len(v)/max(len(rr),1),
            signed_median=float(np.median(v)) if len(v) else None,
            signed_mean=float(np.mean(v)) if len(v) else None,
            absolute_median=float(np.median(np.abs(v))) if len(v) else None)
    coverage=len(values)/max(len(rows),1)
    med=float(np.median(np.abs(values))) if len(values) else None
    bias=float(np.median(values)) if len(values) else None
    safe=sum(r['signed_median'] is not None and abs(r['signed_median'])<=5 for r in per.values())/max(len(per),1)
    passed=bool(coverage>=.8 and med is not None and med<=3 and safe>=.8 and abs(bias)<=3)
    return dict(status='PASS' if passed else 'FAIL',attempted=len(rows),usable=len(values),coverage=coverage,
        missing_boundary_rate=1-coverage,median_absolute_px=med,median_signed_px=bias,
        mean_signed_px=float(np.mean(values)) if len(values) else None,session_bias_safe_fraction=safe,per_session=per)

def main():
    population=PRIOR/'SENSOR_VALIDATION_POPULATION.csv'
    write(OUT/'PROTOCOL_LOCK.json',dict(status='LOCKED_BEFORE_MEASUREMENT',population_sha256=sha(population),
        prior_bad_reference='capturepallet11, independently predesignated; report all and excluded, gate on excluded',
        exclusion_source_sha256=sha(PRIOR/'SENSOR_VALIDATION_RESULT.json'),
        boundary='convex hull of manual 8 corners; retain semantic cuboid silhouette edges except edges whose BOTH endpoints are bottom (2,3,6,7); no internal/amodal or bottom edges',
        samples_per_edge=20,sample_fraction=[.05,.95],search_normal_px=[-12,12],step_px=.5,
        normal_sign='outward from manual hull centroid',nearest_tie='minimum abs displacement then negative signed displacement',
        depth_edge='unchanged prior Sobel magnitude>=97th percentile of valid depths; require valid metric depth',
        scale=.001,scale_fitting=False,offset_fitting=False,
        gates=dict(coverage_min=.8,median_abs_max_px=3,session_bias_max_px=5,session_safe_min=.8,global_bias_max_px=3),
        evidence_level='MECHANISM_ONLY',no_teacher_or_student_before_PASS=True))
    rows=[];frame_notes=[]
    semantic={frozenset(e) for e in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]}
    bottom={2,3,6,7}
    for frame in csv.DictReader(population.open()):
        path=ROOT/'data/evaluation/pallet_eval_v1'/frame['annotation_path']
        payload=json.loads(path.read_text());points=np.array([p if p else [np.nan,np.nan] for p in payload['objects'][0]['projected_cuboid']],float)[:8]
        valid=np.isfinite(points).all(-1);indices=np.flatnonzero(valid)
        raw=cv2.imread(str(ROOT/frame['depth_path']),cv2.IMREAD_UNCHANGED)
        if raw is None or len(indices)<3:
            frame_notes.append(dict(id=frame['frame_id'],status='UNUSABLE_INPUT'));continue
        z=raw.astype(np.float32)*.001;good=(raw>0)&(raw<np.iinfo(raw.dtype).max)
        gx=cv2.Sobel(np.where(good,z,0),cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(np.where(good,z,0),cv2.CV_32F,0,1,ksize=3)
        mag=np.hypot(gx,gy);edge=(mag>=np.percentile(mag[good],97))&good if good.any() else np.zeros_like(good)
        hull=indices[cv2.convexHull(points[valid].astype(np.float32),returnPoints=False).ravel()]
        center=points[hull].mean(0);kept=[]
        for a,b in zip(hull,np.roll(hull,-1)):
            if frozenset([a,b]) not in semantic or (a in bottom and b in bottom):continue
            d=points[b]-points[a];normal=np.array([-d[1],d[0]])/np.linalg.norm(d)
            if np.dot(normal,(points[a]+points[b])/2-center)<0:normal=-normal
            kept.append([int(a),int(b)])
            for alpha in np.linspace(.05,.95,20):
                origin=points[a]*(1-alpha)+points[b]*alpha
                offsets=np.arange(-12,12.001,.5);sample=origin+offsets[:,None]*normal
                xy=np.rint(sample).astype(int);inside=(xy[:,0]>=0)&(xy[:,0]<edge.shape[1])&(xy[:,1]>=0)&(xy[:,1]<edge.shape[0])
                found=[float(t) for t,(x,y),ok in zip(offsets,xy,inside) if ok and edge[y,x]]
                signed=min(found,key=lambda v:(abs(v),v)) if found else None
                rows.append(dict(id=frame['frame_id'],session=frame['source_recording'],edge=[int(a),int(b)],
                    alpha=float(alpha),normal=normal.tolist(),signed_px=signed,prior_bad_reference=frame['source_recording']=='capturepallet11'))
        frame_notes.append(dict(id=frame['frame_id'],status='MEASURED',eligible_edges=kept))
    all_result=summarize(rows);clean=summarize([r for r in rows if not r['prior_bad_reference']])
    write(RAW/'E_privileged_depth_teacher/BOUNDARY_SAMPLES.json',rows)
    write(RAW/'E_privileged_depth_teacher/PER_FRAME.json',frame_notes)
    write(OUT/'MECHANISM_RESULT.json',dict(status=clean['status'],all_references=all_result,excluded_prior_bad_reference=clean,
        verdict='E_ALIGNMENT_PASS_REQUIRES_TEACHER' if clean['status']=='PASS' else 'E_RGBD_ALIGNMENT_NOT_READY',
        teacher_status='NOT_RUN',student_optimizer_updates=0,evidence_level='MECHANISM_ONLY'))
    print(dict(all=all_result['status'],clean=clean['status'],coverage=clean['coverage'],median_abs=clean['median_absolute_px']),flush=True)

if __name__=='__main__':main()

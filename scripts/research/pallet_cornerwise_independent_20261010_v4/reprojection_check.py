"""Check real stored R,t/H output independently with scalar matrix arithmetic."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

DOC=Path(__file__).resolve().parents[3]/'_docs/experiments/pallet_cornerwise_independent_20261010_v4'
SIGNS=((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=DOC)
    p.add_argument('--output',type=Path,default=DOC/'REPROJECTION_CHECKS.json');a=p.parse_args()
    assert not a.output.exists() and not a.output.is_symlink(),'preserve completed check'
    path=a.input/'PREDICTIONS.jsonl.gz'
    with gzip.open(path,'rt') as f:rows=[json.loads(line) for line in f if line.strip()]
    frames=0;new=0;hidden=0;coordinatechecks=0;maximum=0.;failures=[]
    for row in rows:
        frames+=1;prior=row['evaluation_reference']['initial_N3_native_points'];out=row['native_points'];inputs=row['input_points']
        assert out[8]==prior[8],row['id']+' center changed'
        assert row['reprojections_reused_as_observations'] is False
        if not row['new_pose_estimated']:
            assert out==prior,row['id']+' fallback coordinates changed'
            continue
        new+=1;H=row['hidden_initial'];solver=row['solver'];pose=row['actual_pose']
        assert set(H).isdisjoint(solver['fit_input_ids'])
        assert set(H)==set(row['reprojected_ids'])
        assert pose['available']
        R=pose['R_cf'];t=pose['centroid'];dims=pose['cf_extents'];K=row['K']
        for k in range(8):
            if k not in H:
                if all(v is not None and math.isfinite(v) for v in inputs[k]) and inputs[k]!=[-1,-1]:
                    assert out[k]==inputs[k],row['id']+' accepted observation was replaced by projection'
                    coordinatechecks+=1
                continue
            xyz=[SIGNS[k][j]*dims[j]/2 for j in range(3)]
            camera=[math.fsum(R[i][j]*xyz[j] for j in range(3))+t[i] for i in range(3)]
            pixel=[math.fsum(K[i][j]*camera[j] for j in range(3)) for i in range(3)]
            assert camera[2]>0 and pixel[2]!=0
            q=[pixel[0]/pixel[2],pixel[1]/pixel[2]]
            delta=max(abs(q[j]-out[k][j]) for j in range(2));maximum=max(maximum,delta);hidden+=1
            if delta>1e-9:failures.append(dict(id=row['id'],method=row['method'],corner=k,max_abs_px=delta))
            assert row['output_coordinate_sources'][k]=='FINAL_POSE_REPROJECTION'
    result=dict(passed=not failures,scored_rows=frames,new_pose_rows=new,actually_reprojected_corners=hidden,
        nonhidden_observation_coordinate_checks=coordinatechecks,center_checks=frames,
        max_Rt_projection_output_difference_px=maximum,tolerance_px=1e-9,failures=failures,
        raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),independent_scalar_matrix_arithmetic=True,
        actual_model_PnP_optimization_training_RGB_calls=0,physical_GT_or_accuracy_certification=False)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
    if failures:raise SystemExit(1)

if __name__=='__main__':main()

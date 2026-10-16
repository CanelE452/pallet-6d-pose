"""Check real stored R,t/H output independently with scalar matrix arithmetic."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

DOC=Path(__file__).resolve().parents[3]/'_docs/experiments/pallet_three_head_observation_20261010_v7'
SIGNS=((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1))

def compact_saved_witnesses(value):
    if isinstance(value,dict):
        return {key:compact_saved_witnesses(item) for key,item in value.items()
                if key not in ('all_candidate_solutions','alternatives')}
    if isinstance(value,list):return [compact_saved_witnesses(item) for item in value]
    return value

def compact_projection_row(row):
    row.pop('observation_contract',None)
    if 'solver' in row:
        solver=row.get('solver') or {}
        row['solver']={'fit_input_ids':solver.get('fit_input_ids',[])}
    return compact_saved_witnesses(row)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=DOC)
    p.add_argument('--output',type=Path,default=DOC/'REPROJECTION_CHECKS.json');a=p.parse_args()
    assert not a.output.exists() and not a.output.is_symlink(),'preserve completed check'
    path=a.input/'PREDICTIONS.jsonl.gz'
    receipt=json.loads((a.input/'SCORING_RECEIPT.json').read_text())
    assert receipt['complete'] is True
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    assert path.stat().st_size==receipt['predictions']['bytes'] and digest.hexdigest()==receipt['predictions']['sha256']
    with gzip.open(path,'rt') as f:
        rows=[compact_projection_row(json.loads(line)) for line in f if line.strip()]
    protocol=json.loads((a.input/'PROTOCOL.json').read_text())
    methods=set(protocol['methods'])
    assert len(methods)==8 and len(rows)==1960
    assert len({(r['method'],r['id']) for r in rows})==1960 and {r['method'] for r in rows}==methods
    for method in methods:assert len({r['id'] for r in rows if r['method']==method})==245
    frames=0;new=0;hidden=0;coordinatechecks=0;unobservedchecks=0;maximum=0.;failures=[]
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
                else:
                    assert out[k]==prior[k],row['id']+' unobserved non-H display changed'
                    assert row['output_coordinate_sources'][k]=='NATIVE_N3_RGB'
                    unobservedchecks+=1
                continue
            xyz=[SIGNS[k][j]*dims[j]/2 for j in range(3)]
            camera=[math.fsum(R[i][j]*xyz[j] for j in range(3))+t[i] for i in range(3)]
            pixel=[math.fsum(K[i][j]*camera[j] for j in range(3)) for i in range(3)]
            assert camera[2]>0 and pixel[2]!=0
            q=[pixel[0]/pixel[2],pixel[1]/pixel[2]]
            delta=max(abs(q[j]-out[k][j]) for j in range(2));maximum=max(maximum,delta);hidden+=1
            if delta>1e-9:failures.append(dict(id=row['id'],method=row['method'],corner=k,max_abs_px=delta))
            assert row['output_coordinate_sources'][k]=='FINAL_POSE_REPROJECTION'
    result=dict(passed=not failures,methods=sorted(methods),scored_rows=frames,new_pose_rows=new,actually_reprojected_corners=hidden,
        nonhidden_observation_coordinate_checks=coordinatechecks,center_checks=frames,
        unobserved_nonhidden_native_display_checks=unobservedchecks,
        max_Rt_projection_output_difference_px=maximum,tolerance_px=1e-9,failures=failures,
        raw_sha256=digest.hexdigest(),independent_scalar_matrix_arithmetic=True,
        actual_model_PnP_optimization_training_RGB_calls=0,physical_GT_or_accuracy_certification=False)
    result['saved_row_memory_policy']=dict(original_full_serialized_rows_preserved=True,
        complete_raw_SHA_checked_before_loading=True,
        observation_contract_and_unused_solver_witnesses_discarded_in_RAM=True,
        unused_keys_recursively_discarded_in_RAM=['all_candidate_solutions','alternatives'])
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
    if failures:raise SystemExit(1)

if __name__=='__main__':main()

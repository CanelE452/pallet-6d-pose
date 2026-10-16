import sys
from pathlib import Path
sys.path.insert(0,"/home/minjae/Documents/github/pallet-pose")
from copy import deepcopy
import shutil
import time
import numpy as np
from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C
from scripts.research.pallet_visibility_curriculum_20261013_v1.joint_ot_pnp import solve_joint_ot_pnp


STUDY=C.DOC/"nonsevere_v1/statewise_joint_ot_20261014_v2"
OUTPUT=STUDY/"jacobian_benchmark_v2"
ORIGIN=STUDY/"runtime_smoke_guarded_v2"
SOURCES=[Path("scripts/research/pallet_visibility_curriculum_20261013_v1")/name
         for name in ("joint_ot_pnp.py","joint_ot_jacobian_v2.py","common.py")]


def main():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    seal=C.read(ORIGIN/"PREDICTION_SEAL.json")
    assert C.sha(ORIGIN/"PREDICTIONS.jsonl.gz")==seal["predictions_sha256"]
    row=next(C.rows(ORIGIN/"PREDICTIONS.jsonl.gz"));runtime=row["runtime"]
    coarse=runtime["original_coarse_pose"]
    inputs=dict(xyz=runtime["initial_geometry"]["native_xyz"],K=runtime["initial_geometry"]["K"],
        states=runtime["states"],visible=runtime["frozen_visible_measurements"],
        domains=runtime["ot_domains"]["domains"],image_hw=runtime["initial_geometry"]["image_hw"],
        initializations=[dict(name="REGISTERED_COARSE_INITIALIZATION_ONLY",R=coarse["R_cf"],t=coarse["centroid"])])
    C.write(OUTPUT/"FIXED_PREDICTION_INPUT.json",inputs)
    config=dict(max_p3p_triples=56,global_rotation_starts=24,max_seed_nfev=160,
                max_joint_iterations=250,maximum_union_branches=256,maximum_visible_mahalanobis=3.)
    source_bindings={str(p):C.sha(C.ROOT/p) for p in SOURCES}
    for p in SOURCES:
        dest=OUTPUT/"code_snapshot"/p;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(C.ROOT/p,dest)
    shutil.copyfile(__file__,OUTPUT/"benchmark_run.py")
    protocol=dict(schema="EXACT_JOINT_JACOBIAN_BENCHMARK_V2",source_record_id=row["source_record_id"],partition="CAL",
        input_origin="sealed guarded first-head seen-CAL runtime smoke, no target coordinates",
        input_sha256=C.sha(OUTPUT/"FIXED_PREDICTION_INPUT.json"),source_prediction_seal_sha256=C.sha(ORIGIN/"PREDICTION_SEAL.json"),
        source_sha256=source_bindings,config=config,modes=[False,True],
        complete_same_P3P_global_starts_retained=True,GT_reads=0,score_invoked=False,
        timing_scope="pose solver only; cached immutable actual RGB B/I and O/T observations")
    C.write(OUTPUT/"PROTOCOL.json",protocol)
    rows=[]
    for analytic in [False,True]:
        start=time.perf_counter()
        result=solve_joint_ot_pnp(**deepcopy(inputs),config={**config,"analytic_jacobian":analytic})
        elapsed=time.perf_counter()-start
        assert result["accepted"]
        assert all(a["feasible"] for a in result["joint_constraint_audit"])
        assert all(a["measurement_budget_pass"] and a["protection_pass"] for a in result["visible_protection_audit"])
        rows.append(dict(id=row["source_record_id"]+"/"+str(analytic),analytic_jacobian=analytic,
                         elapsed_seconds=elapsed,result=result))
        print(dict(analytic=analytic,elapsed_seconds=elapsed,objective=result["objective"],
                   starts=result["search"]["initialization_count"],joint_starts=result["search"]["joint_initializations_per_branch"],
                   candidates=len(result["candidates"])),flush=True)
    numeric,exact=[r["result"] for r in rows]
    assert numeric["search"]["initialization_count"]==exact["search"]["initialization_count"]
    assert numeric["search"]["p3p_triples_searched"]==exact["search"]["p3p_triples_searched"]
    C.save_rows(OUTPUT/"PREDICTIONS.jsonl.gz",rows)
    if {str(p):C.sha(C.ROOT/p) for p in SOURCES}!=source_bindings:raise RuntimeError("Source changed during benchmark")
    C.write(OUTPUT/"PREDICTION_SEAL.json",dict(schema="JACOBIAN_BENCHMARK_SEAL_V2",before_scoring=True,GT_reads=0,
        score_invoked=False,rows=2,predictions_sha256=C.sha(OUTPUT/"PREDICTIONS.jsonl.gz"),
        protocol_sha256=C.sha(OUTPUT/"PROTOCOL.json"),source_sha256=source_bindings))
    summary=dict(schema="JACOBIAN_BENCHMARK_POSTSEAL_CHECKS_V2",GT_reads=0,score_invoked=False,
        numeric_seconds=rows[0]["elapsed_seconds"],analytic_seconds=rows[1]["elapsed_seconds"],
        speedup=rows[0]["elapsed_seconds"]/rows[1]["elapsed_seconds"],
        objective_numeric=numeric["objective"],objective_analytic=exact["objective"],
        rotation_matrix_difference=float(np.linalg.norm(np.array(numeric["R"])-np.array(exact["R"]))),
        translation_difference_m=float(np.linalg.norm(np.array(numeric["t"])-np.array(exact["t"]))),
        all_individual_joint_constraints_pass_both=True,
        same_finite_P3P_and_global_initializations=True,
        numeric_joint_initializations=numeric["search"]["joint_initializations_per_branch"],
        analytic_joint_initializations=exact["search"]["joint_initializations_per_branch"],
        source_prediction_seal_sha256=C.sha(OUTPUT/"PREDICTION_SEAL.json"),
        limitation="one preselected seen-CAL solver-only benchmark; exact gradients change convergence traces")
    C.write(OUTPUT/"BENCHMARK_CHECKS.json",summary);print(summary,flush=True)

if __name__=="__main__":main()

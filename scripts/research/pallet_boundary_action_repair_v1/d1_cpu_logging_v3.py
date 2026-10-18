"""CPU-only reporting facade: preserve every original solver failure record.

V2 observation source files stay byte-identical while their GPU job runs.
This module changes no inputs, math, starts, candidates, feature, or selection.
It copies original returned diagnostics before otherwise unused fields leave
scope. Original arguments, arithmetic order, return objects and exceptions are
preserved. Both domains use this same V3 CPU facade.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
from types import FunctionType,SimpleNamespace
import traceback

from . import d1_batch as B,d1_producer as P
from scripts.research.pallet_statewise_end_to_end_recovery_v1.common import canonical,sha,write


def source_bindings():
    result=B.source_bindings();root=Path(__file__).parents[3]
    result[str(Path(__file__).relative_to(root))]=sha(__file__);return result


def _summary(solution):
    keys=('schema','accepted','status','fallback_reason','search','parent_search_audit',
        'candidates','objective','objective_components','minimum_constraint_slack','rank',
        'ambiguity','prior_audit','fallback_diagnostics','boundary_factor_audit',
        'boundary_observation_audit','global_outline_audit','OT_constraints_satisfied',
        'requested_prior_weight','effective_prior_weight')
    return {key:deepcopy(solution[key])for key in keys if key in solution}


def logged_solve_e(row,frame,mask,config,model,interval):
    from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_partial_boundary_u_solver_v4_6 as PS
    from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_two_wd_family_v4_9 as W
    from scripts.research.pallet_visibility_curriculum_20261013_v1.common import finite
    original_p,original_w=PS.solve_boundary_u_region_joint_ot_pnp,W.solve_two_wd_families
    calls=[];families=[]
    def partial(*args,**kwargs):
        answer=original_p(*args,**kwargs)
        cfg=kwargs.get('config',args[8]if len(args)>8 else {})
        calls.append(dict(call_index=len(calls),input_prior_weight=cfg.get('prior_weight'),returned_original_solution=_summary(answer)))
        return answer
    def two_family(*args,**kwargs):
        answer=original_w(*args,**kwargs)
        label=('WD0_SEEDS','G10','G13')[len(families)]if len(families)<3 else'OTHER_ORIGINAL_FAMILY_CALL'
        families.append(dict(stage=label,selected_family=answer['selected_family'],family_ambiguity=deepcopy(answer.get('family_ambiguity')),
            families=[dict(name=f['name'],accepted=f['accepted'],objective=f['objective'],solution=_summary(f['solution']))for f in answer['families']]))
        return answer
    PS.solve_boundary_u_region_joint_ot_pnp=partial;W.solve_two_wd_families=two_family
    try:
        result=P.solve_e(row,frame,mask,config,model,interval)
        result['stage_execution_diagnostics']=finite(dict(scope='all original returned P1/WD0/G10/G13 outcomes, including rejected and unsuccessful optimizer records',
            P1_and_WD_backend_calls=calls,family_stage_calls=families,
            feasible_pose_coordinates_not_duplicated_here=True,metadata_only=True))
        result['proof']['CPU_reporting_facade']=dict(version='V3',source_sha256=sha(__file__),numerical_changes=False,extra_solver_calls=0)
        return result
    except BaseException as exc:
        exc.d1_stage_execution_diagnostics=finite(dict(P1_and_WD_backend_calls=calls,family_stage_calls=families,
            status='FAILED_TECHNICAL',error=repr(exc)))
        raise
    finally:PS.solve_boundary_u_region_joint_ot_pnp=original_p;W.solve_two_wd_families=original_w


def solver_job(frame,observation_folder,output_folder,old_run):
    operation=FunctionType(B.solver_job.__code__,{**B.solver_job.__globals__,
        'P':SimpleNamespace(**{**vars(P),'solve_e':logged_solve_e}),'source_bindings':source_bindings},name='_unchanged_CPU_job_with_reporting_v3')
    try:return operation(frame,observation_folder,output_folder,old_run)
    except BaseException as exc:
        if hasattr(exc,'d1_stage_execution_diagnostics'):
            folder=Path(output_folder);folder.mkdir(parents=True,exist_ok=True)
            write(folder/'STAGE_FAILURE_DIAGNOSTICS.json',exc.d1_stage_execution_diagnostics)
        raise


def solve_batch(*args,**kwargs):
    operation=FunctionType(B.solve_batch.__code__,{**B.solve_batch.__globals__,
        'solver_job':solver_job,'source_bindings':source_bindings},name='_unchanged_CPU_batch_with_reporting_v3')
    operation.__kwdefaults__=deepcopy(B.solve_batch.__kwdefaults__)
    return operation(*args,**kwargs)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--observation-dir',required=True)
    p.add_argument('--output',required=True);p.add_argument('--old-run',required=True);p.add_argument('--domain',choices=['real','synthetic'],required=True)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--limit',type=int);p.add_argument('--technical',action='store_true');p.add_argument('--root-GO',action='store_true')
    a=p.parse_args();print(canonical(solve_batch(a.inputs,a.observation_dir,a.output,a.old_run,a.domain,
        workers=a.workers,limit=a.limit,technical=a.technical,GO=a.root_GO)))

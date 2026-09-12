"""Geometry-only diagnostic: hold observations, starts, dimensions and population fixed.

Oracle lines are allowed ONLY as marked diagnostic inputs, never as deployment
measurements, training data for an arm comparison, or checkpoint-selection data.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
import torch
from .solver import Measurements,PointLineRefiner,SolverConfig
from .geometry import lines_from_points,cuboid,project,so3_exp
from .fixtures import geometry_fixture
from .io import write_json


def compare_line_sources(measured:Measurements,dims,K,R0,t0,gt_R,gt_t,iterations=12):
    target,z=project(cuboid(dims),gt_R,gt_t,K)
    gt_lines,gt_line_valid=lines_from_points(target)
    oracle=replace(measured,lines=gt_lines[:,:,None],line_valid=gt_line_valid[:,:,None],
                   line_logprob=torch.zeros_like(gt_line_valid[:,:,None],dtype=target.dtype),
                   line_sigma=measured.line_sigma[:,:,:1])
    result={}
    for name,measurement,lw in [('point_only',measured,0.),('predicted_line',measured,.25),('ORACLE_line_DIAGNOSTIC_ONLY',oracle,.25)]:
        out=PointLineRefiner(SolverConfig(iterations=iterations,line_weight=lw))(measurement,dims,K,R0,t0)
        err=torch.linalg.vector_norm(out['points'][:,:8]-target[:,:8],dim=-1).mean(-1)
        result[name]={'mean_corner_error_px':err.detach().cpu().tolist(),
                      'translation_error_m':torch.linalg.vector_norm(out['t']-gt_t,dim=-1).detach().cpu().tolist(),
                      'pose_valid':out['pose_valid'].cpu().tolist(),'selected_start':out['selected'].cpu().tolist(),
                      'objective_trace':out['objective_trace'].detach().cpu().tolist()}
    return {'scope':'ORACLE_GEOMETRY_DIAGNOSTIC_NOT_DEPLOYMENT','arms':result,
            'start_count':R0.shape[1],'same_initialization':True,
            'interpretation':'An oracle failure can indicate initialization/correspondence/objective/conditioning problems, NOT absence of line information.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    a=p.parse_args()
    if Path(a.output).exists():raise ValueError('Output exists')
    torch.set_num_threads(2)
    dims,K,R,t,target,m=geometry_fixture(3,noise=1.5,seed=17)
    biased=replace(m,lines=m.lines+torch.tensor([0.,0.,4.],dtype=m.lines.dtype))
    R0=(so3_exp(torch.tensor([[.08,-.07,.03]],dtype=R.dtype))@R)[:,None]
    t0=(t+torch.tensor([[.04,-.03,.08]],dtype=t.dtype))[:,None]
    result=compare_line_sources(biased,dims,K,R0,t0,R,t)
    result['input']='generated mathematical fixtures'
    result['initialization']='GT-near perturbation for local solvability test ONLY; does not validate global/RGB initialization.'
    write_json(a.output,result)
    print(result['scope'])
if __name__=='__main__':main()

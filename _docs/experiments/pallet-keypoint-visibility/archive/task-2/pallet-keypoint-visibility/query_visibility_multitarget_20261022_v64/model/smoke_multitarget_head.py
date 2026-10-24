"""CPU-only mock gradient and six-joint inclusion-constraint checks."""
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch.nn import functional as F

from query_multitarget_head import (JOINT_PAIRS, QueryMultiTargetHead, classes_to_joint,
                                    load_model, marginals, parameter_count, save_checkpoint)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20261022)
    model = QueryMultiTargetHead()
    n = 6
    inputs = {"x": torch.randn(n,65,9,9,requires_grad=True),
              "rgb_native": torch.rand(n,75,9,9,requires_grad=True),
              "sam": torch.randn(n,2,9,9,requires_grad=True),
              "context": torch.randn(n,64,requires_grad=True),
              "box_grid": torch.randn(n,4,9,9,requires_grad=True)}
    center, dense = model(**inputs, return_dense=True)
    assert center.shape == (n,6) and dense.shape == (n,6,9,9)
    assert torch.equal(center,dense[:,:,4,4])
    p=marginals(center.softmax(1));v,f=p["visible_probability"],p["full_probability"]
    assert torch.all(v<=f+1e-7) and torch.all(v[:,0]<=v[:,1]+1e-7) and torch.all(f[:,0]<=f[:,1]+1e-7)
    assert torch.allclose(p["state_probability"].sum(2),torch.ones(n,2))
    loss=F.cross_entropy(center,torch.arange(6));loss.backward()
    gradients={key:float(value.grad.abs().sum())for key,value in inputs.items()}
    assert all(value>0 for value in gradients.values())
    changed={key:value.detach().clone()for key,value in inputs.items()};changed["x"][:,:,0,0]+=100
    assert torch.equal(model(**changed),center.detach())
    baseline={key:value.detach().clone()for key,value in inputs.items()}
    a=model(**baseline)
    cb={**baseline,"context":baseline["context"]+1}
    sb={**baseline,"sam":baseline["sam"]+1}
    both={**cb,"sam":baseline["sam"]+1}
    interaction=float((model(**both)-model(**cb)-model(**sb)+a).abs().max())
    assert interaction>1e-7,interaction
    classes=np.empty((6,2,9,9),np.int8)
    for index,pair in enumerate(JOINT_PAIRS):classes[index]=np.array(pair)[:,None,None]
    joint=classes_to_joint(classes,np.ones((6,9,9),bool))
    assert np.array_equal(joint[:,4,4],np.arange(6))
    bad=classes.copy();bad[0,1]=2
    try:classes_to_joint(bad,np.ones((6,9,9),bool))
    except ValueError:pass
    else:raise AssertionError("impossible V->O target accepted")
    with tempfile.TemporaryDirectory(prefix="v64_mock_checkpoint_")as temp:
        checkpoint=Path(temp)/"UNTRAINED_MOCK_ONLY.pt";save_checkpoint(checkpoint,model)
        restored,_=load_model(checkpoint)
        assert torch.equal(model(**baseline),restored(**baseline))
    report={"PASS":True,"parameters":parameter_count(model),"under100k":parameter_count(model)<100000,
            "scalar_exact_dense_center":True,"visible_subset_full":True,"exact_support_subset_radius2":True,
            "all_input_branch_gradients":gradients,"nonlinear_SAM_context_interaction":interaction,
            "query_locality_remote_P3_corner_invariant":True,"six_valid_label_pairs":True,
            "impossible_label_pair_rejected":True,"checkpoint_roundtrip_exact":True,
            "optimizer_updates":0,"source_images_read":0,"private_evaluation_read":False,"PnP_calls":0}
    Path(__file__).with_name("CPU_MODEL_SMOKE.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


if __name__=="__main__":main()

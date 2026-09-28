"""CPU-only inventory of the exact frozen R0 architecture/trainable predicate."""
from pathlib import Path
import time
from . import common as C


def main():
    started=time.perf_counter()
    import torch
    from scripts.research.pallet_type_selftrain_v1 import recovery_pose_trainer as T
    materials={};bindings=[]
    for material in ('PLASTIC','WOOD'):
        binding=C.checkpoint(material,'R0');C.verify(binding);bindings.append(binding)
        model=torch.load(C.ROOT/binding['path'],map_location='cpu',weights_only=False)['model'].float()
        parameters=dict(model.named_parameters());buffers=dict(model.named_buffers());state=model.state_dict()
        selected={name:value for name,value in parameters.items() if T.pose_parameter(name)}
        protected={name:value for name,value in state.items() if name not in selected}
        assert not set(selected)&set(buffers)
        groups={}
        for name,value in selected.items():
            prefix='.'.join(name.split('.')[:3]);row=groups.setdefault(prefix,dict(parameter_tensors=0,scalar_parameters=0))
            row['parameter_tensors']+=1;row['scalar_parameters']+=value.numel()
        materials[material]=dict(total_parameter_tensors=len(parameters),total_scalar_parameters=sum(x.numel() for x in parameters.values()),
            trainable_parameter_tensors=len(selected),trainable_scalar_parameters=sum(x.numel() for x in selected.values()),
            frozen_parameter_tensors=len(parameters)-len(selected),frozen_scalar_parameters=sum(x.numel() for n,x in parameters.items() if n not in selected),
            state_tensors=len(state),all_buffers_frozen=len(buffers),protected_state_tensors=len(protected),groups=groups,
            criterion='Exact PoseOnlyTrainer._setup_train predicate applied to named_parameters; saved checkpoint requires_grad flags do not define new training scope.')
        assert C.bind(C.ROOT/binding['path'])==binding
    result=dict(status='CPU_EXACT_PARAMETER_INVENTORY',materials=materials,sources=bindings,
        predicate=C.bind(Path(T.__file__)),implementation=C.bind(Path(__file__)),
        extra_inference_components=0,new_manual=0,GPU_seconds=0,new_fits=0,optimizer_updates=0,
        CPU_wall_seconds=time.perf_counter()-started)
    C.save(C.DOC/'TRAINABLE_PARAMETER_INVENTORY.json',result)
    import json
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

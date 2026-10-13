"""Run the new S/B/I/O/T method on one actual image, without dataset labels."""
import argparse
from pathlib import Path

import cv2

from . import common as C
from .state_visibility_v2 import load_state_model
from .statewise_joint_api_v2 import StatewiseJointV2
from .statewise_masks_v2 import StatewiseMaskProvider


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image",required=True,type=Path)
    parser.add_argument("--input",required=True,type=Path)
    parser.add_argument("--config",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--device",default="cuda")
    args=parser.parse_args()
    config_path=args.config.resolve();config=C.read(config_path)
    if config.get("schema")!="STATEWISE_JOINT_RUNTIME_CONFIG_V2":raise ValueError("New runtime config required")
    resolved={name:(config_path.parent/config[name]).resolve()
              for name in ("state_model","OT_model","mask_descriptor","boundary_mask_config")}
    for name,path in resolved.items():
        if C.sha(path)!=config["bindings_sha256"][name]:raise ValueError("Frozen config binding changed: "+name)
    inputs=C.read(args.input)
    pose=inputs["initial_pose"]
    image=cv2.imread(str(args.image))
    if image is None:raise ValueError("Actual image could not be decoded")
    model=load_state_model(str(resolved["state_model"]),args.device)
    masks=StatewiseMaskProvider.from_configs(C.read(resolved["mask_descriptor"]),resolved["boundary_mask_config"],args.device)
    runtime=StatewiseJointV2(model,C.read(resolved["OT_model"]),masks,
        config=dict(config.get("runtime",{}),device=args.device,record_id=inputs.get("record_id","actual_image")))
    result=runtime.refine(image,inputs["raw_points"],inputs["K"],
        inputs.get("native_xyz",C.cuboid(*pose["cf_extents"])),pose,
        independent_truncation=inputs.get("independent_truncation"),crop_transform=inputs.get("crop_transform"))
    if args.output.exists():raise FileExistsError("Preserve existing inference output")
    C.write(args.output,result)
    print("states="+"".join(result["states"])+" joint_feasible="+str(result["accepted"])+" pose_source="+result["actual_pose"]["source"])


if __name__=="__main__":main()

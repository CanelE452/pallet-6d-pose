"""Bind pretraining provenance and both actual runtime environments."""
import platform
import torch
from env import *
def run():
    p=ROOT/'challenge/weights/pretrained_yolo/yolo26n-pose.pt';c=torch.load(p,map_location='cpu',weights_only=False)
    args=R0.parent.parent/'args.yaml'
    write(DOC/'R0_PRETRAINING_PROVENANCE.json',dict(baseline=bound(R0),run_args=bound(args),initialization_checkpoint=bound(p),upstream_training_dataset=c['train_args']['data'],upstream_pretrained=c['train_args']['pretrained'],upstream_version=c['version'],upstream_date=c['date'],pallet_run_pretrained=True,claim='P additional training is synthetic-only; full estimator includes prior real COCO-pose supervision; DEV/QA use real labels',directory_name_not_evidence='scratch in path does not override pretrained:true'))
    locks={}
    for name,python in [('torch',Path(sys.executable)),('TF1',RAW/'env_tf1/bin/python')]:
        freeze_text=subprocess.check_output([str(python),'-m','pip','freeze'],text=True)
        (DOC/('ENVIRONMENT_'+name+'.txt')).write_text(freeze_text)
        locks[name]=dict(python=str(python),version=subprocess.check_output([str(python),'--version'],text=True).strip(),packages=bound(DOC/('ENVIRONMENT_'+name+'.txt')))
    write(DOC/'ENVIRONMENT_LOCK.json',dict(platform=platform.platform(),environments=locks,torch_cuda=torch.version.cuda,driver='580.178.04',device='NVIDIA GeForce RTX3080',current_YOLO_environment_replaced=False,TF1_CPU_separate_environment=True,system_driver_reboot_power_changes=False))
    license_text=(OFFICIAL/'LICENSE').read_text()
    (HERE/'THIRD_PARTY_NOTICES.md').write_text('# Third-party notices\n\nPoseFix-derived architecture: official source commit5556364bb0f43b0743a5fcd820de48f34b3d4360.\nThe port is not an unmodified official TF1 GPU implementation.\n\n'+license_text+'\n\nTensorFlow ResNet v1 bottleneck/stride/padding organization:\nCopyright 2016 The TensorFlow Authors. All Rights Reserved.\nLicensed under the Apache License, Version2.0: https://www.apache.org/licenses/LICENSE-2.0\nChanges: PyTorch operators, pallet9 adapter, source-derived initialization and explicit executed BN epsilon. The original sources and their notices remain in the bound external checkout.\n\nExternal weights, TF runtime and TeX assets are not redistributed by this task.\n')
    print('ENVIRONMENT_AND_PRETRAINING_BOUND')
if __name__=='__main__':run()

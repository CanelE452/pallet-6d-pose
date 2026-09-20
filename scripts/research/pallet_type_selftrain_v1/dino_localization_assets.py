"""Pinned official DINOv2 small backbone, isolated from existing checkpoints."""
import subprocess
import sys
from pathlib import Path
import torch
from . import recovery_common as R

C=R.C
ASSETS=C.OUT/'selftrain_recovery_v1/dino_localization_assets'
REPO=ASSETS/'dinov2'
COMMIT='7764ea0f912e53c92e82eb78a2a1631e92725fc8'
DOC=R.DOC/'dino_localization'


def load(device='cuda', download=False):
    head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    assert head==COMMIT
    torch.hub.set_dir(str(ASSETS/'torch_hub'))
    sys.path.insert(0,str(REPO))
    from dinov2.hub.backbones import dinov2_vits14
    weights=ASSETS/'torch_hub/checkpoints/dinov2_vits14_pretrain.pth'
    if not download: assert weights.exists(), 'Run the explicitly authorized asset preparation first'
    model=dinov2_vits14(pretrained=True).eval().requires_grad_(False).to(device)
    assert model.embed_dim==384 and model.patch_size==14
    return model,weights


def main():
    C.N.setup();print('GPU',C.N.E.gpu(),flush=True)
    model,weights=load(download=True)
    with torch.no_grad():
        features=model.forward_features(torch.zeros(1,3,392,294,device='cuda'))['x_norm_patchtokens']
    assert features.shape==(1,588,384) and torch.isfinite(features).all()
    sources=[C.bound(p) for p in sorted(REPO.rglob('*.py'))]
    receipt=dict(commit=COMMIT,repository='https://github.com/facebookresearch/dinov2',
        model='dinov2_vits14',checkpoint=C.bound(weights),code=sources,
        license=C.bound(REPO/'LICENSE'),loader=C.bound(__file__),patch_shape=[28,21,384],
        parameters=sum(p.numel() for p in model.parameters()),all_backbone_parameters_frozen=True,
        environment_installations=0,no_data_uploaded=True)
    C.freeze(DOC/'BACKBONE.json',receipt)
    print('DINO_BACKBONE_READY',receipt['parameters'],receipt['checkpoint'],flush=True)


if __name__=='__main__':main()

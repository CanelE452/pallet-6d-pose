"""Local official factory. No model is constructed by importing this file."""
import sys


VENDOR_ROOT = '/home/minjae/Documents/github/pallet-pose/outputs/pallet_type_selftrain_v1/selftrain_recovery_v1/dino_localization_assets/dinov2'


def build_backbone():
    if 'v112' in VENDOR_ROOT.lower():
        raise ValueError('Forbidden path')
    sys.path.insert(0, VENDOR_ROOT)
    from dinov2.hub.backbones import dinov2_vits14_reg
    return dinov2_vits14_reg(pretrained=False)


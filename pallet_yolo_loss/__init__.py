"""PSPC — Pallet Symmetry-Role and Projective Consistency loss (프로젝트 로컬).

site-packages 를 수정하지 않는다.  Ultralytics 의 PoseLoss26 을 subclass 하고
PoseModel.init_criterion 만 갈아끼운다.
"""
from .loss import PSPCPoseLoss26, PSPCConfig          # noqa: F401
from .model import PSPCPoseModel                       # noqa: F401

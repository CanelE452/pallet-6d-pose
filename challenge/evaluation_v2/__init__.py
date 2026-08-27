"""Paper-facing real-pallet evaluation contract (version 2).

The modules in this package deliberately do not import or modify the historical
evaluators under ``challenge/yolo_pose_one_model/runs_*``.  Population
membership, PnP selection and pose-metric gating are explicit here so a paper
result cannot silently inherit a legacy development-set convention.
"""

from .real_dataset_contract import (
    ContractError,
    MembershipStatus,
    PopulationId,
    PopulationKind,
    PopulationManifest,
    PopulationRole,
)

__all__ = [
    "ContractError",
    "MembershipStatus",
    "PopulationId",
    "PopulationKind",
    "PopulationManifest",
    "PopulationRole",
]

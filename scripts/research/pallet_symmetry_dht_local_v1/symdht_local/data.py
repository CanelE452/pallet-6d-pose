"""Hash-bound observation and separately loaded supervision datasets."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .util import read_json, sha256

OBSERVATION_KEYS = (
    "features", "content", "box", "image_hw", "dims", "base_points",
    "point_valid", "point_sigma",
)
TARGET_KEYS = ("points", "valid")


def _safe_load(path: Path) -> dict:
    value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError(f"tensor artifact must contain a dictionary: {path}")
    return value


def resolve_record_path(manifest_path: Path, relative: str) -> Path:
    return (manifest_path.parent / relative).resolve()


class ObservationDataset(Dataset):
    def __init__(self, manifest: str | Path, *, targets: bool):
        self.manifest_path = Path(manifest).resolve()
        self.manifest = read_json(self.manifest_path)
        if self.manifest.get("schema") != "symdht_local_export_v1":
            raise ValueError("wrong export manifest schema")
        self.records = self.manifest["records"]
        self.targets = targets
        self._verified_observations: set[Path] = set()
        self._verified_targets: set[Path] = set()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        obs_path = resolve_record_path(self.manifest_path, record["observation"])
        if obs_path not in self._verified_observations:
            if sha256(obs_path) != record["observation_sha256"]:
                raise RuntimeError(f"observation hash changed: {record['frame_id']}")
            self._verified_observations.add(obs_path)
        obs = _safe_load(obs_path)
        if tuple(obs) != OBSERVATION_KEYS:
            raise ValueError(f"observation key contract changed: {record['frame_id']}")
        item = {key: obs[key].squeeze(0) for key in OBSERVATION_KEYS}
        item["frame_id"] = record["frame_id"]
        item["symmetry_permutations"] = torch.as_tensor(record["symmetry_permutations"], dtype=torch.long)
        if self.targets:
            target_path = resolve_record_path(self.manifest_path, record["supervision"])
            if target_path not in self._verified_targets:
                if sha256(target_path) != record["supervision_sha256"]:
                    raise RuntimeError(f"supervision hash changed: {record['frame_id']}")
                self._verified_targets.add(target_path)
            target = _safe_load(target_path)
            if tuple(target) != TARGET_KEYS:
                raise ValueError(f"target key contract changed: {record['frame_id']}")
            item["target_points"] = target["points"].squeeze(0)
            item["target_valid"] = target["valid"].squeeze(0)
        return item


def collate(items: list[dict]) -> dict:
    tensor_keys = [key for key, value in items[0].items()
                   if isinstance(value, torch.Tensor) and key != "symmetry_permutations"]
    result = {key: torch.stack([item[key] for item in items]) for key in tensor_keys}
    result["frame_id"] = [item["frame_id"] for item in items]
    result["symmetry_permutations"] = [item["symmetry_permutations"] for item in items]
    return result


def observation_batch(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {key: batch[key].to(device, non_blocking=True) for key in OBSERVATION_KEYS}

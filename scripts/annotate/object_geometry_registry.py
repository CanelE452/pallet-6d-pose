"""Validated object-type registry for real-pallet GT and paper evaluation.

The registry is the only paper-facing mapping from a manifest ``object_type``
to physical X/Y/Z dimensions.  It deliberately has no frame-name or session
fallback: callers must declare the object type before opening GT pose fields.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .pallet_geometry import PhysicalDimensionsXYZ
except ImportError:  # Direct script execution.
    from pallet_geometry import PhysicalDimensionsXYZ  # type: ignore[no-redef]


SCHEMA_VERSION = "pallet_pose_object_geometry_registry_v1"
PLASTIC_OBJECT_TYPE = "plastic_standard_110x130x11"
WOOD_OBJECT_TYPE = "wood_small_80x59x14"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = (
    REPO_ROOT / "challenge" / "real_gt_v2" / "OBJECT_GEOMETRY_REGISTRY.json"
)
_ALLOWED_SYMMETRY_STATUS = frozenset({"FROZEN", "UNREVIEWED"})


class ObjectGeometryRegistryError(ValueError):
    """A geometry registry or lookup violates the explicit object contract."""


@dataclass(frozen=True)
class ObjectGeometrySpec:
    object_type: str
    display_name: str
    aliases: tuple[str, ...]
    physical_dimensions: PhysicalDimensionsXYZ
    source_measurement_cm: tuple[float, float, float]
    source_measurement_order: str
    legacy_tuple_order: str
    canonical_axis_semantics: Mapping[str, str]
    geometry_status: str
    symmetry_status: str
    symmetry_contract: str | None

    @property
    def physical_dimensions_m(self) -> dict[str, float]:
        return self.physical_dimensions.as_dict()

    @property
    def legacy_wdh_tuple(self) -> tuple[float, float, float]:
        """Return explicit compatibility order ``(width, depth, height)``."""

        return (
            self.physical_dimensions.x_m,
            self.physical_dimensions.z_m,
            self.physical_dimensions.y_m,
        )


@dataclass(frozen=True)
class ObjectGeometryRegistry:
    source_path: Path
    sha256: str
    default_object_type: str
    objects: Mapping[str, ObjectGeometrySpec]
    aliases: Mapping[str, str]

    def resolve(self, object_type: str) -> ObjectGeometrySpec:
        if not isinstance(object_type, str) or not object_type.strip():
            raise ObjectGeometryRegistryError("object_type must be a non-empty string")
        key = object_type.strip()
        canonical = key if key in self.objects else self.aliases.get(key)
        if canonical is None:
            raise ObjectGeometryRegistryError(f"UNKNOWN_OBJECT_TYPE: {key}")
        return self.objects[canonical]

    @property
    def default(self) -> ObjectGeometrySpec:
        return self.resolve(self.default_object_type)


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ObjectGeometryRegistryError(f"{field} must be a non-empty string")
    return value.strip()


def _sequence(value: Any, field: str, *, length: int | None = None) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ObjectGeometryRegistryError(f"{field} must be a list")
    if length is not None and len(value) != length:
        raise ObjectGeometryRegistryError(f"{field} must contain exactly {length} values")
    return value


def _spec(raw: Any, index: int) -> ObjectGeometrySpec:
    if not isinstance(raw, Mapping):
        raise ObjectGeometryRegistryError(f"objects[{index}] must be an object")
    object_type = _nonempty(raw.get("object_type"), f"objects[{index}].object_type")
    aliases = tuple(
        _nonempty(value, f"objects[{index}].aliases[{alias_index}]")
        for alias_index, value in enumerate(_sequence(raw.get("aliases"), "aliases"))
    )
    if len(set(aliases)) != len(aliases) or object_type in aliases:
        raise ObjectGeometryRegistryError(f"{object_type}: aliases must be unique")
    dimensions = raw.get("physical_dimensions_m")
    if not isinstance(dimensions, Mapping) or set(dimensions) != {"x", "y", "z"}:
        raise ObjectGeometryRegistryError(
            f"{object_type}: physical_dimensions_m must contain exactly x/y/z"
        )
    try:
        physical = PhysicalDimensionsXYZ(
            x_m=float(dimensions["x"]),
            y_m=float(dimensions["y"]),
            z_m=float(dimensions["z"]),
        )
    except (TypeError, ValueError) as exc:
        raise ObjectGeometryRegistryError(f"{object_type}: invalid dimensions: {exc}") from exc
    measurement = tuple(
        float(value)
        for value in _sequence(
            raw.get("source_measurement_cm"),
            f"{object_type}.source_measurement_cm",
            length=3,
        )
    )
    if any(value <= 0.0 for value in measurement):
        raise ObjectGeometryRegistryError(f"{object_type}: source measurements must be positive")
    expected_cm = (
        physical.x_m * 100.0,
        physical.z_m * 100.0,
        physical.y_m * 100.0,
    )
    if any(abs(a - b) > 1e-9 for a, b in zip(measurement, expected_cm)):
        raise ObjectGeometryRegistryError(
            f"{object_type}: source_measurement_cm does not match canonical X/Z/Y"
        )
    semantics = raw.get("canonical_axis_semantics")
    if (
        not isinstance(semantics, Mapping)
        or set(semantics) != {"x", "y", "z"}
        or any(not isinstance(value, str) or not value.strip() for value in semantics.values())
    ):
        raise ObjectGeometryRegistryError(
            f"{object_type}: canonical_axis_semantics must define x/y/z"
        )
    symmetry_status = _nonempty(
        raw.get("symmetry_status"), f"{object_type}.symmetry_status"
    )
    if symmetry_status not in _ALLOWED_SYMMETRY_STATUS:
        raise ObjectGeometryRegistryError(
            f"{object_type}: symmetry_status must be one of {sorted(_ALLOWED_SYMMETRY_STATUS)}"
        )
    symmetry_contract = raw.get("symmetry_contract")
    if symmetry_contract is not None:
        symmetry_contract = _nonempty(symmetry_contract, f"{object_type}.symmetry_contract")
        if Path(symmetry_contract).is_absolute():
            raise ObjectGeometryRegistryError(
                f"{object_type}: symmetry_contract must be repository-relative"
            )
    if (symmetry_status == "FROZEN") != (symmetry_contract is not None):
        raise ObjectGeometryRegistryError(
            f"{object_type}: FROZEN symmetry requires a contract and UNREVIEWED forbids one"
        )
    return ObjectGeometrySpec(
        object_type=object_type,
        display_name=_nonempty(raw.get("display_name"), f"{object_type}.display_name"),
        aliases=aliases,
        physical_dimensions=physical,
        source_measurement_cm=measurement,
        source_measurement_order=_nonempty(
            raw.get("source_measurement_order"), f"{object_type}.source_measurement_order"
        ),
        legacy_tuple_order=_nonempty(
            raw.get("legacy_tuple_order"), f"{object_type}.legacy_tuple_order"
        ),
        canonical_axis_semantics=dict(semantics),
        geometry_status=_nonempty(raw.get("geometry_status"), f"{object_type}.geometry_status"),
        symmetry_status=symmetry_status,
        symmetry_contract=symmetry_contract,
    )


def load_object_geometry_registry(
    path: str | Path = DEFAULT_REGISTRY_PATH,
) -> ObjectGeometryRegistry:
    source = Path(path).expanduser().resolve()
    try:
        raw_bytes = source.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ObjectGeometryRegistryError(f"registry unreadable: {source}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ObjectGeometryRegistryError("registry root must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ObjectGeometryRegistryError(
            f"schema_version must equal {SCHEMA_VERSION!r}"
        )
    if payload.get("unit") != "metre":
        raise ObjectGeometryRegistryError("registry unit must equal 'metre'")
    specs = tuple(
        _spec(value, index)
        for index, value in enumerate(_sequence(payload.get("objects"), "objects"))
    )
    if len(specs) < 2:
        raise ObjectGeometryRegistryError("registry must contain at least two object types")
    objects = {spec.object_type: spec for spec in specs}
    if len(objects) != len(specs):
        raise ObjectGeometryRegistryError("duplicate canonical object_type")
    aliases: dict[str, str] = {}
    for spec in specs:
        for alias in spec.aliases:
            if alias in objects or alias in aliases:
                raise ObjectGeometryRegistryError(f"duplicate object alias: {alias}")
            aliases[alias] = spec.object_type
    default_object_type = _nonempty(payload.get("default_object_type"), "default_object_type")
    if default_object_type not in objects:
        raise ObjectGeometryRegistryError("default_object_type is not registered")
    if set(objects) != {PLASTIC_OBJECT_TYPE, WOOD_OBJECT_TYPE}:
        raise ObjectGeometryRegistryError(
            "paper registry must contain exactly the locked plastic and wood object types"
        )
    return ObjectGeometryRegistry(
        source_path=source,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        default_object_type=default_object_type,
        objects=objects,
        aliases=aliases,
    )


def get_geometry_spec(
    object_type: str,
    *,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> ObjectGeometrySpec:
    return load_object_geometry_registry(registry_path).resolve(object_type)


__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "ObjectGeometryRegistry",
    "ObjectGeometryRegistryError",
    "ObjectGeometrySpec",
    "PLASTIC_OBJECT_TYPE",
    "SCHEMA_VERSION",
    "WOOD_OBJECT_TYPE",
    "get_geometry_spec",
    "load_object_geometry_registry",
]

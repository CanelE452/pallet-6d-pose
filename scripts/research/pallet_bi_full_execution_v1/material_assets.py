"""Explicit structured PROXY selection, registration and visible rasterization.

No pose/reference labels are accepted.  Named training assets identify source
geometry only; the signed physical registration of photographed instances is
not independently certified.  The affine proxy registration is deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import time

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('EGL_DEVICE_ID', '1')

import numpy as np
from PIL import Image
import trimesh


ASSETS = {
    'plastic_scan': ('data/pallet/scan_cleanup/pallet_full.obj',
                     'data/pallet/scan_cleanup/pallet_full.mtl',
                     'data/pallet/scan_cleanup/laydown_u1_v1_diffuse.png'),
    'scene.usd': ('data/pallet/raw_data/models_usd/scene.usd',),
    'scene_1.usd': ('data/pallet/raw_data/models_usd/scene_1.usd',
                    'data/pallet/raw_data/models_usd/textures/blinn1_baseColor.png'),
}
EXPECTED_SHA = {
    ASSETS['plastic_scan'][0]: '78c823802f7f861038fac247adfd93c59846fa2c029b848bdd10c0cff2203ff3',
    ASSETS['plastic_scan'][1]: 'cd2b7ca8dc378ea6cfc3410ede4aa8a8b6f3dddca969342ae3019f2d352dd043',
    ASSETS['plastic_scan'][2]: '0feca0e5ed72f4bb7b55714cc02a8d2e3ca665e94ddbe6207434c3107815793c',
    ASSETS['scene.usd'][0]: '6d4094a1d3188f9c9a9f1803baf7cceacbff62166635b132ebc0b492d299066e',
    ASSETS['scene_1.usd'][0]: '814a977030f7d456cc51f6f6cabea8661dbafcf41a5ab64290efbb3ace12fa28',
    ASSETS['scene_1.usd'][1]: '255ceb90dd07aadb4ac4a2454679291d81e683c3ea139a5fd0b1718c05102c94',
}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def choose_asset(dimensions, source_asset=None):
    """Known source identity or known registered object dimensions, never GT."""
    dims = np.asarray(dimensions, float)
    if dims.shape != (3,) or not np.isfinite(dims).all() or np.min(dims) <= 0:
        raise ValueError('Positive known XYZ dimensions required')
    if source_asset in ('scene.usd', 'scene_1.usd'):
        return source_asset, 'KNOWN_SOURCE_ASSET_IDENTITY_PROXY_AXES'
    if source_asset is not None:
        return 'scene_1.usd', 'MISSING_NAMED_GLB_FIXED_STRUCTURED_WOOD_PROXY'
    if np.allclose(dims, [1.1, .11, 1.3], atol=1e-6, rtol=0):
        return 'plastic_scan', 'KNOWN_PLASTIC_DIMENSIONS_FIXED_SCAN_PROXY'
    return 'scene_1.usd', 'KNOWN_OBJECT_DIMENSIONS_FIXED_STRUCTURED_WOOD_PROXY'


def axis_registration(vertices, target_dimensions):
    """Centered proper axis map; vertical down; footprint rank preservation.

    Returned unit coordinates have bbox [-.5,.5] on each axis.  Multiplying
    them by known XYZ dimensions produces object/body coordinates in metres.
    The map is deliberately a registered proxy, not a recovered metric CAD.
    """
    vertices = np.asarray(vertices, float)
    dims = np.asarray(target_dimensions, float)
    center = (vertices.min(0) + vertices.max(0)) / 2
    extent = np.ptp(vertices, axis=0)
    thin = int(np.argmin(extent))
    foot = sorted((i for i in range(3) if i != thin), key=lambda i: (extent[i], i))
    target_foot = sorted((0, 2), key=lambda i: (dims[i], i))
    order = [None, thin, None]
    order[target_foot[0]], order[target_foot[1]] = foot
    rotation = np.eye(3)[order]
    rotation[1] *= -1  # raw thin positive -> registered physical Y down
    if np.linalg.det(rotation) < 0:
        rotation[0] *= -1
    assert np.linalg.det(rotation) > .999999
    canonical_extent = extent[order]
    unit = ((vertices - center) @ rotation.T) / canonical_extent
    affine = np.diag(dims / canonical_extent) @ rotation
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = affine, -affine @ center
    return unit, dict(raw_bbox_center=center.tolist(), raw_bbox_extent=extent.tolist(),
        raw_axis_order_for_body_xyz=order, proper_axis_rotation=rotation.tolist(),
        axiswise_scale_to_metres=(dims / canonical_extent).tolist(),
        source_to_object_affine=transform.tolist(), source_raw_units='SOURCE_EXPORT_UNITS',
        object_units='metres', origin='REGISTERED_PROXY_BBOX_CENTER',
        signed_axis_policy='thin positive -> body Y down; fixed X sign enforces proper rotation',
        physical_CAD_identity_verified=False, independent_physical_owner_labels=False,
        physical_owner_accuracy_status='NOT_APPLICABLE_WITH_REASON_NO_INDEPENDENT_LABELS')


def _usd_meshes(path, texture_path=None):
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError('USD stage could not be opened')
    cache = UsdGeom.XformCache()
    parts = []
    image = Image.open(texture_path).convert('RGB') if texture_path else None
    for prim in stage.Traverse():
        mesh = UsdGeom.Mesh(prim)
        if not mesh:
            continue
        points = np.asarray(mesh.GetPointsAttr().Get(), float)
        if not len(points):
            continue
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim), float)
        points = (np.c_[points, np.ones(len(points))] @ matrix)[:, :3]
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), int)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), int)
        uvvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar('st')
        uv = np.asarray(uvvar.ComputeFlattened(), float) if uvvar else None
        interpolation = str(uvvar.GetInterpolation()) if uvvar else None
        faces, uv_faces, cursor = [], [], 0
        for count in counts:
            for j in range(1, int(count)-1):
                corners = np.array([cursor, cursor+j, cursor+j+1])
                faces.append(indices[corners])
                if uv is not None and len(uv):
                    uv_faces.append(uv[corners if interpolation == 'faceVarying' else indices[corners]])
            cursor += int(count)
        faces = np.asarray(faces, int)
        if not len(faces):
            continue
        if uv_faces and image is not None:
            # Expanding the face-varying UVs preserves seams and mesh topology.
            verts = points[faces].reshape(-1, 3)
            render_faces = np.arange(len(verts)).reshape(-1, 3)
            visual = trimesh.visual.texture.TextureVisuals(uv=np.asarray(uv_faces).reshape(-1, 2), image=image)
            part = trimesh.Trimesh(verts, render_faces, visual=visual, process=False)
        else:
            display_color = mesh.GetDisplayColorAttr().Get()
            color = np.asarray(display_color if display_color is not None else [], float).reshape(-1,3)
            rgb = np.clip((color[0] if color.size else np.array([.6, .6, .6]))*255, 0, 255).astype(np.uint8)
            part = trimesh.Trimesh(points, faces, process=False)
            part.visual = trimesh.visual.ColorVisuals(mesh=part, vertex_colors=np.r_[rgb, 255])
        parts.append(part)
    if not parts:
        raise ValueError('USD contains no triangular surface')
    return parts


@dataclass
class RegisteredProxy:
    key: str
    render_meshes: list
    unit_vertices: np.ndarray
    faces: np.ndarray
    edge_indices: np.ndarray
    registration: dict
    bindings: dict

    def body_vertices(self, dimensions):
        return self.unit_vertices * np.asarray(dimensions)

    def body_edges(self, dimensions):
        return self.unit_vertices[self.edge_indices] * np.asarray(dimensions)


class AssetRepository:
    def __init__(self, root, max_edges=512, crease_degrees=35):
        self.root, self.max_edges, self.crease_degrees = Path(root), max_edges, crease_degrees
        self.cache = {}
        self.bindings = {}
        for rel, expected in EXPECTED_SHA.items():
            path = self.root / rel
            value = sha(path)
            if value != expected:
                raise ValueError('Frozen structural proxy changed: ' + str(path))
            self.bindings[str(path)] = value

    def get(self, dimensions, source_asset=None):
        key, reason = choose_asset(dimensions, source_asset)
        rank = bool(dimensions[0] > dimensions[2])
        cache_key = (key, rank)
        if cache_key not in self.cache:
            paths = [self.root / p for p in ASSETS[key]]
            raw = ([trimesh.load(str(paths[0]), force='mesh', process=False)] if paths[0].suffix == '.obj'
                   else _usd_meshes(paths[0], paths[1] if len(paths) > 1 else None))
            all_vertices = np.vstack([np.asarray(p.vertices) for p in raw])
            _, registration = axis_registration(all_vertices, dimensions)
            center = np.asarray(registration['raw_bbox_center'])
            rotation = np.asarray(registration['proper_axis_rotation'])
            extent = np.asarray(registration['raw_bbox_extent'])[registration['raw_axis_order_for_body_xyz']]
            normalized = []
            for part in raw:
                result = part.copy()
                result.vertices = ((np.asarray(part.vertices)-center) @ rotation.T) / extent
                normalized.append(result)
            # Geometry-only welding avoids UV seams becoming fictitious edges.
            geometry = trimesh.util.concatenate([trimesh.Trimesh(p.vertices, p.faces, process=False) for p in normalized])
            geometry.merge_vertices(merge_tex=True, merge_norm=True)
            # Source material crease identity uses the source Euclidean metric.
            # Unit-bbox normalization is anisotropic and must not manufacture
            # sharp edges or change the geometry-only longest-edge ranking.
            metric_geometry=geometry.copy()
            metric_geometry.vertices=np.asarray(geometry.vertices)*extent
            edges = metric_geometry.face_adjacency_edges[metric_geometry.face_adjacency_angles >= np.deg2rad(self.crease_degrees)]
            unique, count = np.unique(np.sort(geometry.edges, axis=1), axis=0, return_counts=True)
            boundary = unique[count == 1]
            edges = np.unique(np.sort(np.vstack((edges, boundary)), axis=1), axis=0)
            length = np.linalg.norm(metric_geometry.vertices[edges[:, 0]] - metric_geometry.vertices[edges[:, 1]], axis=1)
            order = np.lexsort((edges[:, 1], edges[:, 0], -length))
            edges = edges[order[:self.max_edges]]
            registration.update(proxy_key=key, geometry_vertices=len(geometry.vertices), geometry_triangles=len(geometry.faces),
                selected_geometric_edges=len(edges), coplanar_tessellation_edges_excluded=True,
                crease_and_edge_length_metric='SOURCE_EUCLIDEAN_BEFORE_ANISOTROPIC_REGISTRATION',
                longest_edges_before_any_RGB_or_pose_scoring=True,
                texture_mode='ORIGINAL_DIFFUSE_UV' if key != 'scene.usd' else 'ORIGINAL_USD_DISPLAY_COLOR_NO_TEXTURE')
            self.cache[cache_key] = RegisteredProxy(key, normalized, np.asarray(geometry.vertices),
                np.asarray(geometry.faces), edges, registration,
                {str(p): self.bindings[str(p)] for p in paths})
        model = self.cache[cache_key]
        registration = dict(model.registration)
        # Cache geometry depends only on footprint rank; metric affine is row-specific.
        raw_extent = np.asarray(registration['raw_bbox_extent'])
        axis_order = registration['raw_axis_order_for_body_xyz']
        rotation = np.asarray(registration['proper_axis_rotation'])
        affine = np.diag(np.asarray(dimensions)/raw_extent[axis_order]) @ rotation
        transform = np.eye(4); transform[:3,:3] = affine
        transform[:3,3] = -affine @ np.asarray(registration['raw_bbox_center'])
        registration.update(known_physical_dimensions_xyz_m=list(dimensions), selection_reason=reason,
            named_source_asset=source_asset, exact_named_source_geometry_present=source_asset in ('scene.usd', 'scene_1.usd'),
            source_to_object_affine=transform.tolist(), axiswise_scale_to_metres=(np.asarray(dimensions)/raw_extent[axis_order]).tolist(),
            evidence_level='REGISTERED_STRUCTURED_PROXY_AUTOMATIC_EXPERIMENT',
            bindings=model.bindings)
        return model, registration


class VisibleRenderer:
    """One EGL device/context; callers serialize it with all other GPU work."""
    def __init__(self):
        import pyrender
        self.pyrender, self.renderer, self.mesh_cache = pyrender, None, {}
        self.calls, self.raster_passes, self.seconds = 0, 0, 0.

    def render(self, model, dimensions, R, t, K, hw):
        tick = time.monotonic(); p = self.pyrender
        h, w = map(int, hw)
        if self.renderer is None:
            self.renderer = p.OffscreenRenderer(w, h)
        self.renderer.viewport_width, self.renderer.viewport_height = w, h
        scene = p.Scene(bg_color=[0,0,0,0], ambient_light=[.7,.7,.7])
        key = id(model)
        if key not in self.mesh_cache:
            self.mesh_cache[key] = [p.Mesh.from_trimesh(m, smooth=False) for m in model.render_meshes]
        pose = np.eye(4)
        pose[:3,:3] = np.asarray(R) @ np.diag(dimensions)
        pose[:3,3] = t
        pose = np.diag([1,-1,-1,1]) @ pose
        mesh_nodes=[scene.add(mesh, pose=pose)for mesh in self.mesh_cache[key]]
        K = np.asarray(K)
        # OpenGL samples pixel centres at index+.5.  The production/OpenCV
        # observations use integer-index pixel centres.  This is a raster
        # convention conversion, never a change to the stored/input K.
        half_depth=float(np.abs(np.asarray(R)[2])@(np.asarray(dimensions)/2))
        near=max(1e-5,.5*(float(np.asarray(t)[2])-half_depth))
        far=max(2*near,float(np.asarray(t)[2])+half_depth+float(np.linalg.norm(dimensions)))
        scene.add(p.IntrinsicsCamera(K[0,0],K[1,1],K[0,2]+.5,K[1,2]+.5,znear=near,zfar=far))
        scene.add(p.DirectionalLight(color=[1.,1.,1.], intensity=2.))
        rgb, _ = self.renderer.render(scene)
        # Pyrender's default 4x-MSAA depth resolve returns a hardware sample,
        # rather than the centre ray. SEG disables MSAA and supplies the depth
        # at the registered centre pixel. Keep the actual textured RGB pass.
        _, depth = self.renderer.render(scene,flags=p.RenderFlags.SEG,
            seg_node_map={node:np.array([128,128,128])for node in mesh_nodes})
        self.calls += 1; self.raster_passes += 2; self.seconds += time.monotonic()-tick
        return rgb, depth, depth > 0

    def close(self):
        if self.renderer is not None:
            self.renderer.delete(); self.renderer = None


def project(points, R, t, K):
    camera = np.asarray(points) @ np.asarray(R).T + np.asarray(t)
    homogeneous = camera @ np.asarray(K).T
    return homogeneous[..., :2] / homogeneous[..., 2:3], camera[..., 2]


def unproject(pixels, depth, R, t, K):
    rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(K).T
    camera = rays * np.asarray(depth)[:,None]
    return (camera - np.asarray(t)) @ np.asarray(R)


def hardware_info():
    """Query the actual GL implementation; an EGL device is not proof of GPU."""
    import pyrender
    from OpenGL.GL import glGetString,GL_VENDOR,GL_RENDERER,GL_VERSION
    context=pyrender.OffscreenRenderer(64,64)
    try:
        context._platform.make_current()
        result={name:glGetString(value).decode()for name,value in
            [('GL_VENDOR',GL_VENDOR),('GL_RENDERER',GL_RENDERER),('GL_VERSION',GL_VERSION)]}
        result['software_renderer']='llvmpipe'in result['GL_RENDERER'].lower()
        result['hardware_GPU_render_jobs']=0 if result['software_renderer']else 1
        return result
    finally:context.delete()

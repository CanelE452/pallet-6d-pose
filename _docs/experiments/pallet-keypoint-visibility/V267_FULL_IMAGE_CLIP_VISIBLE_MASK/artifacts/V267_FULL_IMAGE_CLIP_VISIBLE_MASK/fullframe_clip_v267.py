"""Pinned official CLIP/CLIP Surgery image encoders for full-frame M4 training.

The encoder receives the entire RGB frame resized to 512 square.  It never
accepts a keypoint, crop, pose, segmentation target, visibility label or frame
identity.  Both arms share OpenAI ViT-B/16 generic weights.  CLIP_SURGERY_ARCH
uses the official image architecture surgery, not text feature surgery.
"""
from pathlib import Path
import ast
import hashlib
import json
import os
import sys
import types
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).absolute().parent
SOURCE = HERE / 'official_source'
OFFICIAL_COMMIT = 'd4696d47f49cfe70f49140afe5eb94f94c5f59bc'
WEIGHTS_SHA256 = '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'
WEIGHTS_URL = 'https://openaipublic.azureedge.net/clip/models/' + WEIGHTS_SHA256 + '/ViT-B-16.pt'
OFFICIAL_CODE = {
    'PLAIN_CLIP': ('clip/clip_model.py', 'd471cf80e274cbbf498ced439de29bd0757ed7b07cc6fbaf1b8f055102dc60a9'),
    'CLIP_SURGERY_ARCH': ('clip/clip_surgery_model.py', '8a55207f040c5c260d1499c067a9f7a49f071e6a73d3ea1a39d2a65940648742'),
}
ARMS = tuple(OFFICIAL_CODE)
RESOLUTION = 512
PATCH_SIDE = 32
MEAN = (.48145466, .4578275, .40821073)
STD = (.26862954, .26130258, .27577711)


def safe(path):
    p = Path(path)
    if not p.is_absolute() or any(s in str(p).lower() for s in ('v112', 'cache', '.pyc', '..')):
        raise ValueError(str(p))
    for a in reversed((p, *p.parents)):
        if a.is_symlink():
            raise ValueError('Symlink: ' + str(a))
    return p


def binding(path, expected_sha256=None):
    p = safe(path)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    result = dict(path=str(p), sha256=h.hexdigest(), size_bytes=p.stat().st_size)
    if expected_sha256 is not None and result['sha256'] != expected_sha256:
        raise ValueError('SHA mismatch: ' + str(p))
    return result


def load_pinned_module(relative, sha256, name):
    p = safe(SOURCE / relative)
    binding(p, sha256)
    module = types.ModuleType(name)
    module.__file__ = str(p)
    sys.modules[name] = module
    exec(compile(p.read_bytes(), str(p), 'exec'), module.__dict__)
    return module


def state_sha(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        x = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(x.dtype).encode())
        h.update(str(tuple(x.shape)).encode())
        h.update(x.numpy().tobytes())
    return h.hexdigest()


def configure_runtime():
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def resize_position_once(visual):
    """Match official CS bilinear position interpolation, before any data read.

    Plain CLIP's official dense-token class has no size interpolation.  Apply
    exactly the CS interpolation to both arms once, then keep resolution fixed.
    No future image can mutate the position embedding based on its dimensions.
    """
    position = visual.positional_embedding.detach()
    old_side = int(round((len(position) - 1) ** .5))
    assert old_side * old_side + 1 == len(position)
    if old_side != PATCH_SIDE:
        spatial = position[1:].reshape(1, old_side, old_side, position.shape[1]).permute(0, 3, 1, 2)
        spatial = F.interpolate(spatial, (PATCH_SIDE, PATCH_SIDE), mode='bilinear', align_corners=False)
        spatial = spatial.flatten(2).transpose(1, 2)[0]
        visual.positional_embedding = nn.Parameter(torch.cat((position[:1], spatial), 0), requires_grad=False)
    return old_side


def load_visual(weights_path, arm, device='cuda:0'):
    """Return (frozen official visual model, provenance/warm-up proof).

    First CS inference creates six V-V attention modules from pretrained QKV
    tensors.  Do that on an artificial zero tensor, then freeze all parameters
    again.  This initialization changes module structure, with no optimization.
    """
    if arm not in ARMS:
        raise ValueError(arm)
    configure_runtime()
    path = safe(weights_path)
    weights = binding(path, WEIGHTS_SHA256)
    module = load_pinned_module(*OFFICIAL_CODE[arm], '_v267_official_' + arm.lower())
    archive = torch.jit.load(str(path), map_location='cpu').eval()
    visual_state = {name[len('visual.'):]: tensor for name, tensor in archive.state_dict().items()
                    if name.startswith('visual.')}
    width = visual_state['conv1.weight'].shape[0]
    patch = visual_state['conv1.weight'].shape[-1]
    layers = len([name for name in visual_state if name.endswith('.attn.in_proj_weight')])
    embed_dim = visual_state['proj'].shape[1]
    original_side = int(round((len(visual_state['positional_embedding']) - 1) ** .5))
    assert (width, patch, layers, embed_dim, original_side) == (768, 16, 12, 512, 14)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        visual = module.VisionTransformer(input_resolution=224, patch_size=16, width=768,
                                         layers=12, heads=12, output_dim=512)
    visual.load_state_dict(visual_state, strict=True)
    del visual_state, archive
    visual.float().eval()
    pretrained_visual_state_sha256 = state_sha(visual)
    old_side = resize_position_once(visual)
    visual.to(device)
    with torch.inference_mode():
        warm = visual(torch.zeros((1, 3, RESOLUTION, RESOLUTION), dtype=torch.float32, device=device))
    assert warm.shape == (1, 1025, 512) and bool(torch.isfinite(warm).all())
    visual.eval().requires_grad_(False)
    # The official CS routine creates new Attention modules inside forward.
    # Their default training flag can otherwise remain True despite outer eval.
    visual.eval()
    frozen_sha = state_sha(visual)
    assert not any(p.requires_grad for p in visual.parameters()) and not any(m.training for m in visual.modules())
    proof = dict(arm=arm, official_repository='https://github.com/xmed-lab/CLIP_Surgery',
                 official_commit=OFFICIAL_COMMIT, official_code=binding(SOURCE / OFFICIAL_CODE[arm][0]),
                 generic_weights=weights, pretrained_visual_state_sha256=pretrained_visual_state_sha256,
                 frozen_ready_state_sha256=frozen_sha, parameters=sum(p.numel() for p in visual.parameters()),
                 source_input_resolution=224, full_frame_resolution=512, old_position_grid=old_side,
                 frozen_position_grid=32, position_interpolation='bilinear align_corners=False once before data',
                 artificial_initialization_forwards=1, optimizer_updates=0,
                 architecture_surgery=arm == 'CLIP_SURGERY_ARCH', text_feature_surgery=False,
                 output='L2-normalized image patch tokens Bx1024x512, CLS discarded for binary mask decoder',
                 dtype='float32 encoder, caller may store tokens as float16')
    return visual, proof


def preprocess_full_frame(rgb):
    """Exact official 512 demonstration's PIL bicubic square resize, no crop."""
    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Full-frame uint8 HxWx3 RGB required.')
    resized = Image.fromarray(image).resize((RESOLUTION, RESOLUTION), resample=Image.Resampling.BICUBIC)
    x = torch.from_numpy(np.array(resized, copy=True)).permute(2, 0, 1).float().div_(255)
    mean = x.new_tensor(MEAN)[:, None, None]
    std = x.new_tensor(STD)[:, None, None]
    return (x - mean) / std


def extract_fullframe_tokens(visual, rgb_arrays, device=None, batch_size=2):
    """Full-frame frozen image features; batch size only changes memory usage.

    Returns dense_tokens Nx1024x512 float32, cls_token Nx512, raw_hw Nx2.
    The caller chooses storage dtype and binds the original RGB inputs itself.
    No segmentation labels or keypoint metadata are accepted by this function.
    """
    if device is None:
        device = next(visual.parameters()).device
    if batch_size < 1 or not len(rgb_arrays):
        raise ValueError('At least one full frame and positive batch size required.')
    assert len(visual.positional_embedding) == 1025
    assert not any(p.requires_grad for p in visual.parameters()) and not any(m.training for m in visual.modules())
    dense, cls, shapes = [], [], []
    with torch.inference_mode():
        for start in range(0, len(rgb_arrays), batch_size):
            images = rgb_arrays[start:start + batch_size]
            x = torch.stack([preprocess_full_frame(rgb) for rgb in images]).to(device)
            features = visual(x)
            assert tuple(features.shape[1:]) == (1025, 512) and bool(torch.isfinite(features).all())
            features = F.normalize(features.float(), p=2, dim=-1, eps=1e-12)
            dense.append(features[:, 1:].cpu().numpy())
            cls.append(features[:, 0].cpu().numpy())
            shapes.extend([np.asarray(rgb).shape[:2] for rgb in images])
    return dict(dense_tokens=np.concatenate(dense), cls_token=np.concatenate(cls), raw_hw=np.asarray(shapes, np.int32))


def official_feature_surgery(image_features, text_features, redundant_features=None):
    """Optional exact official text feature surgery, without importing tokenizer.

    This is unused by the supervised architecture-only comparison unless a
    separately frozen text protocol explicitly supplies the text embeddings.
    """
    p = safe(SOURCE / 'clip/clip.py')
    provenance = json.loads(safe(HERE / 'OFFICIAL_SOURCE_PROVENANCE.json').read_text())
    binding(p, provenance['source_files']['clip/clip.py']['sha256'])
    tree = ast.parse(p.read_bytes(), filename=str(p))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'clip_feature_surgery')
    module = types.ModuleType('_v267_official_feature_surgery')
    module.__file__ = str(p)
    module.__dict__.update(torch=torch, np=np)
    sys.modules[module.__name__] = module
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(p), 'exec'), module.__dict__)
    return module.clip_feature_surgery(image_features, text_features, redundant_features)

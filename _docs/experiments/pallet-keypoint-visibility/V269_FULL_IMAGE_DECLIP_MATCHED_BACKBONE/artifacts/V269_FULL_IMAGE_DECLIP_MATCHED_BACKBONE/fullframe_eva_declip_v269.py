"""Pinned official EVA/DeCLIP whole-image dense encoders, image-only interface.

EVA_STANDARD: original generic EVA weights, official dense mode='vanilla'.
EVA_CSA: same original weights, official dense mode='csa'.
EVA_DECLIP: released COCO-adapted DeCLIP weights, same mode='csa'.
The only new pallet training belongs to a separate SYN maskhead experiment.
"""
from pathlib import Path
from functools import partial
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
ARMS = ('EVA_STANDARD', 'EVA_CSA', 'EVA_DECLIP')
OFFICIAL_ZSSS_COMMIT = '3b65d49e6cbb2f68c3a1fdb2c96f108941d50ee1'
SOURCE_PINS = {
    'utils.py': ('66f88f6e8695d586068e4b6612b88ed7686efd52e386f8a77b565d13fb148287', 14654),
    'rope.py': ('835e73945d12ce0e8d9ffdcfb21eb69c3ca6a1e4a09d1a2b9d3db004dc3755c1', 8804),
    'transformer.py': ('a2081c1d53b1e78df283b398af0815f391533b413f059587266652b4b7a056d9', 26894),
    'eva_vit_model.py': ('4785ad0cfab85814ceabdb29dc1d0621ad6fa5404ab22860b76a1f5a6ae58a05', 34304),
    'model_configs/EVA02-CLIP-B-16.json': ('34d8b8b4ed7ed08b797f7a69b04088be016a785579e6775bd142e3103d60bbc5', 651),
    'constants.py': ('4289fac298908e2fb814ff84b4594ff49094a7eab074fed1b2ca6429339e923b', 116),
}
WEIGHT_PINS = {
    'original': ('EVA02_CLIP_B_psz16_s8B.pt', '4aab21dd652dc19aab7ff2302e5a07582d7838e8a3b26c80d697115fb0c6ddf0', 299522626),
    'declip': ('evab_dinov2B_csa_560_0.25_seg.pt', '2196da6579e23df4f756617d98f46d46c605c630491966e106e0306e813397c4', 1243582799),
}
MEAN = (.48145466, .4578275, .40821073)
STD = (.26862954, .26130258, .27577711)


def safe(path):
    path = Path(path)
    assert path.is_absolute() and not any(x in str(path).lower() for x in ('v112', 'cache', '.pyc', '..')), str(path)
    for ancestor in (path, *path.parents):
        assert not ancestor.is_symlink(), str(ancestor)
    return path


def binding(path):
    path = safe(path); h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return dict(path=str(path), sha256=h.hexdigest(), size_bytes=path.stat().st_size)


def checked_source(root, relative):
    path = safe(root / relative); raw = path.read_bytes(); sha, size = SOURCE_PINS[relative]
    assert len(raw) == size and hashlib.sha256(raw).hexdigest() == sha, str(path)
    return raw


def official_modules(source_root=None):
    """Execute exact pinned official modules; no artifact importlib loaders."""
    root = safe(source_root or HERE / 'official_source/ZSSS/src/open_clip/eva_clip')
    assert os.environ.get('ENV_TYPE') != 'deepspeed'
    name = '_v269_pinned_official_eva'
    package = types.ModuleType(name); package.__path__ = []; package.__package__ = name
    sys.modules[name] = package
    modules = {}
    for relative in ('utils.py', 'rope.py', 'transformer.py', 'eva_vit_model.py'):
        short = Path(relative).stem; qualified = name + '.' + short
        module = types.ModuleType(qualified); module.__package__ = name; module.__file__ = str(root / relative)
        sys.modules[qualified] = module
        exec(compile(checked_source(root, relative), module.__file__, 'exec'), module.__dict__)
        setattr(package, short, module); modules[short] = module
    cfg = json.loads(checked_source(root, 'model_configs/EVA02-CLIP-B-16.json'))
    assert cfg['embed_dim'] == 512 and cfg['vision_cfg']['layers'] == 12
    return modules, cfg, root


def state_sha(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


class FrozenEVADense(nn.Module):
    def __init__(self, visual, arm):
        super().__init__(); assert arm in ARMS
        self.visual = visual; self.arm = arm
        self.mode = 'vanilla' if arm == 'EVA_STANDARD' else 'csa'

    def forward(self, normalized_full512):
        assert normalized_full512.ndim == 4 and normalized_full512.shape[1:] == (3, 512, 512)
        features = self.visual.encode_dense(normalized_full512, keep_shape=False, mode=self.mode)
        assert features.shape[1:] == (1024, 512)
        return F.normalize(features, dim=-1)


def load_visual(arm, device='cpu', pretrained_root=None, source_root=None):
    """Strict official visual-state load + deterministic artificial RoPE warmup.

    Warmup uses a generated zero tensor only and seals official lazy, nonpersistent
    32-grid RoPE buffers before any real or SYN image may enter the encoder.
    """
    assert arm in ARMS
    modules, cfg, root = official_modules(source_root)
    c = cfg['vision_cfg']; key = 'declip' if arm == 'EVA_DECLIP' else 'original'
    filename, expected_sha, expected_size = WEIGHT_PINS[key]
    path = safe((pretrained_root or HERE / 'pretrained') / filename)
    weight_binding = binding(path)
    assert weight_binding['sha256'] == expected_sha and weight_binding['size_bytes'] == expected_size
    # Author-published, LFS-verified checkpoints; only visual state tensors are used.
    checkpoint = torch.load(path, map_location='cpu')
    full_state = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    assert isinstance(full_state, dict)
    visual_state = {name[len('visual.'):]: value for name, value in full_state.items() if name.startswith('visual.')}
    assert visual_state and all(isinstance(value, torch.Tensor) for value in visual_state.values())
    visual = modules['eva_vit_model'].EVAVisionTransformer(
        img_size=c['image_size'], patch_size=c['patch_size'], num_classes=cfg['embed_dim'], use_mean_pooling=False,
        init_values=None, patch_dropout=0., embed_dim=c['width'], depth=c['layers'],
        num_heads=c['width'] // c['head_width'], mlp_ratio=c['mlp_ratio'], qkv_bias=True,
        drop_path_rate=c['drop_path_rate'], norm_layer=partial(modules['transformer'].LayerNorm, eps=1e-6),
        xattn=False, rope=c['rope'], postnorm=False, pt_hw_seq_len=c['pt_hw_seq_len'],
        intp_freq=c['intp_freq'], naiveswiglu=c['naiveswiglu'], subln=c['subln'])
    incompatible = visual.load_state_dict(visual_state, strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    visual_keys = len(visual_state); total_checkpoint_keys = len(full_state)
    del visual_state, full_state, checkpoint
    model = FrozenEVADense(visual, arm).to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    os.environ['RoPE'] = '1'
    initial = state_sha(model)
    with torch.inference_mode():
        generated = torch.zeros((1, 3, 512, 512), dtype=torch.float32, device=device)
        warm = model(generated)
    assert warm.shape == (1, 1024, 512) and bool(torch.isfinite(warm).all())
    assert state_sha(model) == initial
    assert all(not p.requires_grad for p in model.parameters())
    proof = dict(arm=arm, official_mode=model.mode, checkpoint=weight_binding, official_ZSSS_commit=OFFICIAL_ZSSS_COMMIT,
                 source_bindings=[dict(path=str(root / rel), sha256=pin[0], size_bytes=pin[1]) for rel, pin in SOURCE_PINS.items()],
                 strict_visual_state_load=True, visual_state_keys=visual_keys, full_checkpoint_state_keys=total_checkpoint_keys,
                 ignored_checkpoint_components='Text tensors, optimizer, scaler, epoch metadata are not part of image encoder.',
                 parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=0,
                 state_sha256=initial, artificial_zero_warmup_only=True, state_unchanged_by_artificial_warmup=True,
                 normalization='Official exact OPENAI_DATASET_MEAN/STD; same all three EVA arms.',
                 whole_frame_transform='Entire original RGB -> PIL bicubic512x512 stretch -> /255 -> exact CLIP normalization.',
                 projection='Official visual.norm and visual.head(768->512), CLS discarded inside encode_dense, L2 normalize per token.',
                 position='Official dynamic bicubic positional rescale align_corners=False; official dynamic32x32 RoPE.',
                 numerical_engine='float32, official xattn=False softmax/matmul fallback, official LayerNorm eps1e-6; no xformers/apex.',
                 released_pretraining_history='EVA generic pretraining' if key == 'original' else 'EVA generic pretraining plus public COCO DeCLIP dense adaptation',
                 full_zero_shot_text_segmentation_executed=False, original_dataset_or_current_GT_reads=0)
    return model, proof


def preprocess_fullframe(rgb_arrays):
    """Accept only complete uint8 RGB arrays, never query coordinates or masks."""
    tensors = []; raw_hw = []
    for rgb in rgb_arrays:
        raw = np.asarray(rgb)
        assert raw.dtype == np.uint8 and raw.ndim == 3 and raw.shape[2] == 3
        raw_hw.append(tuple(raw.shape[:2]))
        resized = np.array(Image.fromarray(raw).resize((512, 512), Image.Resampling.BICUBIC), copy=True)
        tensors.append(torch.from_numpy(resized).permute(2, 0, 1).float() / 255.)
    batch = torch.stack(tensors)
    mean = torch.tensor(MEAN, dtype=torch.float32)[None, :, None, None]
    std = torch.tensor(STD, dtype=torch.float32)[None, :, None, None]
    return (batch - mean) / std, np.asarray(raw_hw, dtype=np.int32)


@torch.inference_mode()
def extract_fullframe_tokens(model, rgb_arrays, device='cpu', batch_size=2):
    assert model.arm in ARMS and batch_size > 0 and not model.training
    values = []; sizes = []
    for start in range(0, len(rgb_arrays), batch_size):
        batch, hw = preprocess_fullframe(rgb_arrays[start:start + batch_size])
        tokens = model(batch.to(device=device, dtype=torch.float32))
        assert bool(torch.isfinite(tokens).all())
        values.append(tokens.cpu().numpy()); sizes.append(hw)
    return dict(dense_tokens=np.concatenate(values, axis=0), raw_hw=np.concatenate(sizes, axis=0))

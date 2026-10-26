"""Authorized official encoder throughput on synthetic random RGB only."""
import ast
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time
import types

os.environ['XFORMERS_DISABLED'] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.nn import functional as F

H = Path(__file__).absolute().parent
CLOSURE = {'path': str(H.parent/'V260_SINGLE_QUERY_FEATURE_TRANSPORT_B/DEPENDENCY_CLOSURE.json'),
           'sha256': '900ea2373160158c25ef324f5171e3a49ee0bbac52776d32f93d1eae99ab5b17', 'size_bytes': 8660}


def safe(p):
    p = Path(p)
    assert p.is_absolute() and '..' not in p.parts
    assert not any(t in str(p).casefold() for t in ('v112', 'cache', '.pyc'))
    for a in [*reversed(p.parents), p]:
        assert not stat.S_ISLNK(os.lstat(a).st_mode)
    return p


def raw(b):
    with os.fdopen(os.open(safe(b['path']), os.O_RDONLY | os.O_NOFOLLOW), 'rb') as f:
        data = f.read()
    assert len(data) == b['size_bytes'] and hashlib.sha256(data).hexdigest() == b['sha256']
    return data


def binding(p):
    data = safe(p).read_bytes()
    return {'path': str(p), 'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}


def module(b, name):
    m = types.ModuleType(name)
    m.__file__ = b['path']
    sys.modules[name] = m
    exec(compile(raw(b), b['path'], 'exec'), m.__dict__)
    return m


def run():
    out = H/'RANDOM_OFFICIAL_RUNTIME_PROBE.json'
    assert not out.exists(), 'One fresh runtime probe, preserve failures'
    closure = json.loads(raw(CLOSURE))
    started = time.time()
    try:
        module(closure['code']['crop_module'], 'crop_modules')
        loader = module(closure['code']['model_loader'], 'v263_official_model_loader')
        head_code = binding(H/'spatial_head.py')
        spatial = module(head_code, 'v263_spatial_head')
        assert torch.cuda.is_available(), 'No implicit runtime substitution'
        model = loader.load_backbone(closure['model_loader_config'], 'cuda:0', treatment=False)
        model.eval()
        assert not any(p.requires_grad for p in model.parameters()), 'Official encoder frozen'
        before = loader.tensor_state_sha(model.state_dict(), base=True)
        assert before == closure['expected_encoder_state_sha256']
        source = closure['code']['preprocess']
        nodes = [n for n in ast.parse(raw(source)) .body if isinstance(n, ast.FunctionDef) and n.name == 'preprocess_views']
        assert len(nodes) == 1
        ns = {'torch': torch, 'F': F, 'MEAN': (.485,.456,.406), 'STD': (.229,.224,.225)}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), source['path'], 'exec'), ns)
        generator = torch.Generator().manual_seed(17)
        rgb = torch.randint(0, 256, (8, 3, 96, 96), dtype=torch.uint8, generator=generator)
        valid = torch.ones((8, 1, 96, 96), dtype=torch.bool)
        views, _ = ns['preprocess_views'](rgb, valid)
        views = views[:8].to('cuda:0')
        support = torch.ones((8, 16, 16), dtype=torch.bool, device='cuda:0')
        torch.manual_seed(17)
        head = spatial.SpatialPointHead().cuda().eval()
        times = []
        with torch.inference_mode():
            for i in range(12):
                torch.cuda.synchronize()
                t = time.perf_counter()
                tokens = model.get_intermediate_layers(views, n=[8], reshape=False,
                                                       return_class_token=False, norm=True)[0]
                output = head(tokens, support)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-t
                if i >= 2:
                    times.append(elapsed)
            assert tokens.shape == (8, 256, 384) and tokens.dtype == torch.float32
            assert torch.isfinite(tokens).all() and torch.isfinite(output['visibility_E_logit']).all()
            stored = tokens.cpu().to(torch.float16)
            quantization = (stored.float()-tokens.cpu()).abs()
        after = loader.tensor_state_sha(model.state_dict(), base=True)
        assert before == after
        result = {'completed': True, 'PASS': True, 'code': binding(Path(__file__).absolute()),
                  'head_code': head_code, 'closure': CLOSURE,
                  'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__,
                              'torch': torch.__version__, 'cuda': torch.version.cuda,
                              'GPU': torch.cuda.get_device_name(0)},
                  'encoder_file': closure['model_loader_config']['official_state'],
                  'encoder_state_before': before, 'encoder_state_after': after,
                  'encoder_frozen': True, 'layer_index_zero_based': 8, 'official_norm': True,
                  'spatial_shape': list(tokens.shape), 'stored_dtype': 'float16', 'head_forward_dtype': 'float32',
                  'trainable_head_parameters': sum(p.numel() for p in head.parameters()),
                  'random_seed': 17, 'random_RGB96_rows': 8, 'warmup_batches': 2, 'timed_batches': 10,
                  'mean_batch_seconds': float(np.mean(times)), 'rows_per_second': float(8/np.mean(times)),
                  'fp16_storage_maxabs_error': float(quantization.max()),
                  'fp16_storage_meanabs_error': float(quantization.mean()),
                  'actual_source_current_or_held_input_reads': 0, 'optimizer_updates': 0,
                  'new_download_install_or_model_selection': 0,
                  'vendor_scope': 'Existing official revision + four declared critical CODE pins; not every transitive runtime import certified.',
                  'elapsed_seconds': time.time()-started}
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)+'\n')
        print(json.dumps({'completed': True, 'PASS': True, 'probe': binding(out),
                          'rows_per_second': result['rows_per_second'],
                          'head_parameters': result['trainable_head_parameters']}, sort_keys=True), flush=True)
        return result
    except BaseException as exc:
        failure = H/'RANDOM_OFFICIAL_RUNTIME_PROBE_FAILURE.json'
        failure.write_text(json.dumps({'completed': False, 'PASS': False, 'error_type': type(exc).__name__,
                                      'error': str(exc), 'actual_source_current_or_held_input_reads': 0,
                                      'optimizer_updates': 0, 'automatic_retry': False}, indent=2)+'\n')
        raise


if __name__ == '__main__':
    run()

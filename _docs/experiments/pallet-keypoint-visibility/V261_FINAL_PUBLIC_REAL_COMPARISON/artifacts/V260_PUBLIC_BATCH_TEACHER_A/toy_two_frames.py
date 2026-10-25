"""Exactly two invented RGBD frames; no public frame/mesh bytes or network."""
from pathlib import Path
import gzip
import hashlib
import json
import tempfile
import types
import numpy as np
from PIL import Image

H = Path(__file__).absolute().parent
R = H.parent
code = (H / 'compile_public_batch.py').read_bytes()
m = types.ModuleType('invented_public_batch')
m.__file__ = str(H / 'compile_public_batch.py')
exec(compile(code, m.__file__, 'exec'), m.__dict__)


def b(path):
    data = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}


checks = []
try:
    with tempfile.TemporaryDirectory(prefix='v260_invented_two_') as temp:
        root = Path(temp)
        K = root / 'K.txt'
        K.write_text('20 0 12\n0 20 12\n0 0 1\n')
        mesh = root / 'plane.obj'
        mesh.write_text('v -2 -2 1\nv 2 -2 1\nv 2 2 1\nv -2 2 1\nf 1 2 3\nf 1 3 4\n')
        frames = []
        for i, millimeters, mask_value in ((0, 1000, 255), (1, 700, 0)):
            folder = root / str(i); folder.mkdir()
            Image.fromarray(np.zeros((24, 24, 3), np.uint8)).save(folder / 'rgb.png')
            depth = np.full((24, 24), millimeters, np.uint16); depth[0, 0] = 0
            Image.fromarray(depth).save(folder / 'depth.png')
            Image.fromarray(np.full((24, 24), mask_value, np.uint8)).save(folder / 'sam.png')
            np.savetxt(folder / 'pose.txt', np.eye(4))
            frames.append({'frame_id': f'INVENTED_{i}', 'inputs': {
                'RGB': b(folder / 'rgb.png'), 'DEPTH': b(folder / 'depth.png'),
                'SAM': b(folder / 'sam.png'), 'POSE': b(folder / 'pose.txt'),
                'K': b(K), 'OBJ': b(mesh)}, 'provenance': {'invented': True}})
        entry = {'approved_public_surface_batch_teacher': True, 'training_or_model_inference': False,
            'driver_sha256': hashlib.sha256(code).hexdigest(),
            'projection_code': b(R / 'V258_PUBLIC_RGBD_PROJECTION_C/rgbd_geometry_v2.py'),
            'compiler_code': b(R / 'V259_PUBLIC_SURFACE_LABEL_COMPILER_B/surface_label_compiler.py'),
            'policy': b(R / 'V259_ROOT_PUBLIC_OCCLUSION_PILOT/PROSPECTIVE_SURFACE_LABEL_POLICY.json'),
            'output_directory': str(root / 'out'), 'frames': frames}
        ep = root / 'invented_entry.json'; ep.write_text(json.dumps(entry) + '\n')
        e, raw, eb = m.validate_entry(ep)
        result = m.run(e, raw, eb)
        assert result['PASS'] and result['listed_frames'] == 2
        out = root / 'out'
        summary = json.loads((out / 'SUMMARY.json').read_bytes())
        assert summary['totals_over_compiled_frames_only']['reference_rows'] == 18
        assert summary['totals_over_compiled_frames_only']['query_rows'] == 162
        checks.append('exact two-frame streaming flow and all reference/query records retained')
        for i, target in ((0, 0), (1, 1)):
            folder = out / 'frames' / f'F{i:06d}'
            s = json.loads((folder / 'SUMMARY.json').read_bytes())
            assert s['reference_grid_counts_D_E_U'] == {'-1': 8, str(target): 1}
            assert s['full_pixel_counts']['labels'][('D', 'E')[i]] == 195
            checks.append(f'invented frame{i} known D/E and edge/zero-depth U with unchanged11x11 policy')
            with np.load(folder / 'T_GEOMETRY_MAPS.npz', allow_pickle=False) as pack:
                assert pack['T_RENDER_DEPTH_M'].dtype == np.float64
                assert np.isnan(pack['T_OBSERVED_MINUS_RENDER_M'][0, 0])
            checks.append(f'frame{i} lossless floating geometry and invalid-depth NaN retained')
            data = (folder / 'QUERY_ALL.jsonl.gz').read_bytes()
            rows = [json.loads(line) for line in gzip.decompress(data).splitlines()]
            assert len(rows) == 81 and all(r['q_pixel_DE_gold'] is None and not r['r_label_copied_as_query_label'] for r in rows)
            assert data[4:8] == b'\0'*4 and not (data[3] & 8)
            cb = json.loads((folder / 'COMPLETE.json').read_bytes())
            qbind = cb['outputs']['QUERY_ALL.jsonl.gz']
            uncompressed = gzip.decompress(data)
            assert len(uncompressed) == qbind['uncompressed_size_bytes']
            assert hashlib.sha256(uncompressed).hexdigest() == qbind['uncompressed_sha256']
            twin = root / f'twin{i}.jsonl.gz'; m.write_gzip_records(twin, rows)
            assert twin.read_bytes() == data
            checks.append(f'frame{i} deterministic nameless mtime0 gzip and original-stream SHA/size')
            assert not s['inverse_T_full_arrays_saved'] and s['inverse_T_arrays_status'] == 'NOT_SAVED_DIAGNOSTIC_STATS_ONLY'
            g = json.loads((folder / 'GEOMETRY_DIAGNOSTIC.json').read_bytes())
            assert set(g['interpretations']) == {'T', 'inverse_T'} and g['winner_selected'] is None
            checks.append(f'frame{i} both pose diagnostics retained without inverse arrays or pose selection')
        assert all(r['PASS'] for r in json.loads((out / 'FRAME_INDEX.json').read_bytes())['frames'])
        checks.append('no frame filtering and no model/training/admission')
    proof = {'completed': True, 'PASS': True, 'driver_sha256': hashlib.sha256(code).hexdigest(),
             'invented_frames': 2, 'checks': checks, 'real_public_payload_decodes_or_hashes': 0,
             'pinned_CODE_and_policy_metadata_reads_only': True,
             'models_weights_training_Git_or_network_operations': 0}
    (H / 'INVENTED_TWO_FRAME_CHECKS.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(proof, ensure_ascii=False))
except BaseException as exc:
    (H / 'INVENTED_TWO_FRAME_FAILURE.json').write_text(json.dumps({'completed': False, 'PASS': False,
        'error_type': type(exc).__name__, 'error': str(exc), 'checks_completed': checks,
        'real_public_payload_reads': 0, 'automatic_retry': False}, ensure_ascii=False, indent=2) + '\n')
    raise

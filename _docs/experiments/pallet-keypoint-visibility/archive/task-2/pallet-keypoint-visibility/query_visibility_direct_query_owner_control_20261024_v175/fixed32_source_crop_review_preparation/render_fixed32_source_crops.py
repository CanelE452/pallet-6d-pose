#!/usr/bin/env python3
"""Fixed pre-fit VAL examples: metadata preparation, then saved-output-only rendering.

--prepare never opens crop/output NPZs. --entry needs root's completed-fit trigger.
No source RGB, features, weights, encoder, head, labels, q or L are generated/changed.
"""
import argparse
import hashlib
import html
import io
import json
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont

P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V = P / 'query_visibility_direct_query_owner_control_20261024_v175'
HERE = Path(__file__).resolve().parent
SCOPE = V / 'FIXED32_SOURCE_CROP_REVIEW_SCOPE_BEFORE_FIT.json'
FACTS = P / 'query_visibility_actual_initializer_source_20261024_v132/frozen_C16_actual_initializer_v1/INPUT_ROWS_AND_CROP_FACTS.json'
CENSUS = P / 'query_visibility_full_source_query_causal_census_20261024_v173/literal_raster_census_v1/ALL6144_LITERAL_RASTER_SOURCE_DIAGNOSTICS.jsonl'
ARMS = ['C47_PUBLIC_WEAK253_ONLY', 'C48_PUBLIC_WEAK253_PLUS_LITERAL_SYNTHETIC']
CLASSES = ['TARGET_VISIBLE_SUPPORTED', 'FOREIGN_VISIBLE_SUPPORTED', 'BACKGROUND_SUPPORTED']
SHORT = ['TARGET', 'FOREIGN', 'BG']
OWNER6 = ['INCIDENT_TARGET_DECLARED_CF', 'OTHER_TARGET', 'FOREIGN', 'BG', 'AMBIGUOUS', 'INVALID_SUPPORT']
EXPECTED_SCOPE_SHA = '7ea5d7ca846b3f3f285028813b6c2cffeceb4044bf6e48ba455db0637fc49b91'
PROTOCOL = HERE/'PROTOCOL_BEFORE_FIXED32_PIXEL_OR_NEW_OUTPUT_BODIES.json'
PYTHON = '/home/minjae/anaconda3/envs/pallet-yolo26/bin/python'
TRAINER = V/'source_fit_preparation/train_direct_owner_control.py'
TRAINER_PROTOCOL = V/'source_fit_preparation/PROTOCOL_BEFORE_SOURCE_FEATURE_OR_TRAIN_SOFT_BODIES.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def ash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def binding(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def read_bound(b):
    path = Path(b['path']).resolve()
    assert path.is_relative_to(P) and not any(x in str(path).lower() for x in ['_v112','_v169','current223','private'])
    assert sha(path) == b['sha256'], str(path)
    return path


def load_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def point_truth(owner6):
    """Soft exact-point mass; no hard label or truth probability cutoff."""
    if owner6 is None:
        return None
    x = np.asarray(owner6, dtype=np.float64)
    assert x.shape == (6,) and np.all(np.isfinite(x)) and np.all(x >= 0)
    assert abs(float(x.sum()) - 1) <= 1e-12
    return [float(x[0] + x[1]), float(x[2]), float(x[3]), float(x[4] + x[5])]


def diagnostic_probabilities(raw):
    """Saved FP32 logits -> float64 softmax for display only; U remains NaN."""
    raw = np.asarray(raw)
    assert raw.dtype == np.float32 and raw.ndim == 2 and raw.shape[1] == 3
    out = np.full(raw.shape, np.nan, dtype=np.float64)
    good = np.isfinite(raw).all(1)
    assert np.all(good | np.isnan(raw).all(1)), 'mixed finite/NaN owner logits'
    z = raw[good].astype(np.float64)
    e = np.exp(z - z.max(1, keepdims=True))
    out[good] = e / e.sum(1, keepdims=True)
    return out


def validate_predictions(raw, pred, available):
    assert raw.dtype == np.float32 and raw.shape == (6144, 3)
    assert pred.dtype == np.int8 and pred.shape == (6144,)
    assert available.dtype == np.bool_ and available.shape == (6144,)
    assert np.isfinite(raw[available]).all() and np.isnan(raw[~available]).all()
    expected = np.full(6144, -1, np.int8)
    expected[available] = raw[available].argmax(1).astype(np.int8)
    assert np.array_equal(pred, expected), 'saved prediction must be raw argmax with original U first'


def marked_copy(rgb, scale=3):
    """The only marker is on an enlarged display copy, q=(47.5,47.5)."""
    assert rgb.dtype == np.uint8 and rgb.shape == (96, 96, 3)
    im = Image.fromarray(rgb.copy()).resize((96 * scale, 96 * scale), Image.Resampling.NEAREST)
    d = ImageDraw.Draw(im)
    c = (47.5 + 0.5) * scale - 0.5
    # Black outline and white central cross remain visible on wood, plastic and padding.
    for w, color in ((5, '#111111'), (2, '#ffffff')):
        d.line((c - 10, c, c + 10, c), fill=color, width=w)
        d.line((c, c - 10, c, c + 10), fill=color, width=w)
    return im


def invented_checks():
    checks = []
    def check(name, ok):
        assert bool(ok), name
        checks.append({'name': name, 'PASS': True})
    rgb = np.arange(96 * 96 * 3, dtype=np.uint32).reshape(96, 96, 3).astype(np.uint8)
    before = rgb.copy()
    display = marked_copy(rgb)
    check('display marker never changes saved raw96', np.array_equal(rgb, before))
    check('display is exactly 3x nearest-neighbor size', display.size == (288, 288))
    check('q center mapping is47.5 before display scale', (47.5 + .5) * 3 - .5 == 143.5)
    check('central32 is exact unmodified32:64 slice', np.array_equal(rgb[32:64,32:64], before[32:64,32:64]))
    png = io.BytesIO(); Image.fromarray(rgb).save(png,format='PNG'); png.seek(0)
    check('clean PNG decode is byte-exact raw RGB96', np.array_equal(np.asarray(Image.open(png)),before))
    check('literal target sums incident+other without threshold', point_truth([.1,.2,.3,.15,.1,.15]) == [.30000000000000004,.3,.15,.25])
    check('unknown mass remains explicit', point_truth([0,0,0,0,.2,.8]) == [0,0,0,1])
    check('missing owner remains missing', point_truth(None) is None)
    raw = np.array([[2.,2.,0.],[1000.,0.,-1000.],[np.nan]*3], np.float32)
    pr = diagnostic_probabilities(raw)
    check('softmax finite and sum1 including extreme logits', np.isfinite(pr[:2]).all() and np.allclose(pr[:2].sum(1), 1))
    check('unavailable logits yield NaN probabilities', np.isnan(pr[2]).all())
    r = np.zeros((6144,3),np.float32); av = np.ones(6144,bool); av[[0,5]] = False; r[~av] = np.nan
    pd = np.zeros(6144,np.int8); pd[~av] = -1
    validate_predictions(r,pd,av)
    check('argmax ties preserve firstclass and original U first', pd[1] == 0 and pd[0] == -1)
    broken = pd.copy(); broken[0] = 0
    try:
        validate_predictions(r,broken,av)
    except AssertionError:
        rejected = True
    else:
        rejected = False
    check('invalid-row argmax resurrection rejected', rejected)
    check('two-class owner truth never implies semanticE', point_truth([0,0,1,0,0,0]) == [0,1,0,0])
    fixture = {'original_global_index':0,'original_occurrence_id':'invented:native7','base_index':9,
               'native_slot':7,'query_xy':[13.25,19.5],'L':37.0,'split':'VALIDATION',
               'original_input_flags':{'query_available':True,'crop_constructed':True}}
    output = {k:np.asarray([fixture[v]]) for k,v in [('global_index','original_global_index'),
              ('occurrence_id','original_occurrence_id'),('base_index','base_index'),('native_slot','native_slot'),
              ('query_xy','query_xy'),('L','L'),('split','split')]}
    output.update({k:np.asarray([v]) for k,v in fixture['original_input_flags'].items()})
    validate_output_identity(output,0,fixture)
    check('shuffled native7 metadata accepted without modulo inference', True)
    output['query_xy'] = np.asarray([[13.25,19.50001]])
    try:
        validate_output_identity(output,0,fixture)
    except AssertionError:
        rejected = True
    else:
        rejected = False
    check('even small q metadata correction rejected', rejected)
    return {'completed': True, 'PASS': True, 'actual_source_pixel_or_new_output_reads': 0,
            'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'Pillow': PIL.__version__},
            'checks': checks, 'count': len(checks)}


def prepare():
    assert sha(SCOPE) == EXPECTED_SCOPE_SHA
    scope, facts = load_json(SCOPE), load_json(FACTS)
    ids = scope['original_global_indices']
    assert len(ids) == len(set(ids)) == 32 and len(scope['rows']) == 32
    assert facts['completed'] and len(facts['rows']) == 6144
    census = {}
    for line in CENSUS.open():
        r = json.loads(line)
        if r['original_global_index'] in ids:
            census[r['original_global_index']] = r
    assert len(census) == 32
    rows = []
    for s in scope['rows']:
        i = s['original_global_index']; f = facts['rows'][i]; c = census[i]
        assert f['occurrence_index'] == c['original_global_index'] == i
        for k, fk in [('original_occurrence_id','occurrence_id'), ('base_index','base_index'),
                      ('native_slot','native_slot'), ('split','split'), ('variant','variant'), ('query_xy','query_xy'), ('L','L')]:
            assert s[k] == f[fk] == c[k], (i,k)
        assert s['original_input_flags'] == c['original_input_flags']
        assert all(f[k] == v for k,v in s['original_input_flags'].items())
        assert c['owner_class_order'] == OWNER6
        assert c['rgb96_SHA_inherited'] == f['rgb96_sha256'] and c['valid96_SHA_inherited'] == f['valid96_sha256']
        rows.append({**s, 'rgb96_chunk': f['rgb96_chunk'], 'rgb96_row': f['rgb96_row'],
                     'rgb96_sha256': f['rgb96_sha256'], 'valid96_sha256': f['valid96_sha256'],
                     'original_semantic_state': c['original_semantic_state'],
                     'point_owner6': c['soft_distributions']['scene_owner']['point'],
                     'point_owner_T_F_BG_unknown_mass': point_truth(c['soft_distributions']['scene_owner']['point']),
                     'owner_class_order': OWNER6, 'correspondence_boundary': c['correspondence_boundary'],
                     'q_reference_error_px': c['q_reference_error_px'], 'training_admission': False})
    assert [r['original_global_index'] for r in rows] == ids
    assert sorted({r['base_index'] for r in rows}) == [128,129,144,145]
    assert sorted({r['native_slot'] for r in rows}) == list(range(8))
    for base in [128,129,144,145]:
        br = [r for r in rows if r['base_index'] == base]
        assert len(br) == 8 and [r['variant'] for r in br] == ['BEFORE','COVER','NEAR_SHAM','FAR_SHAM'] * 2
    inputs = {'schema': 'V175_FIXED32_RENDER_INPUTS_v1', 'completed': True, 'UTC': now(),
              'scope': binding(SCOPE), 'crop_facts': binding(FACTS), 'saved_census': binding(CENSUS),
              'rows': rows, 'actual_crop_or_new_output_bodies_read': False,
              'crop_bindings_inherited_not_rehashed': True, 'selection_changed': False}
    save_json(HERE/'FIXED32_INPUT_BINDINGS_BEFORE_PIXELS_OR_OUTPUTS.json', inputs)
    checks = invented_checks(); checks['code'] = binding(__file__)
    save_json(HERE/'GENERATED_MEANINGFUL_CHECKS.json', checks)
    protocol = {
        'schema':'V175_FIXED32_SAVED_SOURCE_CROP_READOUT_PROTOCOL_v1', 'UTC':now(), 'prepared_only':True,
        'scope':binding(SCOPE), 'code':binding(__file__),
        'fixed32_inputs':binding(HERE/'FIXED32_INPUT_BINDINGS_BEFORE_PIXELS_OR_OUTPUTS.json'),
        'invented_checks':binding(HERE/'GENERATED_MEANINGFUL_CHECKS.json'),
        'trainer_code':binding(TRAINER), 'trainer_protocol':binding(TRAINER_PROTOCOL),
        'runtime':{'executable':PYTHON,'python':platform.python_version(),'numpy':np.__version__,'Pillow':PIL.__version__,'device':'CPU only'},
        'selection':'Exact root pre-fit32 in original order; bases128/129/144/145, two native rows each, four variants each; no result selection/replacement.',
        'pixel_source':'V132 inherited cropchunk binding/row only; rehash actual chunk after root trigger, verify raw96/valid96 array SHA and exact original globalindex. No source scene RGB or masks.',
        'output_schema':{'arms':ARMS,'family':'OLD6144','path_template':str(V/'runs/C47_C48_DIRECT_OWNER_FIXED8000/{arm}/FINAL_OLD6144_OWNER_OUTPUTS.npz'),
                         'raw_owner_three_logits':'float32[6144,3], unavailable allNaN','owner_prediction':'int8[6144], rawargmax0/1/2 and originalinputU=-1 first',
                         'class_order':CLASSES,'identity_keys':['global_index','occurrence_id','base_index','native_slot','query_xy','L','split','all original input flags'],
                         'display_probability':'float64 softmax of unchanged raw FP32 logits; diagnostic only, no probability calibration or new threshold'},
        'truth_display':'Keep original semantic D/E/S/U separately from exact-q owner6. T=incident+other target; F=foreign; BG=background; unknown mass=ambiguous+invalid. No truth hard cutoff. OwnerF never implies nativeE.',
        'crop_display':'RawRGB96 and exact[32:64,32:64] PNGs decode equal originals; enlarged q-cross only display copy at raw(47.5,47.5). All q/L variant-specific immutable.',
        'execution_gate':'Root approved_saved_output_readout/source_only, completed+all_two_models_frozen trigger, exact code/protocol/input/fitCOMPLETE/all2freeze/output bindings; both8000 fits freeze before saved outputs.',
        'outputs':['four BASE_***_8_FIXED_CROPS.png','32 clean96 PNG','32 exact central32 PNG','index.html','RENDERED32_SAVED_VALUES_AND_PIXEL_PROOFS.json','COMPLETE.json'],
        'limits':['fixed illustrative source VAL examples, not representative performance','literal raster registration does not certify initializer physical native identity',
                  'original semantic CF/V4 correspondence boundary is not a pixel boundary','no label correction, new labels/admission, model/threshold selection or success claim',
                  'HTML source generated; browser render and actual PNG inspection require separate real tool receipts'],
        'forbidden_reads':['source RGB/mask/depth','feature NPZ','weights/checkpoints','models/encoder/head forward','current/private/V112/V169'],
        'actual_pixel_reads_during_preparation':0,'actual_new_output_reads_during_preparation':0,'automatic_retry':False}
    save_json(PROTOCOL,protocol)
    save_json(HERE/'ROOT_COMPLETED_FIT_RENDER_ENTRY_TEMPLATE_APPROVED_FALSE.json',{
        'approved_saved_output_readout':False,'source_only':True,'code':binding(__file__),'protocol':binding(PROTOCOL),
        'fixed32_inputs':protocol['fixed32_inputs'],'root_both_fit_completed_trigger':None,'fit_complete':None,'all2_freeze':None,
        'saved_outputs':{a:None for a in ARMS},'output_directory':str(V/'fixed32_source_crop_review_v1')})
    save_json(HERE/'PREPARATION_COMPLETE.json',{
        'completed':True,'PASS':True,'prepared_only':True,'UTC':now(),'code':binding(__file__),'protocol':binding(PROTOCOL),
        'fixed32_inputs':protocol['fixed32_inputs'],'invented_checks':protocol['invented_checks'],
        'entry_template':binding(HERE/'ROOT_COMPLETED_FIT_RENDER_ENTRY_TEMPLATE_APPROVED_FALSE.json'),
        'rows':32,'bases':[128,129,144,145],'native_slots':list(range(8)),
        'actual_crop_or_new_output_bodies_read':False,'models_or_features_read_or_executed':False,
        'actual_render_performed':False,'training_admission':False})
    print(json.dumps({'prepared_metadata_rows':32, 'invented_checks':checks['count'], 'actual_pixel_reads':0, 'actual_new_output_reads':0}))


def font(size):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)


def display_lines(r, outputs):
    mass = r['point_owner_T_F_BG_unknown_mass']
    truth = 'UNKNOWN' if mass is None else '/'.join(f'{v:.3f}' for v in mass)
    lines = [f"#{r['original_global_index']} native{r['native_slot']} {r['variant']}",
             f"q=({r['query_xy'][0]:.3f},{r['query_xy'][1]:.3f}) L={r['L']:.3f}",
             f"Declared native semantic: {r['original_semantic_state']}",
             f'Exact-point mass T/F/BG/U: {truth}']
    for name in ARMS:
        o = outputs[name]
        p = o['probabilities'][r['original_global_index']]
        pred = int(o['prediction'][r['original_global_index']])
        call = 'U' if pred == -1 else SHORT[pred]
        ps = 'U' if not np.isfinite(p).all() else '/'.join(f'{x:.3f}' for x in p)
        lines.append(f'{name[:3]}: {call} | T/F/BG {ps}')
    lines.append(f"Original inputU={not r['original_input_flags']['query_available']} | q at raw(47.5,47.5)")
    return lines


def render(entry_path):
    entry = load_json(entry_path)
    assert entry['approved_saved_output_readout'] is True and entry['source_only'] is True
    assert entry['code'] == binding(__file__)
    protocol = load_json(read_bound(entry['protocol']))
    assert entry['protocol'] == binding(PROTOCOL) and protocol['code'] == entry['code']
    assert Path(sys.executable).resolve() == Path(protocol['runtime']['executable']).resolve()
    assert platform.python_version() == protocol['runtime']['python'] and np.__version__ == protocol['runtime']['numpy'] and PIL.__version__ == protocol['runtime']['Pillow']
    inputs_path = read_bound(entry['fixed32_inputs']); inp = load_json(inputs_path)
    assert entry['fixed32_inputs'] == protocol['fixed32_inputs']
    assert inp['scope']['sha256'] == EXPECTED_SCOPE_SHA
    trigger_path = read_bound(entry['root_both_fit_completed_trigger']); trigger = load_json(trigger_path)
    assert trigger['completed'] is True
    assert trigger['source_only'] is True
    assert trigger.get('all_two_models_frozen', trigger.get('all2_completed',False)) is True
    complete_path = read_bound(entry['fit_complete']); complete = load_json(complete_path)
    assert complete['completed'] is True and complete['PASS'] is True and complete['all2_frozen_before_outputs'] is True
    assert complete['updates_each'] == 8000 and complete['total_updates'] == 16000
    assert complete['source_only'] is True and complete['code'] == protocol['trainer_code'] and complete['protocol'] == protocol['trainer_protocol']
    freeze = load_json(read_bound(complete['all2_freeze']))
    assert freeze['completed'] is True and set(freeze['arms']) == set(ARMS) and freeze['saved_outputs_emitted'] is False
    assert complete['all2_freeze'] == entry['all2_freeze']
    assert set(entry['saved_outputs']) == set(ARMS)
    outdir = Path(entry['output_directory']).resolve()
    assert outdir == V/'fixed32_source_crop_review_v1' and not outdir.exists(), 'preserve previous attempt; no automatic retry'
    # Only completed output NPZs and the two bound crop chunks are opened below.
    outputs = {}
    for name in ARMS:
        b = entry['saved_outputs'][name]
        assert b == complete['arms'][name]['outputs']['OLD6144']
        assert complete['arms'][name]['updates'] == 8000
        assert Path(b['path']).resolve() == V/'runs/C47_C48_DIRECT_OWNER_FIXED8000'/name/'FINAL_OLD6144_OWNER_OUTPUTS.npz'
        a = np.load(read_bound(b), allow_pickle=False)
        outputs[name] = {k: a[k] for k in a.files}; a.close()
        o = outputs[name]
        assert o['owner_class_order'].tolist() == CLASSES
        validate_predictions(o['raw_owner_three_logits'], o['owner_prediction'], o['query_available'])
        o['probabilities'] = diagnostic_probabilities(o['raw_owner_three_logits'])
        o['prediction'] = o['owner_prediction']
        for r in inp['rows']:
            i = r['original_global_index']
            validate_output_identity(o, i, r)
    outdir.mkdir(); (outdir/'clean96').mkdir(); (outdir/'central32').mkdir()
    save_json(outdir/'STARTED.json', {'UTC':now(), 'entry':binding(entry_path), 'all2_fit_complete':binding(complete_path)})
    crops = {}; manifests = []
    for r in inp['rows']:
        b = r['rgb96_chunk']; key = b['path']
        if key not in crops:
            path = read_bound(b); z = np.load(path,allow_pickle=False)
            crops[key] = {k:z[k] for k in z.files}; z.close()
        z = crops[key]; j = r['rgb96_row']; i = r['original_global_index']
        assert z['rgb96'].dtype == np.uint8 and z['rgb96'].shape[1:] == (96,96,3)
        assert z['valid96'].dtype == np.bool_ and z['valid96'].shape[1:] == (96,96)
        assert int(z['global_index'][j]) == i and bool(z['crop_constructed'][j]) == r['original_input_flags']['crop_constructed']
        rgb = z['rgb96'][j].copy(); valid = z['valid96'][j]
        assert ash(rgb) == r['rgb96_sha256'] and ash(valid) == r['valid96_sha256']
        r['_rgb'] = rgb
        clean = outdir/'clean96'/f'{i:06d}.png'; center = outdir/'central32'/f'{i:06d}.png'
        Image.fromarray(rgb).save(clean); Image.fromarray(rgb[32:64,32:64]).save(center)
        assert np.array_equal(np.asarray(Image.open(clean)), rgb)
        assert np.array_equal(np.asarray(Image.open(center)), rgb[32:64,32:64])
        manifests.append({'original_global_index':i, 'original_occurrence_id':r['original_occurrence_id'],
                          'rgb96_array_sha256':ash(rgb), 'valid96_array_sha256':ash(valid),
                          'clean96':binding(clean), 'central32':binding(center), 'q_L_changed':False,
                          'semantic':r['original_semantic_state'], 'point_owner_mass':r['point_owner_T_F_BG_unknown_mass'],
                          'saved_outputs':{n:{'raw_logits':outputs[n]['raw_owner_three_logits'][i].tolist(),
                                             'probabilities':outputs[n]['probabilities'][i].tolist(),
                                             'prediction':int(outputs[n]['prediction'][i])} for n in ARMS}})
    sheets = []
    for base in [128,129,144,145]:
        rows = [r for r in inp['rows'] if r['base_index'] == base]
        W,H,cw,ch = 1720,1260,420,530
        im = Image.new('RGB',(W,H),'#f4f6f8'); d = ImageDraw.Draw(im)
        factor = rows[0]['factors']
        d.text((20,14), f"V175 fixed VAL base{base} | {factor['material']} / {factor['aspect']}",fill='#101828',font=font(25))
        d.text((20,51),'Pre-fit fixed examples, not representative performance | RGB96 original q/L; center cross is display only',fill='#344054',font=font(19))
        for k,r in enumerate(rows):
            x=20+(k%4)*cw; y=90+(k//4)*ch
            d.rectangle((x,y,x+cw-10,y+ch-10),fill='white',outline='#cbd5e1')
            im.paste(marked_copy(r['_rgb']), (x+60,y+10))
            for n,line in enumerate(display_lines(r,outputs)):
                d.text((x+10,y+306+n*27),line,fill='#111827',font=font(16))
        footer = ['Original declared D/E/S/U is a different task from literal point owner; owner FOREIGN does not mean native E.',
                  'T = incident + other target; U mass = ambiguous + invalid. Soft masses have no truth cutoff.',
                  'Source raster registration only; initializer physical native identity uncertified. No new labels, admission or model selection.']
        for n,line in enumerate(footer):
            d.text((20,1160+n*27),line,fill='#344054',font=font(18))
        path=outdir/f'BASE_{base:03d}_8_FIXED_CROPS.png'; im.save(path); sheets.append(binding(path))
    save_json(outdir/'RENDERED32_SAVED_VALUES_AND_PIXEL_PROOFS.json', {'rows':manifests,'fixed_scope':inp['scope']})
    body=''.join(f'<section><h2>base {base}</h2><a href="{Path(b["path"]).name}"><img src="{Path(b["path"]).name}" alt="fixed8 crops base{base}"></a></section>' for base,b in zip([128,129,144,145],sheets))
    links=''.join(f'<tr><td>{r["original_global_index"]}</td><td>{html.escape(r["original_occurrence_id"])}</td><td>{r["semantic"]}</td><td><a href="clean96/{r["original_global_index"]:06d}.png">clean RGB96</a></td><td><a href="central32/{r["original_global_index"]:06d}.png">exact central32</a></td></tr>' for r in manifests)
    page='<!doctype html><meta charset="utf-8"><title>V175 fixed32 source crops</title><style>body{font-family:sans-serif;max-width:1740px;margin:24px auto;padding:0 16px;color:#17212e}img{width:100%;height:auto}td,th{padding:5px 12px;border-bottom:1px solid #ddd}</style><h1>V175 고정 source VAL32 crop 비교</h1><p>학습 전 고정한 네 base의 32개 예시입니다. 전체 성능의 대표 표본이 아닙니다. 원 semantic native 가시성, literal q 소속 soft mass, 두 모델 owner 예측을 구분합니다. owner FOREIGN을 native 차폐 E로 바꾸지 않습니다.</p><p>흰 중심 표시는 표시 복사본에만 있으며 q는 raw96 (47.5,47.5)입니다. 아래 clean96·central32 링크는 원 배열 그대로 저장했습니다. 이 readout은 새 라벨·학습 편입·모델 또는 문턱 선택을 하지 않습니다. CF/V4 대응과 initializer 물리 native identity는 별개이며 인증되지 않았습니다.</p><p><a href="RENDERED32_SAVED_VALUES_AND_PIXEL_PROOFS.json">원 FP32 logits·표시 확률·저장 prediction·pixel 검증 JSON</a></p>'+body+'<h2>원 crop 링크</h2><table><tr><th>index</th><th>occurrence</th><th>원 semantic</th><th>RGB96</th><th>central32</th></tr>'+links+'</table>'
    (outdir/'index.html').write_text(page)
    save_json(outdir/'COMPLETE.json', {'completed':True, 'PASS':True, 'UTC':now(), 'entry':binding(entry_path),
              'code':binding(__file__), 'protocol':entry['protocol'], 'fit_complete':binding(complete_path),
              'root_trigger':binding(trigger_path), 'fixed32_inputs':binding(inputs_path),
              'contacts':sheets, 'HTML':binding(outdir/'index.html'),
              'rendered_values_and_pixel_proofs':binding(outdir/'RENDERED32_SAVED_VALUES_AND_PIXEL_PROOFS.json'),
              'count':32,'contact_count':4,'clean96_count':32,'central32_count':32,
              'new_labels_or_training_admission':False, 'new_model_or_threshold_selection':False,
              'actual_visual_inspection':'pending separate actual tool view', 'HTML_browser_rendered':False})
    print(json.dumps({'completed':True,'output':str(outdir),'contacts':sheets}))


def validate_output_identity(o, i, r):
    """Exact field aliases are frozen from final producer schema before execution."""
    for output_key, row_key in [('global_index','original_global_index'), ('occurrence_id','original_occurrence_id'),
                                ('base_index','base_index'), ('native_slot','native_slot'),
                                ('query_xy','query_xy'), ('L','L'), ('split','split')]:
        a = np.asarray(o[output_key][i]); b = np.asarray(r[row_key])
        assert np.array_equal(a,b), (i,output_key,a,b)
    for key,value in r['original_input_flags'].items():
        assert key in o and bool(o[key][i]) == value, (i,key)


if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--prepare',action='store_true'); ap.add_argument('--entry',type=Path)
    args=ap.parse_args(); assert args.prepare != bool(args.entry), 'exactly one of --prepare/--entry'
    if args.prepare:
        prepare()
    else:
        existed = (V/'fixed32_source_crop_review_v1').exists()
        try:
            render(args.entry)
        except BaseException:
            out = V/'fixed32_source_crop_review_v1'
            if not existed and out.exists():
                save_json(out/'FAILED_NO_AUTOMATIC_RETRY.json', {'completed':False,'UTC':now(),'entry':str(args.entry),
                          'traceback':traceback.format_exc(),'automatic_retry':False})
            raise

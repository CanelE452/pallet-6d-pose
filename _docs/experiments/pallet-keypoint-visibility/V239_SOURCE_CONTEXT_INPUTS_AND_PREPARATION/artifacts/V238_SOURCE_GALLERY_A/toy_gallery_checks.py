"""Invented CSV/cache arrays only; the prior fixed policy is authorized text."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import numpy as np

HERE = Path(__file__).absolute().parent


def run():
    path = HERE/'build_gallery.py'
    assert 'v112' not in str(path).lower() and '__pycache__' not in str(path).lower()
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream: code = stream.read()
    m = {'__name__':'toy_helpers_only','__file__':str(path)}
    exec(compile(code, str(path), 'exec'), m)
    raw = m['read'](m['POLICY_PIN'][0]); assert hashlib.sha256(raw).hexdigest() == m['POLICY_PIN'][1]
    policy = json.loads(raw)
    target = [0,0,1,1,0,0,1,1,1,1]
    old = ['E','D','D','E','E','D','D','E','E','U']
    new = ['D','E','E','D','D','E','D','D','E','U']
    def rows(predictions):
        result = []
        for i, pred in enumerate(predictions):
            family = 0 if i < 4 else 1
            z = [.25, -.25]
            if pred != 'U': z[family] = .5 if pred == 'E' else -.5
            result.append({'sample_index':str(i),'row_id':'INVENTED:'+str(i),'family':str(family),
                'role':'FIT' if i == 8 else 'DEV','available':'False' if i == 9 else 'True',
                'target':str(target[i]),'logit0':'' if pred == 'U' else str(z[0]),
                'logit1':'' if pred == 'U' else str(z[1]),'selected_family_logit':'' if pred == 'U' else str(z[family]),'prediction':pred})
        s = io.StringIO(); writer = csv.DictWriter(s, fieldnames=list(result[0])); writer.writeheader(); writer.writerows(result)
        return m['parse_predictions'](s.getvalue().encode())
    a, b = rows(old), rows(new)
    cases = m['select_cases'](policy, a, b)
    assert len(cases) == 12 and cases[6]['missing'] and cases[6]['sample_index'] is None
    assert cases[8]['sample_index'] == 0 and cases[9]['sample_index'] == 1
    assert cases[10]['sample_index'] == 4 and cases[11]['sample_index'] == 5
    unique = sorted({x['sample_index'] for x in cases if not x['missing']})
    assert unique == [0,1,2,3,4,5,6]  # FIT8/Unavailable9 never fill the empty E-correct bucket.
    bad = [dict(x) for x in b]; bad[0]['row_id'] = 'WRONG'
    try: m['select_cases'](policy, a, bad)
    except ValueError: pass
    else: raise AssertionError('Paired row mismatch accepted')
    for text in ('/never/V112/x','/never/__pycache__/x','/never/x.pyc'):
        try: m['safe'](text)
        except ValueError: pass
        else: raise AssertionError('Forbidden lexical path')
    rng = np.random.Generator(np.random.PCG64(238))
    rgb = rng.integers(0, 256, (8,3,96,96), dtype=np.uint8)
    valid = np.ones((8,96,96), bool); valid[3,47,47] = False; valid[7] = False
    context = rng.integers(0, 256, (3,224,224), dtype=np.uint8).astype(np.float32)
    context[0,0,:4] = [0., .49, .5, 254.5]
    display = m['display_context'](context)
    assert display[0,:4,0].tolist() == [0,0,1,255]
    q8 = [[10.+10*k,20.+10*k] for k in range(8)]
    group = {'q8':q8,'L8':[64.]*8,'original_available8':[True]*7+[False],'box_xyxy':[10.,20.,80.,90.]}
    item = {'case_number':1,'bucket_name':'YCB_AUX targetD TREAT_correct','eligible_population':1,
        'sample_index':0,'row_id':'INVENTED:0','active_slot':0,'family':1,'target_state':'D',
        'control':a[0],'treatment':b[0],'paired_change':'RECOVERED',
        'cache_pixel_proof':{'context_valid_fraction':1.},'invented_toy_only':True}
    with tempfile.TemporaryDirectory(prefix='toy_gallery_', dir=HERE) as tmp:
        os.environ['MPLCONFIGDIR'] = str(Path(tmp)/'matplotlib')
        proof = m['save_rgb_png'](rgb[0], Path(tmp)/'toy_raw96.png')
        assert proof['cache_CHW_uint8_sha256'] == proof['decoded_CHW_uint8_sha256']
        m['render_card'](HERE/'INVENTED_TOY_CARD.png', item, rgb, valid, display, group)
    result = {'completed':True,'PASS':True,'fixture_groups':5,'code_sha256':hashlib.sha256(code).hexdigest(),
        'policy_sha256':m['POLICY_PIN'][1],'fixed12_min_index_no_substitution':True,'missing_toy_bucket':7,
        'duplicates_preserved_and_unique_count':len(unique),'FIT_and_unavailable_excluded_from_selection':True,
        'CHW_PNG_roundtrip_and_F32_context_display_quantization':True,
        'invented_layout_card':m['binding'](HERE/'INVENTED_TOY_CARD.png'),
        'actual_SOURCE_CSV_pixel_array_weight_feature_GT_or_model_reads':0,'new_model_forward_or_training':0}
    m['write_json'](HERE/'TOY_RESULT.json',result); print(json.dumps(result))


if __name__ == '__main__': run()

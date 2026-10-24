#!/usr/bin/env python3
"""Independent exact-PNG weak-pair QA. No producer functions imported."""
import argparse
import collections
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import numpy as np
from PIL import Image

H = Path(__file__).absolute().parent
V = H.parent


def rb(p):
    p = Path(p)
    assert p.is_absolute() and '..' not in p.parts and 'v112' not in str(p).casefold()
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for x in p.parts[1:-1]:
            n = os.open(x, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = n
        n = os.open(p.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    finally:
        os.close(fd)
    assert stat.S_ISREG(os.fstat(n).st_mode)
    with os.fdopen(n, 'rb') as f:
        return f.read()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def bind(p):
    return {'path': str(p), 'sha256': sha(rb(p))}


def checked(b):
    assert set(b) <= {'path', 'sha256', 'size_bytes'} and {'path', 'sha256'} <= set(b)
    raw = rb(b['path'])
    assert sha(raw) == b['sha256']
    assert 'size_bytes' not in b or (type(b['size_bytes']) is int and len(raw) == b['size_bytes'])
    return raw


def put(p, value):
    assert p.parent == H or H in p.parents
    with os.fdopen(os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644), 'w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n')


def image(raw, rgb=False):
    with Image.open(io.BytesIO(raw)) as im:
        assert im.format == 'PNG'
        if rgb:
            assert im.mode == 'RGB' and im.size == (96, 96)
        a = np.array(im)
    return a


def boolean_image(raw):
    a = image(raw)
    if a.ndim == 3:
        assert a.shape == (96, 96, 3) and np.array_equal(a[:,:,0], a[:,:,1]) and np.array_equal(a[:,:,0], a[:,:,2])
        a = a[:,:,0]
    assert a.shape == (96,96) and a.dtype == np.uint8 and set(np.unique(a)) <= {0,255}
    return a != 0


def independent_OR(s):
    a = np.zeros((96,96), bool)
    b = np.zeros((96,96), bool)
    for x0,y0,x1,y1 in s['original_ROIs_preserved']:
        assert all(type(x) is int for x in (x0,y0,x1,y1)) and 0 <= x0 <= x1 < 96 and 0 <= y0 <= y1 < 96
        for y in range(y0,y1+1):
            for x in range(x0,x1+1):
                a[y,x] = True
    for y,x0,x1 in s['positive_runs_y_x0_x1']:
        assert 0 <= y < 96 and 0 <= x0 <= x1 < 96
        b[y,x0:x1+1] = True
    assert np.array_equal(a,b) and int(a.sum()) == s['positive_pixels']
    return a


def audit(entry_path):
    e = json.loads(rb(entry_path))
    assert e['approved_independent_weak_pixel_QA'] is True and e['code'] == bind(Path(__file__))
    comp = json.loads(checked(e['COMPLETE']))
    assert comp['completed'] and comp['PASS'] and comp['training_admission'] is False
    scope = json.loads(checked(e['root_scope']))
    original = scope['exact_PNG18']
    assert len(original) == 18 and len({x['path'] for x in original}) == 18
    plan = json.loads(checked(comp['input_plan']))
    attempts = json.loads(checked(comp['attempts']))['attempts9']
    proofs = json.loads(checked(comp['proofs']))['proofs9']
    variants = [json.loads(x) for x in checked(comp['variants']).decode().splitlines()]
    assert len(attempts) == len(proofs) == len(plan['cases9']) == 9 and len(variants) == 27
    plan_original = {b['path']:b['sha256'] for b in plan['original_PNG_bindings18']}
    assert plan_original == {b['path']:b['sha256'] for b in original}
    owned = {}
    for a in attempts:
        for b in [*a['images'].values(), a['card']]:
            assert Path(b['path']).parent == V/'weak_part_pair_controls_v1'
            assert Path(b['path']).suffix == '.png'
            if b['path'] in owned:
                assert owned[b['path']] == b
            owned[b['path']] = b
    # Close every original/output opaque hash before the first PNG decoder.
    data = {b['path']:checked(b) for b in [*original,*owned.values()]}
    out = H/'actual_weak_v1'; os.mkdir(out)
    put(out/'OPAQUE_SHA_BEFORE_ANY_DECODE.json', {'completed':True,'PASS':True,
        'original18':original,'owned_PNGs':list(owned.values()),'decoder_calls_before_receipt':0})
    rows = []
    by_variant = collections.defaultdict(dict)
    for r in variants:
        by_variant[r['target_id']][r['variant']] = r
    for c,a,p in zip(plan['cases9'],attempts,proofs):
        assert c['case_key'] == a['case_key'] == p['case_key']
        assert a['input'] == c['input'] and a['original_target_support'] == c['finite_target_support']
        assert a['constructed'] is True and a['training_admission'] is False
        before = image(data[c['clean96']['path']], True)
        valid = boolean_image(data[c['valid96']['path']])
        assert sha(np.ascontiguousarray(valid).tobytes()) == c['valid96_decoded_bool_sha256']
        target = independent_OR(c['finite_target_support'])
        images = {k:image(data[a['images'][k]['path']],True) for k in ('BEFORE','COVER','SHAM')}
        ca = boolean_image(data[a['images']['COVER_alpha']['path']])
        sa = boolean_image(data[a['images']['SHAM_alpha']['path']])
        assert data[a['images']['BEFORE']['path']] == data[c['clean96']['path']]
        assert np.array_equal(images['BEFORE'],before) and np.array_equal(ca,target)
        ys,xs = np.where(target); sy,sx = np.where(sa)
        assert len(sx) == len(xs) > 0
        dx,dy = int(sx.min()-xs.min()),int(sy.min()-ys.min())
        moved = np.zeros_like(target)
        assert np.all((0 <= xs+dx)&(xs+dx < 96)&(0 <= ys+dy)&(ys+dy < 96))
        moved[ys+dy,xs+dx] = True
        assert np.array_equal(moved,sa) and not np.any(sa&target)
        assert np.all(valid[ca]) and np.all(valid[sa])
        cover,sham = images['COVER'],images['SHAM']
        assert np.array_equal(cover[~ca],before[~ca]) and np.array_equal(sham[~sa],before[~sa])
        assert np.array_equal(sham[target],before[target])
        assert np.array_equal(cover[ys,xs],sham[ys+dy,xs+dx])
        # Reconstruct the declared checker without a producer helper.
        palette = np.asarray([[32,160,230],[230,64,160]],np.uint8)
        expected_pattern = palette[((xs-int(xs.min()))//4+(ys-int(ys.min()))//4)%2]
        assert np.array_equal(cover[ys,xs],expected_pattern)
        cd,sd = np.any(cover != before,axis=2),np.any(sham != before,axis=2)
        assert not np.any((cd|sd)&~valid) and cd[target].any() and sd.any()
        actual = {'target_OR_pixels':int(target.sum()),'COVER_alpha_pixels':int(ca.sum()),'SHAM_alpha_pixels':int(sa.sum()),
            'COVER_target_whole_OR_opaque':bool(np.all(ca[target])),
            'both_alpha_all_original_valid':bool(np.all(valid[ca]) and np.all(valid[sa])),
            'alpha_same_mass_pattern_translation':bool(np.array_equal(ca[ys,xs],sa[ys+dy,xs+dx])),
            'RGB_pattern_exact_same':bool(np.array_equal(cover[ys,xs],sham[ys+dy,xs+dx])),
            'SHAM_alpha_disjoint_target_OR':not bool(np.any(sa&target)),
            'SHAM_target_RGB_byte_unchanged':bool(np.array_equal(sham[target],before[target])),
            'COVER_actual_changed_target_pixels':int(cd[target].sum()),'COVER_actual_changed_all_pixels':int(cd.sum()),
            'SHAM_actual_changed_target_pixels':int(sd[target].sum()),'SHAM_actual_changed_pixels':int(sd.sum()),
            'all_actual_changes_only_original_valid':not bool(np.any((cd|sd)&~valid)),
            'COVER_outside_alpha_byte_unchanged':bool(np.array_equal(cover[~ca],before[~ca])),
            'SHAM_outside_alpha_byte_unchanged':bool(np.array_equal(sham[~sa],before[~sa])),
            'SHAM_translation_xy':[dx,dy],'native_or_query_support_label_certified':False}
        assert actual == p['proof']
        vv = by_variant[a['target_id']]; assert set(vv) == {'BEFORE','COVER','SHAM'}
        for v,r in vv.items():
            assert r['input'] == c['input'] and r['native_DE_target'] == -1 and r['training_admission'] is False
            assert r['image'] == a['images'][v]
            assert r['finite_part_DE_weak_target_D0_E1'] == (1 if v=='COVER' else 0)
            assert r['future_fit_role_eligible'] == (c['input']['internal_source_role']=='SOURCE_FIT')
            assert r['SOURCE_DEV_excluded_every_training_branch'] == (c['input']['internal_source_role']=='SOURCE_DEV')
        with Image.open(io.BytesIO(data[a['card']['path']])) as im:
            assert im.mode == 'RGB' and im.size == (1500,620)
        rows.append({'case_key':a['case_key'],'role':c['input']['internal_source_role'],'proof':actual})
    # Rehash original and owned closed lists after decode, without discovery.
    assert all(checked(b) == data[b['path']] for b in [*original,*owned.values()])
    summary={'cases':9,'triplets':9,'variants':27,'roles':dict(collections.Counter(r['role'] for r in rows)),
             'original_PNGs':18,'owned_PNGs':len(owned),'all_exact_OR_alpha_valid_same_pattern_mass':True,
             'COVER_changed_target_pixels':[r['proof']['COVER_actual_changed_target_pixels'] for r in rows],
             'SHAM_changed_target_pixels':[r['proof']['SHAM_actual_changed_target_pixels'] for r in rows],
             'native_labels_created':0,'training_admission':False,'actual_tool_image_views':0}
    put(out/'REVIEW.json',{'completed':True,'PASS':True,'code':bind(Path(__file__)),'protocol':e['protocol'],
        'entry':bind(entry_path),'producer_COMPLETE':e['COMPLETE'],'summary':summary,'rows9':rows,
        'source_original_after_SHA_unchanged':True,'producer_imported':False,
        'limits':'Pixel/mass/validity relations for weak finite ROI bag only; no material pixel gold, intended-native certification, natural occlusion or learning claim.'})
    print(json.dumps({'REVIEW':bind(out/'REVIEW.json'),'summary':summary}))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',type=Path,required=True)
    audit(p.parse_args().entry)

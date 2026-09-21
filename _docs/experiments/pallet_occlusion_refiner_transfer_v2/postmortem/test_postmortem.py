"""Independent arithmetic, identity, provenance and non-mutation checks."""
from collections import Counter
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from analyze import C, OUT, ROOT, ARMS, transition, band, visibility

D=C.read(OUT/'D1_TRANSITIONS.json')
ROWS=D['rows']


def test_denominator_transitions_and_original_metric_parity():
    assert len(ROWS)==713 and len({r['frame_id'] for r in ROWS})==93
    assert len({(r['frame_id'],r['corner_id']) for r in ROWS})==713
    for t in (10,20):
        counts=Counter(transition(r['A10']['error_px'],r['A11']['error_px'],t) for r in ROWS)
        assert dict(counts)==D['transition'][str(t)]
    assert D['transition']['10']['GOOD->BAD']-D['transition']['10']['BAD->GOOD']==6
    original=C.read(C.DOC/'E2_REAL_RESULTS.json')['summary']['PRIMARY_OCC96']
    for a in ARMS:
        for t in (10,20):
            assert abs(sum(r[a]['error_px']<=t for r in ROWS)/713-original[a]['PCK'][str(t)])<1e-12


def test_csv_matches_json_exactly():
    import csv,json
    csvrows=list(csv.DictReader((OUT/'D1_TRANSITIONS.csv').open()))
    assert len(csvrows)==len(ROWS)
    for r,c in zip(ROWS,csvrows):
        assert (c['frame_id'],int(c['corner_id']))==(r['frame_id'],r['corner_id'])
        for a in ARMS:
            assert float(c[a+'_error_px'])==r[a]['error_px']
            assert json.loads(c[a+'_xy'])==r[a]['xy']


def test_approved_whole_object_mapping_and_coordinate_errors():
    contract=C.read(C.N.C.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')
    perms={r['object_type']:r['permutations'] for r in contract['objects']}
    pred=C.read(C.V.RAW/'FROZEN_PREDICTIONS.json')['predictions']
    for a in ('A10','A11'):pred[a]=C.read(C.RAW/f'PREDICTIONS_{a}.json')['predictions']
    for r in ROWS:
        for a in ARMS:
            m=r[a];p=C.P.top(pred[a][r['frame_id']]);native=m['native_index']
            assert perms[r['object_type']][m['branch']][native]==r['corner_id']
            assert p['keypoints_xy'][native]==m['xy']
            if m['matched']:np.testing.assert_allclose(np.linalg.norm(np.array(m['xy'])-r['gt_xy']),m['error_px'],atol=1e-7,rtol=0)


def test_detection_failure_retained():
    bad=[r for r in ROWS if not r['R0']['matched']]
    assert len(bad)==54 and len({r['frame_id'] for r in bad})==8
    assert all(r[a]['error_px']==800 and not r[a]['PCK10'] for r in bad for a in ARMS)
    assert all(r['band']=='B5_DETECTION_MATCH_FAILURE' for r in bad)


def test_band_boundaries_counts_and_damage():
    assert [band(x,True) for x in (0,5,5.01,10,10.01,20,20.01,40,40.01)]==['B0','B0','B1','B1','B2','B2','B3','B3','B4']
    assert band(1,False)=='B5_DETECTION_MATCH_FAILURE'
    b=C.read(OUT/'D2_ERROR_BANDS.json')
    assert sum(v['count'] for v in b.values())==713
    for k,v in b.items():
        rr=[r for r in ROWS if r['band']==k]
        assert len(rr)==v['count']
        for name,condition in [('gained',lambda r:r['transition10']=='BAD->GOOD'),('lost',lambda r:r['transition10']=='GOOD->BAD'),('hard_recovery',lambda r:r['R0']['error_px']>20 and r['A11']['error_px']<=10)]:
            assert sum(condition(r) for r in rr)==v[name]
    assert sum(r['hard_recovery'] for r in ROWS)==9
    assert sum(r['good_damage'] for r in ROWS)==1


def test_visibility_not_invented():
    d=C.read(OUT/'D3_VISIBILITY_STATS.json')
    assert not any(r['external_subtype_verified'] for r in ROWS)
    assert d['status']=='EXTERNAL_OCCLUSION_CORNER_RECOVERY_UNVERIFIED'
    assert d['manual_reviewed_corners']==sum(r['manual_reviewed'] for r in ROWS)
    for r in ROWS:
        if r['manual_visibility']=='OCCLUDED_UNSPECIFIED':assert r['verified_visibility_class']=='UNKNOWN'
        if r['verified_visibility_class']=='VISIBLE':assert r['human_record']['state']=='v'
    fake=dict(visibility=2,reason='visible',in_frame=True,source='unknown')
    assert visibility(fake,None,None)['verified_visibility_class']=='UNKNOWN'
    assert visibility(dict(visibility=1,reason='occluded',in_frame=True),dict(state='o'),None)['verified_visibility_class']=='UNKNOWN'


def test_review_gallery_complete_and_local_assets():
    from html.parser import HTMLParser
    d=C.read(OUT/'D3_REVIEW_SELECTION.json');keys={(r['frame_id'],r['corner_id']) for r in d['rows']}
    assert len(keys)==d['corners']
    required={(r['frame_id'],r['corner_id']) for r in ROWS if r['hard_recovery'] or r['good_damage'] or r['transition10'] in ('GOOD->BAD','BAD->GOOD')}
    assert required<=keys
    assert sum('RANDOM_CONTROL' in r['reasons'] for r in d['rows'])==20
    class Parser(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ('href','src') and v and not v.startswith('#'):assert (OUT/v).is_file(),v
    Parser().feed((OUT/'D3_REVIEW_GALLERY.html').read_text())


def test_frame_oracle_is_one_candidate_and_min_mean():
    d=C.read(OUT/'D4_ORACLE_HEADROOM_VALIDATED.json');m=C.read(C.RAW/'E2_FRAME_METRICS.json');ids=C.read(OUT/'INPUTS.json')['frames']
    selected=[]
    for fid in ids:
        best=min(ARMS,key=lambda a:m[a][fid]['frame_mean_px'])
        assert d['chosen_frames'][fid]==best
        selected.append(m[best][fid])
    assert C.V.M.summary(selected)==d['ORACLE_FRAME']
    e=[min(r[a]['error_px'] for a in ARMS) for r in ROWS]
    observed=[min(r[a]['error_px'] for a in ARMS) for r in ROWS if r['R0']['matched']]
    s=d['ORACLE_CORNER']
    for t in (5,10,20):assert abs(s['PCK'][str(t)]-np.mean(np.array(e)<=t))<1e-12
    np.testing.assert_allclose(s['matched_pooled_corner8_median_px'],np.median(observed),atol=1e-12)
    np.testing.assert_allclose(s['matched_pooled_corner8_P90_px'],np.quantile(observed,.9),atol=1e-12)
    assert 'E_fixed' not in s


def test_d5_groups_and_no_validated_selector_claim():
    d=C.read(OUT/'D5_FAILURE_CLUSTERS.json')
    assert sum(v['count'] for v in d['groups'].values())==713
    assert not d['classifier_trained'] and not d['selector_signal_validated']
    for g,v in d['groups'].items():
        rr=[r for r in ROWS if r['transition10']==g]
        assert v['count']==len(rr)
        np.testing.assert_allclose(v['features']['native_disagreement_px']['median'],np.median([r['native_disagreement_px'] for r in rr]),atol=1e-12)


def test_existing_artifacts_unchanged_and_prediction_freeze_verified():
    for binding in C.read(OUT/'INPUTS.json')['bindings']:C.verify(binding)
    lock=C.read(C.DOC/'E2_PREDICTIONS_LOCK.json')
    for b in [lock['baseline'],*lock['predictions'].values()]:C.verify(b)


def test_reports_generated_from_numbers_and_router():
    d=C.read(OUT/'DECISION_REINTERPRETATION.json');o=C.read(OUT/'D4_ORACLE_HEADROOM_VALIDATED.json')
    assert d['A11_minus_A10_PCK10_pp']==100*((D['correct10']['A11']-D['correct10']['A10'])/713) or abs(d['A11_minus_A10_PCK10_pp']-100*((D['correct10']['A11']-D['correct10']['A10'])/713))<1e-12
    assert d['HISTORICAL_GATE_RESULT']==C.read(C.DOC/'E2_DECISION.json')['decision']
    report=(OUT/'POSTMORTEM_KO.md').read_text()
    for v in (D['transition']['10']['GOOD->BAD'],D['transition']['10']['BAD->GOOD']):assert f'{v}개' in report
    assert f'{o["headroom"]["N2"]:.6f}' in report
    route=C.read(OUT/'NEXT_STAGE_PLAN.json');assert not route['new_training_started']
    assert route['primary']=='C_DATA_OCCLUSION_GAP' and route['secondary'] is None
    for p in OUT.glob('*.md'):
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if link=='AUDIT.json':continue
            assert (p.parent/link).exists(),(p,link)

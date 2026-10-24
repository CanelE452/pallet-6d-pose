"""Finite independent CSV/JSON/generated-PNG QA; original NPZ and models forbidden."""
from pathlib import Path
from collections import Counter, defaultdict
import csv, hashlib, json, math, os, struct

V = Path(__file__).parent.parent
OUT = V / 'all360_source_holdout_viewer_v1'
READS = {}
CHECKS = []
VARS = ('BEFORE', 'COVER', 'SHAM')
OBSERVATIONS = {
    8: 'BASE075/native1: dark gray component; small gray COVER at marked locality and shifted SHAM visible. Captions report D/D/D for both heads, including COVER mismatch to the stored weak E.',
    9: 'BASE075/native2: lower corner and small moved gray patch distinguish COVER/SHAM at display scale. Both heads show D/D/D; no human native identity inferred.',
    10: 'BASE075/native5: upper-right end of a dark component; small COVER and displaced SHAM patch are visible. Both heads show D/D/D.',
    11: 'BASE083/native0: small edge extension under COVER, subtle same-colored intervention. C56 displays D/E/D while C57 displays D/D/D; saved response disagreement is readable.',
    12: 'BASE083/native1: marked interior edge and cyan q are visibly distinct; gray COVER and shifted SHAM changes are small. Both heads display D/D/D.',
    13: 'BASE083/native2: small bottom-edge COVER and displaced dark square SHAM visible. Both heads display E/E/E, so BEFORE/SHAM differ from the stored weak D labels.',
    14: 'BASE104/native3: brown component with small gray COVER at the marked end and displaced SHAM. Both heads display D/E/D under the stored weak labels.'
}


def safe(path):
    path = Path(os.path.abspath(path))
    assert 'v112' not in str(path).lower()
    assert path == V or V in path.parents
    for q in [path, *path.parents]:
        assert not q.is_symlink()
    return path


def data(path):
    path = safe(path)
    assert path.suffix in {'.json', '.csv', '.png'}
    raw = path.read_bytes()
    READS[str(path)] = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    return raw


def binding(path):
    raw = data(path)
    return dict(path=str(safe(path)), sha256=hashlib.sha256(raw).hexdigest())


def checked(b):
    assert binding(b['path']) == b


def check(ok, text):
    assert ok, text
    CHECKS.append(dict(PASS=True, check=text))


def main():
    complete = json.loads(data(OUT/'COMPLETE.json'))
    check(binding(OUT/'COMPLETE.json')['sha256'] == '40346a715b4742de063c82b8d31b57687c69d8ba1b9e0b71cb256eee461bc694', 'Exact root-run completed viewer binding')
    check(complete['completed'] and complete['PASS'], 'Producer completed PASS; no new producer run')
    for key in ('mapping', 'associations'):
        checked(complete[key])
    rows = list(csv.DictReader(data(OUT/'ALL360_HOLDOUT_TARGET_MAPPING.csv').decode().splitlines()))
    mapping = json.loads(data(OUT/'ALL360_HOLDOUT_TARGET_MAPPING.json'))
    associations = json.loads(data(OUT/'ALL15_TARGET_ASSOCIATIONS.json'))['triplets']
    images = json.loads(data(OUT/'IMAGE_MANIFEST.json'))
    check(len(rows) == len(mapping['rows']) == len(images['images']) == 360, 'All360 CSV/JSON/image records retained')
    for row, original in zip(rows, mapping['rows']):
        assert set(row) == set(original)
        for key, value in original.items():
            assert row[key] == ('' if value is None else str(value))
    check(True, 'CSV and generated JSON all fields/values/order exact')
    check(len({r['cache_index'] for r in rows}) == len({r['row_id'] for r in rows}) == 360, '360 unique cache indices and occurrence IDs; original global IDs not forced unique')
    check([int(r['cache_index']) for r in rows] == sorted(int(r['cache_index']) for r in rows), 'Original cache-index order retained')
    truth = Counter(int(r['direct_weak_target']) for r in rows)
    check(truth == {-1:315, 0:30, 1:15}, 'All360: known45(D30/E15), truth UNKNOWN315')
    check(all(r['role'] == 'HOLDOUT' and r['training_admission'] == 'False' for r in rows), 'All source HOLDOUT roles unchanged and no training admission')
    check(all(r['original_query_available'] == 'True' for r in rows), 'Original inputU0 preserved for this fixed HOLDOUT cohort')
    check(Counter(int(r['native_slot']) for r in rows) == {k:45 for k in range(8)}, 'Native0..7 each45 input rows retained, not eight known-gold claims')
    groups = defaultdict(list)
    for row in rows:
        groups[row['attempt_id']].append(row)
    check(len(groups) == len(associations) == len(images['contacts']) == 15, 'All15 source triplets retained without error-based selection')
    for attempt, rr in groups.items():
        assert len(rr) == 24 and {(r['variant'], int(r['native_slot'])) for r in rr} == {(v,k) for v in VARS for k in range(8)}
        selected = [r for r in rr if int(r['direct_weak_target']) >= 0]
        assert len(selected) == 3 and len({r['native_slot'] for r in selected}) == 1
        assert {r['variant']:int(r['direct_weak_target']) for r in selected} == {'BEFORE':0,'COVER':1,'SHAM':0}
        for native in range(8):
            same = [r for r in rr if int(r['native_slot']) == native]
            assert len({(r['original_query_xy'],r['original_L'],r['original_query_available'],r['original_global_index']) for r in same}) == 1
    check(True, 'Each triplet preserves3variants x8native, one weak selected slot and same original q/L/ID/availability')
    cms = {}
    for arm, expected in [('C56',[[28,2,0],[11,4,0]]),('C57',[[28,2,0],[8,7,0]])]:
        cm = [[0,0,0],[0,0,0]]
        for row in rows:
            logit = float(row[arm+'_raw'])
            answer = 'E' if logit >= 0 else 'D'
            assert math.isfinite(logit) and row[arm+'_D_E_U'] == answer
            t = int(row['direct_weak_target'])
            if t >= 0:
                cm[t][{'D':0,'E':1,'U':2}[row[arm+'_D_E_U']]] += 1
        check(cm == expected, arm+' independent45known CSV CM equals frozen source HOLDOUT CM; strictzero decoder/U preserved')
        cms[arm] = cm
    for association in associations:
        rr = groups[association['attempt_id']]
        selected = [r for r in rr if int(r['direct_weak_target']) >= 0]
        assert int(selected[0]['native_slot']) == association['selected_native_slot']
        assert json.loads(selected[0]['original_query_xy']) == association['original_query_xy']
        assert float(selected[0]['original_L']) == association['original_L']
        assert all(r['target_authority'] == association['authority'] for r in rr)
        changes = association['end_mask_RGB_changed_samples_vs_BEFORE']
        assert changes['BEFORE'] == changes['SHAM'] == 0 and changes['COVER'] > 0
        assert changes['COVER'] <= association['end_related_original96_samples']
        proof = association['existing_proof']
        assert proof['status'] == 'CONTROLLED_DIRECT_END_WEAK_PAIR_SUPPORTED'
        assert all(value is True for key,value in proof['actual_proof_metadata'].items() if key != 'evidence_bindings')
        assert proof['new_native_visibility_target'] == proof['new_quality_target'] == -1
        assert association['association_only_not_nativephysicalgold'] and not association['training_admission']
    check(True, 'All15 saved end-mask COVER changed, SHAM0, same q/L/native/authority, closed14proof metadata and no gold/admission promotion')
    images_by_cache = {image['cache_index']:image for image in images['images']}
    assert len(images_by_cache) == 360
    for row in rows:
        image = images_by_cache[int(row['cache_index'])]
        assert Path(image['clean']['path']) == OUT/row['clean_path']
        assert Path(image['valid']['path']) == OUT/row['valid_path']
        checked(image['clean']); checked(image['valid'])
    check(True, 'All720 generated clean/valid PNG byteSHA and row-relative links agree with IMAGE_MANIFEST; no source array replay')
    assert images['contacts'] == complete['contacts']
    for contact in images['contacts']:
        checked(contact)
    check(True, 'All15 generated contact PNG SHA bindings exact; only fixed seven later cards actually viewed by C')
    viewed = []
    for i in range(8,15):
        b = complete['contacts'][i]
        assert Path(b['path']) == OUT/f'TRIPLET_{i:02d}'/'CONTACT_SELECTED_WEAK_END.png'
        header = data(b['path'])
        assert header[:8] == b'\x89PNG\r\n\x1a\n'
        size = struct.unpack('>II',header[16:24])
        assert size == (1488,980)
        viewed.append(dict(triplet=i, image=b, dimensions=list(size), actual_tool='tools.view_image via functions.exec', requested_detail='original', tool_output='Original1488x980 PNG displayed inline; one call for this exact image', observation=OBSERVATIONS[i], layout='Readable title/weak labels/qL/3columns/rawlogits; clean display separated from magenta/yellow/cyan diagnostic copy; no visible overlap'))
    review = dict(completed=True, PASS=True, code=dict(path=str(Path(__file__)),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        producer_complete=binding(OUT/'COMPLETE.json'), checks=CHECKS, independently_recounted_CM=cms,
        full_rows=360,known_truth=45,truth_UNKNOWN=315,original_inputU=0,
        actual_visual_QA=dict(completed=True,PASS=True,fixed_rule='Parent prospectively assigned sorted TRIPLET08..14 before C opened images; root handles00..07.',actual_views=7,tool_batches=[dict(indices=[8,9,10],tool='functions.exec -> tools.view_image'),dict(indices=[11,12,13,14],tool='functions.exec -> tools.view_image')],images=viewed,HTML_browser_rendered=False,all15_or_all360_individually_viewed_by_C=False),
        inputs=dict(mapping=binding(OUT/'ALL360_HOLDOUT_TARGET_MAPPING.csv'),JSON=binding(OUT/'ALL360_HOLDOUT_TARGET_MAPPING.json'),associations=binding(OUT/'ALL15_TARGET_ASSOCIATIONS.json'),IMAGE_MANIFEST=binding(OUT/'IMAGE_MANIFEST.json')),
        generated_image_hashes_checked=735,
        limits=['Source constructive finite-end witness remains its original weak authority. A tiny mask or visible edit in a crop is not a new physical/native identity or broad-presence gold.','Numeric checks use saved CSV/JSON only. Existing V194 source-input and fit audits close original arrays/proofs; neither was reexecuted.','The7fixed tool-viewed contacts show readable label/logit facts, small COVER edits and displaced SHAM examples, including saved successes and mismatches. They do not establish a causal factor or frequency beyond these fixed views.','The white valid96 panels blend into the white page where all samples are valid; labels identify their position. No missing/input-invalid rows are silently removed.','All720 generated clean/valid PNGs and15contacts were byte-hashed, not individually tool-viewed by C; no browser render claim.'],
        original_NPZ_header_weight_model_current_publicheld_COMMON_oldVAL_V195_or_new_fit_access=0,
        new_labels_or_training_or_selection=False,existing_producer_files_modified=False,training_admission=False,selected_model=None,fully_solved=False)
    target=Path(__file__).parent/'REVIEW.json'
    with target.open('x') as stream:
        json.dump(review,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(completed=True,PASS=True,checks=len(CHECKS),actual_views=7,review=dict(path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest()))))


if __name__ == '__main__':
    main()

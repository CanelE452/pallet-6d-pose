"""Finite independent saved-report QA; never opens arrays, weights, logs or pixels."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
import hashlib
import json
import os
import re

PHASE = Path(__file__).parent.parent
P = PHASE.parent
READS = {}
CHECKS = []


def require(ok, name, detail=None):
    if not ok:
        raise AssertionError(name)
    CHECKS.append(dict(check=name, PASS=True, detail=detail))


def safe(path):
    path = Path(os.path.abspath(path))
    if 'v112' in str(path).lower():
        raise AssertionError('Forbidden lexical path')
    if path != P and P not in path.parents:
        raise AssertionError('Outside exact project provenance')
    for component in [path, *path.parents]:
        if component.is_symlink():
            raise AssertionError('Symlink refused')
    return path


def read(path):
    path = safe(path)
    if path.suffix.lower() not in {'.json', '.md', '.html', '.py'}:
        raise AssertionError('Non-text body refused')
    data = path.read_bytes()
    READS[str(path)] = dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
    return data


def obj(path):
    return json.loads(read(path))


def bound(binding):
    data = read(binding['path'])
    assert hashlib.sha256(data).hexdigest() == binding['sha256'], binding['path']
    return data


def ratio(a, b):
    return a / b if b else None


def metric(arm, role, authority, src):
    cm = src['CM_D_E_by_D_E_U']
    assert len(cm) == 2 and all(len(x) == 3 for x in cm)
    assert all(type(x) is int and x >= 0 for row in cm for x in row)
    truth = [sum(row) for row in cm]
    pred = [sum(cm[i][j] for i in range(2)) for j in range(2)]
    assert sum(truth) == src['known']
    assert src['rows'] == src['known'] + src['truth_UNKNOWN']
    assert src['known_abstentions'] == cm[0][2] + cm[1][2]
    classes = [dict(class_D0_E1=i, truth=truth[i], predicted_knowntruth=pred[i],
                    correct=cm[i][i], precision=ratio(cm[i][i], pred[i]),
                    recall=ratio(cm[i][i], truth[i])) for i in range(2)]
    assert classes == src['classes']
    return dict(model=arm, cohort=role, authority=authority, N_all=src['rows'],
                known=src['known'], truth_UNKNOWN=src['truth_UNKNOWN'], D=truth[0], E=truth[1],
                CM_D_E_by_D_E_U=cm, D_precision=classes[0]['precision'],
                D_recall=classes[0]['recall'], E_precision=classes[1]['precision'],
                E_recall=classes[1]['recall'], known_prediction_U=src['known_abstentions'])


def fmt(value):
    if value is None:
        return 'UNKNOWN/미정의'
    if type(value) is float:
        return format(value, '.6f')
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def md_tables(text):
    blocks, cur = [], []
    for line in text.splitlines() + ['']:
        if line.startswith('|'):
            cur.append([x.strip() for x in line.strip('|').split('|')])
        elif cur:
            blocks.append(cur)
            cur = []
    return [(b[0], b[2:]) for b in blocks]


class Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.links = [], []
        self.table = self.row = self.cell = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'a' and 'href' in a:
            self.links.append(a['href'])
        if tag == 'table':
            self.table = []
        elif tag == 'tr':
            self.row = []
        elif tag in ('td', 'th'):
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ('td', 'th'):
            self.row.append(''.join(self.cell))
            self.cell = None
        elif tag == 'tr':
            self.table.append(self.row)
            self.row = None
        elif tag == 'table':
            self.tables.append((self.table[0], self.table[1:]))
            self.table = None


def check_table(table, rows):
    columns, actual = table
    expected = [[fmt(r[k]) for k in columns] for r in rows]
    assert actual == expected


def links_exist(source, links):
    for ref in links:
        u = urlsplit(ref)
        assert not u.scheme and not u.netloc and u.path and not u.path.startswith('/')
        target = safe(source.parent / unquote(u.path))
        assert target.is_file(), str(target)


def main():
    build = obj(PHASE / 'ROOT_SOURCE_REPORT_BUILD_COMPLETE.json')
    assert build['completed'] and build['PASS']
    for spec in [build['code'], build['protocol'], build['entry'], *build['inputs'].values(), *build['outputs'].values()]:
        bound(spec)
    require(True, 'All explicit build code/protocol/entry/input16/output4 text SHA bindings exact')
    docs = {k: bound(v).decode() for k, v in build['outputs'].items()}
    result = json.loads(docs['ROOT_RESULTS_KO.json'])
    require(result['evidence'] == build['inputs'], 'Result and builder provenance maps identical')
    source = {k: json.loads(bound(build['inputs'][k])) for k in [
        'source_fit_review', 'source_direct_readout', 'source_fit_complete', 'all2_freeze',
        'source_cache_audit', 'direct_cache_complete', 'component_split', 'partition_QA']}
    review, readout, fit, freeze = [source[k] for k in ['source_fit_review', 'source_direct_readout', 'source_fit_complete', 'all2_freeze']]
    assert all(x['completed'] and x['PASS'] for x in [review, readout, fit, source['source_cache_audit'], source['direct_cache_complete'], source['component_split'], source['partition_QA']])
    require(True, 'Bound independent fit/readout/cache/partition receipts completed PASS')
    assert review['source_fit_complete'] == build['inputs']['source_fit_complete']
    assert review['source_readout'] == build['inputs']['source_direct_readout']
    assert fit['all2_freeze'] == build['inputs']['all2_freeze']
    require(True, 'Fit independent-review readout and barrier closure exact')

    arms = result['arms']
    assert len(arms) == 2 and set(arms) == set(fit['arms']) == set(readout['models'])
    direct, authority = [], []
    for arm in arms:
        m = readout['models'][arm]
        direct.extend(metric(arm, r, 'ALL_EXISTING_AUTHORITIES', m[r]) for r in ('ALL', 'TRAIN', 'HOLDOUT'))
        for auth, sub in m['by_authority'].items():
            authority.extend(metric(arm, r, auth, sub[r]) for r in ('ALL', 'TRAIN', 'HOLDOUT'))
        for sub in [m, *m['by_authority'].values()]:
            for field in ['rows', 'known', 'truth_UNKNOWN', 'known_abstentions']:
                assert sub['ALL'][field] == sub['TRAIN'][field] + sub['HOLDOUT'][field]
            for i in range(2):
                for j in range(3):
                    assert sub['ALL']['CM_D_E_by_D_E_U'][i][j] == sum(sub[r]['CM_D_E_by_D_E_U'][i][j] for r in ('TRAIN', 'HOLDOUT'))
        for r in ('ALL', 'TRAIN', 'HOLDOUT'):
            for field in ['rows', 'known', 'truth_UNKNOWN', 'known_abstentions']:
                assert m[r][field] == sum(x[r][field] for x in m['by_authority'].values())
            for i in range(2):
                for j in range(3):
                    assert m[r]['CM_D_E_by_D_E_U'][i][j] == sum(x[r]['CM_D_E_by_D_E_U'][i][j] for x in m['by_authority'].values())
    require(direct == result['direct_metrics'], 'Direct6 records independently recomputed CM P/R U UNKNOWN exact')
    require(authority == result['authority_metrics'], 'Authority12 records independently recomputed CM P/R U UNKNOWN and zero-denominator null exact')
    require(True, 'ALL equals TRAIN plus HOLDOUT; separate authority sums equal full CM and denominators')

    domains, natives = [], []
    for domain, roles in fit['cohort_denominators'].items():
        for role in ('TRAIN', 'HOLDOUT'):
            d = roles[role]
            domains.append(dict(domain=domain, cohort=role, N_all=d['rows'], original_input_U=d['original_inputU'], original_available=d['rows']-d['original_inputU']))
            assert set(d['native']) == set(map(str, range(8))) and sum(d['native'].values()) == d['rows']
            natives.extend(dict(domain=domain, cohort=role, native_slot=i, N_all=d['native'][str(i)]) for i in range(8))
            for arm in arms:
                a = review['results'][arm]['source_output_domains'][domain]['cohorts'][role]
                assert a == dict(rows=d['rows'], inputU=d['original_inputU'])
                meta = fit['arms'][arm]['source_cohorts'][domain][role]
                assert meta['rows'] == d['rows'] and meta['original_inputU'] == d['original_inputU']
        partition = source['partition_QA']['all_domains_full_cache_indices_and_native_U_preserved'][domain]
        assert sum(x['rows'] for x in roles.values()) == partition['rows']
        assert sum(x['original_inputU'] for x in roles.values()) == partition['inputU']
        assert all(roles[r]['rows'] == partition['roles'].get(r, 0) for r in roles)
    require(domains == result['all4_domain_cohort_rows_U'], 'All4 domains both roles original U112/194/360/1264 exact; empty CF HOLDOUT retained')
    require(natives == result['all_native8_cohort_denominators'], 'All64 native8 domain-role denominator cells exact')
    require(sum(x['N_all'] for x in domains) == 73976, 'Full73976 per arm and147952 across arms exact')
    assert fit['steps_each'] == 8000 and fit['total_updates'] == 16000
    assert fit['all2_frozen_before_any_TRAIN_or_HOLDOUT_outputs'] and freeze['completed']
    assert not freeze['source_TRAIN_or_HOLDOUT_prediction_emitted'] and not freeze['protected_real_COMMON_VAL_access'] and not freeze['checkpoint_or_arm_selection']
    assert fit['protected_real_COMMON_VAL_access'] == 0 and not fit['original_native_owner_quality_or_qL_changed']
    for arm in arms:
        a = fit['arms'][arm]
        assert a['updates'] == 8000 and freeze['checkpoints'][arm] == a['checkpoint']
        for key, common in [('initial_state_sha', 'same_initial_state_sha'), ('old_draw_sha', 'same_filtered_old192_draw_sha'), ('direct_draw_sha', 'same_direct32_draw_sha')]:
            assert a[key] == fit[common] == freeze[common]
        assert len(a['outputs']) == 4 and len(a['source_cohorts']) == 4
        assert all(set(r) == {'TRAIN', 'HOLDOUT'} for r in a['source_cohorts'].values())
    require(True, 'Two fixed8000 checkpoints and24 source outputs declared only after shared initialization/draw/final-state barrier; protected access0')
    require(review['all16000_log_rows_verified'] and review['all192_and_direct32_RNG_exact_and_TRAIN_only'] and review['all147952_full_output_rows_META6_U_and_scalar_decoders_exact'], 'Independent A source-fit arithmetic/RNG/final saved-output proof closed by receipt, no replay here')
    require(source['source_cache_audit']['all_raw96_valid_qL_META6_sourcebytes_exact'] and source['source_cache_audit']['all3144_SAM96_mask_and_availability_F64_exact'] and source['source_cache_audit']['old_native_quality_owner_allUNKNOWN'], 'Original qL/native/raw inputs/SAM grid and auxiliary UNKNOWN authority preserved by independent cache receipt')
    assert readout['heldout_direct45_synthetic_only_real_end_holdout0']
    for arm in arms:
        assert readout['models'][arm]['TRAIN']['known'] == 327
        assert readout['models'][arm]['HOLDOUT']['known'] == 45
        assert readout['models'][arm]['by_authority']['SOURCE_RGB_SEMANTIC_END_WEAK']['HOLDOUT']['known'] == 0
    require(result['source_grouped_split']['real_end_HOLDOUT_known'] == 0 and result['source_grouped_split']['HOLDOUT_direct_synthetic_only'], 'TRAIN327 vs synthetic-only HOLDOUT45 explicitly separate; real-end HOLDOUT0')
    require(result['selected_model'] is None and not result['adopted'] and not result['fully_solved'] and result['goal_status'] == 'active_unsolved', 'No model adoption/whole-goal completion')
    require(result['source_TRAIN_is_apparent'] and result['source_design_historically_known_not_freshblind'] and result['no_C56_C57_real_improvement_claim'] and result['old_C54_C55_not_retrospectively_validated'], 'Source apparent/historically-known design and no real-transfer or retrospective validation claims')

    md = docs['ROOT_REPORT_KO.md']
    require(md == docs['PORTABLE_REPORT_KO.md'], 'Root and portable Markdown byte-identical')
    mdblocks = md_tables(md)
    assert len(mdblocks) == 4
    for block, rows in zip(mdblocks, [direct, authority, domains, natives]):
        check_table(block, rows)
    require(True, 'All90 rendered Markdown table rows match saved JSON with explicit six-decimal rounding')
    parser = Tags()
    parser.feed(docs['ROOT_READOUT_KO.html'])
    assert len(parser.tables) == 3
    for block, rows in zip(parser.tables, [direct, authority, domains]):
        check_table(block, rows)
    require(True, 'All26 actual HTML table rows match saved JSON; full native8 Markdown embedded escaped')
    mdlinks = re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)', md)
    assert len(mdlinks) == 16 and len(parser.links) == 17
    links_exist(PHASE/'ROOT_REPORT_KO.md', mdlinks)
    links_exist(PHASE/'PORTABLE_REPORT_KO.md', mdlinks)
    links_exist(PHASE/'ROOT_READOUT_KO.html', parser.links)
    require(True, 'All16 local relative Markdown links per copy and17 HTML links exist; no symlink/forbidden paths')
    require(result['fit_audit_limits'] == review['limits'], 'Independent audit limitations reproduced exactly')

    own = Path(__file__)
    receipt = dict(completed=True, PASS=True, scope='Independent finite source-only report QA from bound text/JSON; no source metric recomputation from model arrays.',
        code=dict(path=str(own), sha256=hashlib.sha256(own.read_bytes()).hexdigest()),
        report_build=READS[str(PHASE/'ROOT_SOURCE_REPORT_BUILD_COMPLETE.json')], checks=CHECKS,
        read_text_bindings=list(READS.values()),
        manual_semantic_review=dict(
            source_includes_synthetic_and_public_real_not_synthetic_only=True,
            direct_truth_authorities_separate_constructive_finite_end_and_source_RGB_semantic_weak=True,
            known372_D248_E124_TRAIN327_HOLDOUT45_synthetic_only_real_end_holdout0=True,
            triplets124_are_not_independent124_frames=True,
            public_real_weak_control27_from12_frames_not_natural_real_native_gold=True,
            original_qL_initializer_sampler_unchanged_and_new_SAM_uses_changed_fullRGB_originalq8box=True,
            all_native8_denominators_preserved_not_all_native8_known_gold=True,
            no_owner_or_broad_crop_presence_promoted_to_native_end_truth=True,
            V195_only_described_as_separate_pending_benchmark_not_read=True),
        limits=[
            'This is arithmetic, rendering, metadata closure and wording QA, not a fresh model/log/checkpoint/array audit.',
            'The A source-fit receipt closes the16000 logged draws, final states and saved output checks; no such bodies were opened here.',
            'The source cache/partition receipts close qL/native/U/grid and source roles; no source RGB, masks, targets or SAM arrays were read.',
            'No browser render or new image view/decode was performed. Linked gallery text only was SHA checked.',
            'Known source labels are constructive or assistant weak controls. Source apparent and group-withheld synthetic direct45 do not establish natural real performance or adoption.'
        ], actual_array_header_weight_log_pixel_model_fit_eval_or_V195_reads=0,
        inventory_copy_git_or_network_calls=0, existing_source_files_modified=False,
        selected_model=None, adopted=False, fully_solved=False)
    target=Path(__file__).parent/'REVIEW.json'
    with target.open('x') as f:
        json.dump(receipt,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(completed=True,PASS=True,checks=len(CHECKS),review=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),text_bindings=len(READS)),ensure_ascii=False))


if __name__ == '__main__':
    main()

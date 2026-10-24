"""Independent saved-public-source arithmetic audit; no RGB/model/current data."""
from pathlib import Path
from collections import Counter, defaultdict
import ast, hashlib, json
import numpy as np

H = Path(__file__).resolve().parent
V = H.parent
P = V.parent
V141 = P / 'query_visibility_public_native_initializer_association_20261024_v141'
OUT = H / 'root_manifest_independent_review'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p): return dict(path=str(p), sha256=sha(p))
def read(p): return json.loads(Path(p).read_text())
def lines(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def check(b):
    assert sha(b['path']) == b['sha256'], b['path']
    return Path(b['path'])
def clean(x):
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, float): return x if np.isfinite(x) else None
    if isinstance(x, dict): return {k: clean(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [clean(v) for v in x]
    return x

rootp = V / 'ROOT_ADJUDICATED_SELECTED_SOURCE_NATIVE128_WEAK_LABELS.jsonl'
assert sha(rootp) == '537aeef4b22fad6ddf1e4e90388cd47003e97f6c60f351bd3e11288a65be4529'
summaryp = V / 'ROOT_PUBLIC_NATIVE_WEAK_LABEL_ADJUDICATION.json'
summary = read(summaryp)
assert summary['completed'] and check(summary['manifest']) == rootp
proposalp = check(summary['independent_proposals'])
assert sha(proposalp) == '2ac633445a58e9176f2706125c41d4dfc6519aef032afec108decdff3ee8602e'
queryp = check(summary['source_original'])
sealpath = V141 / 'actual_public_initializer_canonical_transport_v2/PREDICTION_SEAL.json'
assert sha(sealpath) == 'b8e26a9b76d6e1c9084902cbd63e692ab4a7606a97033a93088d7d1d021d2dce'
seal = read(sealpath)
assert seal['completed'] and seal['PASS']
outputs = {Path(b['path']).name:b for b in seal['outputs']}
assert queryp == check(outputs['ORIGINAL_NATIVE_QUERY128_INPUT_ONLY.jsonl'])
propsp = check(outputs['FROZEN_PROPOSALS_ALL16_INPUT_ONLY.jsonl'])
props = lines(propsp)
queries, original, rows = lines(queryp), lines(proposalp), lines(rootp)
with np.load(check(outputs['ORIGINAL_NATIVE_QUERY128_AND_VALID9X9_INPUT_ONLY.npz']), allow_pickle=False) as f:
    arrays = {k:f[k].copy() for k in f.files}
with np.load(check(outputs['ALL_CANDIDATES_RAW9_AND_BOXES_INPUT_ONLY.npz']), allow_pickle=False) as f:
    candidate_arrays = {k:f[k].copy() for k in f.files}
assert len(props)==16 and len(queries)==len(original)==len(rows)==128

# Recompute original q/L and native-valid math from the existing pure functions.
code = V141 / 'initializer_preparation/infer_public_fixed16.py'
roi = P.parent / 'external_ray_proxy_head_20261022_v60/head_and_roi.py'
nodes = [n for n in ast.parse(code.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('pure_native','query_arrays')]
assert len(nodes)==2
ns = dict(np=np, ast=ast, ROI=roi)
exec(compile(ast.Module(body=nodes,type_ignores=[]), str(code)+'::pure_only', 'exec'), ns)
flagmap = dict(finite_non_sentinel='finite_non_sentinel', native_continuous_valid='continuous_valid',
    query_nearest_pixel_valid='nearest_valid', query_RGB96_centre_domain_valid='RGBcentre_valid')
firstchoices=[]
for i,p in enumerate(props):
    assert p['frame_index']==i
    selected = None if not p['candidates'] else int(np.argmax([c['score'] for c in p['candidates']]))
    assert selected==p['selected_index']
    start,end = candidate_arrays['frame_offsets'][i:i+2]
    assert end-start==len(p['candidates'])
    assert int(candidate_arrays['selected_index'][i]) == (-1 if selected is None else selected)
    for local,c in enumerate(p['candidates']):
        j=int(start)+local
        assert c['candidate_index']==local
        assert clean(candidate_arrays['keypoints_xy'][j])==c['keypoints_xy']
        assert clean(candidate_arrays['keypoints_conf'][j])==c['keypoints_conf']
        assert clean(candidate_arrays['box_xyxy'][j])==c['box_xyxy']
        assert float(candidate_arrays['score'][j])==c['score']
    qa=ns['query_arrays'](p,p['raw_hw'])
    for key,a in qa.items():
        assert np.array_equal(a, arrays[key][i*8:i*8+8], equal_nan=True), (i,key)
    firstchoices.append(dict(frame_index=i,selected_index=selected,candidates=len(p['candidates'])))

factpaths=[H/'fixed_source96_contact_v1/CROP_FACTS96.json',H/'fixed_pallox32_contact_v1/CROP_FACTS32.json']
facts=[]
for fp in factpaths:
    f=read(fp); assert f['completed'];facts.extend(f['rows'])
assert len(facts)==128
assert read(factpaths[0])['all_official_instances']==424
roles=['FRONT_TOP_LEFT','FRONT_TOP_RIGHT','FRONT_BOTTOM_RIGHT','FRONT_BOTTOM_LEFT','REAR_TOP_LEFT','REAR_TOP_RIGHT','REAR_BOTTOM_RIGHT','REAR_BOTTOM_LEFT']
changes=[]
for j,(r,o,q,c) in enumerate(zip(rows,original,queries,facts)):
    i,k=divmod(j,8);p=props[i]
    assert r['global_index']==o['global_index']==c['global_index']==j
    assert r['frame_index']==o['frame_index']==c['frame_index']==q['frame_index']==i
    assert r['native_slot']==o['native_slot']==c['native_slot']==q['native_slot']==k
    assert r['native_role']==roles[k]
    assert r['id']==o['id']==c['id']==q['frame_id']==p['id']
    assert r['source_id']==o['source_id']==c['source_id']==q['source_id']==p['source_id']
    assert r['split']==q['split']==p['split']
    assert r['original_input']==q
    assert r['query_xy']==o['query_xy']==c['query_xy']==q['query_xy']==clean(arrays['query_xy'][j])
    assert r['L']==o['L']==c['L']==q['L']==clean(arrays['L'][j])
    assert q['query_id']==f'{p["id"]}:native{k}'
    assert q['actual_original_initializer'] and not q['source_emulation']
    assert r['no_detection']==o['no_detection']==c['no_detection']==q['no_detection']==bool(arrays['no_detection'][j])
    assert r['original_input_valid']==o['original_input_valid']==c['original_input_valid']==q['native_continuous_valid']
    for jk,nk in flagmap.items(): assert q[jk]==bool(arrays[nk][j])
    assert q['native_grid_any_valid']==bool(arrays['native_grid_valid'][j].any())
    for key in ('rgb96_SHA256','valid96_SHA256','crop_valid_fraction','crop_constructed'):
        assert r[key]==o[key]==c[key], (j,key)
    assert r['original_assistant_proposal_state']==o['semantic_state']
    if r['semantic_state']!=o['semantic_state']:
        changes.append(dict(global_index=j,frame_index=i,native_slot=k,before=o['semantic_state'],after=r['semantic_state']))
    assert r['root_adjudicated'] and not r['human_gold'] and not r['trainready'] and not r['training_now']
    assert not r['current_data_used'] and not r['precise_physical_vertex_gold']
    assert r['no_coordinate_error_filter'] and r['visibility_label_independent_of_input_validity']
    assert r['source_weak_binary_available']==(r['semantic_state'] in ('D','E'))
    assert r['source_weak_TRAIN_eligible']==(r['semantic_state'] in ('D','E') and r['split'] in ('TRAIN','TRAIN_LAB'))
    assert not r['full_native8_ontology_certified']

expected=[(16,'D','U'),(17,'D','U'),(80,'E','U'),(83,'E','U'),(123,'U','E')]
assert [(c['global_index'],c['before'],c['after'])for c in changes]==expected
counts=dict(Counter(r['semantic_state']for r in rows)); assert counts==dict(U=115,D=10,E=3)
bysplit={s:dict(Counter(r['semantic_state']for r in rows if r['split']==s))for s in sorted({r['split']for r in rows})}
assert bysplit=={'TRAIN':dict(U=88,D=8),'TRAIN_LAB':dict(U=12,D=2,E=2),'VAL_HALL':dict(U=15,E=1)}
status=dict(valid=sum(r['original_input_valid']for r in rows),noquery=sum(r['no_detection']for r in rows),finite_OOB=sum(not r['original_input_valid']and not r['no_detection']for r in rows))
assert status==dict(valid=107,noquery=8,finite_OOB=13)
assert summary['counts']==counts and summary['by_split']==bysplit
assert summary['input_valid']==107 and summary['noquery']==8 and summary['inputOOB']==13
assert summary['weak_TRAIN_rows']==12 and summary['weak_holdout_rows']==1
assert summary['source_partial_weak_supervision_available'] and not summary['dataset_general_visibility_training_sufficiency_proven']
assert summary['new_classifier_fit']==0
OUT.mkdir(exist_ok=False)
review=dict(completed=True,PASS=True,scope='SAVED_PUBLIC_SOURCE_ONLY_NO_RGB_MODEL_CURRENT_PRIVATE_READ',code=bind(__file__),
    root_manifest=bind(rootp),root_summary=bind(summaryp),original_source_prediction_seal=bind(sealpath),original_queries=bind(queryp),
    original_proposals=bind(propsp),unchanged_assistant128_proposals=bind(proposalp),crop_facts=[bind(p)for p in factpaths],
    full16x8_ID_native_order_q_L_original_input_flags_exact=True,all120_candidates_and_original_first_argmax_exact=True,
    original_pure_native_query_math_all128_exact=True,crop_SHA_and_constructed_metadata_all128_exact=True,
    semantic_counts=counts,by_split=bysplit,input_status=status,root_semantic_changes_exact=changes,
    conditional_Loco10_E2_primary_U=True,Loco02_top2_primary_U=True,Hall15_native3_diagnostic_E=True,
    source_weak_TRAIN_eligible=12,source_weak_TRAIN_D10_E2=True,VAL_HALL_E1_diagnostic_only=True,
    source_partial_weak_usable_not_full_gold_or_training_sufficiency=True,
    original_assistant_proposals_unchanged=True,current_model_fit_usage_declared_zero=True,
    newly_read_RGB_model_weight_current223_V112_PnP=0,
    limitations_KO='저장 조인·분모·판정 변경만 검산했습니다. 약한 source 역할 라벨12 TRAIN과1 HALL 진단이며 사람 gold·정밀 꼭짓점·8채널 전체 ontology·실사 일반화 성능을 인증하지 않습니다.')
(OUT/'REVIEW.json').write_text(json.dumps(review,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(bind(OUT/'REVIEW.json')))

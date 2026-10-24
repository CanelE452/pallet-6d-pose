"""Saved V136 source initializer diagnosis; no image/model imports or reads.

CF is the declared primary convention. Historical V4 is sensitivity only.
No predicted-coordinate matching or better-performing convention selection.
"""
from pathlib import Path
import argparse, ast, collections, hashlib, json, sys
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V = HERE.parent
P = V.parent
POLICY = V/'ROOT_CONTINUOUS_CORPUS_POLICY_BEFORE_PIXELS.json'
POLICY_SHA = '546f3d28d4454ba8a28a5eecdbfb0d30d91e6513e2ecd3209fb8945be8dd9ec4'
CORPUS = V
RENDER = CORPUS/'actual_corpus_v1/RENDER_COMPLETE.json'
SEAL = V/'actual_initializer_v1/PREDICTION_SEAL.json'
OUT = V/'saved_initializer_diagnosis_v1'
HISTORICAL = Path('/tmp/pallet-bi-full-execution-20261019-v1/scripts/annotate/convert_to_camera_facing_v4.py')
HISTORICAL_SHA = '3633dacafdfb4370eec57c7abcb627d32f80c671fbff1708966ebf51b4c452b1'
VARIANTS = ('BEFORE','COVER','NEAR_SHAM','FAR_SHAM')
NAMES = ('CF_PRIMARY_DECLARED','V4_SENSITIVITY_ONLY')
BINS = ('LE3_PX','GT3_LE8_PX','GT8_LE20_PX','GT20_PX','NO_QUERY')
PROTOCOL=HERE/'PROTOCOL_BEFORE_SAVED_DIAGNOSIS.json'
CONFIGS=V/'render_preparation/BASE_CONFIGS640_BEFORE_RGB.json'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p): return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p): return json.loads(Path(p).read_text())
def check(b):
    assert sha(b['path']) == b['sha256'], b['path']
    return Path(b['path'])
def same(a,b): return a['path'] == b['path'] and a['sha256'] == b['sha256']
def write(p,d):
    with Path(p).open('x') as f: json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')
def lines(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def identity(r): return tuple(r[k] for k in ('scene_index','base_index','id','base_id','split','variant'))
def output_in(seal,b): return any(same(x,b) for x in seal['outputs'])
def historical_function():
    assert sha(HISTORICAL) == HISTORICAL_SHA
    funcs = [x for x in ast.parse(HISTORICAL.read_text()).body
             if isinstance(x,ast.FunctionDef) and x.name in ('polyarea','compute_perm_v4')]
    assert len(funcs) == 2
    ns = dict(np=np)
    exec(compile(ast.Module(body=funcs,type_ignores=[]),str(HISTORICAL)+'::pure_GT_only','exec'),ns)
    return ns['compute_perm_v4']


def fixed_ledger(configs):
    fn = historical_function(); out = []
    for bi,c in enumerate(configs):
        assert c['base_index'] == bi and c['base_id'] == f'V136_BASE_{bi:04d}'
        cf = np.asarray(c['permCF_to_canonical'],np.int64)
        assert sorted(cf.tolist()) == list(range(8))
        canonical_xy = np.empty((8,2),np.float64)
        canonical_xy[cf] = np.asarray(c['native8_CF_xy'],np.float64)
        xyz = np.asarray(c['native8_canonical_xyz'],np.float64)
        # Renderer has X right, Y gravity-down, Z horizontal. V4 expects Z up.
        origin_z_up = xyz[:,[0,2,1]].copy(); origin_z_up[:,2] *= -1
        raw_v4 = fn(origin_z_up,canonical_xy)
        assert raw_v4 is not None and raw_v4[8] == 8 and sorted(raw_v4[:8]) == list(range(8))
        v4 = raw_v4[:8]; boundary = cf.tolist() != v4
        out.append(dict(base_index=bi,base_id=c['base_id'],split=c['split'],
                        CF_slot_to_canonical=cf.tolist(),V4_slot_to_canonical=v4,
                        correspondence_boundary=boundary,factors=c['factors']))
    # CF has been certified against the source-label normal-facing rule.
    # Obsolete area-V4 disagreements may occur in either fold and are never used to choose correspondence.
    assert all(c['source_label_mapping_certificate']['perm']==r['CF_slot_to_canonical']for c,r in zip(configs,out))
    return out


def error_bin(error):
    if error is None: return 'NO_QUERY'
    return 'LE3_PX' if error <= 3 else 'GT3_LE8_PX' if error <= 8 else 'GT8_LE20_PX' if error <= 20 else 'GT20_PX'


def summarize(rows):
    a = [r['error_px'] for r in rows if r['error_px'] is not None]
    return dict(original_slots=len(rows),teacher_states=dict(collections.Counter(r['teacher_state'] for r in rows)),
                input_reasons=dict(collections.Counter(r['input_status'] for r in rows)),
                position_bands=dict(collections.Counter(r['position_band'] for r in rows)),
                continuous_inbounds=sum(r['continuous_inbounds'] for r in rows),
                nearest_pixel_inbounds=sum(r['nearest_inbounds'] for r in rows),
                finite_raw_query_count=len(a),finite_query_OOB=sum(r['error_px'] is not None and not r['continuous_inbounds'] for r in rows),
                error_px_quantiles_0_25_50_75_90_95_100=np.quantile(a,[0,.25,.5,.75,.9,.95,1]).tolist() if a else None,
                all_slots_within3=sum(r['position_band']=='LE3_PX' for r in rows),
                within3_rate_all_original_slots=sum(r['position_band']=='LE3_PX' for r in rows)/len(rows) if rows else None,
                teacher_by_input_and_position=dict(collections.Counter(
                    r['teacher_state']+'::'+r['input_status']+'::'+r['position_band'] for r in rows)))


def run(entry):
    e=read(entry);assert e['approved_saved_diagnosis']is True
    assert all(e[k]is False for k in('model_forward','fit','current223','V112','PnP','q_correction','automatic_retry'))
    for k,path in(('code',Path(__file__)),('protocol',PROTOCOL),('policy',POLICY),('configs',CONFIGS),('render_seal',RENDER),('initializer_seal',SEAL)):assert same(e[k],bind(path))
    prep=read(PROTOCOL);assert same(prep['code'],e['code'])and same(prep['policy'],e['policy'])and same(prep['configs'],e['configs'])
    # Verify completed input-only predictions before hashing or reading any GT.
    assert SEAL.exists(), 'Actual initializer prediction seal required; no prediction/GT diagnosis before it.'
    seal = read(SEAL)
    assert seal['completed'] and seal['PASS'] and seal['all2560_scenes20480_slots_preserved']
    assert seal['GT_read'] is False and seal['training_admission'] is False
    result_path = check(seal['result']); result = read(result_path)
    assert output_in(seal,seal['result']) and result['completed'] and result['PASS']
    assert result['frames'] == 2560 and result['slots'] == 20480
    assert result['after_state_unchanged'] and result['checkpoint_unchanged']
    assert result['all_raw9_candidates_retained'] and result['first_argmax_no_query_correction']
    assert result['GT_mixed_buffers_mapping_crop_head_fit_current_V112_PnP_access'] == 0
    assert sha(POLICY)==POLICY_SHA and same(result['render_seal'],e['render_seal'])
    for k in ('proposals','queries'):
        assert output_in(seal,result[k]); check(result[k])
    proposals = lines(result['proposals']['path']); queries = lines(result['queries']['path'])
    assert len(proposals) == 2560 and len(queries) == 20480
    assert not OUT.exists(), 'Preserve any previous diagnosis; no silent replacement.'

    # GT metadata starts only after the complete initializer seal and exact rows.
    render = read(RENDER); assert render['completed'] and render['PASS']
    cfgb,gtb,inputs_b = (render[k] for k in ('base_configs','GT_scenes','input_scenes'))
    assert same(inputs_b,result['inputs'])and same(cfgb,e['configs'])
    configs = read(check(cfgb))['rows']; gt = read(check(gtb))['rows']; inputs = read(check(inputs_b))['rows']
    assert len(configs) == 640 and len(gt) == len(inputs) == 2560
    ledger = fixed_ledger(configs); allrows = []; primary = []; coverage = collections.Counter()
    for si,(pr,g,im) in enumerate(zip(proposals,gt,inputs)):
        bi,vi = divmod(si,4); c=configs[bi]; m=ledger[bi]
        assert identity(pr) == identity(g) == identity(im) == (si,bi,f'V136_SCENE_{si:04d}',f'V136_BASE_{bi:04d}',c['split'],VARIANTS[vi])
        assert pr['raw_hw'] == im['raw_hw'] == [512,512]
        cand=pr['candidates']; selected=None if not cand else int(np.argmax([r['score'] for r in cand]))
        assert selected == pr['selected_index']
        q9=None if selected is None else np.asarray(cand[selected]['keypoints_xy'],np.float64)
        if q9 is not None: assert q9.shape==(9,2) and np.isfinite(q9).all()
        canonical=g['records_canonical']; cf=g['records_CF']
        assert len(canonical)==len(cf)==8 and [r['native_canonical_index'] for r in canonical]==list(range(8))
        for k in range(8):
            assert canonical[m['CF_slot_to_canonical'][k]]['state']==g['state8_CF'][k]==cf[k]['state']
            assert cf[k]['canonical_index']==m['CF_slot_to_canonical'][k] and cf[k]['CF_index']==k
            assert np.array_equal(cf[k]['projected_reference_xy'],c['native8_CF_xy'][k])
            qr=queries[si*8+k]
            assert identity(qr)==identity(pr) and qr['native_slot']==k and qr['query_id']==pr['id']+':native'+str(k)
            assert qr['selected_index']==selected and qr['actual_original_initializer'] is True
            q=None if qr['query_xy'] is None else np.asarray(qr['query_xy'],np.float64)
            if selected is None:
                assert q is None and qr['roi_side'] is None and qr['reason']=='NO_DETECTION_U'
                assert not qr['original_query_continuous_valid'] and not qr['query_nearest_pixel_valid'] and not qr['query_rgb96_centre_domain_valid']
                cont=near=False; status='NO_DETECTION_U'
            else:
                assert q.shape==(2,) and np.array_equal(q,q9[k])
                cont=bool((q>=0).all() and (q<512).all()); pix=np.floor(q+.5)
                near=bool((pix>=0).all() and (pix<512).all())
                assert cont==qr['original_query_continuous_valid'] and near==qr['query_nearest_pixel_valid']==qr['query_rgb96_centre_domain_valid']
                finite=q9[:8][~(q9[:8]==-1).all(1)]; L=float(np.clip(.25*np.max(np.ptp(finite,axis=0)),16,512)) if len(finite)>=2 else 16.
                assert L==qr['roi_side']
                status='INBOUNDS_BOTH' if cont and near else 'CONTINUOUS_ONLY_NEAREST_OOB' if cont else 'NEAREST_ONLY_CONTINUOUS_OOB' if near else 'OOB_BOTH'
            coverage[status]+=1
            for convention,mapkey in zip(NAMES,('CF_slot_to_canonical','V4_slot_to_canonical')):
                ci=m[mapkey][k]; record=canonical[ci]; ref=np.asarray(record['projected_reference_xy'],np.float64)
                assert ref.shape==(2,) and np.isfinite(ref).all() and record['state'] in 'DESU'
                error=None if q is None else float(np.linalg.norm(q-ref))
                row=dict(scene_index=si,base_index=bi,id=pr['id'],base_id=pr['base_id'],split=c['split'],variant=VARIANTS[vi],
                         native_slot=k,query_id=qr['query_id'],convention=convention,canonical_index=ci,
                         correspondence_boundary=m['correspondence_boundary'],teacher_state=record['state'],
                         query_xy=qr['query_xy'],roi_side=qr['roi_side'],reference_xy=ref.tolist(),
                         error_px=error,delta_xy=(q-ref).tolist() if q is not None else None,position_band=error_bin(error),
                         input_status=status,continuous_inbounds=cont,nearest_inbounds=near,
                         raw_RGB_PNG=im['raw_RGB_PNG'],factors=c['factors'],human_gold=False)
                allrows.append(row)
                if convention==NAMES[0]: primary.append(row)
    assert len(primary)==20480 and len(allrows)==40960 and sum(coverage.values())==20480
    assert sum(r['input_status']=='NO_DETECTION_U' for r in primary)==8*result['no_detection_frames']
    declared_counts={s:sum(render['state_counts_before_query_expansion'][f][s]for f in('TRAIN','VALIDATION'))for s in'DESU'}
    assert {s:sum(r['teacher_state']==s for r in primary)for s in'DESU'}==declared_counts
    disagreement_counts={f:sum(r['correspondence_boundary']and r['split']==f for r in ledger)for f in('TRAIN','VALIDATION')}
    summaries={}
    for name in NAMES:
        rr=[r for r in allrows if r['convention']==name]; groups={}
        for split in ('ALL','TRAIN','VALIDATION'):
            sr=[r for r in rr if split=='ALL' or r['split']==split]; groups[split]=summarize(sr)
            for state in 'DESU': groups[split+'::teacher'+state]=summarize([r for r in sr if r['teacher_state']==state])
            for vi in VARIANTS: groups[split+'::'+vi]=summarize([r for r in sr if r['variant']==vi])
            for flag in (False,True): groups[split+'::boundary'+str(flag)]=summarize([r for r in sr if r['correspondence_boundary']==flag])
            for factor in ('geometry_family','aspect','material','foreign_family'):
                for value in sorted({r['factors'][factor] for r in sr}):
                    groups[split+'::'+factor+'::'+value]=summarize([r for r in sr if r['factors'][factor]==value])
        summaries[name]=groups
    # Deterministic first source-VAL representative in each teacher/error cell.
    # Complete selection is saved before any future image/crop reader runs.
    representatives=[]
    for state in 'DESU':
        for band in BINS:
            chosen=next((r for r in primary if r['split']=='VALIDATION' and r['teacher_state']==state and r['position_band']==band),None)
            if chosen: representatives.append(dict(selection_group=state+'::'+band,**chosen,
                companion_scene_indices_same_base=list(range(chosen['base_index']*4,chosen['base_index']*4+4))))
    for flag in (False,True):
        chosen=next((r for r in primary if r['split']=='VALIDATION' and r['correspondence_boundary']==flag and r['error_px'] is not None),None)
        if chosen: representatives.append(dict(selection_group='convention_boundary_'+str(flag),**chosen,
                companion_scene_indices_same_base=list(range(chosen['base_index']*4,chosen['base_index']*4+4))))
    OUT.mkdir()
    write(OUT/'FIXED_CONVENTION_LEDGER640_GT_ONLY.json',dict(completed=True,primary=NAMES[0],sensitivity=NAMES[1],rows=ledger,
          obsolete_area_disagreement_base_counts=disagreement_counts,no_prediction_matching_or_convention_selection=True,
          historical_code=bind(HISTORICAL),canonical_to_historical_origin='[X,Z,-Y]',centroid_excluded=True))
    with (OUT/'ALL20480_SLOTS_TWO_CONVENTIONS_GT_ONLY.jsonl').open('x') as f:
        for row in allrows: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    write(OUT/'REPRESENTATIVE_SELECTION_BEFORE_RGB_READ.json',dict(completed=True,rule='first VAL row per primary teacher/error cell; first available boundary/nonboundary',rows=representatives,
          RGB_or_crop_pixels_read=0,selection_changes_model_or_gate=False))
    report=dict(completed=True,PASS=True,goal_status='active',fully_solved=False,source_only=True,
        original_scenes=2560,original_slots=20480,convention_ledger_rows=640,reported_convention_rows=40960,
        conventions_not_independent_samples=True,position_errors_are_declared_correspondence_diagnostics=True,
        obsolete_area_disagreement_base_counts=disagreement_counts,model_semantics_at_boundary='CF_NORMAL_FACING_SOURCE_LABEL_ORDER_CERTIFIED; OBSOLETE_AREA_V4_DIAGNOSTIC_ONLY',
        original_teacher_STATES_preserved=True,teacher_U_is_not_prediction_U=True,
        primary_teacher_mapping_present_slots=20480,primary_teacher_mapping_missing_slots=0,
        sensitivity_teacher_mapping_present_slots=20480,sensitivity_teacher_mapping_missing_slots=0,input_coverage=dict(coverage),
        summaries=summaries,prediction_seal=bind(SEAL),prediction_result=bind(result_path),
        configs=cfgb,GT_scenes=gtb,input_scenes=inputs_b,code=bind(__file__),policy=bind(POLICY),
        RGB_head_model_fit_private_V112_PnP_q_correction=0,entry=bind(entry),
        limitations_KO=['새 합성 형상·렌더링의 실제 초기 추론 진단이며 실사 전이 성능이 아니다.',
            '정답상 보임/외부가림/자기가림/U와 좌표 오류를 별도 보존한다. 초기 추론 성공과 가림 분류 성능은 다르다.',
            'CF normal-facing 순서는 전 설정에서 소스 라벨 규칙으로 검증됐다. 구식 투영면적 V4는 불일치 진단으로만 보존하며 결과에 따라 대응을 바꾸지 않는다.',
            '위치 오차는 finite OOB 좌표도 포함한다. NO_DETECTION 8슬롯은 원분모에 남으며 오차는 null이다.'])
    write(OUT/'SOURCE_RESULTS_KO.json',report)
    write(OUT/'DIAGNOSIS_SEAL.json',dict(completed=True,PASS=True,prediction_seal=bind(SEAL),
        outputs=[bind(x) for x in sorted(OUT.iterdir()) if x.is_file()],RGB_read=False,source_only=True,code=bind(__file__)))
    print(json.dumps(dict(completed=True,PASS=True,seal=bind(OUT/'DIAGNOSIS_SEAL.json'),input_coverage=dict(coverage)),ensure_ascii=False),flush=True)


def prepare():
    assert sha(POLICY)==POLICY_SHA
    fixed_ledger(read(CONFIGS)['rows']) # GT-coordinate-only historical math; no predictions or RGB/teacher payload.
    protocol=dict(completed=True,prepared_only=True,code=bind(__file__),policy=bind(POLICY),configs=bind(CONFIGS),historical_area_rule=bind(HISTORICAL),
        original_scenes=2560,original_slots=20480,base_count=640,GT_only_after_actual_initializer_seal=True,CF_normal_facing_primary_certified=True,
        obsolete_area_V4_sensitivity_only=True,no_predicted_matching_or_convention_selection=True,no_RGB_model_fit_private_V112_PnP=True,
        expected_render_seal=str(RENDER),expected_initializer_seal=str(SEAL),output=str(OUT))
    write(PROTOCOL,protocol)
    template=dict(approved_saved_diagnosis=False,model_forward=False,fit=False,current223=False,V112=False,PnP=False,q_correction=False,automatic_retry=False,
        code=bind(__file__),protocol=bind(PROTOCOL),policy=bind(POLICY),configs=bind(CONFIGS),render_seal=None,initializer_seal=None,
        fill_after_complete='Bind actual V136 render and actual original initializer seals; never substitute old seals.')
    write(HERE/'ROOT_DIAGNOSIS_ENTRY_TEMPLATE.json',template)
    print(json.dumps(dict(completed=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),template=bind(HERE/'ROOT_DIAGNOSIS_ENTRY_TEMPLATE.json'))),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.prepare and a.entry is None:prepare()
    elif a.entry and not a.prepare:run(a.entry)
    else:raise SystemExit('Use --prepare, or root-approved --entry after actual initializer seal.')

"""Input-only G42 from unchanged actual native initializer channels.

No RGB/crop/feature body, GT, model, solver, or normalization-fit access. The
exact pinned V74 pure function is extracted by AST, without its GT helpers.
"""
from pathlib import Path
import argparse, ast, hashlib, json, traceback
import numpy as np

H = Path(__file__).resolve().parent
V = H.parent
POLICY = V / 'ROOT_MATCHED_GEOMETRY_POLICY_BEFORE_INPUTS.json'
POLICY_SHA = '68a48836f65f43d37a9e96c1acc892060b924ca91f2a079755bb061564a4bfe8'
PROTOCOL = H / 'INPUT_GEOMETRY_PROTOCOL_BEFORE_BUILD.json'
GENERATED = H / 'GENERATED_GEOMETRY_CHECKS.json'
OUT = V / 'input_geometry_v1'
N = 6144


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def bind(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def check(binding):
    assert sha(binding['path']) == binding['sha256'], binding['path']
    return Path(binding['path'])


def same(a, b):
    return a['path'] == b['path'] and a['sha256'] == b['sha256']


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x]


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def pure(path, names):
    nodes = [n for n in ast.parse(Path(path).read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    ns = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path)+'::pure', 'exec'), ns)
    return ns


def function_ast_sha(path):
    node = next(n for n in ast.parse(Path(path).read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'input_context_features')
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def load_input_metadata(p):
    # Metadata bindings only. Features/crop/RGB payload bindings are never checked.
    for key in ('native_feature_seal', 'native_initializer_seal', 'initializer_result',
                'initializer_proposals', 'initializer_queries', 'input_rows', 'visibility_protocol'):
        check(p[key])
    vis = read(p['native_feature_seal']['path'])
    init = read(p['native_initializer_seal']['path'])
    result = read(p['initializer_result']['path'])
    assert vis['completed'] and vis['PASS'] and vis['mode'] == 'c16' and vis['rows'] == N
    assert vis['head_forwarded'] == 6088 and vis['available'] == 6056 and vis['GT_read'] is False
    assert same(vis['initializer_seal'],p['native_initializer_seal']) and same(vis['input_crop_facts'],p['input_rows'])
    assert same(vis['initializer_proposals'],p['initializer_proposals']) and same(vis['initializer_queries'],p['initializer_queries'])
    assert same(vis['features'],p['original_visual_features_binding_not_opened'])
    assert init['completed'] and init['PASS'] and init['GT_read'] is False and init['all768_scenes6144_slots_preserved']
    assert same(init['result'],p['initializer_result']) and result['completed'] and result['PASS']
    assert result['frames'] == 768 and result['slots'] == N
    assert same(result['proposals'],p['initializer_proposals']) and same(result['queries'],p['initializer_queries'])
    packet=read(p['input_rows']['path'])
    assert packet['completed'] and packet['GT_read'] is False and len(packet['rows']) == N
    source=packet['rows']; proposals=lines(p['initializer_proposals']['path']); queries=lines(p['initializer_queries']['path'])
    assert len(proposals)==768 and len(queries)==N
    native_grid=pure(p['native_grid_code']['path'],('native_grid',))['native_grid']
    target=pure(p['target_extent_code']['path'],('query_extent','valid_box'))
    rows=[]
    for si,s in enumerate(proposals):
        assert s['scene_index']==si and s['base_index']==si//4
        assert s['id']=='V129_SCENE_%04d'%si and s['split']==('TRAIN'if si<512 else'VALIDATION')
        assert s['variant']==('BEFORE','COVER','NEAR_SHAM','FAR_SHAM')[si%4] and s['raw_hw']==[512,512]
        cs=s['candidates']; selected=None if not cs else int(np.argmax([c['score']for c in cs]))
        assert selected==s['selected_index']
        if selected is None:
            q8=box=side=None
        else:
            all9=np.asarray(cs[selected]['keypoints_xy'],np.float64)
            assert all9.shape==(9,2)and np.isfinite(all9).all()
            q8=all9[:8].copy();box=target['query_extent'](q8)
            _,_,side=native_grid(all9,s['raw_hw']);side=float(side)
            assert target['valid_box'](box)
        for k in range(8):
            i=si*8+k;r=source[i];q=queries[i]
            assert r['occurrence_index']==i and r['occurrence_id']==q['query_id']==s['id']+':native'+str(k)
            for key in ('scene_index','base_index','native_slot','split','variant'):
                expected=k if key=='native_slot'else s[key]
                assert r[key]==q[key]==expected,(i,key)
            expected_xy=None if q8 is None else q8[k].tolist()
            assert r['query_xy']==q['query_xy']==expected_xy and r['L']==q['roi_side']==side
            assert r['initial_points']==(None if q8 is None else q8.tolist()) and r['context_box_xyxy']==box
            constructed=bool(r['feature_constructed'])
            assert constructed==(selected is not None) and r['actual_original_initializer']is True and r['source_emulation']is False
            if constructed:
                x,y=expected_xy;continuous=bool(0<=x<512 and 0<=y<512)
                nearest=bool(0<=np.floor(x+.5)<512 and 0<=np.floor(y+.5)<512)
            else:
                continuous=nearest=False
                assert expected_xy is None and not r['query_available']
            assert bool(r['native_continuous_valid'])==q['original_query_continuous_valid']==continuous
            assert bool(r['native_nearest_valid'])==q['query_nearest_pixel_valid']==nearest
            rows.append(dict(global_index=i,source_occurrence_index=i,occurrence_id=r['occurrence_id'],
                             scene_index=si,base_index=si//4,native_slot=k,split=r['split'],variant=r['variant'],
                             query_xy=expected_xy,initial_points=None if q8 is None else q8.tolist(),L=side,
                             context_box_xyxy=box,raw_hw=s['raw_hw'],geometry_constructed=constructed,
                             C16_feature_constructed=constructed,query_available=bool(r['query_available']),
                             native_continuous_valid=continuous,native_nearest_valid=nearest,
                             actual_original_initializer=True,source_emulation=False))
    assert len(rows)==N and len({r['occurrence_id']for r in rows})==N
    assert sum(r['geometry_constructed']for r in rows)==6088 and sum(r['query_available']for r in rows)==6056
    assert sum(r['geometry_constructed']and not r['native_continuous_valid']for r in rows)==32
    return rows


def generated_checks(p):
    f=pure(p['geometry_code']['path'],('input_context_features',))['input_context_features']
    q=np.array([[20.,30.],[80.,30.],[80.,90.],[20.,90.],[35.,15.],[95.,15.],[95.,75.],[35.,75.]])
    box=[20.,15.,95.,90.];g=f(q,0,box,[128,160])
    assert g.shape==(42,)and g.dtype==np.float32 and np.isfinite(g).all()
    assert np.array_equal(g[:8],np.eye(8,dtype=np.float32)[0])
    assert np.array_equal(g[8:24],((q-q[0])/75).ravel().astype(np.float32)) and np.all(g[24:32]==1)
    assert np.array_equal(g[32:36],((np.array(box).reshape(2,2)-q[0])/75).ravel().astype(np.float32))
    assert np.array_equal(g[36:38],(q[0]/[160,128]).astype(np.float32))
    assert np.array_equal(g[38:40],[1.,1.]) and np.array_equal(g[40:42],(np.array([128,160])/75).astype(np.float32))
    g2=f(q*2,0,np.array(box)*2,[256,320]);assert np.array_equal(g,g2)
    shift=np.array([4.,6.]);gs=f(q+shift,0,np.tile(shift,2)+box,[128,160])
    assert np.array_equal(g[:36],gs[:36])and np.array_equal(g[38:],gs[38:])
    assert np.allclose(gs[36:38]-g[36:38],shift/[160,128],rtol=0,atol=2e-8)
    edge=q.copy();edge[0]=[159.75,30.];ge=f(edge,0,[20.,15.,159.75,90.],[128,160])
    assert ge[24]==1 and np.floor(edge[0,0]+.5)==160 # continuous, NOT nearest validity
    outside=q.copy();outside[0]=[191.,30.];go=f(outside,0,[20.,15.,191.,90.],[128,160])
    assert go[24]==0 and go[36]>1 and np.isfinite(go).all() # no coordinate clipping
    # Centroid is excluded by slicing before the exact42 helper, never supplied.
    q9=np.r_[q,[[1e6,-1e6]]];assert np.array_equal(f(q9[:8],0,box,[128,160]),g)
    zero=np.zeros((56,42),np.float32);assert np.isfinite(zero).all()and not zero.any()
    return dict(completed=True,PASS=True,code=bind(__file__),geometry_AST_sha256=function_ast_sha(p['geometry_code']['path']),
                checks=dict(layout42_native_slot_and_input_only_geometry_exact=True,
                            scale_is_max_native_q_span_not_crop_L=True,uniform_scale_invariance=True,
                            translation_only_changes_query_image_ratio=True,continuous_vs_nearest_bounds_separate=True,
                            finite_OOB_coordinates_preserved_unclipped=True,centroid_excluded_and_missing_rows_zero=True),
                source_RGB_crop_feature_GT_weights_reads_or_model_calls=0)


def authorize(entrypath):
    e=read(entrypath)
    assert e['approved_input_geometry_build']is True
    assert all(e[k]is False for k in ('GT_read','feature_body_read','RGB_read','model_forward','fit','current223','V112','PnP','q_correction','automatic_retry'))
    assert sha(POLICY)==POLICY_SHA and same(e['code'],bind(__file__))and same(e['protocol'],bind(PROTOCOL))
    p=read(PROTOCOL)
    assert same(p['code'],e['code'])and same(p['policy'],bind(POLICY))and same(e['generated'],p['generated'])
    check(p['generated']);assert read(p['generated']['path'])['PASS']and same(read(p['generated']['path'])['code'],e['code'])
    for key in ('geometry_code','native_grid_code','target_extent_code'):
        check(p[key])
    assert function_ast_sha(p['geometry_code']['path'])==p['geometry_AST_sha256']
    assert not OUT.exists(),'Preserve previous output/failure; no automatic retry.'
    return e,p,load_input_metadata(p)


def run(entrypath):
    e,p,rows=authorize(entrypath)
    f=pure(p['geometry_code']['path'],('input_context_features',))['input_context_features']
    OUT.mkdir()
    try:
        g=np.zeros((N,42),np.float32);constructed=np.array([r['geometry_constructed']for r in rows],bool)
        for i,r in enumerate(rows):
            if r['geometry_constructed']:
                g[i]=f(np.asarray(r['initial_points'],np.float64),r['native_slot'],r['context_box_xyxy'],r['raw_hw'])
                assert g[i,r['native_slot']]==1 and np.array_equal(g[i,:8],np.eye(8,dtype=np.float32)[r['native_slot']])
                assert bool(g[i,24+r['native_slot']])==r['native_continuous_valid']
        assert g.shape==(N,42)and g.dtype==np.float32 and np.isfinite(g).all()and not g[~constructed].any()
        common=dict(global_index=np.arange(N,dtype=np.int32),source_occurrence_index=np.arange(N,dtype=np.int32),
                    occurrence_id=np.array([r['occurrence_id']for r in rows]),corpus=np.full(N,'NATIVE_ACTUAL_INITIALIZER'),
                    scene_index=np.array([r['scene_index']for r in rows],np.int32),base_index=np.array([r['base_index']for r in rows],np.int32),
                    native_slot=np.array([r['native_slot']for r in rows],np.int8),
                    query_xy=np.array([r['query_xy']if r['query_xy']is not None else[np.nan,np.nan]for r in rows],np.float64),
                    L=np.array([r['L']if r['L']is not None else np.nan for r in rows],np.float64))
        path=OUT/'G42_INPUT_ONLY.npz'
        np.savez_compressed(path,G42=g,geometry_constructed=constructed,C16_feature_constructed=constructed,
                            query_available=np.array([r['query_available']for r in rows],bool),
                            native_continuous_valid=np.array([r['native_continuous_valid']for r in rows],bool),
                            native_nearest_valid=np.array([r['native_nearest_valid']for r in rows],bool),
                            source_emulation=np.array(False),actual_original_initializer=np.array(True),**common)
        rp=OUT/'GEOMETRY_ROWS_INPUT_ONLY.json'
        write(rp,dict(completed=True,rows=rows,GT_read=False,feature_body_read=False,source_emulation=False,
                      actual_original_initializer=True,old_AUX51004_geometry='UNKNOWN_ZERO_IMPLICIT_NO_DUPLICATED_FILE'))
        seal=dict(completed=True,PASS=True,entry=bind(entrypath),code=e['code'],protocol=e['protocol'],
                  G42=bind(path),input_rows=bind(rp),rows=N,shape=[N,42],dtype='float32',all_G42_finite=True,
                  geometry_constructed=int(constructed.sum()),noquery_zero_rows=int((~constructed).sum()),
                  available=int(sum(r['query_available']for r in rows)),finite_OOB_retained=32,
                  native_feature_seal=p['native_feature_seal'],native_initializer_seal=p['native_initializer_seal'],
                  original_visual_features_binding_not_opened=p['original_visual_features_binding_not_opened'],
                  original_query_L_order_identity_exact=True,centroid_excluded=True,
                  source_emulation=False,actual_original_initializer=True,old_AUX51004_geometry_unknown_zero_implicit=True,
                  no_clipping_or_fitted_normalizer=True,geometry_AST_sha256=p['geometry_AST_sha256'],
                  GT_RGB_crop_feature_body_weight_reads_or_model_calls=0,fit_current_V112_PnP_q_correction=0)
        write(OUT/'INPUT_GEOMETRY_SEAL.json',seal)
        print(json.dumps(dict(completed=True,seal=bind(OUT/'INPUT_GEOMETRY_SEAL.json'),rows=N,constructed=int(constructed.sum()))),flush=True)
    except BaseException:
        write(OUT/'FAILED_NO_AUTOMATIC_RETRY.json',dict(traceback=traceback.format_exc(),automatic_retry=False))
        raise


def prepare():
    assert sha(POLICY)==POLICY_SHA
    policy=read(POLICY)
    # Never read/hash native_fixed_GT_ledger, native_evaluation, or old scores.
    vis=read(check(policy['native_feature_seal']))
    init=read(check(policy['native_initializer_seal']))
    vproto=read(check(vis['protocol']))
    initresult=read(check(init['result']))
    p=dict(completed=True,code=bind(__file__),policy=bind(POLICY),geometry_code=policy['geometry_code'],
           native_feature_seal=policy['native_feature_seal'],native_initializer_seal=policy['native_initializer_seal'],
           initializer_result=init['result'],initializer_proposals=initresult['proposals'],initializer_queries=initresult['queries'],
           input_rows=vis['input_crop_facts'],visibility_protocol=vis['protocol'],
           native_grid_code=vproto['native_grid_code'],target_extent_code=vproto['c16']['target_crop_code'],
           original_visual_features_binding_not_opened=vis['features'],output=str(OUT),rows=N,
           exact_function='Pinned V74 AST-only input_context_features(query8,native_slot,q8minmax,rawHW)',
           layout={'0:8':'slot onehot8','8:24':'relativeq8x2 /scale','24:32':'nativecontinuousvalid8',
                   '32:36':'q8bboxendsrelquery /scale','36:38':'query/[W,H]','38:40':'span/scale','40:42':'[H,W]/scale'},
           scale='max(max(ptp(finite nativeq8)),1), NOT cropL; no clipping/normalizer fit',
           noquery='56zeroG42 with geometry_constructedFalse. Null q/L encoded NaN only in identity side arrays, never G42.',
           old_AUX51004='All geometryUNKNOWN; implicitzero42 model input only, no replicatedfile or nativebboxanchors',
           GT_read=False,RGB_read=False,feature_body_read=False,model_forward=False,
           fit_current_V112_PnP_q_correction=0,root_owns_actual_execution=True)
    for key in ('geometry_code','native_grid_code','target_extent_code'):
        check(p[key])
    p['geometry_AST_sha256']=function_ast_sha(p['geometry_code']['path'])
    # Metadata-only q/L/order parity is allowed; do not construct actual G42 yet.
    rows=load_input_metadata(p)
    p['metadata_only_preflight']=dict(rows=len(rows),constructed=int(sum(r['geometry_constructed']for r in rows)),
                                      noquery=int(sum(r['query_xy']is None for r in rows)),
                                      finite_OOB=int(sum(r['geometry_constructed']and not r['native_continuous_valid']for r in rows)),
                                      actual_G42_built=False)
    write(GENERATED,generated_checks(p));p['generated']=bind(GENERATED)
    write(PROTOCOL,p)
    ep=H/'ROOT_GEOMETRY_ENTRY_TEMPLATE.json'
    write(ep,dict(approved_input_geometry_build=False,GT_read=False,feature_body_read=False,RGB_read=False,
                  model_forward=False,fit=False,current223=False,V112=False,PnP=False,q_correction=False,
                  automatic_retry=False,code=p['code'],protocol=bind(PROTOCOL),generated=p['generated']))
    receipt=dict(completed=True,PASS=True,code=p['code'],protocol=bind(PROTOCOL),generated=p['generated'],
                 entry_template=bind(ep),actual_G42_built=False,source_RGB_GT_feature_body_or_model_reads=0,
                 command='/home/minjae/anaconda3/bin/python '+str(Path(__file__).resolve())+' --entry ROOT_ACTUAL_GEOMETRY_ENTRY.json')
    write(H/'PREPARATION.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--entry',type=Path)
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.entry:run(args.entry)
    else:raise SystemExit('Use --prepare or root --entry')

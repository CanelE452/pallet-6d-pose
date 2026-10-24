"""Bounded V208 preflight: code/declared TRAIN selection only; never follow GT or binaries."""
from pathlib import Path
import ast, collections, copy, hashlib, json, math, os, re, stat
H=Path(__file__).absolute().parent;V=H.parent;A=V/'preparation_a'
READY_SHA='08db6086b90b4471d2e12883566ac9254fcb5dc52fae4e79e0a110de00142b90'

def rb(p):
    p=Path(p);assert V in p.parents and'..'not in p.parts and all('v112'not in z.casefold()for z in p.parts)
    assert p.suffix in {'.py','.json'}and p.name not in {'SCENES_GT_ONLY.json','BASE_CONFIGS192_BEFORE_RGB.json'}
    fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for z in p.parts[1:-1]:n=os.open(z,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    with os.fdopen(n,'rb')as f:assert stat.S_ISREG(os.fstat(f.fileno()).st_mode);return f.read()
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(rb(p)).hexdigest())
def get(b):
    z=rb(b['path']);assert hashlib.sha256(z).hexdigest()==b['sha256'];return json.loads(z)
def main():
    assert bind(A/'FINAL_READY.json')['sha256']==READY_SHA
    ready=json.loads(rb(A/'FINAL_READY.json'));protocol=get(ready['protocol']);sel=get(ready['selection']);white=get(ready['GT_whitelist'])
    texts={k:rb(ready[k]['path']).decode()for k in ['code','selection_code','test_code']}
    for k,s in texts.items():assert hashlib.sha256(s.encode()).hexdigest()==ready[k]['sha256']
    assert get(ready['checks'])['PASS']is True and ready['scope']==sel['scope']==protocol['scope']
    rows=sel['physical_control_rows'];parents=sel['parents'];original=sel['original_BEFORE_native8'];attempts=sel['attempts']
    assert len(rows)==1968 and len(parents)==41 and len(original)==328 and len(attempts)==82
    assert len({tuple(r['control_join_key'])for r in rows})==1968 and len({r['row_id']for r in rows})==1968
    assert all(r['role']=='TRAIN'and r['control_join_key']==['SYNTH_FINITE_END',r['attempt_id'],r['variant'],r['native_slot']]for r in rows)
    bybase={p['base_index']:p for p in parents};byorig={(r['base_index'],r['native_slot']):r for r in original}
    assert len(bybase)==41 and len(byorig)==328 and all(p['role']=='TRAIN'for p in parents+original+attempts)
    for base in bybase:
        oo=[r for r in original if r['base_index']==base]
        assert sorted(r['native_slot']for r in oo)==sorted(r['declared_CF_to_canonical']for r in oo)==list(range(8))
    for t in attempts:
        rr=[r for r in rows if r['attempt_id']==t['attempt_id']]
        assert len(rr)==t['physical_native_rows']==24 and set(r['variant']for r in rr)==set(t['present_variants'])=={'BEFORE','COVER','SHAM'}
        assert all(sorted(r['native_slot']for r in rr if r['variant']==v)==list(range(8))for v in t['present_variants'])
        assert t['base_index']in bybase
    for r in rows:
        o=byorig[r['source_base_index'],r['native_slot']]
        assert all(r[k]==o[k]for k in ['query_xy','L','original_query_available','original_global_index','original_occurrence_id','original_input_flags'])
    assert sum(not r['original_query_available']for r in original)==sel['counts']['original_parent_inputU']==0
    whitelist=sel['GT_json_whitelist_NOT_OPENED'];assert white['bindings']==whitelist==[p['GT_json_binding_NOT_OPENED']for p in parents]
    assert len(whitelist)==len({b['path']for b in whitelist})==41
    for p,b in zip(parents,whitelist):
        q=Path(b['path']) # Lexical examination only: no existence/stat/hash/read of these GT paths.
        assert q.is_absolute()and'..'not in q.parts and all('v112'not in z.casefold()for z in q.parts)
        assert q.name=='SCENES_GT_ONLY.json'and q.parent==Path(p['original_BEFORE_payload_binding_only']['path']).parent==Path(p['source_base_completion']['path']).parent
        assert Path(p['source_base_completion']['path']).name=='BASE_COMPLETE.json'and re.fullmatch('[a-f0-9]{64}',b['sha256'])
    assert sel['config_json_required']is False and sel['no_new_labels_or_admission']is True
    # Three focused functional checks from frozen pure AST functions on invented dictionaries only.
    functions={'need','saved_state','saved_reference','locality','extract_one'}
    tree=ast.parse(texts['code']);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in functions]
    ns={'math':math,'STATES':{'D','E','S','U'}};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<frozen-pure-functions>','exec'),ns)
    o={'native_slot':2,'declared_CF_to_canonical':5,'query_xy':[10.,20.],'L':96.,'original_query_available':False}
    rec={'CF_index':2,'canonical_index':5,'target_witness_states':['D','S'],'combined_witness_states':['E','D'],'state':'U','target_alone_region_state':'U','projected_reference_xy':[1000.,-100.],'all_region_witnesses_in_frame':False}
    before=copy.deepcopy(o);x=ns['extract_one'](o,rec,None)
    assert x['point0_target_alone_state']=='D'and x['point0_combined_state']=='E'and x['region_state']=='U'and x['original_query_available']is False and o==before
    assert x['original_crop_reference_locality']['inside_canvas_extent']is False # Does not alter saved ray state.
    absent=ns['extract_one'](o,{'CF_index':2,'canonical_index':5,'state':'D'},None)
    assert absent['point0_target_alone_state']==absent['point0_combined_state']=='UNKNOWN'and absent['saved_anchor_projected_reference_xy']is None
    choose=next(n for n in ast.parse(texts['selection_code']).body if isinstance(n,ast.FunctionDef)and n.name=='select_train')
    ns.update(META=('original_global_index','native_slot','query_xy','L','original_query_available'))
    exec(compile(ast.Module(body=[choose],type_ignores=[]),'<frozen-controlkey-selection>','exec'),ns)
    r=dict(row_id='invented',domain='SYNTH_FINITE_END',control_join_key=['SYNTH_FINITE_END','a','BEFORE',0],original_global_index=0,native_slot=0,query_xy=[1.,2.],L=4.,original_query_available=True,cache_index=900,root_admitted=False,source_status='UNKNOWN')
    held=dict(r,row_id='held',control_join_key=['SYNTH_FINITE_END','h','BEFORE',0]);chosen=ns['select_train']([held,r],[dict(r,role='TRAIN',cache_index=0),dict(held,role='HOLDOUT')])
    assert len(chosen)==1 and chosen[0][0]==r and chosen[0][0]['root_admitted']is False
    report=dict(completed=True,PASS=True,review_type='CODE_AND_SELECTED_TRAIN_METADATA_ONLY',ready=bind(A/'FINAL_READY.json'),code=ready['code'],selection_code=ready['selection_code'],protocol=ready['protocol'],selection=ready['selection'],GT_whitelist=ready['GT_whitelist'],
      selected_counts=sel['counts'],selection_internal_full_order_META_and_controlkey_checks=True,
      pure_fixtures=['point0 D/E retained with region U, all-region out-of-frame and original inputU','missing point/reference UNKNOWN with no region D fallback','TRAIN controlkey includes unadmitted UNKNOWN and ignores stale cache_index'],
      code_findings=['CF k uses declared canonical c and saved CF_index/canonical_index agreement, never inverse/nearest/GT-best','Only original BEFORE parent is selected by bounded scene/schema; unknown schema remains UNKNOWN, no config fallback','Point0 arrays[0] are distinct from region fields. Region all-in-frame is not used to gate point0','Saved reference is converted by F64 inverse crop indexing; both center and canvas extents are algebraic, not pixel visibility or coordinate gold','Changed V189 COVER/SHAM states remain UNKNOWN and are not copied from original GT/finite-end/owner'],
      access_limits=dict(original_V194_V180_V189_metadata_rescan=0,source_GT_config_file_stat_hash_header_or_body=0,binary_image_array_or_model_reads=0,new_projection_ray_forward_or_fit=0,protected_HOLDOUT_body_or_score_access=0),
      limits=['C checks selected metadata internal arithmetic/joins and selection code; C did not independently reselect from the original shared3144 container or reopen parent completion receipts','Full-image point visibility and actual crop valid-pixel support are not established by this lookup. Missing/unmeasured observability remains UNKNOWN; all-region flag is not a point-inside certificate','GT container shape and saved field availability are deliberately unverified until the separate actual41 root entry; UNKNOWN is permitted and is not a physicalgold blocker'],
      new_labels_or_admission_or_model_selection=False,goal_solved=False)
    with (H/'REVIEW.json').open('x')as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    with (H/'KO.md').open('x')as f:f.write('''# V208 독립 준비 검토\n\nPASS. 현재 V194 own constructed rows를 control_join_key로 원 component TRAIN role에 결합한 뒤 선택하는 코드를 확인했다. 선택된1968행/82attempt/41원BEFORE/328native의 내부 메타데이터를 독립 대조했고, 원8·q/L/native/가용성·flags 및41 GT JSON의 선언 경로/SHA 목록이 일치한다. GT 경로 자체의 stat/hash/read는 하지 않았다.\n\nCF slot k→declared canonical c 방향이며 저장 CF_index/canonical_index가 일치해야 한다. target/combined witness[0] ray 상태, 전체 region 상태, 원 q와 saved reference의 대수 crop 좌표는 별도다. all_region_witnesses_in_frame가 false여도 저장 point0 상태를 바꾸지 않으며, finite q/reference로 원 U를 보충하지 않는다. 원 BEFORE만 조회하고 새 COVER/SHAM native 상태는 UNKNOWN으로 남긴다.\n\n가상3검사는 regionU·원U·out-of-frame과 point0D/E의 분리, missing→UNKNOWN/region복사금지, stale index와 oldunadmitted 상태에 무관한 TRAIN controlkey 선택을 확인했다. 실제 GT schema/field 존재는 앞으로 root 실제 entry에서 확인할 사안이며, 모르는 frame/crop 관측 가능성은 UNKNOWN이다. 현재 crop extent는 대수적 범위이며 실제 픽셀 가시성·좌표 gold가 아니다. 새 threshold·라벨·학습 허가·모델 선택은 없다.\n''')
    print(json.dumps(bind(H/'REVIEW.json')))
if __name__=='__main__':main()

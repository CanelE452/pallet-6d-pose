"""V209 code/text-metadata preflight plus invented scalar tests; no NPZ/GT access."""
from pathlib import Path
from collections import Counter
import ast,hashlib,json,math,os,stat,copy
import numpy as np

H=Path(__file__).parent;V=H.parent;P=V.parent
READY=V/'native_control_authority_preparation_c/FINAL_READY.json'
READY_SHA='d1fa3d0cb63d4cc4849c8b0c9073e0852738fb61ff471df93e12c1606b937ea7'
SCOPE=V/'ROOT_PROSPECTIVE_SOURCE_TRAIN_NATIVE_POINT_CONTROL_AUTHORITY_SCOPE.json'

def rb(path):
    p=Path(path)
    assert p.is_absolute()and P in p.parents and '..'not in p.parts and 'v112'not in str(p).casefold()
    assert p.suffix in {'.json','.jsonl','.py'},'No binary, header or GT payload calls'
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            n=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    assert stat.S_ISREG(os.fstat(n).st_mode)
    with os.fdopen(n,'rb')as f:return f.read()
def bind(path):return dict(path=str(path),sha256=hashlib.sha256(rb(path)).hexdigest())
def get(b):
    raw=rb(b['path']);assert hashlib.sha256(raw).hexdigest()==b['sha256'];return json.loads(raw)
def rows(b):
    raw=rb(b['path']);assert hashlib.sha256(raw).hexdigest()==b['sha256'];return [json.loads(x)for x in raw.splitlines()if x.strip()]
def put(name,value):
    with (H/name).open('x',encoding='utf-8')as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def scalar_oracle(alpha,ref):
    if ref is None or len(ref)!=2 or not all(math.isfinite(x)for x in ref):return'UNKNOWN',[]
    x,y=ref;h,w=alpha.shape;flags=[];taps=[]
    # Triangular basis over neighboring integer centers, independent implementation.
    for iy in range(math.floor(y)-1,math.floor(y)+3):
        for ix in range(math.floor(x)-1,math.floor(x)+3):
            weight=max(0.,1-abs(x-ix))*max(0.,1-abs(y-iy))
            if weight>0:
                flag=bool(alpha[iy,ix])if 0<=ix<w and 0<=iy<h else None
                taps.append(((ix,iy),weight,flag));flags.append(flag)
    if any(x is None for x in flags):return'UNKNOWN',taps
    return ('COVERED'if all(flags)else'CLEAR'if not any(flags)else'EDGE_UNKNOWN'),taps

def main():
    assert bind(READY)['sha256']==READY_SHA
    ready=get(bind(READY));assert ready['completed']and ready['PASS']and ready['preparation_only']
    protocol=get(ready['protocol']);scope=get(protocol['sources']['scope'])
    assert protocol['sources']['scope']==bind(SCOPE)and protocol['code']==ready['code']
    for b in protocol['sources'].values():assert bind(b['path'])==b
    for key in ('code','protocol','checks','false_entry','whitelist','associations'):assert bind(ready[key]['path'])==ready[key]
    selection=get(protocol['sources']['selection']);points=rows(protocol['sources']['point_rows'])
    whitelist=get(ready['whitelist']);associated=rows(ready['associations']);false=get(ready['false_entry']);prior=get(ready['checks'])
    original=selection['physical_control_rows'];point_by={(p['base_index'],p['native_slot']):p for p in points}
    assert len(points)==len(point_by)==328 and len(original)==len(associated)==1968
    assert [r['control_join_key']for r in original]==[r['control_join_key']for r in associated]
    assert len({tuple(r['control_join_key'])for r in associated})==1968
    packet_by={p['attempt_id']:p for p in whitelist['packets']};assert len(packet_by)==82
    source_packets={};fullgroups=Counter()
    for original_row,row in zip(original,associated):
        assert all(row[key]==value for key,value in original_row.items())
        assert row['role']=='TRAIN'and row['control_join_key']==['SYNTH_FINITE_END',row['attempt_id'],row['variant'],row['native_slot']]
        p=point_by[row['source_base_index'],row['native_slot']]
        for key in ('original_global_index','original_occurrence_id','query_xy','L','original_query_available','original_input_flags'):assert row[key]==p[key]
        assert row['canonical_index']==p['canonical_index']and row['raw_hw']==p['raw_hw']
        assert row['saved_anchor_projected_reference_xy']==p['saved_anchor_projected_reference_xy']
        assert row['saved_original_point0_combined_state']==p['point0_combined_state']
        assert row['saved_original_point0_target_alone_state']==p['point0_target_alone_state']
        assert row['saved_original_region_state_context_only']==p['region_state']
        assert row['original_crop_reference_locality']==p['original_crop_reference_locality']
        assert row['requested_native_slot']==int(row['attempt_id'].rsplit('NATIVE',1)[1])
        assert row['new_native_point_state']is None and row['new_visibility_target_created']is False and row['training_admission']is False
        packet=packet_by[row['attempt_id']]
        assert packet['binding']==row['source_packet']and packet['role']=='TRAIN'
        assert packet['allowed_npz_members_to_decode']==['foreign_alpha']and packet['expected_dtype']=='bool'
        assert packet['variant_order_from_frozen_producer']==['BEFORE','COVER','SHAM']and packet['expected_shape']==[3,*p['raw_hw']]
        assert packet['base_index']==row['source_base_index']
        # Payload bindings stay inert, with lexical check before any filesystem operation.
        source=Path(packet['binding']['path']);assert 'v112'not in str(source).casefold()and source.name=='CONTROL_TRIPLET_INPUTS.npz'and '..'not in source.parts
        source_packets[row['attempt_id']]=packet['binding'];fullgroups[(row['attempt_id'],row['variant'])]+=1
    assert len(source_packets)==82 and len(fullgroups)==246 and set(fullgroups.values())=={8}
    assert len({r['source_base_index']for r in associated})==41
    assert all(false[k]is False for k in ('approved_SOURCE_TRAIN_point_alpha_alignment','automatic_retry','new_labels_or_admission','fit','model_forward'))
    assert false['root_actual_scope']is None and false['decode_only_foreign_alpha']is True
    for k in ('code','protocol','checks','whitelist','associations'):assert false[k]==ready[k]
    assert prior['PASS']and prior['invented_arrays_only']
    code=rb(ready['code']['path']).decode();tree=ast.parse(code)
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    # Execute only authorial pure functions on invented arrays, never import module/run prepare.
    env=dict(np=np,math=math,Path=Path)
    ast_code=ast.Module(body=[functions[k]for k in ('require','safe','alpha_relation','run')],type_ignores=[])
    exec(compile(ast_code,'<exact_authorial_functions>','exec'),env)
    checks=[]
    def ok(name,value):assert value,name;checks.append(name)
    zero=np.zeros((5,6),bool);one=~zero;mixed=one.copy();mixed[2,3]=False
    cases=[('clear',zero,[2.25,2.5]),('covered',one,[2.25,2.5]),('integer',mixed,[2.,2.]),('mixed',mixed,[2.25,2.5]),
        ('outside',one,[-.25,2.]),('last_integer',one,[5.,4.]),('missing',one,None),('nonfinite',one,[float('nan'),1.]),('tiny_positive',mixed,[2+2**-50,2.])]
    for label,alpha,reference in cases:
        actual=env['alpha_relation'](alpha,reference);expect,taps=scalar_oracle(alpha,reference)
        ok('Authorial_'+label+'_matches_independent_scalar',actual['status']==expect)
        got={tuple(t['xy']):(t['weight'],t['opaque'])for t in actual['positive_taps']}
        assert set(got)=={xy for xy,weight,flag in taps}
        for xy,weight,flag in taps:assert abs(got[xy][0]-weight)<1e-14 and got[xy][1]==flag
    for label,entry in [('false_approval',dict(approved_SOURCE_TRAIN_point_alpha_alignment=False)),
        ('decode_flag_false',dict(approved_SOURCE_TRAIN_point_alpha_alignment=True,source_TRAIN_only=True,automatic_retry=False,decode_only_foreign_alpha=False,new_labels_or_admission=False,fit=False,model_forward=False)),
        ('missing_actual_scope',dict(approved_SOURCE_TRAIN_point_alpha_alignment=True,source_TRAIN_only=True,automatic_retry=False,decode_only_foreign_alpha=True,new_labels_or_admission=False,fit=False,model_forward=False,root_actual_scope=None))]:
        calls=[];env['read']=lambda ignored,e=entry:e
        env['verify']=lambda ignored:calls.append('verify')
        try:env['run']('/never_opened.json')
        except AssertionError:ok('Actual_run_'+label+'_rejects_before_payload',not calls)
        else:raise AssertionError(label)
    try:env['safe']('/never_stat/V112/forbidden.npz')
    except AssertionError:ok('V112_rejects_before_Path_or_stat',True)
    else:raise AssertionError('Lexical guard')
    run_node=functions['run']
    archive_gets=[n for n in ast.walk(run_node)if isinstance(n,ast.Subscript)and isinstance(n.value,ast.Name)and n.value.id=='archive']
    ok('Only_foreign_alpha_archive_member_decoded',len(archive_gets)==1 and isinstance(archive_gets[0].slice,ast.Constant)and archive_gets[0].slice.value=='foreign_alpha')
    loads=[n for n in ast.walk(run_node)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='np'and n.func.attr=='load']
    ok('Archive_pickle_disabled',len(loads)==1 and any(k.arg=='allow_pickle'and isinstance(k.value,ast.Constant)and k.value.value is False for k in loads[0].keywords))
    source_text=rb(protocol['sources']['source_producer']['path']).decode();source_tree=ast.parse(source_text)
    saves=[n for n in ast.walk(source_tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='savez_compressed']
    alpha_kwargs=[k.value for n in saves for k in n.keywords if k.arg=='foreign_alpha'];assert len(alpha_kwargs)==1
    writer=ast.unparse(alpha_kwargs[0]);ok('Writer_saves_zero_ca_sa_not_RGB_difference',writer=='np.asarray([np.zeros_like(ca), ca, sa])')
    paste=next(n for n in source_tree.body if isinstance(n,ast.FunctionDef)and n.name=='paste_plane')
    alpha_assign=[n for n in ast.walk(paste)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='alpha'for t in n.targets)]
    opaque_formula=ast.unparse(alpha_assign[0].value)
    ok('Writer_opacity_is_depth_gated_foreground_rectangle','paste &'in opaque_formula and 'z < bd - tol'in opaque_formula and '~finite'in opaque_formula)
    results=dict(completed=True,PASS=True,code_only=True,own_code=bind(Path(__file__)),scope=bind(SCOPE),code=ready['code'],protocol=ready['protocol'],prepared_ready=bind(READY),
        whitelist=ready['whitelist'],associations=ready['associations'],metadata_counts=dict(TRAIN_packets=82,physical_rows=1968,parents=41,saved_original_points=328),
        checks=checks,invented_and_static_checks=len(checks),writer_foreign_alpha_expression=writer,writer_alpha_expression=opaque_formula,
        source_payload_stat_hash_header_decode=0,original_GT_reopened=0,producer_or_loader_imports=0,model_forward_or_fit=0,
        original_qL_native_availability_mapping_exact=True,labels_or_admission_created=False,
        limitations=['COVERED/CLEAR are positive integer raster tap-opacity relations, not a continuous 3D ray result or native D/E/S supervision.',
            'CLEAR may include non-front/depth-tie pixels absent from saved alpha; no tie mask is decoded and no visibility label follows.',
            'Future actual run requires a separately approved root scope/entry and exact frozen82 containers; this review decoded no payload.'])
    put('REVIEW.json',results)
    with(H/'KO.md').open('x',encoding='utf-8')as f:
        f.write('V209 코드·메타데이터 사전검토 PASS. 원 TRAIN82 packet/1968행/41parent/328point와 원 q/L/native/U·control-key 순서를 유지하며, writer가 저장한 foreign_alpha만 미래 actualentry 뒤 해독한다. RGB 차분·region·finite-end 라벨을 native 상태로 복사하지 않는다.\n\n')
        f.write('엄격 양수 bilinear tap 전체 True/False/mixed/outside 분류를 독립 scalar와 비교했다. COVERED/CLEAR는 저장 raster의 opacity 관계이지 연속 ray·native gold·새 E가 아니다. 깊이 tie는 alpha에 포함되지 않으므로 CLEAR를 가시성 인증으로 쓰지 않는다. 실제82 NPZ/원GT/이미지/학습 접근은 0이다.\n')
    print(json.dumps(dict(completed=True,PASS=True,review=bind(H/'REVIEW.json'),checks=len(checks),actual_payload_reads=0)))

if __name__=='__main__':main()

"""Seal eight manually observed weak visible-feature rectangles; no pixel computation."""
from pathlib import Path
import ast, collections, copy, csv, hashlib, json, os, stat

H=Path(__file__).absolute().parent
V=H.parent
P=V.parent
ENTRY=V/'ROOT_ACTUAL_INDEPENDENT_SOURCE8_WITNESS_ENTRY.json'
ANNOTATION_FIELDS={
    'visible_intended_part_identified_YES_UNKNOWN','witness_ROI_raw96_xyxy_or_null',
    'visible_feature_description','positive_local_visual_evidence','ambiguity_note',
    'actually_viewed_image_bindings'}

# Independent manual C values, recorded only after actual tools.view_image calls.
MANUAL=[
 ([52,48,82,80], '상단 팔레트 앞 왼쪽 상판 끝과 그 아래 블록의 맞닿는 모서리/세로면',
  '원 CLEAN96 오른쪽 아래에 밝은 상판의 끝 모서리와 갈색 블록 면이 연속해 보인다. full q8 맥락에서 아래쪽 다른 팔레트와 구별되는 상단 팔레트의 앞 왼쪽 끝이다.',
  '정밀 부품 분할이나 3D 점은 아니며 직사각형에는 모서리 주변 배경이 일부 포함될 수 있다. q 중심 포함을 요구하지 않았다.'),
 ([19,32,42,45], '상단 팔레트 앞 오른쪽 끝의 상판 전면/끝 모서리',
  '원 CLEAN96의 유효한 위쪽 영역에 노출된 상판의 밝은 윗면과 갈색 끝면 접합이 보인다. full q8와 contact에서 상단 팔레트 앞 오른쪽 끝임을 확인했다.',
  '원 영상 하단 밖은 검은 padding이다. 이 ROI는 보이는 유효 상부 조각만 포함하며 잘린 전체 끝의 외형을 복원하지 않는다.'),
 ([46,24,73,56], '상단 팔레트 앞 왼쪽 블록의 노출 세로면과 아래쪽 끝',
  'CLEAN96 중앙 오른쪽에서 상판 아래의 갈색 목재 블록 면이 바닥과 구별되고 아래쪽 경계가 보인다. full frame에서 같은 상단 팔레트 왼쪽 지지부로 식별된다.',
  '뒤쪽/아래쪽 다른 목재의 전 영역이나 물리 native 점을 표기한 것이 아니다. 보이는 블록 면의 일부를 근거로 선택했다.'),
 ([46,50,63,63], '상단 팔레트 뒤 왼쪽 긴 상판의 노출 끝 모서리',
  'CLEAN96 아래 중앙에서 긴 목재 상판이 끝나는 밝은 단부와 가장자리가 보인다. full q8/원영상의 뒤 왼쪽 상판 끝과 대응하며 위의 검은 작업대와 구별된다.',
  '상판 아래의 가려진 부위를 추정해 포함하지 않았다. 약한 2D visible feature rectangle이며 정확 native/canonical 정답이 아니다.'),
 ([40,33,61,60], '바닥 팔레트 앞 오른쪽 블록의 끝면/바깥 모서리',
  'CLEAN96의 가운데 근처에서 상판 아래 끝 블록의 밝은 면과 바닥 사이 바깥 경계가 드러난다. 원 frame에서 회색 통 아래쪽과 별개로 앞 오른쪽 끝이 노출된다.',
  '각인 또는 일반 목재 존재만으로 정한 것이 아니라 블록 끝면의 모서리를 근거로 했다. 직사각형이 전체 부품을 분할하지는 않는다.'),
 ([54,39,77,64], '적재물 아래 팔레트 앞 왼쪽 상판 끝/블록 윗모서리',
  'CLEAN96 오른쪽에서 상판이 끝나는 모서리와 바로 아래 EPAL 블록 윗면의 전환이 보인다. full frame에서 흰 포장재 아래 노출된 앞 왼쪽 끝으로 식별된다.',
  '글자 자체가 아니라 보이는 목재 끝 모서리가 근거다. 원 q가 배경에 가까운 사실과 이 visible feature를 분리했다.'),
 ([25,33,44,60], '적재물 아래 팔레트 앞 오른쪽 상판 끝과 블록 상단 모서리',
  'CLEAN96 왼쪽의 밝은 상판 끝에서 세로 블록 면으로 바뀌는 오른쪽 경계가 보인다. 원 frame/q8에서 앞 오른쪽 노출 끝과 대응한다.',
  'q 중심을 포함하도록 옮기지 않았고, 배경을 부품으로 보지 않았다. 전체 블록 또는 물리 점의 정밀 위치를 뜻하지 않는다.'),
 ([29,28,47,48], '적재물 아래 팔레트 앞 오른쪽 블록의 아래/바깥 모서리',
  'CLEAN96 왼쪽 위의 목재 블록 아래 경계가 어두운 바닥과 만나는 모서리가 노출되어 있다. full q8에서 앞 오른쪽 아래 끝 맥락을 확인했다.',
  '직사각형은 보이는 모서리 주변 약영역이며 일부 배경을 포함할 수 있다. q 오차를 수치화하거나 corrected q를 만들지 않는다.'),
]

def rb(p):
    p=Path(p)
    assert p.is_absolute() and P in p.parents and '..'not in p.parts
    assert all('v112'not in s.casefold()for s in p.parts)
    assert p.suffix.lower()in {'.json','.py','.png','.csv','.md'}
    fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for s in p.parts[1:-1]:
            n=os.open(s,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd);fd=n
        n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    with os.fdopen(n,'rb')as f:
        assert stat.S_ISREG(os.fstat(f.fileno()).st_mode)
        return f.read()

def bind(p):return {'path':str(p),'sha256':hashlib.sha256(rb(p)).hexdigest()}
def checked(b):
    z=rb(b['path']);assert hashlib.sha256(z).hexdigest()==b['sha256'];return json.loads(z)
def put(p,x):
    with p.open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')

def main():
    assert bind(ENTRY)['sha256']=='75d345e36bae482c00156ea89fabd1dac2a8d7ad7383d17b15e88cfc7ff69b17'
    entry=json.loads(rb(ENTRY));assert entry['approved_independent_A_C_SOURCE_TRAIN_weak_witnesses']is True
    ready=checked(entry['C_ready']);scope=checked(entry['scope']);selection=checked(entry['selection']);checked(ready['protocol'])
    assert ready['code']['sha256']=='1a943e74ab8bc3bc7b93c2894a082c291c680fcdb9ed3f3aa72961c4ee42f366'
    assert bind(ready['code']['path'])==ready['code']
    template=checked(ready['template']);cases=selection['source_cases8'];assert len(cases)==len(template['rows'])==len(MANUAL)==8
    # Invoke the frozen pure ROI/state validator without importing or rerunning preparation.
    tree=ast.parse(rb(ready['code']['path']).decode())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='validate_witness')
    ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),'<frozen-pure-ROI-validator>','exec'),ns)
    validate=ns['validate_witness']
    whitelist={b['path']:b for b in entry['source_generated_PNG_whitelist']};assert len(whitelist)==25
    assert all(Path(b['path']).suffix=='.png'for b in whitelist.values())
    # Actual tool-view order: 3 contexts, then selected CLEAN96/VALID96 pairs, per frame.
    views=[];seen_frames=[]
    for c in cases:
        f=c['original_frame_inputs'];fk=f['frame_key']
        if fk in seen_frames:continue
        seen_frames.append(fk)
        ordered=[(kind,f[kind])for kind in ['fulloriginal','overlay','all8contact']]
        for same in cases:
            if same['original_slot']['frame_key']==fk:
                ordered.extend((kind,same['original_slot'][kind])for kind in ['clean96','valid96'])
        for kind,b in ordered:
            assert whitelist.get(b['path'])==b and bind(b['path'])==b
            views.append({'sequence':len(views)+1,'frame_key':fk,'kind':kind,'binding':b,
                'actual_tool':'functions.exec -> tools.view_image','detail':'original','result':'displayed and independently inspected by C during V206'})
    assert len(views)==25 and len({v['binding']['path']for v in views})==25
    assert set(v['binding']['path']for v in views)==set(whitelist)
    rows=[];identity=[]
    for i,(old,c,manual)in enumerate(zip(template['rows'],cases,MANUAL)):
        s=c['original_slot'];f=c['original_frame_inputs'];r=copy.deepcopy(old)
        box,feature,evidence,ambiguity=manual
        relevant=[s['clean96'],s['valid96'],f['fulloriginal'],f['overlay'],f['all8contact']]
        r.update(visible_intended_part_identified_YES_UNKNOWN='YES',witness_ROI_raw96_xyxy_or_null=box,
            visible_feature_description=feature,positive_local_visual_evidence=evidence,ambiguity_note=ambiguity,
            actually_viewed_image_bindings=relevant)
        validate(r,final=True)
        # Mandatory seal guard separate from the generic ROI validator.
        assert all(r[k]==old[k]for k in old if k not in ANNOTATION_FIELDS)
        assert r['row_id']==s['row_id'] and r['frame_key']==s['frame_key']
        assert r['original_frame_index']==s['original_frame_index']and r['native_slot']==s['native_slot']
        assert r['original_query_xy']==s['query_xy']and r['original_L']==s['L']and r['original_query_available']==s['original_query_available']
        assert r['prior_relation_unchanged']==c['prior_relation']and r['prior_location_flags_unchanged']==c['prior_location_flags']
        assert r['training_admission']is False and r['corrected_query_or_localization_output']is False and r['physical_native_gold']is False
        vp={v['binding']['path']:v for v in views}
        assert all(whitelist.get(b['path'])==b and vp[b['path']]['binding']==b for b in relevant)
        assert s['clean96']in relevant and (f['fulloriginal']in relevant or f['overlay']in relevant)
        assert all(vp[b['path']]['frame_key']==s['frame_key']for b in relevant)
        rows.append(r);identity.append({'ordinal':i,'row_id':r['row_id'],'original_metadata_exact':True,'prior_flags_exact':True,
            'fitfalse_exact':True,'samecase_CLEAN96_and_sameframe_context_actually_toolviewed':True,'image_bindings':relevant})
    assert [r['row_id']for r in rows]==[c['original_slot']['row_id']for c in cases]==[r['row_id']for r in template['rows']]
    assert len({r['row_id']for r in rows})==8
    put(H/'ACTUALLY_VIEWED_BINDINGS.json',{'completed':True,'observer':'C','root_entry':bind(ENTRY),'selection':entry['selection'],
        'actual_generated_PNG_toolviews':25,'unique_CLEAN96':8,'unique_VALID96':8,'full_frames':3,'q8_overlays':3,'contacts':3,
        'views':views,'prior_V205_views_not_counted_as_new_V206_views':True,'browser_used':False,'other_sourceframes_or_new_source_arrays_viewed':0})
    put(H/'SEAL_IDENTITY_AND_VIEW_PRECONDITION_PASS.json',{'completed':True,'PASS':True,'code':bind(__file__),'selection':entry['selection'],
        'template':ready['template'],'rows8_original_order_ID_qL_native_U_exact':True,'all_prior_flags_unchanged':True,'training_admission_false_exact':True,
        'eachcase_sameCLEAN96_plus_sameframe_fullcontext_actually_toolviewed':True,'all_viewbindings_in_root_whitelist':True,
        'rows':identity,'limits':'Identity/view guard is separate from the generic pure ROI/state validator and does not certify visual semantics or physical native gold.'})
    put(H/'INDEPENDENT8_WITNESS_OBSERVATIONS.json',{'completed':True,'observer':'C','authority':scope['authority'],'root_entry':bind(ENTRY),
        'selection':entry['selection'],'protocol':ready['protocol'],'actual_views':bind(H/'ACTUALLY_VIEWED_BINDINGS.json'),
        'identity_view_precondition':bind(H/'SEAL_IDENTITY_AND_VIEW_PRECONDITION_PASS.json'),'rows':rows,'new_A_ROI_reads':0,'training_admission':False})
    flat=[{k:json.dumps(v,ensure_ascii=False)if isinstance(v,(list,dict))else v for k,v in r.items()}for r in rows]
    with (H/'INDEPENDENT8_WITNESS_OBSERVATIONS.csv').open('x',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    put(H/'VALIDATOR_PASS.json',{'completed':True,'PASS':True,'frozen_pure_validator_code':ready['code'],'actualseal_code':bind(__file__),
        'all8_finalROI_state_checks':True,'raw96_integer_inclusive_0_95':True,'q_containment_or_minimum_area_not_required':True,
        'identity_view_precondition':bind(H/'SEAL_IDENTITY_AND_VIEW_PRECONDITION_PASS.json'),'YES':8,'UNKNOWN':0,
        'semantic_interpretation':'Eight manually identified visible local features, not new D/E targets, corrected q, wholepart segmentation or gold.'})
    put(H/'INDEPENDENCE_AT_SEAL.json',{'completed':True,'observer':'C','new_A_ROI_value_or_file_reads':0,'prior_V205_sealed_selection_context_acknowledged':True,
        'GT_mesh_model_oldscore_protected_eval_reads':0,'original_source_array_or_RGB_reopen':0,'new_sampling_or_qL_native_change':False,
        'fit_or_admission_or_crossobserver_adjustment':False,'actual_pixel_inputs':'Only the 25 exact generated PNG whitelist entries were toolviewed; no other pixels.'})
    with (H/'KO.md').open('x')as f:f.write('''# C 독립 visible-part witness\n\n고정 8사례의 CLEAN96/VALID96 16개와 같은 세 프레임의 원 fullRGB/q8 overlay/contact 9개, 총 25개의 허용 PNG를 이번 V206에서 직접 toolview했다. 이전 V205의 관찰이나 A의 새 ROI를 대체 근거로 삼지 않았다. A 새 ROI 값은 읽지 않았다.\n\n8개 모두 원 crop 안의 노출된 상판 끝·블록 모서리를 식별해 약한 정수 inclusive ROI를 기록했다. 잘린 앞 오른쪽 사례는 padding 아래로 부위를 복원하지 않고 유효 상부의 노출 조각만 잡았다. 박스는 visible feature 주변 직사각형으로 일부 배경을 포함할 수 있으며, 부품 전체 분할·3D 점·corrected q가 아니다. q 중심 포함, 최소 면적, 완벽한 mesh 또는 위치 정답을 요구하지 않았다.\n\n범용 ROI validator와 별도로 원 8행 순서/ID/qL/native/가용성/prior flags/fitfalse 및 각 행의 same-case CLEAN96와 same-frame full context 실제 view 바인딩을 검증했다. 원본 q/L/native와 V205 strict 결과는 바꾸지 않았다. 물리 native gold·합의·학습 허가·새 E 라벨은 생성하지 않았다. 이후 root/B가 양쪽 봉인 뒤 의미/overlap을 검토한다.\n''')
    put(H/'COMPLETE.json',{'completed':True,'PASS':True,'observer':'C','root_entry':bind(ENTRY),'protocol':ready['protocol'],'code':bind(__file__),
        'observations':bind(H/'INDEPENDENT8_WITNESS_OBSERVATIONS.json'),'CSV':bind(H/'INDEPENDENT8_WITNESS_OBSERVATIONS.csv'),
        'actual_views':bind(H/'ACTUALLY_VIEWED_BINDINGS.json'),'identity_view_precondition':bind(H/'SEAL_IDENTITY_AND_VIEW_PRECONDITION_PASS.json'),
        'validator':bind(H/'VALIDATOR_PASS.json'),'independence':bind(H/'INDEPENDENCE_AT_SEAL.json'),'KO':bind(H/'KO.md'),
        'rows':8,'YES':8,'UNKNOWN':0,'new_actual_generated_PNG_views':25,'new_A_ROI_reads':0,'training_admission':False,'own_observations_frozen':True})
    print(json.dumps(bind(H/'COMPLETE.json')))

if __name__=='__main__':main()

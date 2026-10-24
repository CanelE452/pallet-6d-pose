"""Seal C's manual observations after 18 actual generated-PNG tool views. No image decoder."""
from pathlib import Path
import collections, csv, hashlib, json, os, stat

H = Path(__file__).absolute().parent
V = H.parent
ENTRY = V/'ROOT_ACTUAL_INDEPENDENT_A_C_OBSERVATION_ENTRY.json'

def rb(p):
    p = Path(p)
    assert p.is_absolute() and V in p.parents and '..' not in p.parts
    assert all('v112' not in x.casefold() for x in p.parts)
    assert p.suffix.lower() in {'.json', '.py', '.png', '.csv', '.md'}
    fd = os.open(p.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for x in p.parts[1:-1]:
            n = os.open(x, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = n
        n = os.open(p.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    finally: os.close(fd)
    with os.fdopen(n, 'rb') as f:
        assert stat.S_ISREG(os.fstat(f.fileno()).st_mode)
        return f.read()

def binding(p): return {'path': str(p), 'sha256': hashlib.sha256(rb(p)).hexdigest()}
def checked(b):
    z = rb(b['path']); assert hashlib.sha256(z).hexdigest() == b['sha256']
    return json.loads(z)
def put(p, x):
    with p.open('x') as f:
        json.dump(x, f, ensure_ascii=False, indent=2); f.write('\n')

# (visibility, intended-part identity, query relation, possible location error, visual evidence, external evidence)
# Target instance is separately specified per frame. No value from observer A is read.
MANUAL = {
 (0,0): ('D','YES','NEAR','YES','상단 팔레트의 앞 왼쪽 상판 끝과 아래 목재 지지가 crop 안에 드러난다. q는 끝의 바로 왼쪽 배경에 있다.',''),
 (0,1): ('D','YES','NEAR','YES','상단 팔레트의 앞 오른쪽 상판/측면 끝이 crop 상부에 식별된다. q는 원영상 하단 경계 부근이며 패딩을 별도 취급했다.',''),
 (0,3): ('D','YES','NEAR','YES','상단 팔레트 앞 왼쪽의 세로 목재 지지 끝이 crop에 보인다. q는 그 왼쪽 바닥 쪽에 가깝다.',''),
 (0,4): ('D','YES','NEAR','YES','상단 팔레트 뒤 왼쪽의 노출된 상판 끝을 full q8 맥락과 crop에서 식별한다. q는 그 끝의 바로 위/왼쪽이다.',''),
 (0,5): ('UNKNOWN','UNKNOWN','OFF','YES','q와 crop 중심은 팔레트 오른쪽 바닥/추 부근이다. crop 가장자리 목재만으로 해당 native 끝을 확정하지 않았다.','해당 목표 끝을 가리는 외부 전경은 식별하지 못했다.'),
 (0,6): ('UNKNOWN','UNKNOWN','OFF','YES','q는 두 팔레트 오른쪽의 바닥 쪽이다. crop의 목재가 상단 목표의 해당 끝인지 아래 팔레트인지 명확하지 않다.',''),
 (0,7): ('UNKNOWN','UNKNOWN','UNKNOWN','YES','q는 상단 팔레트의 긴 상판 위에 있으나 요청된 아래/뒤 끝 자체는 이 crop에서 분리해 식별하기 어렵다. 주변 목재를 D나 S로 승격하지 않았다.',''),
 (2,1): ('D','YES','ON','NO','앞 오른쪽 노출 목재 블록과 끝면이 q 부근 및 crop에 뚜렷하다. 위 회색 통과 별개로 이 끝은 보인다.',''),
 (2,2): ('D','YES','OFF','YES','앞 오른쪽 아래 끝면은 crop 위쪽에서 보이지만 q 중심은 그 아래 바닥에 놓여 있다. 가시성과 q 위치를 분리했다.',''),
 (2,4): ('D','YES','OFF','YES','왼쪽 뒤의 노출된 팔레트 상판 끝이 crop 위/오른쪽에 보인다. q는 그보다 왼쪽/아래 배경에 있다.',''),
 (2,5): ('E','YES','UNKNOWN','UNKNOWN','full q8와 노출 상판의 이어짐으로 목표 팔레트의 뒤 오른쪽 끝 영역을 약하게 식별한다. 그 영역 앞에 누운 회색 통이 있고 crop은 통 표면/경계와 뒤 나무상자를 보인다. 정확한 q-숨은 끝의 위치 관계는 확정하지 않았다.','목표 상판이 이어질 뒤 오른쪽 영역을 누운 회색 통 및 별도 나무상자가 전경에서 덮는다. 배경/재료 부재만으로 E를 부여하지 않았다.'),
 (2,6): ('E','YES','UNKNOWN','UNKNOWN','뒤 오른쪽 아래 끝 영역은 노출 팔레트 면의 연속성과 q8 맥락으로 약하게 식별되며 crop에서는 누운 회색 통이 그 영역 앞을 차지한다. 가려진 끝의 정확한 좌표는 인증하지 않는다.','별도 회색 통의 불투명한 몸체가 목표 팔레트 뒤 오른쪽 끝 영역 앞에 놓인다. 자체 구조 가림으로 보지 않았다.'),
 (2,7): ('D','YES','NEAR','YES','팔레트 왼쪽 뒤의 아래쪽 가장자리/목재 끝이 crop 오른쪽에 드러난다. q는 그 왼쪽 경계 밖에 가깝다.',''),
 (5,0): ('D','YES','NEAR','YES','앞 왼쪽 상판 끝과 EPAL 목재 블록이 crop에 노출된다. q는 상판 모서리 바로 왼쪽이다.',''),
 (5,1): ('D','YES','NEAR','YES','앞 오른쪽 상판 끝과 블록 끝면이 crop에 드러난다. q는 목재의 바로 오른쪽 배경이다.',''),
 (5,2): ('D','YES','NEAR','YES','앞 오른쪽 아래 목재 모서리가 crop 왼쪽/중앙 부근에 보인다. q는 그 바깥 바닥 경계에 가깝다.',''),
 (5,4): ('D','YES','OFF','YES','뒤 왼쪽 노출 상판 끝이 crop 아래/오른쪽에 보인다. q 중심은 그 위/왼쪽 검은 배경에 있으므로 가시성과 위치 오류를 분리한다.',''),
 (5,5): ('UNKNOWN','UNKNOWN','OFF','YES','q/crop 중심은 회색 상자의 옆면이다. 목표 팔레트는 아래에 있지만 해당 native 끝의 국소 대응은 crop에서 충분히 식별되지 않는다.','회색 상자는 전경에 있으나 해당 끝의 대응이 불충분하여 E로 강제하지 않았다.'),
 (5,6): ('UNKNOWN','UNKNOWN','OFF','YES','q는 회색 상자 하단과 흰 포장재 부근이다. crop 가장자리 목재를 해당 native 아래/뒤 끝이라고 확정하기 어렵다.','상자/흰 포장재가 있으나 목표 부위와의 가림 대응이 불명확하다.'),
 (5,7): ('D','YES','NEAR','YES','뒤 왼쪽의 노출된 상판 가장자리와 끝이 crop에서 확인된다. q는 가장자리 바로 왼쪽이며 흰 포장재와 구별된다.',''),
}
FRAME_DEFAULTS = {
 0: ('YES', '상단 팔레트를 q8 맥락의 목표로 식별하지만 해당 native 부위의 근거가 충분하지 않다.'),
 1: ('UNKNOWN', 'q8이 회색 플라스틱 상자의 오른쪽 좁은 영역/배경에 몰려 있다. 이 맥락에서 목표 팔레트와 요청된 native 부위를 식별하지 못하며 다른 물체/배경을 E로 바꾸지 않는다.'),
 2: ('YES', '바닥의 단일 팔레트는 식별되지만 해당 부위는 원본 U 또는 근거 부족으로 판단하지 않는다.'),
 3: ('YES', '하단 팔레트는 식별되지만 available q는 파란 상자 또는 뒤 쓰레기통 부근이다. 요청된 팔레트 끝이 이 국소 crop에서 식별되지 않아 외부물체 존재만으로 E를 주지 않는다.'),
 4: ('YES', '뒤에 세운 목재 팔레트는 보이지만 q8이 작은 영역에 압축되어 있다. 파란 통 또는 넓은 목재 표면만으로 각 native 끝의 대응을 구분할 수 없다.'),
 5: ('YES', '단일 팔레트와 위의 적재물은 보이나 해당 native 끝의 국소 대응은 별도 근거가 필요하다.'),
}

def validate(row):
    enums = {
      'part_visibility_D_E_S_UNKNOWN': {'D','E','S','UNKNOWN'},
      'query_relation_on_near_off_UNKNOWN': {'ON','NEAR','OFF','UNKNOWN'},
      'target_instance_identified': {'YES','NO','UNKNOWN'},
      'intended_part_identified': {'YES','NO','UNKNOWN'},
      'location_error_possible': {'YES','NO','UNKNOWN'},
    }
    assert all(row[k] in values for k,values in enums.items())
    if not row['original_query_available']:
        assert not row['reviewed'] and all(row[k]=='UNKNOWN' for k in enums)
    else:
        assert row['reviewed'] and row['short_visual_evidence'].strip()
    if row['part_visibility_D_E_S_UNKNOWN'] in {'D','E'}:
        assert row['target_instance_identified']==row['intended_part_identified']=='YES'
    if row['part_visibility_D_E_S_UNKNOWN']=='E': assert row['external_occluder_evidence'].strip()
    assert row['training_admission'] is False and row['physical_native_gold'] is False

def main():
    assert binding(ENTRY)['sha256']=='6dc4a91933c95bb972f7850fa562128c261a3bc6138a6ffd9e759dabb0b3d65f'
    e=json.loads(rb(ENTRY)); assert e['approved_independent_A_and_C_SOURCE_TRAIN_weak_observations']
    preview=checked(e['preview_complete']); manifest=checked(e['preview_manifest']); qa=checked(e['identity_QA']); checked(e['C_contract'])
    assert preview['completed'] and preview['PASS'] and qa['completed'] and qa['PASS']
    assert qa['native_slots48'] and qa['original_inputU8']
    template=checked({'path':str(H/'EMPTY48_OBSERVATION_TEMPLATE.json'),'sha256':'7d9e77a8186d4c1ea4fda9567b87e5498ef81ee61141d9dd818e822f0bf4c787'})
    frames=manifest['frames']; assert len(frames)==6
    viewed=[]
    for fi,f in enumerate(frames):
        for kind in ['fulloriginal','overlay','all8contact']:
            b=f[kind]; assert binding(b['path'])==b
            viewed.append(dict(sequence=len(viewed)+1, frame_ordinal=fi, frame_key=f['frame_key'], kind=kind, binding=b,
                               actual_tool='functions.exec -> tools.view_image', detail='original', result='image displayed and independently inspected by C'))
    assert len(viewed)==18
    byid={s['row_id']:s for s in manifest['slots48']}
    rows=[]
    for old in template['rows']:
        r=dict(old); fi=r['frame_ordinal']; n=r['native_slot']; s=byid[r['row_id']]
        assert all(r[k]==s[k] for k in ['query_xy','L','native_slot','original_query_available'])
        r['viewed_generated_images']=[x['binding'] for x in viewed if x['frame_ordinal']==fi]
        if not r['original_query_available']:
            r['short_visual_evidence']='원본 input U를 그대로 UNKNOWN으로 보존했다. 주변 crop 일부가 보여도 semantic 판단/가용성 보충을 하지 않았다.'
        else:
            d=MANUAL.get((fi,n), ('UNKNOWN','UNKNOWN','UNKNOWN','YES',FRAME_DEFAULTS[fi][1],''))
            r.update(reviewed=True,target_instance_identified=FRAME_DEFAULTS[fi][0], intended_part_identified=d[1],
                     part_visibility_D_E_S_UNKNOWN=d[0],query_relation_on_near_off_UNKNOWN=d[2],location_error_possible=d[3],
                     short_visual_evidence=d[4],external_occluder_evidence=d[5])
        validate(r); rows.append(r)
    assert len(rows)==48 and sum(r['reviewed'] for r in rows)==40
    assert sum(not r['original_query_available'] for r in rows)==8
    put(H/'ACTUALLY_VIEWED_BINDINGS.json', {'completed':True,'observer':'C','root_entry':binding(ENTRY),'manifest':e['preview_manifest'],'views':viewed,
         'unique_generated_images':18,'full_frames6':True,'all8_contact_frames6':True,'other185_sourceframes_or_all114_generatedPNGs_viewed':False,
         'tool_view_notes':'Six ordered batches, each fulloriginal then q8 overlay then all8 contact. Images were displayed with detail=original; no browser used. Original source NPZ/RGB was not reopened or redecoded.'})
    obs={'completed':True,'observer':'C','authority':'INDEPENDENT_ASSISTANT_SOURCE_TRAIN_WEAK_OBSERVATION','root_entry':binding(ENTRY),'protocol':e['C_contract'],
         'preview_complete':e['preview_complete'],'identity_QA':e['identity_QA'],'viewed_bindings':binding(H/'ACTUALLY_VIEWED_BINDINGS.json'),'rows':rows,
         'A_observation_reads':0,'consensus_applied':False,'training_admission':False,'physical_native_gold':False}
    put(H/'INDEPENDENT48_OBSERVATIONS.json',obs)
    flat=[{k:json.dumps(v,ensure_ascii=False)if isinstance(v,(list,dict))else v for k,v in r.items()}for r in rows]
    with (H/'INDEPENDENT48_OBSERVATIONS.csv').open('x',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    counts=dict(collections.Counter(r['part_visibility_D_E_S_UNKNOWN']for r in rows))
    put(H/'VALIDATOR_PASS.json', {'completed':True,'PASS':True,'code':binding(__file__),'rows48':True,'available40_reviewed':True,'original_U8_UNKNOWN':True,
        'original_identity_qL_native_availability_exact':True,'D_E_identity_and_positive_E_foreground_guards':True,'no_UNREVIEWED_remaining_available':True,
        'D_E_S_UNKNOWN_counts':counts,'query_relation_counts':dict(collections.Counter(r['query_relation_on_near_off_UNKNOWN']for r in rows)),
        'consensus_or_label_admission':False,'observation_json':binding(H/'INDEPENDENT48_OBSERVATIONS.json'),'CSV':binding(H/'INDEPENDENT48_OBSERVATIONS.csv')})
    notes='''# Observer C: 독립 SOURCE TRAIN 약관찰\n\n원 fullRGB, 원 q8 overlay, all8 crop contact를 여섯 프레임 각각 직접 도구로 보았다(고유 PNG 18개). 다른 원자료 185프레임 전체나 생성 PNG 114개 전체를 본 것은 아니다. A 관찰값, GT, 기존 약라벨, 모델/점수는 읽지 않았다.\n\nD/E는 식별 가능한 목표 부위에 대한 약판단이다. D와 q OFF가 함께 기록될 수 있다. 이는 부위가 원 crop 안에서 보이지만 q 중심이 배경에 놓인 경우이며 위치 정답이나 학습 허가가 아니다. E 두 건은 팔레트 뒤 오른쪽 영역 앞의 회색 통을 근거로 하되, 가려진 끝에 대한 정확한 q 관계는 UNKNOWN으로 남겼다. 원 U 8개는 UNKNOWN이다. 자체 구조 가림 S를 확신할 사례는 없었으며, 애매한 후면/하단 부위를 S로 추정하지 않았다.\n\n작게 압축된 q8 또는 다른 물체/배경 위 q에서 목표 native 끝을 구분하지 못한 경우 UNKNOWN을 유지했다. 완벽한 mesh나 물리 native 인증을 요구한 것이 아니라, 이 원 crop와 q8 맥락에서 해당 부위를 식별할 수 있는지가 기준이다.\n\n관찰값은 root의 합의 규칙으로 조정하지 않았다. 합의·라벨 생성·fit admission·학습·평가·모델 선택은 수행하지 않았다. 이 여섯 프레임은 자연 source TRAIN 약관찰이며 평가 gold나 독립 실사 성능 증거가 아니다.\n'''
    with (H/'KO.md').open('x')as f:f.write(notes)
    put(H/'COMPLETE.json',{'completed':True,'PASS':True,'observer':'C','root_entry':binding(ENTRY),'protocol':e['C_contract'],'code':binding(__file__),
        'observations':binding(H/'INDEPENDENT48_OBSERVATIONS.json'),'CSV':binding(H/'INDEPENDENT48_OBSERVATIONS.csv'),'actual_views':binding(H/'ACTUALLY_VIEWED_BINDINGS.json'),
        'validator':binding(H/'VALIDATOR_PASS.json'),'KO':binding(H/'KO.md'),'rows':48,'available_reviewed':40,'original_U_UNKNOWN':8,
        'D_E_S_UNKNOWN_counts':counts,'actual_generated_PNG_views':18,'A_values_GT_oldweak_scores_models_read':0,'source_RGB_or_NPZ_reopened':0,
        'fit_evaluation_or_training_admission':False,'other_observer_consensus_applied':False,'own_observations_frozen':True})
    print(json.dumps(binding(H/'COMPLETE.json')))

if __name__=='__main__': main()

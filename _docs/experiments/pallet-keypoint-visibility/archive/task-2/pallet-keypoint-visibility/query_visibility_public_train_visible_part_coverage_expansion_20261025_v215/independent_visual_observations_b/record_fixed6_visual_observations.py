"""Record manual actual tool observations of the six prospectively fixed generated cards.

The six view_image calls happened in the B agent turn. This recorder opens only
the prospective view-scope JSON, never images, source arrays, GT or model scores.
"""
import hashlib
import json
import os
from pathlib import Path
import stat

H = Path(__file__).absolute().parent
P = H.parent.parent


def opened(path, write=False):
    s = str(path)
    assert 'v112' not in s.casefold()
    p = Path(s)
    assert p.is_absolute() and '..' not in p.parts and P in p.parents
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
        out = os.open(p.name, (os.O_WRONLY | os.O_CREAT | os.O_EXCL if write else os.O_RDONLY) | os.O_NOFOLLOW, 0o644, dir_fd=fd)
    finally:
        os.close(fd)
    assert stat.S_ISREG(os.fstat(out).st_mode)
    return os.fdopen(out, 'wb' if write else 'rb')


def rb(path):
    with opened(path) as f:
        return f.read()


def binding(path):
    return dict(path=str(path), sha256=hashlib.sha256(rb(path)).hexdigest())


# Manual descriptions are independent of metadata availability. INPUT U remains
# a separate immutable field even where a valid strip contains visible material.
OBSERVATIONS = {
    1: dict(
        fullimage='회색 열린 플라스틱 상자와 검은 받침/프레임, 바닥 케이블이 함께 보인다. 하단 q0..3은 검은 받침·바닥·케이블 주변이고 상단 q4..7은 회색 상자 테두리/측면 주변이다. 어떤 물체가 declared native의 intended target인지 이 카드만으로 확정하지 않았다.',
        issue='여러 물체의 표면이 같은 crop에 들어온다. 표면이 보임과 intended native 부위가 식별됨은 분리해야 한다.',
        slots=[
            ('MIXED_OBJECT_AND_BACKGROUND','오른쪽 회색 상자 하단 모서리와 왼쪽 어두운 받침/바닥이 함께 보인다. cyan 중심은 어두운 영역 근처다.'),
            ('MIXED_OBJECT_AND_BACKGROUND','상단/왼쪽 회색 상자 아래 모서리와 바닥·흰 케이블이 포함된다. 중심 부근은 어두운 바닥/받침이다.'),
            ('MOSTLY_BACKGROUND','바닥과 흰 케이블이 중심 부근을 지난다. 회색 상자 조각은 좌상단에만 일부 보이고 아래에는 padding이 있다.'),
            ('MIXED_OBJECT_AND_BACKGROUND','검은 받침의 선/프레임과 흰 케이블이 보이고 회색 상자의 일부가 우상단에 있다.'),
            ('LOCAL_OBJECT_SURFACE_OR_EDGE_VISIBLE','회색 상자의 위쪽 테두리와 안쪽/바깥쪽 면 경계가 뚜렷하게 보인다.'),
            ('MIXED_OBJECT_AND_BACKGROUND','왼쪽의 회색 상자 테두리 끝과 오른쪽의 밝은 벽이 함께 보인다. 중심은 그 경계 가까이에 있다.'),
            ('LOCAL_OBJECT_SURFACE_OR_EDGE_VISIBLE','회색 상자 측면의 리브/테두리와 밝은 벽의 경계가 보인다.'),
            ('LOCAL_OBJECT_SURFACE_OR_EDGE_VISIBLE','회색 상자 안쪽의 수직 모서리와 윗 테두리 부근이 보인다.'),
        ]),
    3: dict(
        fullimage='앞쪽에는 EPAL 표시가 있는 목재 pallet가 있고 그 위/뒤로 파란 상자, 나무 상자와 회색 원통형 물체가 놓여 있다. 표시된 q8은 전면 pallet의 끝이 아니라 뒤쪽 나무 상자 상단 부근에 모여 보인다. 이는 화면상의 대응 관찰이며 native 오류/라벨 오류 판정은 아니다.',
        issue='fullimage의 pallet 존재만으로 각 crop을 pallet native 끝의 증거로 취급할 수 없다. selected instance와 intended part를 먼저 독립적으로 기록해야 한다.',
        slots=[
            ('LOCAL_OBJECT_SURFACE_OR_EDGE_VISIBLE','나무 상자 상단의 가로 모서리와 수직 이음선, 위쪽 밝은 벽이 보인다.'),
            ('LOCAL_OBJECT_SURFACE_OR_EDGE_VISIBLE','나무 상자 오른쪽 모서리와 벽의 경계가 보인다.'),
            ('MIXED_OBJECTS','나무 상자 면/오른쪽 모서리와 아래쪽을 가로지르는 회색 원통 물체의 굽은 테두리가 함께 보인다. 이것만으로 가림 native를 판정하지 않는다.'),
            ('LOCAL_OBJECT_SURFACE_OR_EDGE_VISIBLE','나무 상자 면의 세로 이음선과 위쪽 가로 테두리가 보인다.'),
            ('LOCAL_OBJECT_SURFACE_OR_EDGE_VISIBLE','나무 상자 상단 테두리와 검은 손잡이 홈, 위쪽 벽이 보인다.'),
            ('LOCAL_OBJECT_SURFACE_OR_EDGE_VISIBLE','나무 상자 오른쪽 윗 모서리와 벽이 보인다.'),
            ('MIXED_OBJECTS','나무 상자 오른쪽 가장자리와 아래 회색 굽은 테두리 조각이 함께 보인다.'),
            ('LOCAL_OBJECT_SURFACE_OR_EDGE_VISIBLE','상단 나무판과 검은 반원형 홈이 보인다.'),
        ]),
    5: dict(
        fullimage='큰 목재 pallet 위에 파란 상자·세운 나무 상자·회색/금속 통 등이 놓여 있다. q4/7은 왼쪽 목재판 바깥쪽 부근, q5/6은 파란 상자와 나무 상자의 접경 부근이며 q1/2는 오른쪽 바닥/케이블 주변이다. q0/3의 원 INPUT U도 유지된다.',
        issue='판 끝이 국소적으로 보이는 crop, 다른 물체 표면이 섞이는 crop, 바닥과 원 입력 U를 구별해서 기록할 수 있다. 외부 물체가 보인다는 사실만으로 intended 끝의 외부가림 라벨을 부여하지 않는다.',
        slots=[
            ('INPUT_U_RETAINED','원 INPUT U. 위쪽 valid strip에 목재판 일부가 있고 아래 대부분은 검은 padding이다. 이를 available로 승격하지 않는다.'),
            ('MIXED_OBJECT_AND_BACKGROUND','검은 프레임/물체 가장자리, 바닥, 흰색·색상 케이블이 보인다. intended pallet 끝은 이 crop만으로 식별하지 못했다.'),
            ('MOSTLY_BACKGROUND','바닥과 여러 색 케이블이 중심 부근을 지난다. 왼쪽 위의 물체 조각 외에는 배경 위주다.'),
            ('INPUT_U_RETAINED','원 INPUT U. 표시된 RGB/valid가 검은 padding이며 새 내용이나 부위 위치를 만들지 않았다.'),
            ('LOCAL_OBJECT_SURFACE_OR_EDGE_VISIBLE','오른쪽 아래 목재판의 바깥 끝/가장자리와 왼쪽/위 바닥이 보인다. 중심은 판 경계 가까운 바닥 쪽이다.'),
            ('MIXED_OBJECTS','파란 플라스틱 상자 면과 나무 상자의 판/모서리가 함께 보인다.'),
            ('MIXED_OBJECTS','왼쪽 파란 상자 면과 오른쪽 나무 상자 면/세로 경계가 보인다.'),
            ('LOCAL_OBJECT_SURFACE_OR_EDGE_VISIBLE','목재 긴 판의 바깥 가장자리와 바닥이 함께 보인다. 국소 판 표면은 보이지만 canonical/native 끝을 확정하지 않는다.'),
        ]),
    7: dict(
        fullimage='배경에 세운 목재 pallet가 있고 앞쪽에는 열린 나무 상자와 뒤쪽 나무 판/상자, 회색 통이 겹쳐 있다. q0/3은 뒤쪽 나무판 아래/바닥 부근, q1/2는 앞 열린 상자 가로판 부근, q4..7은 뒤쪽/위쪽 목재 구조에 놓여 보인다.',
        issue='유사한 나무 재질의 여러 물체와 틈이 겹친다. 보이는 판을 declared target/native로 자동 대응시키지 않아야 한다.',
        slots=[
            ('MOSTLY_BACKGROUND','왼쪽 위에 나무 상자/판의 하단 조각이 있고 중심 및 대부분은 카펫 바닥이다.'),
            ('LOCAL_OBJECT_SURFACE_OR_EDGE_VISIBLE','밝게 보이는 가로 나무판/상자 가장자리와 위쪽 내부 판이 보인다.'),
            ('LOCAL_OBJECT_SURFACE_OR_EDGE_VISIBLE','여러 가로 나무판과 그 사이 밝고 어두운 틈이 보인다. 단일 물리 native 끝으로 식별하지 않았다.'),
            ('MOSTLY_BACKGROUND','카펫 바닥이 crop을 채우며 식별 가능한 부위 끝은 보이지 않는다.'),
            ('LOCAL_OBJECT_SURFACE_OR_EDGE_VISIBLE','오른쪽의 상단 나무판 끝/세로 면과 왼쪽 카펫 경계가 보인다.'),
            ('LOCAL_OBJECT_SURFACE_OR_EDGE_VISIBLE','경사진 나무 가로판과 고정점, 위쪽 다른 판이 보인다.'),
            ('LOCAL_OBJECT_SURFACE_OR_EDGE_VISIBLE','두 나무 가로판과 그 사이의 어두운 틈이 보인다. 어두운 틈을 foreign occluder나 SELF로 분류하지 않았다.'),
            ('LOCAL_OBJECT_SURFACE_OR_EDGE_VISIBLE','나무 상자/판의 세로 바깥 가장자리와 카펫이 함께 보인다.'),
        ]),
    9: dict(
        fullimage='목재 바닥판 위의 회색 열린 상자와 금속 통, 뒤쪽의 작은 검은 손잡이/프레임이 보인다. q8은 회색 상자/목재판이 아니라 뒤쪽 검은 손잡이 주변에 작은 사각 형태로 모여 보인다.',
        issue='작은 L에서 crop은 흐릿한 검은 손잡이/바닥의 국소 확대처럼 보인다. available=True는 intended visiblepart가 식별됐다는 뜻이 아니다. q8이 어떤 target을 뜻하는지 별도 대응 관찰이 필요하다.',
        slots=[
            ('UNCERTAIN_LOCAL_STRUCTURE','검은 세로/굽은 구조와 카펫 경계가 보이나 작은 crop의 구조 세부 및 intended 부위는 불확실하다.'),
            ('UNCERTAIN_LOCAL_STRUCTURE','어두운 굽은 물체 가장자리와 바닥이 보인다. 대상 부위는 확정하지 못했다.'),
            ('MOSTLY_BACKGROUND','카펫 위주이며 왼쪽에 어두운 구조 가장자리 일부가 보인다.'),
            ('UNCERTAIN_LOCAL_STRUCTURE','어두운 세로/곡선 구조와 바닥의 경계가 보인다. 끝/가림 원인 판정은 하지 않았다.'),
            ('MOSTLY_BACKGROUND','카펫 질감 위주이며 식별 가능한 의도부위 끝은 확인하지 못했다.'),
            ('UNCERTAIN_LOCAL_STRUCTURE','검은 손잡이처럼 보이는 굽은 가장자리와 밝은 바닥이 보이나 intended target 대응은 불명확하다.'),
            ('UNCERTAIN_LOCAL_STRUCTURE','어두운 고리/손잡이의 굽은 모서리와 바닥이 함께 보인다.'),
            ('UNCERTAIN_LOCAL_STRUCTURE','어두운 세로 가장자리와 바닥이 보인다. native 물리 모서리는 확인하지 않았다.'),
        ]),
    11: dict(
        fullimage='긴 목재판/흰색 덮개 위에 회색 상자와 금속 통이 있다. q0은 영상 하단의 흰 면 부근, q1..3은 원 INPUT U인 하단 바깥쪽, q4/7은 회색 상자, q5/6은 오른쪽 목재판과 바닥의 경계 부근이다.',
        issue='같은 native8 안에서도 유효 부분 crop과 padding, 서로 다른 재질의 물체 경계가 섞인다. 원 U와 visible material은 서로 다른 축으로 유지해야 한다.',
        slots=[
            ('MIXED_SURFACE_AND_PADDING','위쪽 valid 영역에 흰 덮개/판 면이 보이고 아래에는 padding이 있다. 원 available=True는 유지하지만 끝의 의미는 확인하지 않았다.'),
            ('INPUT_U_RETAINED','원 INPUT U. 위쪽 목재판/바닥 조각과 아래쪽 검은 padding이 함께 보인다. 가용성이나 라벨을 바꾸지 않는다.'),
            ('INPUT_U_RETAINED','원 INPUT U. 맨 위의 작은 목재/바닥 조각 이외 대부분이 padding이다.'),
            ('INPUT_U_RETAINED','원 INPUT U. 위쪽 흰 면 조각과 아래쪽 검은 padding이 보인다.'),
            ('LOCAL_OBJECT_SURFACE_OR_EDGE_VISIBLE','회색 상자의 윗 테두리/리브와 흰 내부 조각이 보인다.'),
            ('LOCAL_OBJECT_SURFACE_OR_EDGE_VISIBLE','목재판의 바깥 가장자리와 오른쪽 카펫이 보인다.'),
            ('MIXED_OBJECT_AND_BACKGROUND','왼쪽 목재판 긴 가장자리와 오른쪽 카펫이 보이며 중심은 그 접경 근처다.'),
            ('LOCAL_OBJECT_SURFACE_OR_EDGE_VISIBLE','회색 상자 전면 리브와 윗 테두리/측면 모서리가 보인다.'),
        ]),
}


scope_path = H / 'FIXED6_GENERATED_CONTACT_VIEW_SCOPE_BEFORE_VIEWS.json'
scope = json.loads(rb(scope_path))
assert scope['fixed_positions'] == [1,3,5,7,9,11] and scope['actual_toolviews_so_far'] == 0
frames = []
for order, f in enumerate(scope['only_generated_contacts'], 1):
    pos = f['position']
    manual = OBSERVATIONS[pos]
    assert len(f['native8_original_metadata']) == len(manual['slots']) == 8
    slots = []
    for original, (appearance, description) in zip(f['native8_original_metadata'], manual['slots']):
        slots.append(dict(**original, local_appearance=appearance, descriptive_visual_evidence_KO=description,
                          intended_native_part_correspondence='UNRESOLVED_NOT_CERTIFIED',
                          native_DE_S_label_assigned=False, physical_point_gold=False,
                          external_or_self_occlusion_label_assigned=False, fit_admission=False))
    assert [r['native_slot'] for r in slots] == list(range(8))
    frames.append(dict(position=pos, frame_key=f['frame_key'], original_frame_index=f['original_frame_index'],
                       component_id=f['component_id'], contact=f['contact'], actual_view_tool='tools.view_image',
                       actual_tool_call_order=order, actual_tool_view_completed=True, contact_declared_width_px=2304,
                       tool_display_note='Tool-rendered image was shown at2048x1189 after2304x1338 source resizing; header/native8 qL/U/clean/valid/footer remained legible.',
                       readable_header_and_all8_native_metadata=True, contact_layout_overlap_observed=False,
                       fullimage_crop_relation_KO=manual['fullimage'], next_correspondence_issue_KO=manual['issue'],
                       native8_local_appearance_observations=slots))
allslots = [r for f in frames for r in f['native8_local_appearance_observations']]
assert len(frames) == 6 and len(allslots) == 48 and sum(not r['original_query_available'] for r in allslots) == 5
result = dict(
    completed=True, PASS=True, phase='V215', review_kind='ACTUAL_FIXED6_SOURCE_FRAME_LOCAL_APPEARANCE_ONLY',
    code=binding(__file__), fixed_pre_view_scope=binding(scope_path), preview_complete=scope['preview_complete'], manifest=scope['manifest'],
    fixed_positions=[1,3,5,7,9,11], actual_generated_contact_toolviews=6, viewed_native_display_slots=48,
    original_inputU_within_viewed48=5, preview_metadata_total96_U11_known=True, all96_toolview_claimed=False,
    root_complementary_view_positions=[0,2,4,6,8,10], root_views_independently_not_claimed_here=True,
    frames=frames,
    common_SOURCE_correspondence_issues_KO=[
        'fullimage에 여러 물체가 있으며 같은 crop 안에 재질/표면/배경/케이블이 함께 들어온다. generic material presence를 intended native 끝 visibility로 승격하지 않는다.',
        '목재 pallet가 fullimage에 존재해도 q8 crop이 그 pallet의 native 부위인지 이 카드만으로 결정되지 않는 사례가 보인다. native 오류 또는 기존 D/E 라벨 오류를 판정한 것은 아니다.',
        '다음 약관찰에서는 selected-instance 식별과 의도부위 식별, 관찰가능한 국소 feature, q와의 ON/NEAR/OFF 관계, 외부/자기 가림 증거를 분리하고 UNKNOWN/불일치를 보존한다.',
        '원 unavailable/padding이 있거나 crop이 배경/작은 구조를 포함해도 q/L/ROI를 이동하거나 대체 frame을 고르지 않는다. available 또는 U0를 의미적 확신으로 해석하지 않는다.',
        '이6 고정 카드의 기술적 관찰은 prevalence·대표성·일반화·학습가능성/성능 개선 또는 private 실패 원인 추론이 아니다.',
    ],
    actual_original_RGB_cache_NPZ_GT_weights_scores_or_protected_opens=0,
    generated_contact_SHA_verified_before_views=True, original_qL_native_order_inputU_unchanged=True,
    new_native_DE_S_or_corrected_keypoint_or_ROI_labels=False, training_admission=False,
    model_selection_or_training=False, fully_solved=False,
)
path = H / 'ACTUAL_FIXED6_SOURCE_FRAME_OBSERVATIONS.json'
with opened(path, True) as f:
    f.write((json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())
print(json.dumps(dict(completed=True,PASS=True,review=binding(path),actual_views=6,display_native_slots=48,original_inputU=5),ensure_ascii=False))

"""Source-only factual query bridge report; no learned computation."""
from pathlib import Path
from collections import Counter
import hashlib
import html
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

V = Path(__file__).resolve().parent
FONT = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def font(size):
    return ImageFont.truetype(FONT, size)


def main():
    facts = read(V / 'root_saved_query_role_diagnosis/RESULTS.json')
    actual = read(V / 'actual_mr6d_initializer_v1/RESULT.json')
    evidence = read(V / 'original_query_evidence/EVIDENCE_RENDER_COMPLETE.json')
    source = read(V / 'native_reference_audit/SOURCE104_PROPOSAL_SUMMARY_KO.json')
    assert actual['frames'] == 10 and actual['native_slots'] == 80
    assert facts['full_query_inventory'] == 80 and facts['full_GT_native_inventory'] == 104
    for name in ['independent_bridge_review/REVIEW.json', 'independent_reference_review/REVIEW.json',
                 'independent_saved_initializer_review/REVIEW.json']:
        assert read(V / name)['completed'] and read(V / name)['PASS']
    cases = [
        (0, 1, '다른 물체를 선택', '팔레트 앞의 목재 상자다. 나무가 보여도 대상 팔레트의 점이 아니다.'),
        (4, 4, '팔레트가 있어도 q는 배경', '두 팔레트의 점이 섞였다. 이 배경 crop을 외부 가림으로 바꾸지 않는다.'),
        (8, 0, '대상 팔레트의 앞 모서리', '보이는 목재 모서리와 원 native0 역할이 구조적으로 대응한다.'),
        (8, 5, '같은 팔레트의 외부 가림 후보', '뒤 위 모서리 역할을 회색 적재물이 덮는다. source 약한 주석 후보다.'),
    ]
    canvas = Image.new('RGB', (1480, 1100), '#f0f4f8')
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 15), '원 추론 crop에서 구분해야 할 서로 다른 원인', font=font(32), fill='#15283b')
    draw.text((28, 58), '원 q/L의 실제 96×96 RGB · 최근접 확대 · 가림 모델의 예측 결과가 아닌 입력 진단', font=font(20), fill='#334c60')
    for num, (frame, slot, title, reason) in enumerate(cases):
        x, y = (num % 2) * 740 + 20, (num // 2) * 495 + 100
        packet = V / f'original_query_evidence/FRAME_{frame:02d}_FACTUAL96.npz'
        with np.load(packet, allow_pickle=False) as data:
            crop = data['rgb96'][slot].copy()
        draw.rounded_rectangle((x, y, x + 710, y + 475), radius=14, fill='white')
        draw.text((x + 14, y + 10), title, font=font(28), fill='#15283b')
        draw.text((x + 14, y + 48), f'고정 source 프레임 {frame:02d} / native{slot}', font=font(19), fill='#526e82')
        canvas.paste(Image.fromarray(crop).resize((384, 384), Image.Resampling.NEAREST), (x + 16, y + 83))
        # Wrap the Korean explanation by measured text length, not arbitrary clipping.
        words, rows, line = reason.split(), [], ''
        for word in words:
            trial = (line + ' ' + word).strip()
            if draw.textlength(trial, font=font(23)) > 270 and line:
                rows.append(line)
                line = word
            else:
                line = trial
        rows.append(line)
        draw.multiline_text((x + 420, y + 102), '\n'.join(rows), font=font(23), fill='#253f53', spacing=11)
    draw.text((28, 1070), 'Gouda et al., MR6D · CC-BY-4.0 · 원 JPEG·q/L 보존, 실제96 crop 확대 · assistant weak 검토, 사람 gold 아님', font=font(17), fill='#526e82')
    fig = V / 'QUERY_FAILURE_CAUSES_KO.png'
    canvas.save(fig)
    result = dict(completed=True, phase='V152_ACTUAL_SOURCE_QUERY_AND_NATIVE_ROLE_BRIDGE_COMPLETE',
        full_goal_solved=False, selected_model=None, source_training_ready=False,
        conclusion_KO='새 공개 실사에서 원 R0를 실제 실행했다. 6장은 다른 물체/대상 부재 오검출, 2장은 여러 팔레트의 점이 섞여 개체 대응 불명확, 2장은 팔레트 개체가 지지되지만 native별 위치 품질과 역할 가림을 별도 검토해야 한다. 원80 query와104 GT 참조를 모두 보존했으며, 오검출을 외부 가림 E로 학습하면 안 된다는 직접 근거를 얻었다.',
        actual_R0_source_frames=10, explicit_predict_calls=actual['explicit_predict_calls'], total_forward_hooks=actual['total_forward_hooks'],
        actual_training_updates=0, new_DINO_SAM_or_visibility_head_calls=0,
        private_current223_reads_or_PnP_or_query_correction=0,
        all_candidate_count=actual['candidate_count'], firstargmax_unchanged=True,
        original_query_rows=80, finite_queries=80, continuous_valid=actual['continuous_valid_slots'], nearest_valid=actual['nearest_valid_slots'],
        factual_GT_role_rows=104, GT_instance_observations=13,
        entity_inventory=dict(wrong_entity_or_no_visible_target_frames=6, stack_composite_unresolved_frames=2,
            supported_pallet_entity_provisional_native_mapping_frames=2),
        all104_source_assistant_role_proposals=source['counts'],
        role_proposals_NOT_original80_query_labels=True,
        per_output_partial_weak_supervision=dict(target_entity_wrong=48, target_entity_supported=16,
            target_entity_unknown=16, identity_label_nearest_input_available=63,
            supported_role_D=8, supported_role_E=2, visibility_unlabeled=70,
            structural_role_correspondence_supported_including_OOF=11,
            exact_point_coordinate_quality_labels=0),
        confirmed_human_gold_or_exact3px_source_labels=0,
        nominal_source_projection_not_exact_pixel_gold=True,
        JPEG_pinned_cv2_vs_PIL_all10_byte_exact=True,
        latest_private_comparison_remains='V149; no new private evaluation or result-based selection',
        limits_KO=['이 고정10장은 source 준비를 위한 작은 표본이며 초기 모델 정확도나 실제 배포 빈도를 추정하는 표본이 아니다.',
            '104개 source 역할 제안과 원80개 query 라벨은 다르다. 다른 GT 팔레트의 점을 원 q 대신 넣지 않았다.',
            'source D/E/S는 assistant weak 제안이다. 제공 pose/CAD는 센티미터 불확실성이 있어 정확한3px gold가 아니다.',
            '새 공개 자료는 목재 Euro 직사각형이며 정사각형 target의 실사 감독 공백은 남아 있다.',
            '시각 독립 검토는 root의 초안 관측을 전달받은 뒤 실행했으므로 blind adjudication은 아니다.'],
        next_action_KO='개체 오검출/혼합 및 위치 품질과 native 역할 visibility를 별도 출력·별도 loss로 정의하고, 공개 장면 분리 검증에서 비교할 다음 source 실험을 고정한다. 작은16개 대응만으로 전체 문제 해결을 선언하지 않는다.',
        evidence={
            'prediction': bind(V / 'actual_mr6d_initializer_v1/PREDICTION_SEAL.json'),
            'preforward_independent': bind(V / 'independent_bridge_review/REVIEW.json'),
            'saved_independent': bind(V / 'independent_saved_initializer_review/REVIEW.json'),
            'reference_independent': bind(V / 'independent_reference_review/REVIEW.json'),
            'identity_independent': bind(V / 'independent_saved_initializer_review/SELECTED_ENTITY_VISUAL_REVIEW_COMPLETE.json'),
            'source_proposals': bind(V / 'native_reference_audit/SOURCE104_PROPOSAL_SUMMARY_KO.json'),
            'all80_join': bind(V / 'root_saved_query_role_diagnosis/RESULTS.json'),
            'next_label_contract': bind(V / 'ROOT_OPERATIONAL_LABEL_CONTRACT_NEXT_EXPERIMENT.json'),
            'root_partial_supervision': bind(V / 'ROOT_PARTIAL_SUPERVISION_ADJUDICATION.json'),
            'source_focused_correspondence': bind(V / 'native_reference_audit/selected_query_role_review/REVIEW_COMPLETE.json'),
            'crop_boards': bind(V / 'original_query_evidence/EVIDENCE_RENDER_COMPLETE.json'),
            'figure': bind(fig)})
    write(V / 'ROOT_RESULTS_KO.json', result)
    display = {'WRONG_ENTITY_WOOD_CRATE':'목재 상자 오선택', 'NO_PALLET_IN_VIEW_FALSE_POSITIVE':'팔레트 화면 밖 / 오검출',
        'STACK_COMPOSITE_ENTITY_UNRESOLVED':'여러 팔레트 혼합 의심', 'WRONG_ENTITY_WOOD_CARGO':'목재 화물 오선택',
        'PALLET_ENTITY_SUPPORTED_NATIVE_MAPPING_PROVISIONAL':'팔레트 개체 지지 / 역할별 검토',
        'WRONG_ENTITY_SMALL_FLOOR_OBJECT':'바닥 작은 물체 오선택'}
    table = ''.join(f"<tr><td>{r['frame_index']:02d}</td><td>{html.escape(r['source_id'])}</td><td>{display[r['entity_status']]}</td><td>{html.escape(r['reason_KO'])}</td></tr>" for r in facts['frames'])
    gallery = ''
    for item in evidence['sheets']:
        relative = Path(item['sheet']['path']).relative_to(V)
        gallery += f'<details><summary>프레임 {item["frame_index"]:02d} · {html.escape(item["source_id"])}</summary><a href="{relative}"><img loading="lazy" src="{relative}"></a></details>'
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V152 원 추론과 보임·가림 정답 연결 진단</title><style>body{font:17px/1.65 sans-serif;margin:auto;max-width:1400px;padding:24px;color:#17324a;background:#f4f7fa}section{background:white;padding:24px;margin:22px 0;border-radius:12px}h1{font-size:30px}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{text-align:left;vertical-align:top;border-bottom:1px solid #ccd7df;padding:10px}summary{cursor:pointer;padding:12px}a{color:#1261a0}.note{border-left:5px solid #d18a24;padding:16px;background:#fff7e8}</style><h1>V152 · 원 추론 query와 source 역할 정답을 연결한 결과</h1>'''
    page += f'<section><p><strong>{html.escape(result["conclusion_KO"])}</strong></p><p>실제 원 R0: 10프레임 · 모든16후보 · 원80점 · 연속좌표 유효78 / 최근접 픽셀 유효77. 별도 제공 GT native 참조104개와 centroid13개. 학습·PnP·현재 평가셋 접근0.</p><p class="note">현재 평가 성능 개선을 입증한 단계는 아니다. 이전 모델 비교는 V149에 그대로 남아 있다. 이10장의 비율을 배포 정확도로 해석하지 않는다.</p></section>'
    page += '<section><h2>같은 “안 보임”으로 묶으면 안 되는 실제 crop</h2><img src="QUERY_FAILURE_CAUSES_KO.png"><p>잘못 선택한 목재 상자, 다른 팔레트와 섞인 점, 보이는 대상 모서리, 외부 적재물 가림 후보를 나눠 본다. 원96 crop은 색·픽셀 변경 없이 최근접 확대했다.</p></section>'
    page += '<section><h2>10장 전체 선택 결과</h2><table><thead><tr><th>순서</th><th>source</th><th>개체 대응</th><th>근거</th></tr></thead><tbody>'+table+'</tbody></table></section>'
    page += '<section><h2>정답 정의를 분리한 이유</h2><p>오검출/개체 혼합/위치 오류는 query 품질이고, 실제 그 역할이 보이거나 외부 물체에 가리는 상태는 역할 visibility다. 전경 목재 상자 crop을 E로 바꾸면 “다른 물체”와 “외부 가림”을 혼동해 배운다. 명확한 부분 역할은 다른 역할이 불확실해도 별도 약한 감독 후보로 남긴다.</p><p>제공104 역할에 대한 source assistant 제안: D39 / E10 / S2 / 화면밖35 / U18. 이는 원80 query에 곧바로 붙일 정답이 아니다. 원80행을 모두 보존한 별도 약한 주석에는 대상 개체 불일치48 / 개체 지지16 / 불명확16을 기록했다. 그중 구조 대응이 지지되는 역할 D8 / E2만 visibility 감독 후보로 남겼다. 개체 label64개 중 최근접 입력 유효는63개이며, 정확한 점 좌표 품질 label은0개다.</p><p>사람 gold·정확한3px 좌표·학습 준비 완료를 주장하지 않는다. 외부 가림 / 자기 가림 / 화면 밖 / 위치 오류를 분리하고 정사각형 목표 범위도 계속 유지한다.</p></section>'
    page += '<section><h2>모든 원 q/L crop 보기</h2>'+gallery+'</section>'
    links = {'원104개 역할·제공좌표 그림':'native_reference_audit/INDEX.html','원80개 query·개체·역할·local 증거':'root_saved_query_role_diagnosis/ALL80_QUERY_IDENTITY_ROLE_AND_LOCAL_EVIDENCE.jsonl','부분 역할 대응 집중 검토':'native_reference_audit/selected_query_role_review/REVIEW_COMPLETE.json','출력별 부분 약한 주석80행':'ROOT_PARTIAL_WEAK_SUPERVISION_ALL80.jsonl','다음 실험의 정답 정의':'ROOT_OPERATIONAL_LABEL_CONTRACT_NEXT_EXPERIMENT.json','기계 판독 결과':'ROOT_RESULTS_KO.json','독립 검산':'independent_saved_initializer_review/REVIEW.json'}
    page += '<section><h2>근거</h2>'+''.join(f'<p><a href="{path}">{label}</a></p>' for label,path in links.items())+'<p>Gouda et al., MR6D, CC-BY-4.0. 공개 source 개발용 재사용이며 공식 MR6D blind benchmark 결과가 아니다.</p></section></html>'
    (V / 'ROOT_READOUT_KO.html').write_text(page)
    print(json.dumps({'results':bind(V/'ROOT_RESULTS_KO.json'),'readout':bind(V/'ROOT_READOUT_KO.html'),'figure':bind(fig)}))


if __name__ == '__main__':
    main()

"""Build one read-only final review page from independently checked evidence.

The user's document confirmation is separate from annotation approval. This
builder neither asks for coordinates nor writes human-reviewed dataset records.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'_docs/experiments/pallet_combined_closeout_20261003_v1/closeout_20261006_v1'
OUT = BASE/'final_review'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def binding(path):
    return dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    started=time.perf_counter()
    OUT.mkdir(parents=True,exist_ok=True)
    audits=[OUT/'INDEPENDENT_PAPER_AUDIT.json',OUT/'INDEPENDENT_REFERENCE_AUDIT.json']
    for path in audits:
        payload=read(path)
        if payload.get('status') not in ('PASS','PASS_WITH_DOCUMENTED_LIMITATIONS'):
            raise ValueError('Independent audit is not ready: '+str(path))
    static=read(BASE/'static/STATIC_REAGGREGATION.json')
    paper=read(BASE/'paper_patch/CLOSEOUT_VALIDATION.json')
    lifter=read(BASE/'lifter/AUDIT.json')
    preserve=read(BASE/'PRESERVATION_CHECK.json')
    subgroups=read(OUT/'SUBGROUP_REVIEW.json')
    rows=[]
    for key,name in [('yolo','YOLO'),('dope','DOPE'),('resnet18','ResNet-18')]:
        a=static['backbones'][key]['Base']['result']['all']
        b=static['backbones'][key]['N3']['result']['all']
        rows.append(dict(name=name,corner=f'{a["corner_median_px"]:.3f} → {b["corner_median_px"]:.3f}',
            translation=f'{a["translation_median_cm"]:.3f} → {b["translation_median_cm"]:.3f}',
            rotation=f'{a["rotation_median_deg"]:.3f} → {b["rotation_median_deg"]:.3f}',
            pose_frames=f'{a["pose_available_frames"]} / 319 → {b["pose_available_frames"]} / 319'))
    markdown=['# 마지막에 확인할 내용','',
        '추가 분류·코너 클릭·식별자 입력 없이 기존 자료로 가능한 실험 마감과 원고 반영을 마쳤습니다. '
        '아래 내용과 남은 근거 제한을 한 번 확인하면 이번 마감본의 검토가 끝납니다.','',
        '| 기반 | 코너 중앙값 px | 이동 중앙값 cm | 회전 중앙값 ° | 자세 산출 장수 |',
        '|---|---:|---:|---:|---:|']
    for r in rows:markdown.append(f'| {r["name"]} | {r["corner"]} | {r["translation"]} | {r["rotation"]} | {r["pose_frames"]} |')
    markdown+=['',
        '세 Base는 RGB 추정기이며, N3가 이미지 특징·초기 코너·박스·물리 치수를 함께 받습니다. '
        '각 기반에 학습된 보정기를 적용한 결과로, 하나의 가중치가 세 기반에 전이된 결과는 아닙니다. '
        'ResNet Base는 실제 10-epoch CONSTANT-fold RGB 모델입니다.','',
        'DOPE의 코너·회전 P90, ResNet의 이동 P90, 일부 어려움 집단은 악화됩니다. '
        '모든 상황에서 T/R이 함께 개선됐다고 주장하지 않습니다. 숫자는 각 seed 통계의 평균이며, '
        '직사각형 T/R 참조는 코너·기하 재구성 참조입니다.','',
        '정적 319/119장 등급과 3,101개 코너 상태를 그대로 재사용했습니다. '
        '등급의 의미는 사용자 입력 등급으로 제한하여 추가 기준 답변이나 재분류가 필요하지 않게 했습니다. '
        '외부 가림의 독립 검증 완료라고 바꾸지 않았습니다.','',
        '리프터 8,910장 추론과 12장 가시성 96개는 완료입니다. '
        '수동 좌표 67점과 좌표 없는 보임 5점, 자체 가림 24점을 구분합니다. '
        '가시 코너 정확도는 공식 참조·대상 대응이 없어 x, 정지 잡음은 정지 확인이 없어 x, '
        '정사각형·리프터의 독립 물리 T/R은 독립 참조가 없어 x로 유지합니다. '
        '최종 문서 확인으로 이 실험 참조를 승인 처리하지 않습니다.','',
        f'실제 원고 숫자 셀은 본문 {paper["replaced_numeric_cells_main"]}개·보충 '
        f'{paper["replaced_numeric_cells_supplement"]}개를 연결했고, '
        f'{paper["source_json_pointer_checks"]}개 출처를 검증했습니다. '
        '본문의 세 기반·등급별 결과는 실제 보충 표에 반영했습니다. '
        '추가 상세 행은 읽을 수 있는 보조 자료로 제공하므로 사용자가 표를 직접 편집할 필요가 없습니다.','',
        '원본을 보존했고 새 학습·추론·PDF 컴파일·리프터 제어·push·외부 업로드는 실행하지 않았습니다. '
        '이번 확인은 결과와 원고 소스 마감본의 내용 확인이며, PDF 조판이나 투고 완료를 뜻하지 않습니다.','',
        '[최종 원고 읽기](../paper_updated/manuscript_ko.md) · '
        '[보충자료 읽기](../paper_updated/supplement_ko.md) · '
        '[근거와 미확인 항목](../paper_patch/PAPER_GAP_MATRIX.md) · '
        '[세부 집단·각 seed 54행](SUBGROUP_REVIEW_KO.md)','',
        '![최신 입력 구성](../paper_updated/figures/current_review_summary.png)','',
        '**내용이 맞으면 채팅에서 “확인 완료”라고만 답하면 됩니다.** '
        '수정할 내용이 있으면 그 부분만 말씀해 주세요.']
    (OUT/'FINAL_REVIEW_KO.md').write_text('\n'.join(markdown)+'\n',encoding='utf-8')
    table_rows=''.join('<tr>'+''.join(f'<td>{escape(r[k])}</td>' for k in
        ('name','corner','translation','rotation','pose_frames'))+'</tr>' for r in rows)
    gallery=''.join(f'<figure><a href="../paper_updated/figures/example_{i}_frame.png">'
        f'<img src="../paper_updated/figures/example_{i}_frame.png" alt="기존 실제 영상 예시 {i}" loading="lazy"></a>'
        f'<figcaption>{escape(label)}</figcaption></figure>' for i,label in
        [(1,'큰 오류가 일부 줄어든 사례'),(2,'코너 오차가 줄어든 사례'),(3,'원래 양호한 예측이 조금 악화된 사례')])
    subgroup_details=''
    panel_names={'three_backbone_grades':'세 기반 × 세 등급 — 실제 보충 표',
        'material_by_grade':'재질 × 등급 상세', 'seed_all_319':'각 seed 결과',
        'square119_by_grade':'정사각형 등급과602/600모드'}
    for panel in dict.fromkeys(r['panel'] for r in subgroups['rows']):
        selected=[r for r in subgroups['rows'] if r['panel']==panel]
        content=''
        for row in selected:
            identity=row['group_label']
            if row.get('seed') is not None:identity+=' / '+str(row['seed'])
            if row.get('mode'):identity+=' / '+row['mode']
            values=[row['backbone_label'],identity,str(row['frames']),row['corner_median'],
                row['translation_median_cm'],row['rotation_median_deg'],str(row['reference_corners'])]
            content+='<tr>'+''.join('<td>'+escape(str(value))+'</td>' for value in values)+'</tr>'
        subgroup_details+=f'<details><summary>{escape(panel_names.get(panel,panel))} ({len(selected)}행)</summary>'
        subgroup_details+='<div class="scroll"><table><tr><th>기반</th><th>집단</th><th>장수</th><th>코너 px</th><th>이동 cm</th><th>회전 °</th><th>참조점</th></tr>'+content+'</table></div></details>'
    text=f'''<!doctype html>
<html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>팔레트 실험·원고 — 마지막 확인</title>
<style>
body{{margin:0;background:#f3f5f8;color:#1c2735;font:17px/1.65 system-ui,"Noto Sans CJK KR",sans-serif}}
main{{max-width:1120px;margin:30px auto;padding:0 24px 50px}}h1{{font-size:30px}}h2{{font-size:22px}}
section{{background:white;padding:24px;border:1px solid #d7e0e8;border-radius:12px;margin:18px 0}}
.badge{{display:inline-block;background:#e2f2eb;color:#175c40;border-radius:20px;padding:5px 14px;margin:4px}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #dfe5eb}}
th{{background:#edf2f6}}.scroll{{overflow-x:auto}}a{{color:#155b99}}.note{{color:#536272;font-size:15px}}
img{{max-width:100%;height:auto}}.gallery{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}
figure{{margin:0}}figcaption{{font-size:15px}}.finish{{background:#e7f4ec;border-color:#98cdb2}}
details{{margin:12px 0;border:1px solid #dfe5eb;padding:12px;border-radius:8px}}summary{{cursor:pointer;font-weight:600}}
li{{margin:9px 0}}@media(max-width:700px){{.gallery{{grid-template-columns:1fr}}main{{padding:0 12px}}section{{padding:16px}}}}
</style><main>
<h1>마지막에 내용만 확인하세요</h1>
<p>계산·검산·원고 수정을 직접 확인했습니다. 추가 분류나 코너 클릭, 식별자 입력은 요구하지 않습니다.</p>
<span class="badge">세 기반 검산 완료</span><span class="badge">실제 원고 반영 완료</span>
<span class="badge">원본 보존</span><span class="badge">학습 재실행 0회</span>
<section><h2>1. 결과는 이렇게 보고합니다</h2><div class="scroll"><table>
<tr><th>기반</th><th>코너 중앙값 px</th><th>이동 중앙값 cm</th><th>회전 중앙값 °</th><th>자세 산출 장수</th></tr>
{table_rows}</table></div>
<p>세 기반에서 중앙값은 줄었습니다. DOPE의 코너·회전 P90, ResNet의 이동 P90과 일부 어려움 집단은 악화됩니다.</p>
<p class="note">Base → N3. N3는 세 seed 통계의 평균입니다. 실패 프레임을 분모에 유지했습니다.
직사각형 T/R은 코너·기하 재구성 참조에 대한 오차이며 독립 물리 실측 정확도를 뜻하지 않습니다.</p>
<p>Base는 RGB 추정기입니다. N3는 이미지 특징·초기 코너·박스·치수를 함께 받으며 각 기반에 별도로 학습되었습니다.</p></section>
<section><h2>2. 이미 한 입력을 다시 쓰지 않아도 됩니다</h2>
<p>정적 319/119장 등급과 코너 상태 3,101개를 재사용했습니다. 등급은 <b>사용자 입력 등급</b>으로 설명하며,
미확인 기준을 추정해 외부 가림 검증 완료로 바꾸지 않았습니다.</p>
<img src="../paper_updated/figures/current_review_summary.png" alt="직사각형과 정사각형의 입력 등급 및 코너 상태">
<p>리프터 8,910장 추론과 12장 가시성 96개도 완료했습니다. 실제 수동 좌표 67점·좌표 없는 보임 5점·자체 가림 24점을 구분합니다.</p></section>
<section><h2>3. 근거가 없는 정확도는 x로 유지합니다</h2><ul>
<li>리프터 가시 코너 정확도: 공식 참조와 대상 대응 미확인, 보임 5점의 실제 좌표 없음.</li>
<li>리프터 정지 잡음: 사람이 확인한 정지 구간 없음.</li>
<li>정사각형·리프터 독립 물리 T/R: 독립 참조 없음.</li></ul>
<p>이 제한을 원고에 적었습니다. 이번 내용 확인은 주석 승인이나 새 실험 결과를 만들지 않습니다.</p></section>
<section><h2>4. 실제 원고와 근거</h2><p>본문 {paper['replaced_numeric_cells_main']}개·보충 {paper['replaced_numeric_cells_supplement']}개 숫자 셀,
{paper['source_json_pointer_checks']}개 출처 연결을 검증했습니다. 본문에서 설명한 세 기반·등급별 결과를 보충 표에 넣었습니다.</p>
<p><a href="../paper_updated/manuscript_ko.md">최종 원고 읽기</a> · <a href="../paper_updated/supplement_ko.md">보충자료 읽기</a> ·
<a href="../paper_patch/PAPER_GAP_MATRIX.md">근거와 미확인 항목</a> · <a href="INDEPENDENT_PAPER_AUDIT_KO.md">독립 검산 기록</a></p>
{subgroup_details}
<div class="gallery">{gallery}</div><p class="note">기존 원고의 실제 예시입니다. 새로 좋은 사례만 골라낸 그림이 아닙니다.
초록 ×는 참조, 파랑 ○는 Base, 분홍 ◇는 N3 seed1입니다.</p></section>
<section class="finish"><h2>확인은 한 번이면 됩니다</h2><p>이 결과·제한·원고 마감본의 내용이 맞으면 채팅에서 <b>“확인 완료”</b>라고만 답하세요.
수정할 내용이 있으면 그 부분만 말씀해 주세요.</p>
<p class="note">새 학습·PDF 컴파일·리프터 제어·push·업로드는 실행하지 않았습니다.
이번 확인은 실험 결과와 원고 소스 마감본에 대한 내용 확인입니다.</p></section></main></html>'''
    (OUT/'FINAL_REVIEW.html').write_text(text,encoding='utf-8')
    inputs=audits+[BASE/'static/STATIC_REAGGREGATION.json',BASE/'paper_patch/PAPER_CELL_MAP.json',
        BASE/'paper_patch/CLOSEOUT_VALIDATION.json',BASE/'lifter/AUDIT.json',BASE/'PRESERVATION_CHECK.json',
        OUT/'SUBGROUP_REVIEW.json']
    files=[binding(path) for path in inputs]+[binding(path) for path in sorted((BASE/'paper_updated').rglob('*')) if path.is_file()]
    packet=dict(schema='pallet_final_document_review_packet_v1',status='READY_FOR_ONE_USER_DOCUMENT_CONFIRMATION',
        generated_at=datetime.now(timezone.utc).isoformat(),inputs_and_updated_paper=files,
        required_new_annotation_actions=0,automatic_human_confirmation_created=False,
        document_confirmation_is_annotation_approval=False,grade_criterion='NOT_CONFIRMED_GENERIC_USER_ENTERED_GRADES',
        official_lifter_reference_approved_frames=lifter['formal_reference']['approved_frames'],
        original_files_unchanged=preserve['checked_files'],new_training=0,new_model_forward=0,
        pdf_compilation=0,push=0,actual_lifter_control=0,build_wall_seconds=time.perf_counter()-started)
    (OUT/'FINAL_REVIEW_PACKET.json').write_text(json.dumps(packet,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:value for key,value in packet.items() if key!='inputs_and_updated_paper'},ensure_ascii=False))


if __name__=='__main__':
    main()

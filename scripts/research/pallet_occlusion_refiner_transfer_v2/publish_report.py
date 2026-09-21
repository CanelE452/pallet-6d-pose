"""Publish deterministic, portable scientific overlays from frozen predictions."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import run as C

ARMS = ('R0', 'N2', 'REPLAY', 'A11')
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def main():
    dest = C.DOC / 'figures'
    dest.mkdir(exist_ok=True)
    assert not (C.DOC / 'REPORT_WITH_FIGURES_KO.md').exists(), 'Published report is immutable'
    protocol = C.read(C.DOC / 'E2_PROTOCOL.json')
    results = C.read(C.DOC / 'E2_REAL_RESULTS.json')['summary']
    metrics = C.read(C.RAW / 'E2_FRAME_METRICS.json')
    truth = C.read(C.V.RAW / 'E1_FRAME_METRICS.json')['truth_for_display_only']
    pred = C.read(C.V.RAW / 'FROZEN_PREDICTIONS.json')['predictions']
    pred['A11'] = C.read(C.RAW / 'PREDICTIONS_A11.json')['predictions']
    selections = C.read(C.OUT / 'GALLERY_SELECTION.json')['selection']
    meta = {r['id']: r for r in protocol['eval_records']}
    font = ImageFont.truetype(FONT, 18)
    small = ImageFont.truetype(FONT, 14)
    evidence = []
    lines = ['# Clean 실사 pseudo-label을 이용한 보정기 적응 — 이미지 포함 결과 보고', '',
        '[확인] 2026-09-21, `pallet_occlusion_refiner_transfer_v2`. **판정: PARTIAL. 기존 최종 모델을 교체하지 않는다.**', '',
        '## 결론', '',
        '[확인] 사용자가 지정한 낮 플라스틱 264장 중 253장이 기존 필터를 통과했다. 이를 이용해 R0를 동결하고 보정기 네 개를 각각 300 update 학습했다. 인공 가림 입력은 실제 가린 영상에서 R0를 다시 추론했다.', '',
        '[확인] 실제 가림 평가 93장에서 A11은 PCK10 49.09%로 R0 43.20%보다 높지만 기존 N2/Replay 49.09%와 같다. 가림 학습의 추가 효과는 source 없이 −0.28pp, source 병행 시 −0.84pp로 확인되지 않았다. 기존보다 개선된 방법으로 채택할 근거는 부족하다.', '',
        '[확인] A11은 큰 오류 250개 중 9개를 10px 이내로 복구했지만, 정상점 손상과 clean/source 보존 기준을 통과하지 못했다. 따라서 가림 성능 조건은 통과하되 전체 성공은 아닌 PARTIAL이다.', '',
        '## 실험 구성과 평가 범위', '',
        '| [확인] 조건 | 실사 입력 | 합성 GT 병행 |', '|---|---|---|',
        '| A00 | clean RGB + clean R0 | 없음 |', '| A01 | 인공 가림 RGB + 가림 후 R0 | 없음 |',
        '| A10 | clean RGB + clean R0 | 있음 |', '| A11 | 인공 가림 RGB + 가림 후 R0 | 있음 |', '',
        '[확인] 공통: synthetic-only PoseFix에서 초기화, seed 1, 300 update, 실사 batch 8, lr 1e-4, BN 통계 동결, 마지막 checkpoint만 평가. 합성 병행 조건만 source batch 8을 추가했다. 실사 타깃·마스크·순서·2,400회 노출은 같다. compute-matched 실험은 아니다.', '',
        '[확인] 타깃은 frozen R0 → 기존 confidence/flip/LOO → frozen Replay → 기존 self-occlusion PnP → LOO로 만들었다. 필터 통과가 정답을 보장하지 않는다. eval GT는 학습/추론에 쓰지 않았고 출력 고정 뒤 채점했다.', '',
        '[확인] 이번 학습 촬영 전체와 Replay 수동 학습 세션을 평가에서 제외했다. 총 278장(비초록 128 + 초록 150)이며 가림 93장, 비가림 17장, DEV 72장 등 하위 집합은 일부 겹치므로 합산하지 않는다. 검출/매칭 실패도 전체 PCK 분모에 남겼다. median/P90은 matched 코너 기준이다.', '',
        '[확인] PCK10/20은 각각 오차 10/20px 이내 코너 비율이다. 복구는 R0 >20px → 보정 ≤10px, 손상은 R0 <5px → 보정 >10px이다. whole-object 대칭 매칭을 사용하며 점마다 유리한 대칭을 따로 고르지 않는다.', '']
    for pop, title in [('PRIMARY_OCC96', '실제 가림 93장'), ('CLEAN_NONCAD69', '비가림 17장'), ('GREEN150_MANUAL', '초록 manual GT 150장')]:
        lines += ['## ' + title, '', '| [확인] 방법 | PCK10 % ↑ | PCK20 % ↑ | median px ↓ | P90 px ↓ | 복구/큰 오류 | 손상/정상점 |', '|---|---:|---:|---:|---:|---:|---:|']
        for arm, s in results[pop].items():
            d = s['recovery_damage']
            lines.append(f'| {arm} | {100*s["PCK"]["10"]:.2f} | {100*s["PCK"]["20"]:.2f} | {s["matched_pooled_corner8_median_px"]:.2f} | {s["matched_pooled_corner8_P90_px"]:.2f} | {d["recovered"]}/{d["hard"]} | {d["damaged"]}/{d["good"]} |')
        lines += ['', '[확인] 아래는 기존 전체 갤러리의 개선/악화 순위 각 상위 2장 및 seed 1 랜덤 목록 앞 2장이다. 순위는 A11−R0 프레임 평균 오차 기준이며 그룹 간 중복이 가능하다. 개선 순위에도 실제 개선이 없는 경우가 있을 수 있어 부호를 함께 표시한다.', '',
            '[확인] 각 그림: 왼쪽 위 R0, 오른쪽 위 N2, 왼쪽 아래 기존 Replay, 오른쪽 아래 A11. **초록 십자=GT/reference, 파랑 선·점=예측, 주황 선=R0에서 이동.** G0…G7 오차는 대칭 매칭 후 GT 코너 기준, P 번호는 native 예측 번호다. GT는 채점/표시 전용이다.', '']
        for group, fids in selections[pop].items():
            label = {'improvement':'개선 순위', 'damage_or_least_improvement':'악화/최소 개선 순위', 'random_seed1':'고정 랜덤'}[group]
            for rank, fid in enumerate(fids[:2], 1):
                raw = Image.open(C.ROOT / meta[fid]['image']['path']).convert('RGB')
                assert raw.size == (640, 480)
                canvas = Image.new('RGB', (1280, 1152), '#10212b')
                initial = np.asarray(C.P.top(pred['R0'][fid])['keypoints_xy'])
                rows = {}
                for i, arm in enumerate(ARMS):
                    tile = Image.new('RGB', (640, 576), '#10212b'); tile.paste(raw, (0, 76)); draw = ImageDraw.Draw(tile)
                    m = metrics[arm][fid]; p = C.P.top(pred[arm][fid]); pts = np.asarray(p['keypoints_xy']) if p else None
                    draw.text((10, 5), f'{arm} | mean {m["frame_mean_px"]:.2f}px | matched={m["matched"]}', font=font, fill='white')
                    errors = m['canonical_errors']
                    for line in range(2):
                        text = '  '.join(f'G{j}:{errors[j]:.1f}' if errors[j] is not None else f'G{j}:NA' for j in range(line*4, line*4+4))
                        draw.text((10, 29+20*line), text, font=small, fill='#bcebbb')
                    if pts is not None:
                        for a, b in C.V.S.G.EDGES:
                            if np.isfinite(pts[[a,b]]).all():draw.line([(pts[a,0],pts[a,1]+76),(pts[b,0],pts[b,1]+76)], fill='#38baff', width=2)
                        for j, (x,y) in enumerate(pts[:8]):
                            if not np.isfinite([x,y]).all():continue
                            ix,iy=initial[j]; y+=76
                            draw.line([(ix,iy+76),(x,y)],fill='#ffa64d',width=2)
                            draw.ellipse((x-3,y-3,x+3,y+3),fill='#38baff')
                            draw.text((x+4,y-16),f'P{j}',font=small,fill='white',stroke_width=1,stroke_fill='black')
                    for j, (x,y) in enumerate(truth[fid]['gt'][:8]):
                        if truth[fid]['valid'][j]:
                            y+=76;draw.line((x-5,y,x+5,y),fill='#69ff63',width=2);draw.line((x,y-5,x,y+5),fill='#69ff63',width=2)
                    move = float(np.linalg.norm(pts[:8]-initial[:8],axis=1).max()) if pts is not None else None
                    draw.text((10, 558),f'Max movement from R0: {move:.2f}px' if move is not None else 'No detection',font=small,fill='#ffa64d')
                    canvas.paste(tile, ((i%2)*640,(i//2)*576))
                    rows[arm] = dict(metric=m,prediction=p,max_movement_px=move)
                name=f'{pop}_{group}_{rank:02d}.jpg'; path=dest/name
                assert not path.exists();canvas.save(path,quality=88,subsampling=0)
                delta=metrics['A11'][fid]['frame_mean_px']-metrics['R0'][fid]['frame_mean_px']
                lines += [f'### {label} {rank}: `{fid}`', '', f'[확인] A11−R0 평균 코너 오차 **{delta:+.2f}px** (음수=개선).', '', f'![{title} {label} {rank}](figures/{name})', '']
                evidence.append(dict(population=pop,group=group,rank=rank,id=fid,image=meta[fid]['image'],figure=C.bind(path),truth_display_only=truth[fid],arms=rows))
    lines += ['## 실제 학습 입력 예시', '', '[확인] 왼쪽 clean, 오른쪽 인공 가림. 전체 253장 중 185장에 패치가 적용됐다. OCC 조건은 패치 적용 후 R0를 실제 재실행한 예측을 입력으로 썼다. 아래 두 장은 기존 저장 예시의 파일명순 첫 두 장이며 별도 결과 선별은 하지 않았다.', '']
    for i, src in enumerate(sorted((C.OUT/'augmentation').glob('*.png'))[:2],1):
        path=dest/f'training_input_{i:02d}.jpg';assert not path.exists()
        Image.open(src).convert('RGB').save(path,quality=88,subsampling=0)
        lines += [f'![실제 학습 입력 {i}](figures/{path.name})', '']
    lines += ['## 해결된 것과 남은 병목', '',
        '[확인] 합성 stress PCK10은 초기 19.54% → A11 69.14%로 높아졌지만, 합성 clean은 94.02% → 91.00%로 낮아졌다. 실제 가림에서는 기존 보정기와 동률이다. 합성 교란 복구 향상을 실제 가림 일반화 성공으로 해석할 수 없다.', '',
        '[확인] 인공 가림만 학습한 A01의 가림 정상점 손상은 11/116, 합성 병행 A11은 1/116이다. source 병행은 이 비교에서 손상을 줄였지만 전체 clean/GREEN/source 보존 기준까지 해결하지는 못했다.', '',
        '[추정] 현재 병목은 단순히 보정량이 작은 문제가 아니라, 실제 큰 오류를 고치면서 정상점을 보존하는 일반화다. 좁은 연속 촬영 구간, pseudo-target 오차, 인공/실제 가림 차이 중 어느 하나가 주원인인지는 이 실험만으로 확정할 수 없다.', '',
        '## 한계와 무결성', '',
        '- [확인] 253개 고유 이미지지만 연속 한 촬영분이며 253개 독립 장면이 아니다. clean 조건도 사용자 구간 확인으로, 프레임별 정답 검증은 아니다.',
        '- [확인] Replay teacher의 과거 목재 학습 6장은 같은 DAY recording 출신이다. 선택 이미지와 정확한 SHA 중복은 없지만 teacher-independent adaptation 주장은 불가하다.',
        '- [확인] GREEN은 manual GT, 비초록은 provenance가 혼재한 legacy reference다. 반복 사용한 DEV 결과이며 독립 최종 test/통계적 유의성 주장이 아니다. 물리적으로 짝지어진 clean/가림 GT는 없다.',
        '- [확인] 자동 테스트 10개 통과, 네 조건 순서/노출/마스크/BN 및 원본 모델 SHA 보존을 확인했다. 첫 테스트 import 문제로 A00가 검사 완료 전에 시작되어 잠시 정지 후 재개한 순서 일탈은 [실행 기록](EXECUTION_NOTE.md)에 보존했다.',
        '- [확인] 기존 annotation/split/논문표/최종 checkpoint는 변경하지 않았다. E3~E6, 추가 튜닝, 재학습은 실행하지 않았다.', '',
        '## 자료 위치와 업로드 범위', '',
        '- [확인] [전체 수치 보고서](RESULTS_KO.md), [판정 상세](E2_DECISION.json), [최종 감사](AUDIT.json), [그림별 좌표·오차·해시](FIGURE_EVIDENCE.json).',
        '- [확인] [선행 E1 진단](../pallet_occlusion_refiner_transfer_v1/RESULTS_KO.md)은 기존 결과 그대로 보존했다. E1의 STOP은 후보 데이터 확정 전 상태이고, 이번 v2는 사용자 지정 구간 승인 후 별도 실행했다.',
        '- [확인] 이 문서의 비교 그림 18장과 학습 예시 2장은 GitHub에서 직접 표시되는 저장소 내부 상대 경로다. 대용량 원본 RGB/영상, checkpoint, tensor cache, 전체 로컬 HTML 갤러리는 업로드 대상이 아니다. 기존 문서의 outputs/data 경로는 로컬 산출물 참조이며 clean clone만으로 전체 재실행이 가능한 자료 묶음은 아니다.', '']
    C.freeze(C.DOC/'FIGURE_EVIDENCE.json',dict(status='[확인]',selection='First two from each existing frozen improvement/damage/random group; no new ranking',rows=evidence,rendering='PIL overlays, no generated or altered predictions'))
    C.freeze(C.DOC/'REPORT_WITH_FIGURES_KO.md','\n'.join(lines))
    C.freeze(C.DOC/'PUBLISH_MANIFEST.json',dict(status='[확인]',source_HEAD='73bfe38a259b3c846e49a98fb82c44578f6e2248',report=C.bind(C.DOC/'REPORT_WITH_FIGURES_KO.md'),figures=[C.bind(p) for p in sorted(dest.glob('*.jpg'))],evidence=C.bind(C.DOC/'FIGURE_EVIDENCE.json'),builder=C.bind(Path(__file__)),image_synthesis=False))
    print('REPORT_READY',len(evidence),'comparisons',len(list(dest.glob('*.jpg'))),'images')


if __name__ == '__main__':
    main()

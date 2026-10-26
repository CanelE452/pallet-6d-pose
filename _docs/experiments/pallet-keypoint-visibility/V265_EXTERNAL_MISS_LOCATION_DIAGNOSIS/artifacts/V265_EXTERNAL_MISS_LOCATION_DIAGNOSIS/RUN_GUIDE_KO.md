# V265 저장 출력 진단 재현

새 학습·분류기/위치 모델 forward·PnP·threshold 변경 없이 V264 외부 가림 누락67점과 V263 수동 보임 위치854점을 분석했습니다. `ROOT_READOUT_KO.html`이 짧은 결과이고 `diagnosis_v1/INDEX_KO.html`에12개 확대 사례와 원시 수치가 있습니다.

## 휴대용 재현

이 폴더에서 Python3.10, NumPy1.26.4, Pillow 및 NotoSansCJK-Regular 글꼴을 사용합니다. 실제 실행 글꼴 경로는 `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`입니다.

```bash
python -B diagnose_saved_v265.py --replay --out diagnosis_replay_v1
```

`analysis_inputs`의 포함된 CSV·고정 heatmap geometry·선택 crop만 사용합니다. 원본 GB 특징·모델·실사 원본 전체 없이 거리 구간, 수동 보임 비교,12개 사례 그림을 다시 생성합니다. 원래 출력 폴더는 덮어쓰지 않습니다. `SEALED_INPUTS.json`의 상대 파일명으로 SHA/크기를 확인하며 직렬화된 원래 절대 경로를 재현 입력으로 따라가지 않습니다.

기본 실행(리플레이 옵션 없음)은 로컬 원본 바인딩으로 포함 입력을 만들던 원래 준비 코드입니다. 원래 자료가 없는 다른 컴퓨터에서는 위 휴대용 명령을 사용합니다. `all67_gallery_v1`에는 누락67점 전체의 원래 query crop이 포함됩니다. 전체 gallery 생성 코드는 `render_all67_v265.py`입니다.

## 과장하지 않는 해석

- 3/6px 구간은 진단용 거리 구간이며 새 운용 threshold가 아닙니다.
- 가려진 점의 참조 좌표는 투영·외삽·숨은 점 수동 표시입니다. 직접 보이는 위치 정답이 아니므로 근접/이탈로 서술합니다.
- 사후 humanD 집합에서 원시 위치 제안을 모두 적용한 비교는 보임 승인 규칙이나 배포 성능이 아닙니다.
- 기존 버전/병렬 감사에서 일부 crop을 이미 확인했습니다. 대표 선정은 고정 숫자 규칙이지만 독립 블라인드 검증이라고 부르지 않습니다.
- G996 주석 이견과 작은3px 경계 악화를 별도로 표시했고 원 주석을 변경하지 않았습니다.

정사각형에서 위치 후보의 유용성이 확인됐지만 목재는 개선되지 않았습니다. 이번 분석으로 새 모델·게이트를 채택하거나 추가 학습을 시작하지 않았습니다.

## 그림 재현 런타임

수치와 RGB 원본 해시는 표시 글꼴/PNG 인코딩과 구분합니다. 대표12개 그림은 Python3.10.20·NumPy1.26.4·Pillow12.2.0·FreeType2.14.3·Noto CJK에서 생성했습니다. 전체67점 갤러리는 Python3.13.9·NumPy2.3.5·Pillow12.0.0·FreeType2.13.3·DejaVuSans에서 생성했습니다. 두 환경의 글자 픽셀이 다를 수 있으며, 바이트 단위 PNG 재현에는 해당 원래 런타임을 사용합니다. 전체67점 코드의 `DejaVuSans.ttf` 경로도 설치돼 있어야 합니다.

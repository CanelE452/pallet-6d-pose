# V271 저장 비교 자료 재현

새 학습·모델·GPU·원래 데이터 폴더 없이 포함된 저장 표·지도·PNG만 사용합니다. 표·마스크 재계산은 NumPy/Pillow가 있는 Python, JS 단위 검사는 Node가 필요합니다. 합성 PNG 재생에는 모델 없는 PyTorch bilinear 수치 연산도 사용합니다. byte 동일 PNG 검증 환경은 Python3.10/PyTorch2.1.1+cu118/NumPy1.26.4/Pillow12.2.0이며 로컬에서는 /home/minjae/anaconda3/envs/pallet-yolo26/bin/python -B를 사용했습니다. 다른 렌더링 환경에서 PNG bytes와 일부 픽셀이 달라질 수 있습니다. 로컬 파일을 브라우저에서 자동으로 열려는 작업은 정책상 거부됐으므로 브라우저 실물 렌더 검증을 주장하지 않습니다. 그림은 직접 확인했고 JS는 로컬 DOM-shim 단위 검사입니다.

작업 폴더를 artifacts 안 V271으로 정하고 다음을 실행합니다. 출력은 새 경로여야 합니다.

```sh
python -B replay_saved_comparison_v271.py --out /tmp/v271_saved_replay_fresh
```

1784점에서540개 지표 묶음의 CM/precision/recall을 재계산하고, 저장 raw32 logits로50개 전체 마스크를 다시 생성해 binary PNG 및 RGB cutout의 픽셀과 일치하는지 검사합니다. 외부 RGB 경로나 모델은 열지 않습니다.

```sh
node verify_root_interaction_v271.js
python -B make_syn_gt_comparison_v271.py --out /tmp/v271_syn_replay_fresh.png
```

첫 명령은 10장·모델·표시 방식·KP 강조·재질/번호 필터를 local DOM shim에서 검사합니다. 실제 브라우저 화면 검사와 구별합니다. 둘째는 포함된 합성2장 원본 M4 GT/저장 예측으로 합성 비교 PNG를 다시 만듭니다. 새로운 가림 정답이나 실사 pixel GT를 만들지 않습니다.

공개 패키지 별도 경로에서 실행한 결과는 패키지의 PUBLICATION_SAVED_REPLAY_VERIFICATION.json과 publication_verification에 보존합니다. 마스크 문턱0.5·native 대응·사람 주석은 바꾸지 않습니다.

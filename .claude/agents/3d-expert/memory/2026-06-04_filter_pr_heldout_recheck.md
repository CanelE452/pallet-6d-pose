# 필터 P/R held-out 재평가 (누수 교정)

## 누수 발견
이전 필터 P/R 스크리닝(`filter_pr_camfacing.py --tag s2`)의 평가모델 `dope_cropaug_ft_s2`가
평가 GT(outside_combined 129 + night_combined 90)를 **학습데이터로 포함** → train-set 평가 = 누수.
base rate 0.53·필터 P/R 전부 낙관적. **모델 P/R 스크리닝 할 땐 평가셋이 그 모델 train set인지 header.txt로 먼저 확인할 것.**

## held-out 재평가
- 평가모델 = `weights/dope/dope_cropaug_pretrain/final_net_epoch_0060.pth` (학습=mixed_v8_train+truncation_crops_dope/pretrain, manual GT 미포함 = held-out. header.txt가 Namespace(data=[...]) 로 train data 박혀있음).
- GT pool 251 = 219 + forklift gt_manual 32. forklift도 object-frame canonical(HEIGHT-edge shortest 32/32 검증). forklift 이미지는 `rgb/` 서브디렉(JSON은 `gt_manual/`) → 스크립트에 `--include_forklift` + (gt_dir, img_dir) 튜플 처리 추가.

## 핵심 발견 — 검출 빈약(도메인 갭)
held-out: detectable 119(ft_s2 115와 비슷) 인데 **good 61→14, base rate 0.53→0.118**, mean_match median 9.9→16.4px,
gross>20px bucket 5→43. night 거의 사망(good 1/42). **검출은 비슷한데 키포인트 정확도가 무너짐.**
→ "합성+trunc only 일반화 모델의 real 검출 한계" = 논문에 그대로 보고할 발견(self-training 필요 근거).

## diag 검증 결론
- **방향성 유지·절대강도 약화.** ft_s2에서 diag gross 5/5(100%) reject 였던 게 누수 5-표본 산물.
  held-out 43-표본에선 **72%(31/43)** reject, catastrophic>40px 15개 중 4개 통과.
- **"gross 100% 제거" 주장 철회 → "~72% 제거"로 정정.** 절대수치를 5-표본에서 뽑으면 안 됨(표본 확대로 약점 드러남).
- ransac_loo(95%)/cf_strict(100%)/combo(93%)가 gross 제거는 더 강하나 pass 1-8·P=0 = self-training 물량 사망.
  PnP-free·비율 unknown 적용성 종합 → diag가 "volume vs gross-reject" 최선 trade-off, **primary 유지 타당.**
- **base rate 0.118 = P/R 표 거의 무의미**(good 14개뿐), gross-reject%만 정보성 있음(ft_s2 교훈 재확인, 표본 5→43으로 신뢰도↑).

## 교훈 (재사용)
1. 필터/평가 모델 P/R 볼 때 평가셋 ⊂ train set 누수 의심 → header/args 확인.
2. gross-reject 같은 absolute %는 작은 bucket(5)에서 뽑지 말 것 — held-out 큰 표본(43)에서 재서술.
3. 일반화 모델 real 검출 빈약은 숨기지 말고 보고(self-training 동기).
4. paper_base 모델로 동일 251-pool diag 재검증 권장(논문 최종).

산출물: `data/pallet/eval_results/filter_pr_camfacing/{summary,per_frame}_heldout_pretrain.json`,
docs: `_docs/experiments/filter/pr_screening.md` (Held-out 섹션).

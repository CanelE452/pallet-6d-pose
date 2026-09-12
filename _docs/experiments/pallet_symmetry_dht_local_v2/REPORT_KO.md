# Symmetry-aware DHT local fusion v2 — 최종 bounded experiment

## 결론

사전 고정된 판정은 **`DHT_LOCAL_TRACK_CLOSED`** 이다. Hough 세 seed 모두 Point 대비 primary 1% 개선 gate를 통과하지 못했고 median/P90도 모두 악화했다. 따라서 real DEV와 FINAL은 열지 않았으며 추가 DHT architecture/loss/seed/epoch/grid 탐색을 종료한다. C4는 데이터가 없어 `C4_NOT_EVALUATED`이다.

## 관찰

- Point: primary `0.009474144`, median `2.0264px`, P90 `7.3339px`, coverage `4027`.
- Direct seed 1/2/3 primary 변화(Point 대비): `-8.447%`, `-9.292%`, `-13.234%`.
- Hough seed 1/2/3 primary 변화(Point 대비): `-2.895%`, `-2.661%`, `-2.801%` (양수가 개선). Hough median은 `2.1186/2.1241/2.1939px`, P90은 `8.2567/8.0448/8.1023px이다.
- Hough good-point damage는 `0.598%/0.628%/0.389%`; 새 catastrophic frame은 모두 0, coverage는 모두 Point와 동일하다.
- Hough continuous line endpoint error 평균은 `4.458/4.622/4.489px`로 Direct보다 작았지만 point 개선으로 이어지지 않았다.
- Hough utility AUROC는 `0.609/0.601/0.604`, AUPRC는 `0.277/0.279/0.282`이다. signed candidate gain과의 Spearman은 `-0.073/-0.063/-0.072`로 음수였다.

## Oracle 원인 분해

GT를 사용하는 설명용 진단이며 배포 성능 주장이 아니다. primary는 baseline `0.009474144`에서 exact continuous GT line `0.008059417`, 36×65 soft-quantized/decode GT line `0.008242076`, 기존 v1 seed1 predicted line `0.008921087`가 됐다. GT-oracle utility가 있으면 세 단계 모두 baseline을 개선하므로 local geometry에 개선 가능성은 있었다. soft representation은 exact line보다 일부 손실이 있었고, predicted line에서 추가 손실이 있었다.

## 네 수정

1. anchor를 제거하고 corrected-point loss와 one-sided no-harm만 유지했다. 2/5/10px fixture에서 exact < partial < baseline을 확인했다.
2. structural support와 correction utility를 분리했다. utility는 detached selected line의 counterfactual gain이 0.25px를 초과하는지 감독하며 inference는 GT-free다.
3. 36×65 lattice는 그대로 두고 1-bin theta/rho bandwidth의 continuous soft target을 사용했다. hard-bin accuracy와 continuous raw-pixel line error를 별도 보고했다.
4. semantic edge당 MAP-local 한 mode만 WLS에 전달했다. 대안 mode들의 normal equation을 합산하지 않는다.

## 해석 범위

관찰된 결과는 네 수정 후에도 bounded synthetic gate가 실패했다는 뜻이다. 네 불일치가 과거 실패에 각각 얼마나 기여했는지, 또는 모든 가능한 DHT 설계가 실패한다는 인과 명제는 증명하지 않는다. 논문 결론은 현재 고정 데이터·예산·구조 범위에서 DHT line estimation 향상이 안정적인 point 개선으로 이어지지 않았다는 수준으로 제한한다.

# BROAD_FAMILY_V2 — SPEC 초안 템플릿

> ⚠ **이 문서는 WEAK_PASS 이거나 "generic 은 좋은데 target/night 만 실패" 일 때만
> 채운다.** 그 전에 렌더를 시작하지 않는다. 수량을 먼저 정하지 않는다.

## 1. 현재 support (측정값만)

`coverage_analyzer.py` 출력에서 옮긴다. 추정으로 채우지 않는다.

```
asset 수
종횡비   min / median / max
두께비   min / median / max
luma     min / median / max
scene    분포
```

## 2. 부족한 support (DEV 실패와 대조해서)

```
cell                    현재      필요 근거(어느 DEV set 이 어떻게 실패했나)
geometry / 얇은 팔레트
appearance / 밝은 조명
appearance / 야간
viewpoint / 저앙각
```

## 3. 목표 분포

현재 BROAD 분포를 **중심으로 유지**하고 부족한 cell 만 채운다.
모든 축을 균등하게 만들지 않는다.

## 4. target OBJ 제외 계약

```
평가 대상 pallet_full.obj 및 그 alias / 동일 mesh / 동일 geometry signature
= 생성에 사용 금지
검증: TARGET_ASSET_EXCLUSION_AUDIT 를 새 데이터에도 실행해 PASS 확인
```

## 5. 필요 asset / frame 수

**[추정]** 태그를 붙여 적고, 근거(어느 cell 이 몇 프레임 부족한지)를 함께 쓴다.

## 6. 중단 기준

몇 프레임을 넣어도 해당 cell 의 DEV 지표가 개선되지 않으면 그 축을 접는다.
기준을 생성 전에 적는다.

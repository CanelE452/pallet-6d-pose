# PCR-DOPE architecture

```
RGB -> VGG19
   ├─ F100 (vgg tap, 100x100)  ─┐
   └─ F50  (shared, 128x50x50) ─┴─> role encoder -> E_role 32x50x50
                                     prototypes P 8x32 (L2 normalized 양쪽)
                                     S = cos(E,P)/0.10   (8x50x50, sigmoid 없음)
belief stage 1~3 (frozen) ─> H3, A3
stage s in {4,5,6}:
    F_s = FiLM_s([E_role, softmax(S)], F50)   # zero-init -> identity
    입력 = concat(H_{s-1}, A_{s-1}, F_s)
    출력 = 기존 belief/affinity head 그대로
-> 기존 decoder -> centroid 포함 canonical PnP
```

belief 에 residual 을 더하지 않는다.  role score 는 decoder 에 들어가지 않고
**feature modulation 에만** 쓰인다.

trainable 17,950,843 (vgg_last 5,014,912 / stage4~6 12,567,579 /
role encoder 248,736 / FiLM 119,616).
frozen: VGG early, belief 1~3, affinity 전체, segmentation, teacher, decoder/PnP.

## Symmetry

frame 당 identity 또는 yaw+180 permutation (5,4,7,6,1,0,3,2) **하나만** 선택한다.
corner 별 독립 최소화 금지, unrestricted Hungarian 금지, top-bottom inversion 불허.
involution 과 near↔far 교환을 테스트로 강제했다.

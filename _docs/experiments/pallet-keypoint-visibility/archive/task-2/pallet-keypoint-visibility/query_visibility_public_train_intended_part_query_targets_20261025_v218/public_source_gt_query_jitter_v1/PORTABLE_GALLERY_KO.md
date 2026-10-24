# 별도 GT-query SOURCE 입력

GT에서 유래한 새 학습용 q8입니다. 원 initializer 96/U11과 별개이며, 원 query 개선이나 실사 성능을 측정하지 않습니다. 원 L·native alias·FIT/DEV를 계승합니다. nominal cuboid와 물리 부위 대응은 불확실하며 모든 D/E/SELF는 UNKNOWN입니다. DEV 120행은 향후 모든 fit/normalizer에서 제외합니다. pose/ref/ROI/mask는 추가 배포 feature로 주지 않습니다.

원본 학습용 RGB96 360개는 marker가 없습니다. 아래 15 contact는 nominal offset(0,0)만 보여 주며 cyan crop 십자는 표시 전용입니다. 나머지 두 jitter 상태 240개도 보존했습니다. strict validity는 양수 bilinear 탭 전체 inbounds이고 partial source support와 별도입니다.

[요약](SUMMARY.json) · [그림 manifest](IMAGE_MANIFEST.json)

전체 360행과 validity RLE는 복원 가능한 `ALL360_GENERATED_INPUTS.jsonl`에 저장되어 있습니다.

![SOURCE nominal contact 0](F00_C000000_S0_CONTACT.png)

![SOURCE nominal contact 1](F00_C000001_S0_CONTACT.png)

![SOURCE nominal contact 2](F01_C000000_S0_CONTACT.png)

![SOURCE nominal contact 3](F01_C000001_S0_CONTACT.png)

![SOURCE nominal contact 4](F02_C000000_S0_CONTACT.png)

![SOURCE nominal contact 5](F02_C000001_S0_CONTACT.png)

![SOURCE nominal contact 6](F03_C000000_S0_CONTACT.png)

![SOURCE nominal contact 7](F04_C000000_S0_CONTACT.png)

![SOURCE nominal contact 8](F05_C000000_S0_CONTACT.png)

![SOURCE nominal contact 9](F06_C000003_S0_CONTACT.png)

![SOURCE nominal contact 10](F07_C000003_S0_CONTACT.png)

![SOURCE nominal contact 11](F08_C000003_S0_CONTACT.png)

![SOURCE nominal contact 12](F09_C000000_S0_CONTACT.png)

![SOURCE nominal contact 13](F10_C000000_S0_CONTACT.png)

![SOURCE nominal contact 14](F11_C000000_S0_CONTACT.png)

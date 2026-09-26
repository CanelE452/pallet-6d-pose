"""Post-tagging public summary and raw-only reduced contact sheet."""
from collections import Counter
from PIL import Image,ImageDraw
from . import common as C

def enrich(report,summary):
    lock=C.read(C.DOC/'HARD_SELECTION_LOCK.json');C.verify(lock['private_selection'])
    rows=C.read(C.ROOT/lock['private_selection']['path'])['rows']
    initial=[r for r in rows if r['assignment']=='INITIAL']
    sheet=Image.new('RGB',(1600,540),'#15222c');draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(rows):
        C.verify(r['image'])
        with Image.open(C.ROOT/r['image']['path']) as source:
            im=source.convert('RGB');im.thumbnail((312,225))
        x=i%5*320;y=i//5*270
        sheet.paste(im,(x+(320-im.width)//2,y+40+(225-im.height)//2))
        draw.text((x+5,y+3),f'{i+1:02d} {r["assignment"]} {r["tag"]}',fill='white')
        draw.text((x+5,y+19),r['recording']+' / '+r['frame_id'].split(':')[-1],fill='white')
    sheet.save(C.DOC/'figures/03_selected_hard_frames_contact.jpg',quality=88)
    table='\n'.join(f'|{i+1}|{r["assignment"]}|{r["recording"]}|{r["tag"]}|' for i,r in enumerate(rows))
    detail=f'''## 현재 진행: Round1 완료 → 8장 수동 입력 대기

난도 태깅 **123/123 완료**. CLEAN {summary['counts']['CLEAN']}, MODERATE {summary['counts']['MODERATE']}, SEVERE {summary['counts']['SEVERE']}, UNCERTAIN {summary['counts']['UNCERTAIN']}, INVALID {summary['counts']['INVALID']}. Hard **22장 /4 recordings**로 고정 종료 조건을 만족하여 Round2/3은 열지 않는다. INVALID91은 사용자의 판정을 그대로 보존했으며 모델로 재분류하지 않았다.

Initial **8장 = Moderate4 + Severe4 /3 recordings**. Reserve2는 별도로 고정했고 아직 입력 대상이 아니다. 전체10장에서도 recording당 최대3이다. 모델 출력/GT/teacher를 보지 않고 인간 태그·recording·고정 hash만으로 선정했다.

|번호|용도|recording|사람 난도|
|---|---|---|---|
{table}

![선정 원본 축소 모음 — 예측/GT/PnP 없음](figures/03_selected_hard_frames_contact.jpg)

### 지금 입력하는 방법

```bash
python -m scripts.research.pallet_min_hard_ab_v1.annotate_hard
```

1. 앞면/가까운 면 역할을 확신하면 **C**. 모호하면 ‘역할 불확실’ 버튼 후 Enter.
2. 현재 관찰되는 팔레트 범위를 박스로 드래그. 화면 밖·가려진 외곽은 추정하지 않는다.
3. P0..7은 직접 보일 때만 클릭한다. 클릭 즉시 다음 번호. **O** 가림/자체 가림, **X** 화면 밖, **U** 위치 모호는 좌표 없이 다음 점.
4. P7까지 확인 후 **Enter**로 해당 이미지 완료. **Z** 되돌리기, **B** 박스 다시 입력.

작게 보이면 **마우스 휠로 확대**, 오른쪽 버튼 드래그로 이동, **F**로 전체 보기. 확대 상태에서도 원본 좌표로 저장된다.

보이지 않는 점/P8/PnP 보완은 입력하지 않는다. 아직 teacher inference·새 학습·A/B 평가는 하지 않았다. 입력 완료 후 창을 닫고 `python -m scripts.research.pallet_min_hard_ab_v1.cli resume` 또는 대화에 완료를 알린다.

[선정 lock](HARD_SELECTION_LOCK.json) · [태깅 집계](DIFFICULTY_TAG_SUMMARY_PUBLIC.json) · [선정 검증 16항목 PASS](HARD_SELECTION_AUDIT.json)

'''
    if (C.DOC/'ANNOTATION_EDITOR_OVERRIDE.json').exists():
        detail=detail.replace('scripts.research.pallet_min_hard_ab_v1.annotate_hard','scripts.research.pallet_min_hard_ab_v1.open_existing_annotation')
        start=detail.index('1. 앞면/가까운');end=detail.index('[선정 lock]',start)
        detail=detail[:start]+'''사용자 요청으로 전용 Tk 입력기를 닫고 **기존 `scripts/annotate/annotate.py`**로 전환했다. 공유 편집기 소스는 수정하지 않고 이번 프로세스에서만 격리 경로와 직접 클릭 저장을 연결했다.

- 좌클릭: 현재 P0..P7 입력 후 다음 번호.
- 안 보이는 점은 찍지 말고 숫자키로 다음 보이는 번호를 선택.
- **s** 저장 후 다음 이미지. 두 점만 입력해도 PnP 없이 저장 가능.
- **z** 되돌리기 / **d** 현재 점 삭제 / **+,-** 확대·축소 / **q** 종료.

이번 pass는 직접 클릭만 저장한다. PnP 표시·자동채움·외삽·P8은 비활성화했고, 기존 GT/모델 출력은 불러오지 않는다. 정확한 xy는 private 폴더에만 저장하며 기존 입력 기록도 보존했다. **역할/가시성/수동 bbox 확인은 이후 별도 단계로 남아 있고, 클릭 저장만으로 최종 학습 label lock을 만들지 않는다.** 새 학습은 아직 없다.

'''+detail[end:]
        report=report.replace('scripts.research.pallet_min_hard_ab_v1.annotate_hard','scripts.research.pallet_min_hard_ab_v1.open_existing_annotation')
    report=report.replace('현재는 Phase1–2 준비 단계이며','현재는 Phase3 선정 완료 / Phase4 수동 입력 대기 단계이며')
    report=report.replace('### 지금 하는 조작','### 완료된 난도 태깅 조작 (이력용)')
    report=report.replace('지금은 **키포인트/박스를 찍지 않는다**. 난도만 입력한다. 첫 라운드 완료 후:',
                          '위 설명은 완료된 난도 태깅의 이력이다. 지금은 상단 안내에 따라 8장 수동 입력을 진행한다. 입력 후:')
    report=report.replace('지금 사용자는 첫 라운드 난도 태깅만 하면 된다.','난도 태깅은 종료했고 지금은 선정된 8장만 수동 입력하면 된다.')
    report=report.replace('사람 태그/실제 annotation/학습/평가 테스트는 NOT_RUN이다.','기존 준비 감사 시점에는 사람 태그가 없었다. 현재 난도 태깅은 검증·고정됐으며 실제 corner annotation/학습/평가는 아직 NOT_RUN이다.')
    return report.replace('## 1. 한 줄 결론',detail+'## 1. 한 줄 결론',1)

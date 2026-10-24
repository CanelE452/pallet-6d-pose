"""Display-only contact sheet from four already fixed, saved native-query crops."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

V = Path(__file__).resolve().parents[1]
d = json.loads((V/'visual_followup/C16_FIXED4_MAPPING_AND_METRICS.json').read_text())
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
def font(size):
    return ImageFont.truetype(font_path, size)

im = Image.new('RGB', (1200, 900), '#eef3f6')
draw = ImageDraw.Draw(im)
draw.text((28,20), '실제 crop에서 무엇을 맞히고 놓치는가', font=font(32), fill='#183044')
draw.text((28,70), '현재 C16  |  보임 1,137/1,147  ·  외부 가림 35/90', font=font(24), fill='#365b74')
texts = [
    ('보임 성공', ['사람: 보임 → 모델: 보임', '테두리·모서리 구조가', '노출된 경우'], '#216a50'),
    ('외부 가림 성공', ['사람: 가림 → 모델: 가림', '콘이 질의 부근을', '차지하는 경우'], '#216a50'),
    ('외부 가림 실패', ['사람: 가림 → 모델: 보임', '콘 옆에 목재가 남은 경계', '기대한 단서와 주변 재질의', '혼동 가능성 — 원인 가설'], '#aa3934'),
    ('과거 실패 → 현재 성공', ['사람: 가림 → 모델: 가림', '콘이 중심을 크게 차지', '콘을 전혀 못 알아보는', '상태는 아님'], '#216a50'),
]
for i, (c, (title, lines, color)) in enumerate(zip(d['cases'], texts)):
    x, y = 20 + (i%2)*600, 125 + (i//2)*345
    draw.rounded_rectangle((x,y,x+580,y+325), radius=10, fill='white')
    draw.rectangle((x,y,x+580,y+6), fill=color)
    draw.text((x+18,y+18), title, font=font(26), fill=color)
    crop = Image.open(c['raw_RGB96']['path']).convert('RGB')
    assert crop.size == (96,96)
    im.paste(crop.resize((216,216), Image.Resampling.NEAREST), (x+18,y+67))
    draw.text((x+18,y+287), '저장 원 crop · 무표시', font=font(17), fill='#4d667a')
    for j, line in enumerate(lines):
        draw.text((x+252,y+75+j*37), line, font=font(19), fill='#183044')
    draw.text((x+252,y+262), f"고정 행 {c['original_native_row']}", font=font(18), fill='#637a8b')
draw.text((28,833), 'crop 중앙은 초기 예측점. 같은 q·크기 유지, 위치 교정 없음. 작은 crop 외 문맥도 모델 입력에 사용.', font=font(17), fill='#3c5c72')
draw.text((28,863), '네 사례는 진단 예시. 전체 외부 가림 90개의 성공·실패 crop은 연결된 보고서에 수록.', font=font(17), fill='#3c5c72')
out = V/'visual_followup/INLINE_SUCCESS_FAILURE_C16_KO.png'
im.save(out)
print(out)

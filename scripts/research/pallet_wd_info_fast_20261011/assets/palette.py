"""viz-expert 색 단일 진실원천.

색을 손으로 타이핑하는 순간 그림마다 달라진다. 모든 색 결정은 여기를 거친다.

    import sys; sys.path.insert(0, str(Path.home() / '.claude/agents/viz-expert/assets'))
    from palette import color_for, SEMANTIC, CMAP, CV

핵심은 `color_for(label)` 이다. cycle 순서만으로는 계열 색이 고정되지 않는다.
"baseline, ours" 를 그리면 ours 가 하늘색, "ours, baseline, variant" 를 그리면
ours 가 주황이 된다 — 같은 대상인데 그림마다 색이 바뀐다. 이름에 색을 묶어야 한다.

SERIES 순서는 ieee.mplstyle / analysis.mplstyle 의 axes.prop_cycle 과 같아야 한다.
verify() 가 실제 파일을 읽어 대조한다.
"""

import hashlib
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# ── 계열 색 — Okabe-Ito (색각 이상 안전, Nature Methods 권장) ────────────────
# 흰 배경에서 대비가 약한 노랑(F0E442)은 뒤로 뺐다.
SERIES = [
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#009E73',  # bluish green
    '#0072B2',  # blue
    '#D55E00',  # vermilion
    '#CC79A7',  # reddish purple
    '#F0E442',  # yellow
    '#000000',  # black
]

# 이름 해시로 색을 고를 때 쓰는 후보. 노랑(흰 배경에서 저대비)과
# 검정(GT 전용으로 예약)은 뺀다.
_FALLBACK = SERIES[:6]

# 자주 쓰는 이름은 색을 고정한다. 같은 대상이 늘 같은 색으로 나오게.
# 새 이름을 추가할 때 기존 항목의 색을 바꾸지 않는다 — 과거 그림과 안 맞게 된다.
REGISTRY = {
    'ours':      '#D55E00',  # vermilion — 우리 방법은 항상 이 색
    'baseline':  '#0072B2',  # blue
    'gt':        '#000000',  # ground truth 는 검정
    'pred':      '#E69F00',  # orange
    'train':     '#0072B2',
    'val':       '#E69F00',
    'test':      '#009E73',
    'before':    '#56B4E9',
    'after':     '#D55E00',
}

# ── 의미가 정해진 색 — 뜻이 있으니 바꾸지 않는다 ──────────────────────────────
SEMANTIC = {
    'mean':      '0.4',       # 실선
    'median':    '0.4',       # 파선
    'threshold': '#D55E00',   # 점선. 라벨 필수
    'baseline':  '0.6',       # 쇄선
    'zero':      '0.5',
    'diagonal':  '0.3',       # y = x
    'failure':   '#D55E00',   # 실패 구간 음영 (alpha 0.20)
    'annotation':'0.25',      # 화살표·주석 텍스트
}

LINESTYLE = {
    'mean': '-', 'median': '--', 'threshold': ':', 'baseline': '-.',
    'zero': '-', 'diagonal': '-',
}

ALPHA = {'std': 0.15, 'ci': 0.25, 'range': 0.10, 'failure': 0.20, 'overlay': 0.5}

# ── 컬러맵 — 종류별로 하나씩만 ────────────────────────────────────────────────
CMAP = {
    'sequential': 'viridis',   # 두 mplstyle 의 image.cmap 기본값과 동일
    'overlay':    'magma',     # attention / Grad-CAM / saliency 오버레이
    'diverging':  'coolwarm',  # 잔차·차이. vmin=-vmax 로 0 을 중심에 고정할 것
    'ordinal':    'cividis',   # 순서형 카테고리(거리·각도 구간)
}

# ── CV 오버레이 ──────────────────────────────────────────────────────────────
CV = {
    'bbox_pred': '#D55E00',
    'bbox_gt':   '#009E73',
    'keypoint':  '#E69F00',
    'skeleton':  '#56B4E9',
    'occluded':  '#CC79A7',   # 가려진 키포인트
    'text':      '#FFFFFF',
    'text_bg':   '#000000',
}


def color_for(label: str) -> str:
    """계열 이름에 색을 고정한다. 그리는 순서가 바뀌어도 색은 그대로.

    REGISTRY 에 있으면 그 색, 없으면 이름 해시로 SERIES 에서 고른다.
    md5 를 쓰는 이유: 파이썬 내장 hash() 는 프로세스마다 값이 달라져
    (PYTHONHASHSEED) 다음에 돌리면 색이 바뀐다.
    """
    key = label.strip().lower()
    if key in REGISTRY:
        return REGISTRY[key]
    digest = hashlib.md5(key.encode('utf-8')).digest()
    return _FALLBACK[int.from_bytes(digest[:4], 'big') % len(_FALLBACK)]


def colors_for(labels) -> list:
    """여러 계열을 한 번에. 해시 충돌이 나면 뒤 계열을 다음 빈 색으로 밀어낸다.

    한 그림 안에서 두 계열이 같은 색이 되는 것만 막고, 색 배정 자체는
    이름에 묶인 채로 둔다.
    """
    out, used = [], set()
    for lb in labels:
        c = color_for(lb)
        if c in used:
            c = next((s for s in _FALLBACK + SERIES if s not in used), c)
        out.append(c)
        used.add(c)
    return out


def verify() -> bool:
    """SERIES 가 두 mplstyle 의 prop_cycle 과 실제로 같은지 대조한다."""
    ok = True
    want = [c.lower().lstrip('#') for c in SERIES]
    for name in ('ieee.mplstyle', 'analysis.mplstyle'):
        line = next((l for l in (_HERE / name).read_text().splitlines()
                     if l.startswith('axes.prop_cycle')), None)
        got = [t.strip().strip("'\"").lower().lstrip('#')
               for t in line.split('[', 1)[1].rsplit(']', 1)[0].split(',')] if line else []
        if got != want:
            print(f"MISMATCH {name}\n  style : {got}\n  palette: {want}")
            ok = False
    return ok


if __name__ == '__main__':
    print('prop_cycle 일치:', verify())
    names = ['ours', 'baseline', 'gt', 'method-X', 'method-Y', 'method-Z']
    for lb, c in zip(names, colors_for(names)):
        print(f'  {lb:<12} {color_for(lb)}   in-figure: {c}')

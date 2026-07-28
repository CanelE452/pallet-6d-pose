# calib/hud.py
# HUD 텍스트 패널 + 진행바( rel_yaw / 90°, FWD 잔여시간 )
# - 변경점:
#   1) draw_panel(...)에 cmd_status 인자(선택)를 받아, 현재 명령/진행도 텍스트 2줄을 추가(기존 동일).
#   2) cmd_status를 기반으로 상단 패널 하단에 "그래픽 진행바"를 렌더링.
#      - UNTIL 모드(|rel_yaw| 등): 현재값/목표값 비율로 진행바 표시 (예: rel_yaw/90°).
#      - TIMED 모드(FWD 잔여시간): (경과/총시간) 비율로 진행바 표시.
#   3) API 불일치에도 안전하게 동작하도록 try/except 및 특성 존재 여부를 점검.

import os
import cv2
import numpy as np
from typing import Optional, List, Tuple
from PIL import Image, ImageDraw, ImageFont
from .config import COLOR_PANEL_BG, COLOR_PANEL_EDGE

# (선택) CommandStatus 타입 힌트용: 순환 import 방지 위해 런타임 참조만 사용
try:
    from .command_status import CommandStatus  # 존재하지 않아도 런타임 문제 없도록 try
except Exception:
    CommandStatus = None  # type: ignore

FONT_PATH_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothicCoding.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansKR-Regular.otf",
    "/Library/Fonts/AppleGothic.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/NanumGothic.ttf",
]

def _pick_font_path():
    for p in FONT_PATH_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None

_FONT_PATH = _pick_font_path()

def _to_rgb(color_bgr):
    b, g, r = color_bgr
    return (r, g, b)

def _get_font(font_scale: float):
    base_px = int(round(24 * font_scale))
    base_px = max(base_px, 12)
    if _FONT_PATH:
        try:
            return ImageFont.truetype(_FONT_PATH, base_px)
        except:
            pass
    return ImageFont.load_default()

def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont):
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return w, h

def _append_command_lines(
    lines: List[Tuple[str, Tuple[int, int, int]]],
    cmd_status: "Optional[CommandStatus]"
) -> List[Tuple[str, Tuple[int, int, int]]]:
    """
    기존 lines에 [현재 명령], 진행도 2줄을 덧붙여 반환.
    cmd_status가 없으면 원본을 그대로 반환.
    """
    if cmd_status is None:
        return lines

    # 기본 색상: 본문 흰색, 보조 회색 (OpenCV BGR)
    WHITE = (255, 255, 255)
    GREY = (200, 200, 200)

    label = getattr(cmd_status, "label", None) or "-"
    progress_text = ""
    # 안전하게 progress_text() 호출
    try:
        progress_text = cmd_status.progress_text()
    except Exception:
        progress_text = ""

    augmented = list(lines)
    augmented.append((f"[현재 명령] {label}", WHITE))
    if progress_text:
        augmented.append((f"            {progress_text}", GREY))
    else:
        augmented.append(("            ", GREY))
    return augmented

def _extract_progress(cmd_status: "Optional[CommandStatus]"):
    """
    진행바 비율과 캡션을 안전하게 추출.
    반환: (ratio[0~1], caption:str) 또는 (None, None)
    - UNTIL: current/target (예: |rel_yaw|/90°)
    - TIMED: (t_total - t_remain)/t_total (예: FWD 잔여시간)
    """
    if cmd_status is None:
        return None, None

    # 1) 라이브러리에 progress_ratio()가 있으면 우선 사용
    try:
        if hasattr(cmd_status, "progress_ratio"):
            r = cmd_status.progress_ratio()
            if r is not None:
                r = float(max(0.0, min(1.0, r)))
                # 적당한 캡션 추론
                mode = getattr(cmd_status, "mode", "")
                if mode == "until":
                    metric = getattr(cmd_status, "metric_name", "") or ""
                    tgt = getattr(cmd_status, "target_value", None)
                    if tgt:
                        return r, f"{metric} {r*100:.0f}%"
                    return r, f"{metric} 진행 {r*100:.0f}%"
                elif mode == "timed":
                    return r, f"FWD 진행 {r*100:.0f}%"
                else:
                    return r, f"진행 {r*100:.0f}%"
    except Exception:
        pass

    # 2) 수동 계산 (속성 조합으로 복원)
    try:
        mode = getattr(cmd_status, "mode", "")
        if mode == "until":
            cur = getattr(cmd_status, "current_value", None)
            tgt = getattr(cmd_status, "target_value", None)
            metric = getattr(cmd_status, "metric_name", "") or ""
            if cur is not None and tgt and tgt > 0:
                r = float(max(0.0, min(1.0, cur / tgt)))
                return r, f"{metric} {cur:.1f}/{tgt:.1f} ({r*100:.0f}%)"
        elif mode == "timed":
            # 가능한 속성 후보: t_total, t_remain, t_elapsed
            t_total  = getattr(cmd_status, "t_total", None)
            t_remain = getattr(cmd_status, "t_remain", None)
            t_elapsed = getattr(cmd_status, "t_elapsed", None)

            if t_total and t_total > 0:
                if t_elapsed is not None:
                    r = float(max(0.0, min(1.0, t_elapsed / t_total)))
                    return r, f"FWD {t_total - t_elapsed:.1f}s 남음 ({r*100:.0f}%)"
                if t_remain is not None:
                    r = float(max(0.0, min(1.0, (t_total - max(0.0, t_remain)) / t_total)))
                    return r, f"FWD {t_remain:.1f}s 남음 ({r*100:.0f}%)"
            # t_total이 없으면 ratio를 알 수 없으므로 캡션만 구성
            if t_remain is not None:
                return None, f"FWD {t_remain:.1f}s 남음"
    except Exception:
        pass

    return None, None

def _draw_progress_bar(draw: ImageDraw.ImageDraw, x, y, w, h, ratio: float,
                       bg=(60, 60, 60), fg=(40, 180, 255), edge=(95, 95, 95)):
    """간단한 수평 진행바(0~1)를 PIL로 렌더."""
    # 배경 + 테두리
    draw.rectangle((x, y, x+w, y+h), fill=bg, outline=edge, width=1)
    # 전경
    fill_w = int(max(0, min(w, round(w * ratio))))
    if fill_w > 0:
        draw.rectangle((x, y, x+fill_w, y+h), fill=fg)

def draw_panel(
    img_bgr,
    lines: List[Tuple[str, Tuple[int, int, int]]],
    origin=(18, 18),
    pad=(12, 10),
    line_h=26,
    font_scale=0.6,
    thickness=2,
    cmd_status: Optional["CommandStatus"] = None
):
    """
    좌측 HUD 텍스트 패널을 렌더링합니다.
    - lines: [(문자열, BGR색상), ...] 형식
    - cmd_status: 전달 시, '[현재 명령]'과 진행도 2줄이 자동으로 추가되며,
                  패널 하단에 그래픽 진행바가 표시됩니다.
    """
    # 필요 시 현재 명령/진행도 라인 추가
    lines_to_draw = _append_command_lines(lines, cmd_status)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(font_scale)
    x0, y0 = origin
    pad_x, pad_y = pad

    width = 0
    for txt, _ in lines_to_draw:
        tw, th = _text_size(draw, txt, font)
        width = max(width, tw)
    height_text = line_h * len(lines_to_draw)

    # 진행바 정보 추출
    ratio, cap = _extract_progress(cmd_status)
    use_bar = (ratio is not None) or (cap is not None)

    # 진행바 레이아웃(있을 때만 바닥쪽에 1개 렌더)
    bar_h = 12
    bar_gap_top = 6
    extra_h = (bar_gap_top + bar_h + 2*pad_y) if use_bar else 0

    rect0 = (x0 - pad_x, y0 - pad_y, x0 + width + pad_x, y0 + height_text + pad_y + extra_h)
    draw.rectangle(rect0, fill=_to_rgb(COLOR_PANEL_BG), outline=_to_rgb(COLOR_PANEL_EDGE), width=1)

    # 텍스트 그리기
    y = y0
    for txt, color_bgr in lines_to_draw:
        draw.text((x0, y), txt, font=font, fill=_to_rgb(color_bgr))
        y += line_h

    # 진행바 렌더
    if use_bar:
        # 캡션
        cap_text = cap or ""
        if cap_text:
            tw, th = _text_size(draw, cap_text, font)
            draw.text((x0, y + bar_gap_top - th - 2), cap_text, font=font, fill=(230, 230, 230))
        # 바
        bar_x = x0
        bar_y = y + bar_gap_top
        bar_w = width
        if ratio is None:
            # 비율을 모를 때는 루프가 그려지지 않도록 0으로 두되, 배경만 표시
            ratio = 0.0
        _draw_progress_bar(draw, bar_x, bar_y, bar_w, bar_h, ratio)

    out_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img_bgr[:] = out_bgr

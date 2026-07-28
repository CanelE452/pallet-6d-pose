# ui/diagram.py
# FSM 다이어그램 패널 + 상단 우측 "미니 진행바" (rel_yaw / 90°, FWD 잔여시간)
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ===== 스타일 상수 =====
PANEL_BG = (25, 25, 25)
GROUP_BG = (35, 35, 35)
GROUP_BORDER = (70, 70, 70)
GROUP_ACTIVE = (60, 200, 255)

TEXT_DIM   = (165, 165, 165)
TEXT_NORM  = (230, 230, 230)
TEXT_TITLE = (210, 210, 210)

NODE_W, NODE_H = 148, 30
LEVEL_GAP      = 24
ROW_GAP        = 18
COL_GAP        = 18

PANEL_MARGIN_LR = 16
PANEL_TITLE_H   = 40
GROUP_PAD       = 14
EDGE_COLOR      = (110, 110, 110)

# ===== 진행바 유틸 =====
def _bar(img, x, y, w, h, ratio, bg=(60,60,60), fg=(40,180,255), edge=(95,95,95)):
    cv2.rectangle(img, (x, y), (x+w, y+h), bg, -1, lineType=cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x+w, y+h), edge, 1, lineType=cv2.LINE_AA)
    fill = max(0, min(w, int(round(w * ratio))))
    if fill > 0:
        cv2.rectangle(img, (x, y), (x+fill, y+h), fg, -1, lineType=cv2.LINE_AA)

def _progress_from_cmdstatus(cmd_status):
    """
    (ratio, caption) 또는 (None, None)
    - UNTIL: rel_yaw/90° 등 (metric_name 사용)
    - TIMED: FWD 잔여시간
    """
    if cmd_status is None:
        return None, None
    # 1) progress_ratio() 우선
    try:
        if hasattr(cmd_status, "progress_ratio"):
            r = cmd_status.progress_ratio()
            if r is not None:
                r = float(max(0.0, min(1.0, r)))
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
    # 2) 속성으로 복원
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
            if t_remain is not None:
                return None, f"FWD {t_remain:.1f}s 남음"
    except Exception:
        pass
    return None, None

def _put_center(img, text, cx, cy, color, scale=0.48, thick=1):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    cv2.putText(img, text, (int(cx - tw/2), int(cy + th/2 - 4)),
                FONT, scale, color, thick, cv2.LINE_AA)

def _draw_box(img, x, y, w, h, color, fill=None, thick=2):
    if fill is not None:
        cv2.rectangle(img, (x, y), (x+w, y+h), fill, -1, lineType=cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x+w, y+h), color, thick, lineType=cv2.LINE_AA)

def _draw_node(img, name, x, y, active=False, dim=False):
    fill = (60, 60, 60) if dim else (42, 42, 42)
    border = GROUP_ACTIVE if active else (95, 95, 95)
    _draw_box(img, x, y, NODE_W, NODE_H, border, fill=fill, thick=2)
    _put_center(img, name, x + NODE_W//2, y + NODE_H//2,
                TEXT_NORM if not dim else TEXT_DIM, scale=0.50, thick=1)

def _draw_group(img, x, y, w, h, title, active=False):
    _draw_box(img, x, y, w, h, GROUP_ACTIVE if active else GROUP_BORDER,
              fill=GROUP_BG, thick=2)
    _put_center(img, title, x + w//2, y + 18, TEXT_TITLE, scale=0.62, thick=2)

def _arrow(img, p1, p2):
    cv2.arrowedLine(img, p1, p2, EDGE_COLOR, 1, tipLength=0.03, line_type=cv2.LINE_AA)

# ===== 계층형 레이아웃 유틸 =====
def _induced_edges(edges, allowed):
    allowed = set(allowed)
    return [(u, v) for (u, v) in edges if u in allowed and v in allowed]

def _entry_candidates(nodes, edges):
    nodes = set(nodes)
    parents = {v for (u, v) in edges if u in nodes and v in nodes}
    roots = list(nodes - parents)
    if roots:
        return roots
    indeg = {n:0 for n in nodes}
    for u, v in edges:
        if v in nodes and u in nodes:
            indeg[v] += 1
    return [min(nodes, key=lambda n: indeg.get(n, 0))]

def _levels_by_bfs(nodes, edges, entries):
    nodes = list(nodes)
    E = {n: [] for n in nodes}
    for u, v in edges: E.setdefault(u, []).append(v)

    from collections import deque
    level_of = {n: None for n in nodes}
    q = deque()
    for s in entries:
        if s in level_of and level_of[s] is None:
            level_of[s] = 0; q.append(s)
    if not q:
        n = nodes[0]; level_of[n] = 0; q.append(n)

    while q:
        u = q.popleft()
        for v in E.get(u, []):
            if v in level_of and level_of[v] is None:
                level_of[v] = level_of[u] + 1
                q.append(v)

    max_lv = max([lv for lv in level_of.values() if lv is not None] or [0])
    for n in nodes:
        if level_of[n] is None:
            max_lv += 1
            level_of[n] = max_lv

    by_level = {}
    for n, lv in level_of.items():
        by_level.setdefault(lv, []).append(n)
    for lv in by_level:
        by_level[lv].sort()
    return by_level, level_of

def _wrap_row(items, inner_w):
    max_cols = max(1, (inner_w + COL_GAP) // (NODE_W + COL_GAP))
    rows = []
    if max_cols <= 0:
        max_cols = 1
    for i in range(0, len(items), max_cols):
        rows.append(items[i:i+max_cols])
    return rows

def _row_start_x(inner_x, inner_w, cols):
    total_w = cols*NODE_W + (cols-1)*COL_GAP
    return inner_x + max(0, (inner_w - total_w)//2)

def _layout_hier(inner_x, start_y, inner_w, nodes, edges, entries):
    by_level, _ = _levels_by_bfs(nodes, edges, entries)
    pos = {}
    y = start_y

    for lv in sorted(by_level.keys()):
        names = by_level[lv]
        rows = _wrap_row(names, inner_w)

        for r, row in enumerate(rows):
            cols = len(row)
            sx = _row_start_x(inner_x, inner_w, cols)
            for i, name in enumerate(row):
                x = sx + i*(NODE_W + COL_GAP)
                pos[name] = (x, y)
            if r < len(rows)-1:
                y += NODE_H + ROW_GAP
        y += NODE_H + LEVEL_GAP

    return pos, y

def _layout_by_levels(inner_x, start_y, inner_w, levels):
    pos = {}
    y = start_y
    for row in levels:
        if not row:
            continue
        cols = len(row)
        sx = _row_start_x(inner_x, inner_w, cols)
        for i, name in enumerate(row):
            x = sx + i*(NODE_W + COL_GAP)
            pos[name] = (x, y)
        y += NODE_H + LEVEL_GAP
    return pos, y

def draw_fsm_diagram_panel(fsm, panel_size=(820, 1200), col_weights=(0.28, 0.72)):
    """
    새 다이어그램 (2026-05-27 snapshot 모델) 시각화:
      흐름:  SEARCH → ALIGN (composite) → INSERT → DONE
      ALIGN: YAW_CHECK ↔ YAW_CORRECT_R/L → OFFSET_CHECK → DIST_CHECK
             → OFFSET_NEED_CORRECT 분기 → LATERAL_ROTATE_R/L
             → FORWARD_AFTER_R/L → LATERAL_ROTATE_L/R_BACK → YAW_CHECK
             → READY_TO_DONE → INSERT (top-level)

    mermaid stateDiagram 과 동일 — OUTER 그룹 박스 없음, ALIGN 만 composite 박스.
    RECOVER / DETECTED / CHECK / HOLD 모두 제거.
    """
    H, W = panel_size
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = PANEL_BG

    # 상단 타이틀
    _put_center(img, "Calibration FSM (Snapshot Model)", W//2, 22,
                (210,210,210), scale=0.70, thick=2)

    # ===== 상단 우측: 미니 진행바 =====
    cmd_status = getattr(fsm, "cmd_status", None)
    ratio, cap = _progress_from_cmdstatus(cmd_status)
    mini_w = 320
    mini_x = W - PANEL_MARGIN_LR - mini_w
    mini_y = 8
    cap_text = cap or "진행 정보 없음"
    cv2.putText(img, cap_text, (mini_x, mini_y + 12), FONT, 0.45, (230,230,230), 1, cv2.LINE_AA)
    _bar(img, mini_x, mini_y + 18, mini_w, 10, 0.0 if ratio is None else float(max(0.0, min(1.0, ratio))))

    # ===== 노드 정의 — align.py 의 실제 self.sub 이름과 동기화 =====
    # top-level (그룹 박스 없이 독립 노드)
    top_level_nodes = ["SEARCH", "INSERT", "DONE"]

    # ALIGN composite 내부
    ALIGN_CHECKS       = ["YAW_CHECK", "OFFSET_CHECK", "DIST_CHECK", "READY_TO_DONE"]
    ALIGN_YAW_CORRECT  = ["YAW_CORRECT_RIGHT", "YAW_CORRECT_LEFT"]
    ALIGN_DIST_ADJUST  = ["ALIGN_FWD_ADJUST", "ALIGN_BWD_ADJUST"]
    ALIGN_NEED_CORRECT = ["OFFSET_NEED_CORRECT"]   # DIST_CHECK 의 |d_lat|>tol 분기 (mermaid 시각화용)
    ALIGN_RIGHT_BRANCH = ["LATERAL_ROTATE_RIGHT", "FORWARD_AFTER_RIGHT", "LATERAL_ROTATE_LEFT_BACK"]
    ALIGN_LEFT_BRANCH  = ["LATERAL_ROTATE_LEFT",  "FORWARD_AFTER_LEFT",  "LATERAL_ROTATE_RIGHT_BACK"]

    align_nodes = (ALIGN_CHECKS + ALIGN_YAW_CORRECT + ALIGN_DIST_ADJUST
                   + ALIGN_NEED_CORRECT + ALIGN_RIGHT_BRANCH + ALIGN_LEFT_BRANCH)

    # 활성 플래그
    top = getattr(fsm, "state", "SEARCH")
    align_sub = getattr(fsm, "align_sub", None) or getattr(getattr(fsm, "align", None), "sub", None)
    active_align = top in ("ALIGN", "INSERT")   # INSERT 상태도 align.sub 사용

    # ===== 엣지 정의 (mermaid conform) =====
    EDGES = [
        # top-level 흐름
        ("SEARCH", "YAW_CHECK"),        # SEARCH → ALIGN 진입 (YAW_CHECK 부터)
        ("READY_TO_DONE", "INSERT"),    # ALIGN composite 탈출 → top-level INSERT
        ("INSERT", "DONE"),

        # YAW_CHECK → 3 branches
        ("YAW_CHECK", "YAW_CORRECT_RIGHT"),
        ("YAW_CHECK", "YAW_CORRECT_LEFT"),
        ("YAW_CHECK", "READY_TO_DONE"),

        # YAW_CORRECT → OFFSET_CHECK
        ("YAW_CORRECT_RIGHT", "OFFSET_CHECK"),
        ("YAW_CORRECT_LEFT",  "OFFSET_CHECK"),

        # OFFSET_CHECK → DIST_CHECK
        ("OFFSET_CHECK", "DIST_CHECK"),

        # DIST_CHECK → 4 branches (mermaid: |d_lat|>tol 면 OFFSET_NEED_CORRECT 거쳐서 LATERAL)
        ("DIST_CHECK", "ALIGN_FWD_ADJUST"),
        ("DIST_CHECK", "ALIGN_BWD_ADJUST"),
        ("DIST_CHECK", "OFFSET_NEED_CORRECT"),
        ("DIST_CHECK", "YAW_CHECK"),
        # OFFSET_NEED_CORRECT → LATERAL chain 좌/우 분기
        ("OFFSET_NEED_CORRECT", "LATERAL_ROTATE_RIGHT"),
        ("OFFSET_NEED_CORRECT", "LATERAL_ROTATE_LEFT"),

        # ALIGN_*_ADJUST → DIST_CHECK 복귀
        ("ALIGN_FWD_ADJUST", "DIST_CHECK"),
        ("ALIGN_BWD_ADJUST", "DIST_CHECK"),

        # LATERAL 우측 chain
        ("LATERAL_ROTATE_RIGHT", "FORWARD_AFTER_RIGHT"),
        ("FORWARD_AFTER_RIGHT",  "LATERAL_ROTATE_LEFT_BACK"),
        ("LATERAL_ROTATE_LEFT_BACK", "YAW_CHECK"),

        # LATERAL 좌측 chain
        ("LATERAL_ROTATE_LEFT", "FORWARD_AFTER_LEFT"),
        ("FORWARD_AFTER_LEFT",  "LATERAL_ROTATE_RIGHT_BACK"),
        ("LATERAL_ROTATE_RIGHT_BACK", "YAW_CHECK"),
    ]

    # ===== 레이아웃 =====
    # 왼쪽 column 에 SEARCH (위), INSERT (중), DONE (아래) 독립 노드.
    # 가운데~오른쪽에 ALIGN composite 박스.
    inner_left_x  = PANEL_MARGIN_LR
    inner_left_w  = int((W - 2 * PANEL_MARGIN_LR - COL_GAP) * col_weights[0])
    inner_right_x = inner_left_x + inner_left_w + COL_GAP
    inner_right_w = (W - 2 * PANEL_MARGIN_LR) - inner_left_w - COL_GAP
    top_y = PANEL_TITLE_H

    # ALIGN composite 박스 안쪽 좌표
    align_box_x = inner_right_x
    align_inner_x = align_box_x + GROUP_PAD
    align_inner_w = inner_right_w - 2 * GROUP_PAD
    align_inner_start_y = top_y + GROUP_PAD + 18

    align_levels_ordered = [
        ["YAW_CHECK"],
        ["YAW_CORRECT_RIGHT", "YAW_CORRECT_LEFT"],
        ["OFFSET_CHECK"],
        ["DIST_CHECK"],
        ["ALIGN_FWD_ADJUST", "ALIGN_BWD_ADJUST"],
        ["OFFSET_NEED_CORRECT"],
        ["LATERAL_ROTATE_RIGHT", "LATERAL_ROTATE_LEFT"],
        ["FORWARD_AFTER_RIGHT", "FORWARD_AFTER_LEFT"],
        ["LATERAL_ROTATE_LEFT_BACK", "LATERAL_ROTATE_RIGHT_BACK"],
        ["READY_TO_DONE"],
    ]
    flat = [n for row in align_levels_ordered for n in row]
    remain = [n for n in align_nodes if n not in flat]
    if remain:
        align_levels_ordered.append(remain)

    pos_align, y_align_last = _layout_by_levels(align_inner_x, align_inner_start_y,
                                                align_inner_w, align_levels_ordered)
    align_box_y_start = top_y
    align_box_y_end   = y_align_last + GROUP_PAD

    # 왼쪽 column 의 top-level 노드 — ALIGN 박스 높이에 맞춰 균등 배치
    align_box_h = align_box_y_end - align_box_y_start
    cx_left = inner_left_x + inner_left_w // 2 - NODE_W // 2
    n_top = len(top_level_nodes)
    if n_top >= 2:
        spacing = (align_box_h - NODE_H) // (n_top - 1)
    else:
        spacing = 0
    pos_top = {}
    for i, name in enumerate(top_level_nodes):
        y = align_box_y_start + i * spacing
        pos_top[name] = (cx_left, y)

    # 패널 세로 크기 자동 확장
    needed_H = align_box_y_end + GROUP_PAD
    if needed_H > H:
        pad = needed_H - H
        pad_img = np.zeros((pad, W, 3), dtype=np.uint8); pad_img[:] = PANEL_BG
        img = np.vstack([img, pad_img])
        H = needed_H

    # ALIGN composite 박스만 그림 (OUTER 박스 없음)
    _draw_group(img, align_box_x, align_box_y_start, inner_right_w, align_box_h, "ALIGN",
                active=active_align)

    # 포지션 병합
    POS = {}
    POS.update(pos_top)
    POS.update(pos_align)

    # 엣지 그리기 (센터-센터)
    def C(name):
        x, y = POS[name]
        return (x + NODE_W//2, y + NODE_H//2)

    for u, v in EDGES:
        if u in POS and v in POS:
            _arrow(img, C(u), C(v))

    # 노드 그리기 (활성/디밍)
    for name, (x, y) in POS.items():
        active = False
        dim = False

        if name in top_level_nodes:
            active = (top == name)
        elif name in align_nodes:
            if active_align and (align_sub == name):
                active = True
            elif not active_align:
                dim = True

        _draw_node(img, name, x, y, active=active, dim=dim)

    # 하단 상태 표시
    status = f"{top}"
    if top in ("ALIGN", "INSERT") and align_sub:
        status += f" • {align_sub}"
    cv2.putText(img, status, (16, H - 14), FONT, 0.50, (225, 225, 225), 1, cv2.LINE_AA)

    return img

"""Result table figure (publication style) — Phase 1 + Phase 2 종합 정리."""
import os as _os, sys as _sys

# --- data_prep 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 먼저 실행돼야 하므로 최상단에 둔다.
_DP = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_DP] + [_os.path.join(_DP, _d) for _d in sorted(_os.listdir(_DP))
                         if _os.path.isdir(_os.path.join(_DP, _d)) and not _d.startswith(".")]

import os
import matplotlib.pyplot as plt
import numpy as np


def main():
    headers = ["Model", "PL #", "indoor", "outside", "night", "Note"]
    rows = [
        # Phase 1 (Baseline base)
        ["Baseline base (synthetic only)",      "-",     "21.6", "27.9", "21.1", ""],
        ["+ R1 (indoor PL, F5)",               "2",      "60.5", "31.8", "32.2", "R1 self best (indoor)"],
        ["+ R1 (outside PL)",                  "167",   "58.4", "39.5", "33.3", "R1 self best (outside, night)"],
        ["+ R1 (night PL)",                    "105",   "43.9", "22.5", "26.7", ""],
        ["+ R2 (outside PL)",                  "514",    "15.9", "24.8", "11.1", "over-iteration ↓"],
        ["+ R3 (outside PL)",                  "900",    "4.8",  "21.7", "5.6",  "catastrophic decay ↓↓"],
        # Phase 2 (Proposed base)
        ["Proposed base (camera-facing)",      "-",     "63.6", "52.7", "58.9", ""],
        ["+ R1 (indoor strict PL)",            "19",     "77.0", "48.1", "50.0", "★ best indoor"],
        ["+ R1 (outside strict PL)",           "48",     "67.0", "54.3", "48.9", "★ best outside"],
        ["+ R1 (night strict PL)",             "41",     "49.1", "53.5", "60.0", "★ best night"],
    ]

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.axis("off")

    # 색상: phase 1 (light gray), phase 2 baseline (light blue), phase 2 R1 (light green)
    colors = ["#f0f0f0"] * 6 + ["#bbdefb"] + ["#c8e6c9"] * 3

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colWidths=[0.34, 0.07, 0.09, 0.09, 0.09, 0.32],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.7)

    # header style
    for j in range(len(headers)):
        cell = table[(0, j)]
        cell.set_facecolor("#37474f")
        cell.set_text_props(color="white", fontweight="bold")

    # row colors + bold for best cells
    for i, row in enumerate(rows, 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            cell.set_facecolor(colors[i - 1])
            if "★" in row[5] and j in (2, 3, 4):
                # bold best per-domain (Phase 2 self-domain)
                if (j == 2 and i == 8) or (j == 3 and i == 9) or (j == 4 and i == 10):
                    cell.set_text_props(fontweight="bold", color="darkgreen")

    # separator row line
    for i in [6]:  # after Phase 1 R3
        for j in range(len(headers)):
            cell = table[(i, j)]
            cell.set_height(cell.get_height() * 0.9)

    fig.suptitle("Phase 1 (Baseline base) + Phase 2 (Proposed base, camera-facing) — combined NN<20px (%)",
                 fontsize=13, y=0.93)

    # legend
    fig.text(0.5, 0.05, "Light gray: Phase 1 (Baseline base)   |   "
             "Light blue: Phase 2 Baseline   |   "
             "Light green: Phase 2 + 1 round ST",
             ha="center", fontsize=10, color="#555")

    plt.tight_layout()
    os.makedirs("_docs/figures", exist_ok=True)
    out = "_docs/figures/phase_result_table.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

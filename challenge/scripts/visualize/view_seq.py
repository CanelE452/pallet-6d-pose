"""시퀀스 빠른 재생. 키: space=pause, n/p=±1, j/k=±10, q=quit."""
import argparse, os, sys, glob
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--seq", required=True)
ap.add_argument("--fps", type=int, default=15)
args = ap.parse_args()

paths = sorted(glob.glob(os.path.join(args.seq, "rgb", "*.png")))
if not paths:
    print(f"no frames in {args.seq}/rgb")
    sys.exit(1)

print(f"[viewer] {args.seq}  {len(paths)} frames  fps={args.fps}")
print("  space=pause/play  n/p=±1  j/k=±10  q=quit")

win = f"viewer — {os.path.basename(args.seq.rstrip('/\\'))}"
cv2.namedWindow(win)
i, paused = 0, False
delay = max(1, int(1000 / args.fps))

while True:
    img = cv2.imread(paths[i])
    cv2.putText(img, f"{i+1}/{len(paths)}  {os.path.basename(paths[i])}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow(win, img)
    key = cv2.waitKey(0 if paused else delay) & 0xFF
    if key == ord('q'): break
    elif key == ord(' '): paused = not paused
    elif key == ord('n'): i = (i + 1) % len(paths); paused = True
    elif key == ord('p'): i = (i - 1) % len(paths); paused = True
    elif key == ord('j'): i = max(0, i - 10); paused = True
    elif key == ord('k'): i = min(len(paths) - 1, i + 10); paused = True
    elif not paused: i = (i + 1) % len(paths)

cv2.destroyAllWindows()
import os as _os, sys as _sys

# --- challenge/scripts 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_CS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_CS] + [_os.path.join(_CS, _d) for _d in sorted(_os.listdir(_CS))
                         if _os.path.isdir(_os.path.join(_CS, _d)) and not _d.startswith(".")]


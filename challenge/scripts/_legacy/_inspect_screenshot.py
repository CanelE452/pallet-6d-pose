"""스크린샷 우측 색 점 좌표 추출 — Hough/색 매칭 없이 단순 색별 centroid."""
import cv2, numpy as np, os

SS = r"C:\Users\minjae\.claude\image-cache\f9ef0d32-c1ee-489a-85b4-83dc202aa453\38.png"
OUT_DIR = r"C:\Users\minjae\Documents\github\FoundationPose\data\pallet\results\annotate_v8_oblique"
os.makedirs(OUT_DIR, exist_ok=True)

img = cv2.imread(SS)
print(f"screenshot shape: {img.shape}")  # likely (126, 224, 3) or similar
H, W = img.shape[:2]

# annotate_draw 의 색 (BGR):
# 0=red(0,0,255), 1=orange(0,165,255), 2=yellow(0,255,255), 3=green(0,255,0)
# 4=cyan(255,255,0), 5=blue(255,0,0), 6=magenta(255,0,255), 7=white(255,255,255)
# 화면에서 색 점은 작은 disc + 큰 영역 보임. 화면이 너무 작아 정확한 좌표 어렵지만
# 좌측 image area 와 우측 panel 분리해서 처리. image area 는 약 W*2/3 (그 다음 panel).

# 우측 panel 영역 제외 (image area 만): 대체로 image 영역 = 0..0.62*W
img_panel_split_x = int(W * 0.62)
img_only = img[:, :img_panel_split_x].copy()

# enlarge x4 to see detail
big = cv2.resize(img_only, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
cv2.imwrite(os.path.join(OUT_DIR, "screenshot_image_only_6x.png"), big)
print(f"image_only saved (6x): shape {big.shape}")

# Find color blobs
color_targets = {
    "0_red":     ((0, 0, 200),    (60, 60, 255)),
    "1_orange":  ((0, 100, 200),  (100, 200, 255)),
    "2_yellow":  ((0, 200, 200),  (100, 255, 255)),
    "3_green":   ((0, 200, 0),    (100, 255, 100)),
    "4_cyan":    ((200, 200, 0),  (255, 255, 100)),
    "5_blue":    ((200, 0, 0),    (255, 100, 100)),
    "6_magenta": ((200, 0, 200),  (255, 100, 255)),
}

img_h, img_w = img_only.shape[:2]
print(f"\nImage panel (thumbnail-coord): {img_w}x{img_h}")
print(f"Map to 640x480: scale_x={640/img_w:.2f}, scale_y={480/img_h:.2f}")

result_640 = {}
for name, (lo, hi) in color_targets.items():
    mask = cv2.inRange(img_only, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        print(f"  {name}: NOT FOUND")
        result_640[name] = None
        continue
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    # scale to 640x480
    u = cx * 640 / img_w
    v = cy * 480 / img_h
    print(f"  {name}: thumb=({cx:.1f},{cy:.1f}) → 640x480=({u:.1f},{v:.1f})  (n_px={len(xs)})")
    result_640[name] = [u, v]

# Save annotated screenshot
vis = big.copy()
for name, p in result_640.items():
    if p is None: continue
    # back to thumb pixel * 6
    pu = p[0] * img_w / 640 * 6
    pv = p[1] * 480 / 480 * 6  # actually need y: thumb_v * 6
    # recompute: thumb coords = result*img_w/640, then *6
    thumb_u = p[0] * img_w / 640
    thumb_v = p[1] * img_h / 480
    cv2.circle(vis, (int(thumb_u*6), int(thumb_v*6)), 12, (255, 255, 255), 2)
    cv2.putText(vis, name.split("_")[0], (int(thumb_u*6)+10, int(thumb_v*6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
cv2.imwrite(os.path.join(OUT_DIR, "screenshot_color_centroids.png"), vis)

print("\nExtracted clicks (640x480 coord):")
print("  Note: each colored point may map to 1 click — but cuboid wireframe lines are")
print("        also colored, so centroid ≠ click position. Need to filter by area.")
print(result_640)

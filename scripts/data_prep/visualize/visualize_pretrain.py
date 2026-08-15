"""Pretrain 결과 종합 시각화: belief map + cuboid + PnP 6D pose (yaw/pitch/roll)."""

import sys, os, glob, cv2, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Deep_Object_Pose', 'common'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Deep_Object_Pose', 'train'))
from models import DopeNetwork

CORNER_NAMES = ['TFL','TFR','BFR','BFL','TBL','TBR','BBR','BBL','CTR']
EDGES_TOP = [(0,1),(1,5),(5,4),(4,0)]
EDGES_BOT = [(3,2),(2,6),(6,7),(7,3)]
EDGES_VERT = [(0,3),(1,2),(4,7),(5,6)]

# Average pallet 3D cuboid (meters)
CUBOID_3D = np.array([
    [-0.50,  0.075,  0.60],
    [ 0.50,  0.075,  0.60],
    [ 0.50, -0.075,  0.60],
    [-0.50, -0.075,  0.60],
    [-0.50,  0.075, -0.60],
    [ 0.50,  0.075, -0.60],
    [ 0.50, -0.075, -0.60],
    [-0.50, -0.075, -0.60],
], dtype=np.float64)

K = np.array([[615.111,0,320],[0,615.111,240],[0,0,1]], dtype=np.float64)

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def rotation_to_euler(R):
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        roll = np.degrees(np.arctan2(R[2,1], R[2,2]))
        pitch = np.degrees(np.arctan2(-R[2,0], sy))
        yaw = np.degrees(np.arctan2(R[1,0], R[0,0]))
    else:
        roll = np.degrees(np.arctan2(-R[1,2], R[1,1]))
        pitch = np.degrees(np.arctan2(-R[2,0], sy))
        yaw = 0
    return yaw, pitch, roll


def infer(model, img_path, device):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (448, 448))
    img_norm = (img_resized.astype(np.float32) / 255.0 - MEAN) / STD
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    with torch.no_grad():
        out_bel, _ = model(tensor)
    belief = out_bel[-1][0].cpu().numpy()
    bh, bw = belief.shape[1], belief.shape[2]

    kps, confs = [], []
    for i in range(9):
        bmap = belief[i]
        c = float(bmap.max())
        confs.append(c)
        idx = int(bmap.argmax())
        by, bx = divmod(idx, bmap.shape[1])
        kps.append((bx * w / bw, by * h / bh))

    # PnP
    pts2d = np.array(kps[:8], dtype=np.float64)
    yaw = pitch = roll = dist = None
    proj = None
    ret, rvec, tvec = cv2.solvePnP(CUBOID_3D, pts2d, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if ret:
        R, _ = cv2.Rodrigues(rvec)
        yaw, pitch, roll = rotation_to_euler(R)
        dist = float(np.linalg.norm(tvec))
        proj, _ = cv2.projectPoints(CUBOID_3D, rvec, tvec, K, None)
        proj = proj.reshape(-1, 2)

    # GT
    gt_kps = None
    gt_euler = (None, None, None)
    basename = os.path.splitext(os.path.basename(img_path))[0]
    json_path = os.path.join(os.path.dirname(img_path), basename + '.json')
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        obj = data['objects'][0]
        gt_kps = [(gx, gy) for gx, gy in obj['projected_cuboid']]
        gt_kps.append(tuple(obj['projected_cuboid_centroid']))
        if 'euler_angles' in obj:
            e = obj['euler_angles']
            gt_euler = (e.get('yaw'), e.get('pitch'), e.get('roll'))

    return img_rgb, belief, kps, confs, (yaw, pitch, roll, dist), proj, gt_kps, gt_euler


def draw_figure(img_rgb, belief, kps, confs, pose, proj, gt_kps, gt_euler, title):
    yaw, pitch, roll, dist = pose
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(3, 5, figure=fig, hspace=0.35, wspace=0.25)

    # --- Panel 1: Image + cuboid overlay ---
    ax1 = fig.add_subplot(gs[0:2, 0:3])
    ax1.imshow(img_rgb)
    if gt_kps:
        for i, j in EDGES_TOP + EDGES_BOT + EDGES_VERT:
            ax1.plot([gt_kps[i][0], gt_kps[j][0]], [gt_kps[i][1], gt_kps[j][1]],
                     'w--', lw=1.5, alpha=0.7)
    for i, j in EDGES_TOP:
        ax1.plot([kps[i][0], kps[j][0]], [kps[i][1], kps[j][1]], '-', color='lime', lw=2)
    for i, j in EDGES_BOT:
        ax1.plot([kps[i][0], kps[j][0]], [kps[i][1], kps[j][1]], '-', color='red', lw=2)
    for i, j in EDGES_VERT:
        ax1.plot([kps[i][0], kps[j][0]], [kps[i][1], kps[j][1]], '-', color='yellow', lw=2)
    if proj is not None:
        for i, j in EDGES_TOP + EDGES_BOT + EDGES_VERT:
            ax1.plot([proj[i][0], proj[j][0]], [proj[i][1], proj[j][1]], 'c-', lw=1.5, alpha=0.6)
    cmap = plt.cm.rainbow(np.linspace(0, 1, 9))
    for i, (kx, ky) in enumerate(kps):
        ax1.plot(kx, ky, 'o', color=cmap[i], ms=8, mec='k', mew=1)
        ax1.annotate(CORNER_NAMES[i], (kx+5, ky-5), fontsize=6, color='white',
                     bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.5))
    legend = 'green=top  red=bot  yellow=vert'
    if proj is not None:
        legend += '  cyan=PnP'
    if gt_kps:
        legend += '  white=GT'
    ax1.set_title(f'{title}\n{legend}', fontsize=9)
    ax1.axis('off')

    # --- Panel 2: Info ---
    ax2 = fig.add_subplot(gs[0:2, 3:5])
    lines = []
    lines.append(f'Avg conf: {np.mean(confs):.4f}')
    lines.append(f'Max conf: {max(confs):.4f} ({CORNER_NAMES[np.argmax(confs)]})')
    lines.append(f'Min conf: {min(confs):.4f} ({CORNER_NAMES[np.argmin(confs)]})')
    lines.append('')
    lines.append('--- 6D Pose (PnP) ---')
    if yaw is not None:
        lines.append(f'Yaw:   {yaw:+7.1f} deg')
        lines.append(f'Pitch: {pitch:+7.1f} deg')
        lines.append(f'Roll:  {roll:+7.1f} deg')
        lines.append(f'Dist:  {dist:7.2f} m')
    else:
        lines.append('PnP failed')
    if gt_euler[0] is not None:
        lines.append('')
        lines.append('--- GT Pose ---')
        lines.append(f'Yaw:   {gt_euler[0]:+7.1f} deg')
        lines.append(f'Pitch: {gt_euler[1]:+7.1f} deg')
        lines.append(f'Roll:  {gt_euler[2]:+7.1f} deg')
    lines.append('')
    lines.append('--- Confidence ---')
    for i in range(9):
        bar = '#' * int(confs[i] * 40)
        lines.append(f'{CORNER_NAMES[i]:>3}: {confs[i]:.3f} {bar}')
    ax2.text(0.05, 0.95, '\n'.join(lines), transform=ax2.transAxes, fontsize=9,
             va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', fc='wheat', alpha=0.8))
    ax2.axis('off')
    ax2.set_title('Pose & Confidence', fontsize=10)

    # --- Panel 3: Top-4 belief maps ---
    top4 = sorted(range(9), key=lambda i: confs[i], reverse=True)[:5]
    for pi, bi in enumerate(top4):
        ax = fig.add_subplot(gs[2, pi])
        ax.imshow(belief[bi], cmap='hot', vmin=0, vmax=max(0.01, confs[bi] * 1.2))
        ax.set_title(f'{CORNER_NAMES[bi]} ({confs[bi]:.3f})', fontsize=8)
        ax.axis('off')

    return fig


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DopeNetwork()
    state = torch.load('weights/misc/pallet_category/final_net_epoch_0060.pth', map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    print('Model loaded')

    out_dir = 'data/pallet/eval_results/comprehensive'
    os.makedirs(out_dir, exist_ok=True)

    # Synthetic val
    syn_imgs = sorted(glob.glob('data/pallet/training_data/val/*.png'))
    syn_step = max(1, len(syn_imgs) // 5)
    syn_sel = syn_imgs[::syn_step][:5]

    # Real
    real_imgs = sorted(glob.glob('data/pallet/real_data/*.jpg'))
    real_step = max(1, len(real_imgs) // 5)
    real_sel = real_imgs[::real_step][:5]

    for idx, path in enumerate(syn_sel):
        img_rgb, belief, kps, confs, pose, proj, gt_kps, gt_euler = infer(model, path, device)
        fig = draw_figure(img_rgb, belief, kps, confs, pose, proj, gt_kps, gt_euler,
                          f'SYNTHETIC #{idx} - {os.path.basename(path)}')
        fig.savefig(f'{out_dir}/syn_{idx:02d}.png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'[syn_{idx}] conf={np.mean(confs):.3f}')

    for idx, path in enumerate(real_sel):
        img_rgb, belief, kps, confs, pose, proj, gt_kps, gt_euler = infer(model, path, device)
        fig = draw_figure(img_rgb, belief, kps, confs, pose, proj, gt_kps, gt_euler,
                          f'REAL #{idx} - {os.path.basename(path)}')
        fig.savefig(f'{out_dir}/real_{idx:02d}.png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'[real_{idx}] conf={np.mean(confs):.3f}')

    print(f'\nAll saved to {out_dir}/')


if __name__ == '__main__':
    main()

"""
challenge/scripts/live/measure_latency.py

DOPE 모델 inference latency 측정. 우리 GPU 결과로 Jetson 예상 fps 1차 추정.

사용:
  python challenge/scripts/live/measure_latency.py
  python challenge/scripts/live/measure_latency.py --input_size 224 320 448 --n_iter 200
  python challenge/scripts/live/measure_latency.py --our_tflops 12.7  # GPU 자동인식 실패 시

⚠ 추정은 GPU FP32 TFLOPS 비율 기반. 실제 Jetson 추론은 메모리 대역폭/캐시/CUDA
core 수 등으로 ±50% 오차. 정확한 fps 는 Jetson 실측 필요.
"""

from __future__ import annotations
import os as _os, sys as _sys

# --- challenge/scripts 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_CS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_CS] + [_os.path.join(_CS, _d) for _d in sorted(_os.listdir(_CS))
                         if _os.path.isdir(_os.path.join(_CS, _d)) and not _d.startswith(".")]

import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml
from torch.autograd import Variable

_HERE = os.path.dirname(os.path.abspath(__file__))
# 2026-08-15 재편: 이 파일이 challenge/scripts/<계열>/ 로 한 단계 내려갔다.
# dirname 을 하나 더 감싸야 저장소 루트다 (계열 -> scripts -> challenge -> repo).
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.append(os.path.join(_REPO, "Deep_Object_Pose", "common"))

from detector import ModelData


# FP32 peak TFLOPS (NVIDIA spec sheet 기준)
GPU_TABLE = {
    # Desktop / Laptop
    "RTX 5090":    104.8,
    "RTX 4090":    82.6,
    "RTX 4080":    48.7,
    "RTX 4070 Ti": 40.1,
    "RTX 4070":    29.2,
    "RTX 4060 Ti": 22.1,
    "RTX 4060":    15.1,
    "RTX 3090":    35.6,
    "RTX 3080":    29.8,
    "RTX 3070 Ti": 21.7,
    "RTX 3070":    20.3,
    "RTX 3060 Ti": 16.2,
    "RTX 3060":    12.7,
    "RTX 3050":    9.1,
    "RTX 2080 Ti": 13.4,
    "RTX 2070":    7.5,
    "GTX 1660 Ti": 5.4,
    "GTX 1080 Ti": 11.3,
    "GTX 1080":    8.9,
    "GTX 1070":    6.5,
    # Jetson (FP32)
    "Jetson Nano (4GB)":      0.47,
    "Jetson TX2":             1.33,
    "Jetson Xavier NX":       1.41,
    "Jetson AGX Xavier":      2.84,
    "Jetson Orin Nano (4GB)": 0.625,
    "Jetson Orin Nano (8GB)": 1.00,
    "Jetson Orin NX (8GB)":   1.88,
    "Jetson Orin NX (16GB)":  2.50,
    "Jetson AGX Orin (32GB)": 5.32,
    "Jetson AGX Orin (64GB)": 5.32,
}

JETSON_TARGETS = [
    ("Jetson Nano (4GB)",      "Nano (구형 Maxwell)"),
    ("Jetson Orin Nano (4GB)", "Orin Nano 4GB"),
    ("Jetson Orin Nano (8GB)", "Orin Nano 8GB"),
    ("Jetson Xavier NX",       "Xavier NX"),
    ("Jetson Orin NX (8GB)",   "Orin NX 8GB"),
    ("Jetson Orin NX (16GB)",  "Orin NX 16GB"),
    ("Jetson AGX Orin (32GB)", "AGX Orin"),
]


def detect_our_gpu():
    if not torch.cuda.is_available():
        return "CPU (no CUDA)", None
    name = torch.cuda.get_device_name(0)
    for k, v in GPU_TABLE.items():
        if k.lower() in name.lower():
            return name, v
    # try shorter substrings
    for k, v in GPU_TABLE.items():
        token = k.replace("RTX ", "").replace("GTX ", "").replace("Ti", "").strip()
        if token.lower() in name.lower():
            return name, v
    return name, None


def measure(net, input_size=448, n_warmup=10, n_iter=100,
            tf32=True, cudnn_benchmark=False):
    """순수 GPU forward 시간 (CUDA event).

    tf32=False 면 Tensor Core TF32 가속을 끈다 (Jetson Nano Maxwell 과 fair 비교).
    cudnn_benchmark=True 면 cudnn 이 알고리즘 자동 선택 → 보통 더 빠름.
    """
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.benchmark = cudnn_benchmark

    net.eval()
    x = torch.randn(1, 3, input_size, input_size, device="cuda")

    # Warmup (cudnn benchmark 모드면 algorithm 탐색 시간 흡수)
    for _ in range(max(n_warmup, 20 if cudnn_benchmark else n_warmup)):
        with torch.no_grad():
            _ = net(x)
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(n_iter):
        starter.record()
        with torch.no_grad():
            _ = net(x)
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))   # ms
    return np.asarray(times)


def measure_end_to_end(net, input_size=448, n_warmup=10, n_iter=50):
    """전처리 + GPU forward + CPU postprocess 합산 시간 (wall clock).

    run_live.py 의 실제 추론 흐름을 모사:
      cv2.resize → transform(normalize) → cuda upload → forward → CPU retrieve
    """
    import cv2
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img_np = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    net.eval()

    for _ in range(n_warmup):
        small = cv2.resize(img_np, (input_size, input_size))
        rgb = small[..., ::-1].copy()
        t = tf(rgb).unsqueeze(0).cuda()
        with torch.no_grad():
            out = net(t)
        for o in out:
            o[-1][0].cpu().numpy()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        small = cv2.resize(img_np, (input_size, input_size))
        rgb = small[..., ::-1].copy()
        t = tf(rgb).unsqueeze(0).cuda()
        with torch.no_grad():
            out = net(t)
        # CPU 로 결과 retrieve (run_live 가 실제로 함)
        _ = out[0][-1][0].cpu().numpy()
        _ = out[1][-1][0].cpu().numpy()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return np.asarray(times)


def fmt_row(label, ms, fps_trt_mult=1.0):
    if ms is None:
        return f"  {label:25s}  -"
    fps = 1000.0 / ms
    fps_trt = fps * fps_trt_mult
    return f"  {label:25s}  {ms:>9.1f} ms  {fps:>7.2f} fps   (TRT-FP16 ≈ {fps_trt:>6.1f} fps)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",  default=os.path.join(_REPO, "challenge", "config", "task.yaml"))
    ap.add_argument("--weights", default=None)
    ap.add_argument("--input_size", type=int, nargs="+", default=[448],
                    help="입력 해상도 (여러개 가능, 비교용)")
    ap.add_argument("--n_iter",  type=int, default=100)
    ap.add_argument("--n_warmup", type=int, default=10)
    ap.add_argument("--our_tflops", type=float, default=None,
                    help="자동 인식 실패 시 수동 지정 (FP32 peak)")
    ap.add_argument("--trt_speedup", type=float, default=3.0,
                    help="Jetson 에서 TensorRT FP16 변환 시 예상 가속 배수 (보통 2~5)")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        C = yaml.safe_load(f)
    weights = args.weights or os.path.join(_REPO, C["baseline"]["weights"])

    print("=" * 78)
    print(" DOPE Inference Latency Measurement")
    print("=" * 78)

    gpu_name, auto_tflops = detect_our_gpu()
    our_tflops = args.our_tflops if args.our_tflops is not None else auto_tflops
    print(f"  GPU      : {gpu_name}")
    if our_tflops is not None:
        src = "auto" if args.our_tflops is None else "manual"
        print(f"  TFLOPS   : {our_tflops:.2f} (FP32, {src})")
    else:
        print(f"  TFLOPS   : 알 수 없음 — --our_tflops <value> 로 수동 지정 시 Jetson 추정")
    print(f"  Weights  : {weights}")
    print(f"  Iter     : warmup={args.n_warmup}, measure={args.n_iter}")

    # Load model
    sd = torch.load(weights, map_location="cpu")
    parallel = list(sd.keys())[0].startswith("module.")
    print(f"  Parallel ckpt: {parallel}")
    m = ModelData(name="pallet", net_path=weights, parallel=parallel)
    m.load_net_model()

    for size in args.input_size:
        print()
        print("=" * 78)
        print(f" Input  {size} x {size}")
        print("=" * 78)

        # 측정 3 가지 설정 비교
        runs = []
        runs.append(("FP32 (TF32 OFF)",   measure(m.net, size, args.n_warmup, args.n_iter,
                                                    tf32=False, cudnn_benchmark=False)))
        runs.append(("TF32 ON (default)", measure(m.net, size, args.n_warmup, args.n_iter,
                                                    tf32=True,  cudnn_benchmark=False)))
        runs.append(("TF32 + bench",      measure(m.net, size, args.n_warmup, args.n_iter,
                                                    tf32=True,  cudnn_benchmark=True)))
        e2e = measure_end_to_end(m.net, size, args.n_warmup, max(20, args.n_iter // 2))

        print(f"  우리 GPU ({gpu_name}) — pure GPU forward (CUDA event):")
        print(f"  {'mode':<22s}  {'mean':>8s}  {'p95':>8s}    {'fps':>7s}")
        for label, times in runs:
            mean_ms = float(times.mean())
            p95_ms  = float(np.percentile(times, 95))
            print(f"  {label:<22s}  {mean_ms:>5.2f}ms  {p95_ms:>5.2f}ms    {1000/mean_ms:>7.2f}")

        e2e_mean = float(e2e.mean())
        e2e_p95  = float(np.percentile(e2e, 95))
        print(f"\n  End-to-end (resize+normalize+cuda+forward+retrieve, wall clock):")
        print(f"  {'mean':>8s}  {'p95':>8s}    {'fps':>7s}")
        print(f"  {e2e_mean:>5.2f}ms  {e2e_p95:>5.2f}ms    {1000/e2e_mean:>7.2f}")

        if our_tflops is None:
            continue

        # Jetson 추정: 두 가지 비율
        #   - Maxwell Nano: TF32 없음 → FP32 (TF32 OFF) 결과로 추정
        #   - Orin (Ampere): TF32 있음 → TF32 ON 결과로 추정
        fp32_ms = float(runs[0][1].mean())   # TF32 OFF
        tf32_ms = float(runs[1][1].mean())   # TF32 ON

        print()
        print(f"  Jetson 추정 (±50% 오차, TRT-FP16 가속 ≈ ×{args.trt_speedup:.1f}):")
        print(f"  {'Target':25s} {'PyTorch fps':>12s}  {'TRT-FP16 fps':>13s}  비고")
        for key, label in JETSON_TARGETS:
            jt = GPU_TABLE[key]
            # Maxwell Nano 는 TF32 없음 → FP32 결과 사용
            is_maxwell = "Nano (4GB)" in key and "Orin" not in key
            base_ms = fp32_ms if is_maxwell else tf32_ms
            jms = base_ms * (our_tflops / jt)
            jfps = 1000 / jms
            trt_fps = jfps * args.trt_speedup
            note = "TF32 없음" if is_maxwell else "Tensor Core OK"
            print(f"  {label:25s} {jfps:>11.2f}   {trt_fps:>12.1f}   {note}")

    print()
    print("=" * 78)
    print(" 주의: 위 추정은 FP32 GFLOPS 비율 단순 가정. 실측은 ±50% 가능.")
    print("       정확한 fps 는 Jetson 에서 같은 모델을 같은 방식으로 측정해야 함.")
    print("=" * 78)


if __name__ == "__main__":
    main()

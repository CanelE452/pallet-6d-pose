import os, time, datetime, argparse
import numpy as np
import cv2

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def ts():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

def save_color(outdir: str, img_bgr: np.ndarray, quality: int = 92):
    fn = os.path.join(outdir, f"color_{ts()}.jpg")
    cv2.imwrite(fn, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return fn

# ---------------------- RealSense 경로 (RGB만) ----------------------
def run_realsense(outdir: str, fps: float = 2.0, show: bool = True, width=640, height=480, stream_fps=30):
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    # 컬러 스트림만 활성화 (RGB만 저장)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, stream_fps)

    pipeline.start(config)

    last_save = 0.0
    interval = 1.0 / fps
    print("RealSense 캡처 시작 (RGB 저장 2 FPS). 종료: q 또는 ESC")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            c = frames.get_color_frame()
            if not c:
                continue

            color = np.asanyarray(c.get_data())  # BGR
            now = time.time()

            # 프리뷰
            if show:
                cv2.imshow("RGB Preview", color)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):  # ESC or q
                    break

            # 저장 주기(0.5s)
            if now - last_save >= interval:
                last_save = now
                save_color(outdir, color)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        if show:
            cv2.destroyAllWindows()

# ---------------------- OpenCV 일반 경로 (RGB만) ----------------------
def run_opencv(outdir: str, rgb_index: int = 0, fps: float = 2.0, show: bool = True, width: int | None = None, height: int | None = None):
    cap = cv2.VideoCapture(rgb_index, cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError(f"RGB 카메라(index={rgb_index})를 열 수 없습니다.")

    # 해상도 고정 원하면 설정 (기본은 장치 기본값 사용)
    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    last_save = 0.0
    interval = 1.0 / fps
    print("OpenCV 캡처 시작 (RGB 저장 2 FPS). 종료: q 또는 ESC")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            now = time.time()

            # 프리뷰
            if show:
                cv2.imshow("RGB Preview", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):  # ESC or q
                    break

            # 저장 주기(0.5s)
            if now - last_save >= interval:
                last_save = now
                save_color(outdir, frame)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()

# ---------------------- 엔트리포인트 ----------------------
def main():
    parser = argparse.ArgumentParser(description="RGB 프레임을 초당 2장 저장 + 실시간 프리뷰")
    parser.add_argument("--outdir", default="./data", help="저장 경로")
    parser.add_argument("--backend", choices=["auto", "realsense", "opencv"], default="auto",
                        help="카메라 백엔드 선택(기본 auto: RealSense 우선)")
    parser.add_argument("--rgb-index", type=int, default=0, help="OpenCV RGB 카메라 인덱스")
    parser.add_argument("--fps", type=float, default=2.0, help="저장 FPS(기본 2)")
    parser.add_argument("--no-preview", action="store_true", help="프리뷰 창 표시하지 않음")
    # 해상도 고정이 필요할 때만 사용
    parser.add_argument("--width", type=int, default=None, help="OpenCV 해상도 가로")
    parser.add_argument("--height", type=int, default=None, help="OpenCV 해상도 세로")
    # RealSense 컬러 스트림 설정 (원치 않으면 바꾸지 마세요)
    parser.add_argument("--rs-width", type=int, default=640, help="RealSense 컬러 가로")
    parser.add_argument("--rs-height", type=int, default=480, help="RealSense 컬러 세로")
    parser.add_argument("--rs-stream-fps", type=int, default=30, help="RealSense 컬러 스트림 FPS")
    args = parser.parse_args()

    ensure_dir(args.outdir)
    show = not args.no_preview

    if args.backend in ("auto", "realsense"):
        try:
            import pyrealsense2 as _  # 존재 확인
            run_realsense(args.outdir, fps=args.fps, show=show,
                          width=args.rs_width, height=args.rs_height, stream_fps=args.rs_stream_fps)
            return
        except Exception as e:
            if args.backend == "realsense":
                raise
            print(f"RealSense 사용 불가 → OpenCV로 대체: {e}")

    run_opencv(args.outdir, rgb_index=args.rgb_index, fps=args.fps, show=show,
               width=args.width, height=args.height)

if __name__ == "__main__":
    main()

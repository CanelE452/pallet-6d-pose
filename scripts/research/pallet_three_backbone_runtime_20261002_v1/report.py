"""Pure Korean Markdown report renderer for the unified runtime result."""
from __future__ import annotations

from . import common as C


LABELS = {
    "YOLO_R0": "YOLO R0",
    "YOLO_P1": "YOLO + P1",
    "DOPE_BASE": "DOPE",
    "DOPE_P1": "DOPE + P1",
    "RESNET_FULL": "ResNet-18 FULL",
    "RESNET_P0_S1": "ResNet-18 + P0 (seed 1)",
    "RESNET_D0_S1": "ResNet-18 + D0 (seed 1)",
    "RESNET_P5_CONSTANT_S1": "ResNet-18 + P5_CONSTANT (seed 1)",
    "RESNET_P5_S1": "ResNet-18 + P5 (seed 1)",
}


def _table(arms, summary):
    lines = [
        "| 방법 | 2D median ms | 2D P90 ms | PnP median ms | 전체 median ms | 전체 P90 ms | pose 상태 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for arm in arms:
        value = summary[arm]
        status = ", ".join(f"{key}:{count}" for key, count in sorted(value["pose_status_counts"].items()))
        lines.append(
            f"| {LABELS[arm]} | {value['two_d_ms']['median']:.3f} | {value['two_d_ms']['p90']:.3f} | "
            f"{value['pnp_ms']['median']:.3f} | {value['full_ms']['median']:.3f} | "
            f"{value['full_ms']['p90']:.3f} | {status} |")
    return lines


def _overhead(summary):
    lines = [
        "| 보정 방법 | 대응 baseline | 추가 2D median ms | 추가 2D P90 ms | 추가 전체 median ms | 추가 전체 P90 ms |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for arm in C.BASELINE_FOR:
        value = summary[arm]
        lines.append(
            f"| {LABELS[arm]} | {LABELS[value['paired_baseline']]} | "
            f"{value['paired_added_two_d_ms']['median']:.3f} | "
            f"{value['paired_added_two_d_ms']['p90']:.3f} | "
            f"{value['paired_added_full_ms']['median']:.3f} | "
            f"{value['paired_added_full_ms']['p90']:.3f} |")
    return lines


def render_report(result: dict, protocol: dict) -> str:
    if (result.get("complete") is not True or result.get("PASS") is not True
            or result.get("measured_calls") != 1170 or result.get("warmup_calls") != 180):
        raise ValueError("A complete 9-arm runtime result is required")
    summary = result["summary"]
    if set(summary) != set(C.ARMS) or not all(summary[arm]["parity_PASS"] for arm in C.ARMS):
        raise ValueError("Runtime summary/parity is incomplete")
    lines = [
        "# YOLO·DOPE·ResNet-18 통합 속도 측정",
        "",
        "## 측정 결론",
        "",
        "세 기반 추정기를 **같은 26개 DEV 영상 바이트**에서 다시 실행했다. 모든 행은 batch 1, "
        "arm별 20회 warmup, 5회 반복이며 시작점은 RAM에 디코딩된 native BGR이다. 아래 수치는 "
        "모델 로드와 파일 디코딩을 제외하고 2D 출력까지, 그리고 같은 prediction-only MAIN PnP까지를 각각 잰 값이다.",
        "",
        "## 논문 본문용 주 비교",
        "",
        *_table(C.PRIMARY_ARMS, summary),
        "",
        "주 비교의 ResNet 보정기는 합성 source 선택 규칙을 고정한 `P0_S1` 한 개다. 속도 측정에서 "
        "12개 평가 head를 순차 실행하지 않았다.",
        "",
        "## ResNet 보조 ablation",
        "",
        *_table(C.AUXILIARY_ARMS, summary),
        "",
        "`P5_CONSTANT_S1`과 `P5_S1`은 같은 구조에서 correction head의 5차원 치수 문맥만 바뀌는 "
        "보조 비교다. FULL backbone 자체는 이미 이미지와 치수를 함께 사용한다.",
        "",
        "## 대응 추가 비용",
        "",
        *_overhead(summary),
        "",
        "추가 비용은 같은 repeat·같은 frame의 보정 arm에서 해당 baseline 시간을 뺀 대응 차이다. "
        "음수 표본도 그대로 포함했고 가장 빠른 반복을 고르지 않았다.",
        "",
        "## 재현·동일성 검사",
        "",
        f"- warmup {result['warmup_calls']}회와 본 측정 {result['measured_calls']}회를 모두 수행했다.",
        f"- 저장된 DEV 2D 출력 재현의 전체 최대 절대차는 `{result['maximum_2d_parity_abs_px']:.3g}` px다.",
        f"- 저장된 pose 상태·축·지표 재현의 전체 최대 절대차는 `{result['maximum_pose_parity_abs']:.3g}`다.",
        "- parity 계산과 GT 기반 수치 대조는 timer 종료 뒤에만 수행했다. GT는 추론이나 PnP 입력에 들어가지 않았다.",
        "- CPU thread는 Torch intra-op 1, inter-op 1, OpenCV 1로 고정했다.",
        "- 저장 출력 재현을 위해 CUDA 수치 설정도 source 실험대로 복원했다: YOLO cuDNN TF32 on, "
        "DOPE·ResNet-18 cuDNN TF32 off다. 설정 전환은 timer 밖이다.",
        "- 측정 순서는 repeat마다 arm을 순환 이동하고 홀수 repeat에서 뒤집었다.",
        "",
        "## 해석 범위",
        "",
        "이 표는 동일 데스크톱 GPU에서 얻은 기술적 latency 비교다. 26장은 재사용 DEV이며 accuracy의 "
        "독립 반복이 아니다. Jetson 속도, 처리량 최적화, 동시 요청 성능을 뜻하지 않는다. 보정 전후 정확도 "
        "주장은 각 backbone의 별도 DEV 결과와 함께 읽어야 한다.",
        "",
        "## 원자료",
        "",
        "- [고정 측정 계약](PROTOCOL.json)",
        "- [전체 결과와 1,170개 측정 행](RESULTS.json)",
        f"- raw JSONL: `{result['raw_rows']['path']}`",
        "",
        f"GPU 기록: `{result['gpu_before']['gpu']}`",
        "",
    ]
    if protocol.get("measured_calls") != result["measured_calls"]:
        raise ValueError("Protocol/result call counts differ")
    return "\n".join(lines)

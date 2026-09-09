"""Render completed point/line experiments; no model calls or browser launch."""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

from PIL import Image

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "data/pallet/results"
TITLE = "점·선 결합 실험 · 진행 결과"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(out):
    out = out.resolve()
    inputs = {}

    def bind(rel):
        path = (RESULTS / rel).resolve()
        assert path.is_file(), path
        inputs[str(path)] = sha(path)
        return path

    def read(rel):
        return json.loads(bind(rel).read_text())

    def link(rel, label):
        return f'<a href="{html.escape(bind(rel).as_uri())}">{html.escape(label)}</a>'

    def picture(rel, alt, css=""):
        path = bind(rel)
        with Image.open(path) as im:
            im.verify()
        uri = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
        return f'<a href="{path.as_uri()}"><img class="{css}" src="{uri}" alt="{html.escape(alt)}"></a>'

    v2path = "pallet_dht_structured_v2/REAL_RESULTS_seed1.json"
    abpath = "pallet_dht_structured_v2/provenance/proposal_removal_diagnostic/RESULTS.json"
    gapath = "pallet_dht_structured_v2/provenance/cost_gap_diagnosis_seed1/COST_GAP_DIAGNOSIS.json"
    v3hpath = "pallet_dht_local_v3/evaluation/hough_seed1/SYNTHETIC_RESULTS.json"
    v3npath = "pallet_dht_local_v3/evaluation/no_hough_seed1/SYNTHETIC_RESULTS.json"
    mappath = "pallet_dht_local_map_v3b/RESULTS.json"
    v4path = "pallet_dht_deepfield_v4/prototype/PROTOTYPE.json"
    gtpath = "pallet_dht_gt_audit_v1/AUDIT_CONCLUSION.json"
    v2, ab, gaps = read(v2path), read(abpath), read(gapath)
    v3h, v3n, v3b = read(v3hpath), read(v3npath), read(mappath)
    v4, gt = read(v4path), read(gtpath)
    for d in [v2, ab, gaps, v3h, v3n, v3b, v4]:
        assert d["complete"] and d["PASS"], "Incomplete source artifact"
    assert v3h["baseline"] == v3n["baseline"] == v3b["baseline"]
    assert not any(x["advancement"]["advance"] for x in [v3h, v3n, v3b])
    assert v3b["optimizer_steps"] == v3b["real_image_forwards"] == 0
    assert v4["n_actual_synthetic_images"] == 3 and v4["n_actual_real_images"] == 0
    assert not gt["gt_fully_certified"]
    real = {r["arm"]: r for r in v2["summaries"]}
    assert all(r["n_frames"] == 319 and r["n_observed_points"] == 2738 for r in real.values())
    for key in ["mean_px", "median_px", "p90_px"]:
        assert real["baseline"][key] == real["point_segment_hough"][key]
    counts = {k: v["n_selected_nonidentity"] for k, v in gaps["summary"].items()}
    assert counts == {"point_only": 1, "point_segment": 1, "point_segment_hough": 0}

    def table(rows, decimals=3):
        body = "".join('<tr><th>' + html.escape(name) + '</th>' + ''.join(
            f'<td>{m[k]:.{decimals}f}</td>' for k in ["mean_px", "median_px", "p90_px"]
        ) + '<td>' + html.escape(status) + '</td></tr>' for name, m, status in rows)
        return '<div class="scroll"><table><thead><tr><th>Method</th><th>Mean px ↓</th><th>Median px ↓</th><th>P90 px ↓</th><th>Status</th></tr></thead><tbody>' + body + '</tbody></table></div>'

    real_rows = [("Baseline · frozen joint points", real["baseline"], "Reference"),
                 ("v2 · Point score", real["point_only"], "1 frame changed; worse"),
                 ("v2 · Point + segment score", real["point_segment"], "1 frame changed; C4 gain"),
                 ("v2 · Point + segment + Hough score", real["point_segment_hough"], "319 / 319 identity")]
    synthetic_rows = [("Baseline · frozen joint points", v3h["baseline"], "Reference"),
                      ("v3 · Direct local control", v3n["validation"], "Gate failed"),
                      ("v3 · Local Hough", v3h["validation"], "Gate failed"),
                      ("v3b · Fixed-weight MAP prior", v3b["validation"], "Gate failed")]
    c4 = ab["case"]["posthoc_official_GT_metrics"]
    high = "pallet_dht_local_v3/provenance/synthetic_edge_diagnosis/high_harm_VALID_GT_CONTACT.png"
    sample = "pallet_dht_local_v3/provenance/synthetic_edge_diagnosis/sha_id_sample_VALID_GT_CONTACT.png"
    field = "pallet_dht_deepfield_v4/prototype/synthetic_field_comparison.png"
    raw = REPO / "data/evaluation/pallet_eval_v1/dev_existing/sessions/eval_pallet07/rgb/1778652166837872128.png"
    # The image helper accepts an absolute existing path as well as a results-relative path.
    raw_figure = picture(str(raw), "Original user failure image", "raw")
    metric_data = {"real": real, "synthetic_baseline": v3h["baseline"],
                   "v3_no_hough": v3n["validation"], "v3_hough": v3h["validation"],
                   "v3b": v3b["validation"], "source_datasets_are_separate": True}
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{TITLE}</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f5f8;color:#152236;font:16px/1.7 "Noto Sans CJK KR","Noto Sans KR","Malgun Gothic",Arial,sans-serif}}
main{{max-width:1080px;margin:auto;padding:28px 24px 60px}}header,section{{background:white;border:1px solid #dce2eb;border-radius:14px;padding:25px 30px;margin-bottom:20px}}
header{{border-top:6px solid #245cb0}}h1{{font-size:30px;line-height:1.35;margin:6px 0 16px}}h2{{font-size:23px;line-height:1.4;margin:0 0 12px}}h3{{font-size:18px;margin:8px 0}}p{{margin:10px 0}}a{{color:#1455a0}}small,.muted,figcaption{{color:#526175;font-size:14px}}
.tag{{display:inline-block;font-size:12px;font-weight:700;letter-spacing:.04em;background:#eaf0fa;color:#204f8f;padding:3px 9px;border-radius:5px}}.lead{{font-size:20px;font-weight:650;line-height:1.5}}.callout{{background:#fff7e8;border-left:4px solid #b87511;padding:12px 16px;margin:15px 0}}
.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;margin:16px 0;font-size:14px}}th,td{{padding:11px 12px;border-bottom:1px solid #e1e6ec;white-space:nowrap;text-align:right;font-variant-numeric:tabular-nums}}th:first-child{{text-align:left}}thead{{background:#f0f4fa}}tbody th{{font-weight:500}}td:last-child{{font-size:12px;color:#526175;text-align:left}}details{{border-top:1px solid #e1e6ec;padding-top:12px;margin-top:16px}}summary{{cursor:pointer;color:#204f8f;font-weight:600}}figure{{margin:16px 0}}img{{display:block;max-width:100%;height:auto;margin:auto;border-radius:5px}}.contact{{max-height:960px}}.raw{{max-height:360px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}code{{font-size:13px;overflow-wrap:anywhere}}.sources{{font-size:14px;line-height:2.1}}.sources a{{margin-right:14px}}footer{{font-size:13px;color:#637185}}
@media(max-width:700px){{main{{padding:14px 10px}}header,section{{padding:20px 16px}}h1{{font-size:25px}}.grid{{grid-template-columns:1fr}}}}
</style><main><header><span class="tag">COMPLETED RESULTS · 2026-09-09</span><h1>{TITLE}</h1>
<p class="lead">점과 선을 결합하는 계산·학습 경로는 작동하지만,<br>안정적인 실사 성능 향상은 아직 확인되지 않았습니다.</p>
<p>v2는 실사 평가까지 완료했습니다. v3와 v3b는 합성 검증 기준을 통과하지 못해 실사 평가를 진행하지 않았습니다. v4는 사전학습 선 특징의 출력 확인 단계입니다.</p>
<p class="muted">표의 오차는 원래 점 번호를 유지한 원본 이미지 픽셀 거리입니다. 낮을수록 좋습니다. 실사와 합성의 평가 점·분모가 다르므로 두 표의 수치를 직접 비교하지 않습니다.</p></header>

<section><span class="tag">v2 · REAL DEV</span><h2>전체 배치 선택: 완전 결합은 기존 예측을 그대로 유지</h2>
<p>재사용 DEV 319장·13세션. 감독 GT 2,818점 중 매칭된 309장의 2,738점을 평가했습니다. 모서리와 중심점이 포함되며 결측·미매칭 80점은 오차 표에 포함되지 않습니다. 모든 방법의 평가 마스크는 같습니다.</p>
{table(real_rows)}
<p><b>Baseline은 기존 <code>hough_joint_seed1</code> EMA의 점 출력입니다.</b> 순수 YOLO 대조군이 아닙니다. 세 점수 모델은 같은 DHT 후보 배치를 사용하므로 <code>Point score</code>도 DHT 후보가 전혀 없는 비교는 아닙니다.</p>
<div class="callout">Point + segment의 작은 집계 개선은 한 프레임의 C4 번호 회전에 집중됐습니다. 선 교점 후보를 제거한 사후 진단에서도 개선이 유지되어, 이 결과를 선 위치 보정의 이득으로 볼 수 없습니다.</div>
<details><summary>한 프레임 개선을 분해한 진단</summary><p><code>plastic_day_01:005838</code>의 9점 평균 오차는 기존 {c4['baseline']['mean9_px']:.3f}px → C4 번호 회전만 적용하면 {c4['C4_slot3']['mean9_px']:.3f}px → 저장된 선 보정까지 포함하면 {c4['saved_selected']['mean9_px']:.3f}px입니다. 여기서 추가 선 보정은 약 {c4['saved_selected']['mean9_px']-c4['C4_slot3']['mean9_px']:.3f}px 불리했습니다. 결과를 본 뒤 분해한 단일 사례이며, GT는 진단에만 사용했습니다.</p><p>{link(abpath, '실제 후보 제거 진단 JSON')}</p></details>
<p class="muted">한 학습 seed의 탐색적 결과입니다. 재사용 실사의 작은 개선을 독립 데이터의 안정적 향상으로 해석하지 않습니다. {link(v2path, '실사 결과 JSON')} · {link(gapath, '선택·유지 개수의 근거')}</p></section>

<section><span class="tag">v3 / v3b · SYNTHETIC VALIDATION</span><h2>고해상도 국소 보정: 합성 검증의 다음 단계 기준 미통과</h2>
<p>동일 합성 512장 중 감독·매칭이 있는 509장, 모서리 4,021점의 비교입니다. 중심점은 제외합니다. 학습 1,792장 / 보정계수 선택 256장 / 별도 합성 검증 512장을 분리했습니다.</p>
{table(synthetic_rows, 5)}
<p>v3 Local Hough는 평균 오차가 약 {-100*v3h['advancement']['mean_reduction_fraction']:.3f}% 커졌습니다. v3b는 평균 오차를 약 {100*v3b['advancement']['mean_reduction_fraction']:.3f}% 줄였지만, P90이 악화했고 사전 기준인 평균 1% 이상 개선에도 못 미쳤습니다.</p>
<p><b>v3b는 재학습한 모델이 아닙니다.</b> v3의 학습된 가중치를 그대로 두고, 원래 점에서 멀어지는 선 후보에 고정 MAP 사전분포를 추가했습니다. 가중치 변화·학습 업데이트는 0회입니다. 이 변경은 선 위치와 분포의 집중도 기반 가중치에 함께 영향을 줍니다.</p>
<div class="callout">v3와 v3b의 실사 성능은 평가하지 않았습니다. 국소 이동은 입력 영상에서 최대 4px 범위이므로, 원 사례의 수백 픽셀 오차나 점 번호 회전을 해결하는 실험도 아닙니다.</div>
<details><summary>검증 기준과 원래 양호한 점의 변화</summary><p>평균·중앙값·P90 비악화, 평균 1% 이상 개선, 기존 10px 이내 점의 10px 밖 이동률 ≤1%, 0이 아닌 보정계수를 함께 요구했습니다. 기존 양호점 3,743개 중 경계 밖 이동은 direct control 0개, Local Hough 1개, MAP 0개였습니다. 일부 조건 충족만으로 다음 단계 통과를 선언하지 않았습니다.</p></details>
<p class="muted">{link(v3npath, 'v3 direct control 결과')} · {link(v3hpath, 'v3 Hough 결과')} · {link(mappath, 'v3b 고정 가중치 MAP 결과')}</p>
<details><summary>실제 합성 이미지와 GT-valid 선 비교 보기</summary><p>아래는 오차가 더 커진 5장을 골라 본 <b>사후 진단</b>입니다. 전체 성능을 대표하는 무작위 표본이 아닙니다. 초록은 감독 가능한 GT 점 사이의 선, 청록은 기존 예측, 빨강은 출력입니다. GT 유효성은 해당 선이 실제로 보인다는 인증이 아닙니다.</p>
<figure>{picture(high, 'Post-hoc high-harm synthetic examples; GT-valid endpoints only', 'contact')}<figcaption>클릭하면 원본 크기. 작은 이동은 원본 확대에서 확인할 수 있습니다. {link(sample, '별도 SHA-ID 표본 5장 전체 보기')}</figcaption></figure>
<p>검토 사례에서는 GT 근처의 선 신호가 있어도 더 멀리 있는 평행 대비선이 높은 점수를 받거나, 여러 선 봉우리를 평균하면서 경계에서 벗어나는 경우가 있었습니다. 영상 대비가 높다는 사실만으로 팔레트 외곽이라는 뜻은 아닙니다. {link('pallet_dht_local_v3/provenance/synthetic_edge_diagnosis/FINDINGS.md', '국소 선 진단 상세')}</p></details></section>

<section><span class="tag">v4 · FIELD PROTOTYPE ONLY</span><h2>DeepLSD: 선 특징을 꺼낼 수 있음까지 확인</h2>
<p>일반 실사 MegaDepth/MiniDepth로 사전학습된 공식 DeepLSD를 사용했습니다. 합성 학습 집합에서 SHA로 고른 3장에 대해 거리·방향 필드를 추출했고, 공식 구현과 출력이 정확히 일치했습니다. 새 학습·전체 캐시 생성·실사 평가·팔레트 정확도 검증은 진행하지 않았습니다.</p>
<figure>{picture(field, 'Grayscale input, Sobel magnitude, and DeepLSD distance field on three synthetic images')}<figcaption>왼쪽부터 원본 흑백 영상, Sobel 경계 강도, DeepLSD 거리 필드입니다. 방향 필드는 별도로 저장했고 이 그림에는 표시하지 않았습니다. 울타리와 바닥 무늬에도 반응하며, 팔레트 외곽의 정답 확률을 나타내는 그림이 아닙니다.</figcaption></figure>
<p class="muted">그림은 실제 모델 필드이며 attention·인과 설명이 아닙니다. 공식 사전학습의 일반 실사 데이터를 사용하므로 합성 데이터만으로 새로 학습한 비교와 구분해야 합니다. {link(v4path, '구현 일치 검증 JSON')}</p></section>

<section><h2>처음 실패한 원 사례와 GT 검토</h2><div class="grid"><figure>{raw_figure}<figcaption>원본 <code>eval_pallet07:1778652166837872128</code>. 새로운 성공 사례를 보여주는 이미지가 아닙니다.</figcaption></figure><div><p>이 사례의 큰 오차에는 C4 점 번호 대응 문제가 포함됩니다. v2의 완전 결합은 기존 배치를 유지했고, 국소 보정만으로 번호 회전을 해결할 수는 없습니다.</p><p>기존 GT는 수정하지 않았습니다. GT 감사에서 좌표·계산의 재현성과 실제 영상에서의 GT 정확성은 구분했습니다. 가려짐·외삽·경계 정의의 불확실성이 남아 있어 전체 GT의 물리적 정확성을 인증한 상태는 아닙니다.</p><p>{link('pallet_dht_gt_audit_v1/gt_review.html', '기존 GT-only 검토 화면 열기')}<br>{link(gtpath, 'GT 감사 결론 JSON')}</p></div></div></section>

<section><h2>현재 판단</h2><p>결합 경로의 구현과 실제 실행은 확인했습니다. 다만 <b>DHT를 더한 덕분에 실사 점 정확도가 안정적으로 좋아졌다는 근거는 아직 확보하지 못했습니다.</b> 한 프레임의 번호 회전 성공, 합성 평균의 소폭 감소, 일반 선 필드의 출력 확인을 서로 다른 증거로 유지해야 합니다.</p><details><summary>재현 출처와 범위</summary><div class="sources">{link('pallet_dht_structured_v2/PROTOCOL.json', 'v2 protocol')}{link('pallet_dht_local_v3/PROTOCOL.json', 'v3 protocol')}{link('pallet_dht_local_map_v3b/PROTOCOL.json', 'v3b protocol')}{link('pallet_dht_local_v3/provenance/synthetic_edge_diagnosis/RESULTS.json', 'Synthetic edge diagnostics')}{link('pallet_dht_deepfield_v4/prototype/FIGURE_RECEIPT.json', 'DeepLSD figure provenance')}</div><p class="muted">모든 수치는 저장된 완료 결과에서 읽었습니다. 이 진행 보고서 생성 중 새로운 학습·모델 추론·GT 변경은 없습니다. 원래 픽셀 거리의 분포와 서로 다른 분모를 그대로 표시했으며, 새 통합 성공 판정은 만들지 않았습니다.</p></details></section>
<script type="application/json" id="report-data">{json.dumps(metric_data, ensure_ascii=False).replace('<', chr(92)+'u003c')}</script>
<footer>완료된 결과를 모은 진행 보고서 · 원본 이미지·자료 링크는 이 작업 공간의 로컬 파일입니다. 그림과 표는 HTML 안에 포함되어 있습니다.</footer></main></html>'''
    out.mkdir(parents=True, exist_ok=True)
    target = out / "index.html"
    target.write_text(page, encoding="utf-8")
    class Links(HTMLParser):
        def __init__(self):
            super().__init__(); self.links = []; self.images = 0
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "a": self.links.append(attrs.get("href", ""))
            if tag == "img":
                assert attrs["src"].startswith("data:image/png;base64,")
                self.images += 1
    parser = Links(); parser.feed(page)
    for value in parser.links:
        parsed = urlparse(value)
        assert parsed.scheme == "file" and Path(unquote(parsed.path)).is_file(), value
    assert parser.images == 3 and '<title>' + TITLE + '</title>' in page
    assert '<script src=' not in page and 'https://' not in page
    for path, digest in inputs.items(): assert sha(path) == digest, path
    receipt = {"schema": "pallet_dht_progress_render_v1", "complete": True, "PASS": True,
               "scope": "Static data/link/image rendering checks only; no browser or accuracy certification",
               "title": TITLE, "html": str(target), "html_sha256": sha(target),
               "input_sha256": inputs, "source_sha256": {str(Path(__file__).resolve()): sha(__file__)},
               "embedded_images": parser.images, "checked_local_links": len(parser.links),
               "broken_local_links": 0, "new_model_forwards": 0, "browser_launched": False,
               "accuracy_goal_achieved": False, "source_results_unchanged": True,
               "created_at": datetime.now(timezone.utc).isoformat()}
    (out / "REPORT_RENDER.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ["html", "html_sha256", "embedded_images", "checked_local_links", "PASS"]}, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=RESULTS / "pallet_dht_progress_20260909")
    render(ap.parse_args().run_dir)

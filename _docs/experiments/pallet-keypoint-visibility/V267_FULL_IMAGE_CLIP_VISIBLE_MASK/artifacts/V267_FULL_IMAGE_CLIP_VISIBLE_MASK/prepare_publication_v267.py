"""Publish a byte-identical, explicitly inventoried V267 result package locally."""
from pathlib import Path
import hashlib
import json
import os
import shutil
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote

HERE = Path(__file__).absolute().parent
REPO = Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
DEST = REPO / '_docs/experiments/pallet-keypoint-visibility/V267_FULL_IMAGE_CLIP_VISIBLE_MASK'
PAYLOAD = DEST / 'artifacts/V267_FULL_IMAGE_CLIP_VISIBLE_MASK'


def safe(p):
    p = Path(p)
    assert p.is_absolute() and not any(x in str(p).lower() for x in ('v112', 'cache', '.pyc', '..')), str(p)
    for x in reversed((p, *p.parents)):
        assert not x.is_symlink(), str(x)
    return p


def binding(p):
    p = safe(p)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return dict(path=str(p), size_bytes=p.stat().st_size, sha256=h.hexdigest())


def dump(p, value):
    with safe(p).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')


def walk(d):
    for p in sorted(safe(d).iterdir()):
        safe(p)
        if p.name == 'official_source':
            continue
        if p.is_dir():
            yield from walk(p)
        elif p.is_file():
            yield p


def main():
    assert not safe(DEST).exists()
    safe(PAYLOAD).mkdir(parents=True)
    included, excluded = [], []
    for source in walk(HERE):
        rel = source.relative_to(HERE)
        reason = None
        if rel.as_posix() == 'pretrained/ViT-B-16.pt':
            reason = 'Generic official pretrained backbone; official URL and SHA in included provenance; not a new fitted weight.'
        elif rel.parts[0] == 'syn_fullframe_features_v1' and rel.suffix == '.npy':
            reason = 'Large frozen SYN feature array; original data and extraction protocol/SHA retained; compact replay inputs included.'
        elif rel.parts[0] == 'current_inference_v1' and rel.name in ('FULL_CLIP_TOKENS_FP16.npy', 'FULL_CLIP_SURGERY_ARCH_TOKENS_FP16.npy'):
            reason = 'Large real full-frame frozen backbone array; all frame maps/predictions plus fixed5frame head-forward fixture included.'
        elif rel.parts[0] == 'evaluation_v1' and rel.suffix in ('.png', '.npz', '.html'):
            reason = 'Duplicate partial evaluator-v1 display/input payload; corrected evaluator-v2 payload included; v1 metrics/log/code retained unchanged.'
        record = dict(relative_path=rel.as_posix(), **binding(source))
        if reason:
            record['reason'] = reason
            excluded.append(record)
            continue
        target = safe(PAYLOAD / rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        after = binding(target)
        assert after['sha256'] == record['sha256'] and after['size_bytes'] == record['size_bytes']
        included.append(dict(package_path=target.relative_to(DEST).as_posix(), **record))

    wrapper = '''# V267 — 원래 합성 팔레트 마스크로 전체 이미지 CLIP 분할

**콘을 제외하는 전체 영상 분할은 진전했지만, 보이는 모서리 누락이 커서 보임/가림 판별기로 채택하지 않았습니다.** 기존 V264의 외부 가림 누락67점 중61점을 회복했고, 기존에 맞히던23점의 새 누락은0입니다. 보임 recall은89.63→51.87%로 낮아졌습니다.

![전체 영상 콘 회복과 보이는 모서리 누락](artifacts/V267_FULL_IMAGE_CLIP_VISIBLE_MASK/RESULT_OVERVIEW_KO.png)

## 같은 평가 대상에서의 비교

실사223장, PRIMARY1237점(보임 D1147/외부 가림 E90). 새 모델의 비대상 마스크 출력을 E에 대응한 **MASK_PRESENCE_PROXY** 평가입니다. 백분율이며 혼동행렬의 열 순서는 D/E/U입니다.

| 방법 | D→D/E/U | E→D/E/U | 보임 precision | 보임 recall | E precision | E recall | 보류율 |
|---|---|---|---:|---:|---:|---:|---:|
| 이전 V264 heatmap 주변 특징 | 1028/119/0 | 67/23/0 |93.88|89.63|16.20|25.56|0.00|
| 전체 일반 CLIP |414/733/0|9/81/0|97.87|36.09|9.95|90.00|0.00|
| 전체 CLIP 개선형 |595/552/0|6/84/0|99.00|51.87|13.21|93.33|0.00|
| 이전 v61 |260/887/0|2/88/0|99.24|22.67|9.03|97.78|0.00|
| 과거 실사학습 모델 |655/319/173|3/83/4|99.54|57.11|20.65|92.22|14.31|

과거 실사학습 모델의 CAL 노출 이력은 유지합니다. 새 두 모델만 합성 전용 적합입니다. 자기 가림은 E에 합치지 않았습니다. 전체1784점의 입력불가45U도 보존했습니다.

| 개선형 집단 | 보임 정답/전체 D | E 정답/전체 E | 보임 recall | E recall | E precision |
|---|---:|---:|---:|---:|---:|
| 목재105장 |307/616|27/29|49.84|93.10|8.04|
| 정사각형118장 |288/531|57/61|54.24|93.44|19.00|

보임 precision99.0%는 선택한601점 중595점이 D라는 뜻입니다. 전체 정확도99%가 아니며, 실제 보임1147점 중552점을 버리는 문제가 남습니다. 같은 동결0.5 기준의 결과이고 사후 confidence gate를 선택하지 않았습니다.

## 데이터와 실제 학습

- 사용자의 원래 `v2_prod40k_clean_merged` RGB/M4 visible-target mask1003장을 전수 감사했습니다. 학습768/합성 검증235이며 장면 ID/RGB SHA 중복0입니다. 별도 내부 진단용 SYN440은 쓰지 않았습니다.
- **처음 입력부터 전체 RGB를512×512로 resize합니다. KP crop/중앙 crop은 없습니다.** 전체32×32 지도를 저장·동결한 다음 기존 초기 native q를 원영상 좌표로 읽었습니다. 갤러리의 확대는 표시 전용입니다.
- 일반 OpenAI CLIP ViT-B/16과 [공식 CLIP Surgery](https://github.com/xmed-lab/CLIP_Surgery/tree/d4696d47f49cfe70f49140afe5eb94f94c5f59bc)의 영상 architecture surgery(V–V attention/dual path)를 동결했습니다. 텍스트 feature surgery/zero-shot pipeline을 실행한 것은 아닙니다.
- 같은26721개 파라미터 mask head, 같은 초기값·학습 배치·seed17, 각1000업데이트입니다. M4>127→전체512→patch16 평균 soft target에 학습 마스크에서만 정한 weighted BCE+soft Dice를 적용했습니다. M0는 새 학습에 쓰지 않았습니다.
- 두 final1000 체크포인트를 DEV 평가 전에 동결했습니다. 실사223장은 모델·통계·threshold·체크포인트 적합에 사용하지 않았고, 지도와1784점 예측을 고정한 뒤 주석을 붙였습니다. 기존223장은 반복 개발 평가로 독립 블라인드 검증이 아닙니다. 새 PnP/위치 보정 학습0입니다.

일반 CLIP↔개선형만 동일 조건의 영상 구조 비교입니다. V264와의 차이에는 데이터·입력 범위·기반 특징·감독도 달라져 모든 회복을 전체 영상 하나의 인과 효과라고 단정하지 않습니다.

## 실제로 되는 것과 남은 문제

G370은 콘의 흰 띠, G908은 큰 콘 몸체를 팔레트에서 제외합니다. 개선형에도 콘 받침 일부의 오분할은 남습니다. G996은 사용자의 보임 판단과 원래 E 주석이 다른 사례라 명백한 성공 근거로 단정하지 않고 주석을 유지했습니다.

초기 좌표 오차≤3px인 수동 보임376점 중182점도 비대상입니다. 이182점의 예측 마스크까지 거리는≤3/3–6/6–16/>16px/빈 마스크가47/52/57/24/2점입니다. G200/G224/G852는 몸체를 찾으나 외곽 q를 놓치고, G314/G332는 작은 목재 일부, G665는 어두운 대상 전체를 놓칩니다. 실사 픽셀 GT가 없어 이 거리는 원인 확정이 아닌 예측 영역과의 근접성 진단입니다.

원해상도 M4 합성 DEV235의 micro IoU는 일반66.02%, 개선형65.29%(평균 영상56.06/57.40%)입니다. 작은 팔레트에서 두 방식 모두 약합니다. 실사 segmentation IoU는 픽셀 GT가 없어 보고하지 않습니다.

**M4는 팔레트/비대상 정답입니다.** 비대상 내부의 외부 물체·배경·빈틈을 별도 정답으로 학습하지 않았으므로 마스크 밖을 곧바로 외부 가림으로 확정할 수 없습니다. 후속 개선 가설은 전체 문맥을 유지한 고해상도 경계 분할과, 별도의 native 대응/외부 덮임 검증입니다. 이 결과 뒤 추가 fit이나 마스크 팽창·좌표 이동·문턱 탐색은 하지 않았습니다.

## 자료 열기와 검산

- [전체 보고서](artifacts/V267_FULL_IMAGE_CLIP_VISIBLE_MASK/ROOT_READOUT_KO.html) — 폴더째 내려받아 브라우저에서 열면 HTML 갤러리가 작동합니다.
- [실사104사례: 전체 RGB/확률/마스크/표시 전용 확대](artifacts/V267_FULL_IMAGE_CLIP_VISIBLE_MASK/evaluation_v2/INDEX_KO.html), [혼동행렬·P/R/U 전체60행](artifacts/V267_FULL_IMAGE_CLIP_VISIBLE_MASK/evaluation_v2/CLASSIFICATION.csv), [1784점 결과](artifacts/V267_FULL_IMAGE_CLIP_VISIBLE_MASK/evaluation_v2/ALL1784_COMPARISON.csv), [자기 가림 별도 집계](artifacts/V267_FULL_IMAGE_CLIP_VISIBLE_MASK/evaluation_v2/SELF_RESPONSES.csv).
- [합성 원본 RGB/M4/분할 예측](artifacts/V267_FULL_IMAGE_CLIP_VISIBLE_MASK/syn_mask_diagnostic_v1/INDEX_KO.html), [실사 보임 누락 경계 진단](artifacts/V267_FULL_IMAGE_CLIP_VISIBLE_MASK/mask_boundary_diagnostic_v1/INDEX_KO.html).
- [학습 전 프로토콜](artifacts/V267_FULL_IMAGE_CLIP_VISIBLE_MASK/PROTOCOL_BEFORE_FIT.json), [학습 코드](artifacts/V267_FULL_IMAGE_CLIP_VISIBLE_MASK/train_fullmask_v267.py), [전체 영상 encoder](artifacts/V267_FULL_IMAGE_CLIP_VISIBLE_MASK/fullframe_clip_v267.py), [공식 소스 commit/SHA](artifacts/V267_FULL_IMAGE_CLIP_VISIBLE_MASK/OFFICIAL_SOURCE_PROVENANCE.json).
- [CPU compact 재현 안내](artifacts/V267_FULL_IMAGE_CLIP_VISIBLE_MASK/RUN_GUIDE_KO.md), [학습 독립 검토](artifacts/V267_FULL_IMAGE_CLIP_VISIBLE_MASK/MASK_TRAINING_INDEPENDENT_REVIEW.json), [입력 범위 독립 검토](artifacts/V267_FULL_IMAGE_CLIP_VISIBLE_MASK/FULLFRAME_SCOPE_INDEPENDENT_REVIEW.json), [60행 독립 검산](artifacts/V267_FULL_IMAGE_CLIP_VISIBLE_MASK/ROOT_INDEPENDENT_METRICS_CHECK.json).
- [패키지 파일별 SHA·대형 파일 제외 목록](RESULTS_PACKAGE_MANIFEST.json). 일반 pretrained350.8MB·대형 frozen 특징은 제외하고 URL/SHA/경로를 보존했습니다. 신규 head 가중치·전체223장 지도/1784점 예측·고정5영상 compact forward fixture·선택한 합성6장의 정확한 원본 RGB/M4 쌍·표/실제 시각화는 포함했습니다. 저장소만으로 원래 데이터 전체 재학습을 즉시 재현한다고 주장하지 않습니다.

첫 입력 감사의 JPEG decoder 차이는 원래 환경으로 해결했습니다. 평가 HTML append 오류는 표시 코드만 수정했고 이전·최종 지표6파일 및104개 사례 PNG는 모두 바이트 동일합니다. 원래 실패 코드·로그를 보존했습니다. 새 학습2회/2000업데이트 외 추가 fit·실사 적합·새 PnP·사후 문턱 선택0입니다.
'''
    safe(DEST / 'README_KO.md').write_text(wrapper)
    dump(DEST / 'RESULTS_PACKAGE_MANIFEST.json', dict(schema='V267_FULL_FRAME_ORIGINAL_SYN_RESULTS',
        completed=True, payload_files=len(included), payload_bytes=sum(x['size_bytes'] for x in included),
        included=included, excluded_local_files=excluded, excluded_source_repository=dict(
            path=str(HERE/'official_source'), reason='Do not vendor third-party Git clone; official URL/commit and code SHA are included.',
            url='https://github.com/xmed-lab/CLIP_Surgery', commit='d4696d47f49cfe70f49140afe5eb94f94c5f59bc'),
        full_original_1003_RGB_M4_corpus_not_duplicated=True, six_selected_exact_RGB_M4_pairs_included=True, fixed_cpu_replay_fixtures_included=True,
        real_fits_or_threshold_selection=0, additional_fits_after_frozen_comparison=0))

    class Links(HTMLParser):
        def __init__(self):
            super().__init__(); self.links=[]
        def handle_starttag(self, tag, attrs):
            self.links += [v for k,v in attrs if k in ('src','href') and v]
    checked=[]
    for f in sorted(PAYLOAD.rglob('*.html')):
        safe(f); parser=Links();parser.feed(f.read_text())
        for link in parser.links:
            u=urlsplit(link)
            if u.scheme or u.netloc or not u.path: continue
            # Resolve lexical navigation before guarded filesystem access, then confine to package.
            target=safe(Path(os.path.abspath(f.parent / unquote(u.path))))
            assert target.is_relative_to(DEST) and target.exists(), (str(f),link,str(target))
            checked.append(dict(source=f.relative_to(DEST).as_posix(),link=link,target=target.relative_to(DEST).as_posix()))
    dump(DEST/'LOCAL_LINK_VERIFICATION.json',dict(PASS=True, checked=len(checked), links=checked))
    print(json.dumps(dict(completed=True, package=str(DEST),payload_files=len(included),
        payload_bytes=sum(x['size_bytes'] for x in included),excluded_files=len(excluded),local_html_links=len(checked))))


if __name__ == '__main__':
    main()

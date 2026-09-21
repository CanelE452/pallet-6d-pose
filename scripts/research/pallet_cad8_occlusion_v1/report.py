"""Small report with actual training augmentation snapshots, never generated imagery."""
import html
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.research.pallet_cad8_occlusion_v1 import run as R


def main():
    result=R.read(R.DOC/'RESULTS.json');audit=R.read(R.DOC/'AUDIT.json');preview=R.read(R.OUT/'AUGMENTATION_PREVIEWS.json')
    labels={'R0':'R0 기준','CLEAN':'포즈만 학습 · 깨끗한 입력','OCCLUDED':'포즈만 학습 · 인공 가림 입력'}
    sections=[]
    for name,title in [('PRIMARY_OCC96','주 평가 · occlusion96장'),('ALL_OCC107','보조 · occlusion107장'),('ALL_NONCAD176','CAD 제외 전체176장'),('CLEAN69','가림 없는 평가69장')]:
        table=''
        for a,s in result['summary'][name].items():
            table+=f'<tr><td>{labels[a]}</td><td>{s["matched_pooled_corner8_median_px"]:.2f}</td><td>{s["matched_pooled_corner8_P90_px"]:.2f}</td><td>{100*s["PCK"]["10"]:.2f}%</td><td>{100*s["PCK"]["20"]:.2f}%</td><td>{s["matched"]}/{s["total_frames"]}</td></tr>'
        sections.append(f'<h2>{title}</h2><table><tr><th>모델</th><th>중앙값 px ↓</th><th>P90 px ↓</th><th>PCK10 ↑</th><th>PCK20 ↑</th><th>매칭</th></tr>{table}</table>')
    cards=[]
    for i,p in enumerate(preview,1):
        cards.append(f'<section class="card"><h3>학습 배치 {p["step"]+1} · {html.escape(Path(p["file"]).name)}</h3><p>왼쪽: 공통 기본 augmentation 후 입력 · 오른쪽: RGB 가림 추가. 가린 코너 {p["plan"]["covered_corners"]}. 좌표/학습 마스크는 동일.</p><img src="augmentation_{i:02}.png"></section>')
    head=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>포즈 전용 · 깨끗한 입력 vs 인공 가림</title><style>body{{background:#101c24;color:#eaf2f5;font:17px system-ui;max-width:1400px;margin:auto;padding:24px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #647987;text-align:left}}img{{width:100%}}section{{margin-top:32px;border-top:2px solid #687a86}}a{{color:#78d5ff}}</style><h1>CAD8 수도레이블 · 포즈 전용 인공 가림 실험</h1><p>두 실험 모두 동일한 Replay+PnP 수도레이블. R0에서 시작,320회 업데이트,seed42. 검출부/공유 backbone/BN 통계 고정.</p><p>실사 노출 {audit['real_exposures']}회 중 {audit['occluded_exposures']}회 가림 적용. 입력 순서·가림 전 RGB 샘플·모든 타깃·가림 계획 일치 확인. 차이는 RGB 패치를 칠했는지뿐.</p><p>CAD 전체는 평가에서 제외. 주 평가96장은 Replay teacher의 실사 학습 세션도 제외. 평가에는 학생만 사용하며 PnP/보정기/필터를 적용하지 않았습니다.</p><p>중앙값/P90은 매칭 성공 코너만. PCK는 실패도 포함한 전체 분모. 단일 seed/재사용 DEV의 bounded screen이며 독립 확인이 아닙니다.</p>'''
    R.write(R.OUT/'index.html',head+''.join(sections)+'<h2>실제로 학습에 사용한 인공 가림 예시</h2>'+''.join(cards)+'</html>')
    print(R.OUT/'index.html')


if __name__=='__main__':main()

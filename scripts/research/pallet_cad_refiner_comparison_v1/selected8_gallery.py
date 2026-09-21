"""Display only the eight user-selected CAD frames; no prediction changes."""
import argparse
import re
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.research.pallet_cad_refiner_comparison_v1 import self_occlusion as S

SELECTED=[2,6,7,8,9,10,11,12]


def main(open_page=False):
    inputs=S.G.read(S.BASE/'INPUTS.json')
    ids={inputs[i-1]['id'] for i in SELECTED}
    metrics=S.G.read(S.OUT/'METRICS.json')
    original=(S.OUT/'index.html').read_text()
    cards=dict((int(i),card) for card,i in re.findall(r'(<section class="card" id="frame-(\d+)">.*?</section>)',original,re.S))
    assert len(cards)==18
    table=[];rows=[]
    for arm in S.ARMS:
        selected=[r for r in metrics['rows'] if r['arm']==arm and r['id'] in ids]
        assert len(selected)==8
        for variant in ['before','after']:
            e=np.concatenate([np.array(r[variant+'_errors'])[:8][np.array(r['valid'])[:8]] for r in selected])
            assert len(e)==64
            row=dict(arm=arm,variant=variant,median_px=float(np.median(e)),p90_px=float(np.quantile(e,.9)),PCK10=float(np.mean(e<=10)),PCK20=float(np.mean(e<=20)))
            rows.append(row)
            name=arm+(' + 가림 PnP' if variant=='after' else '')
            table.append(f'<tr><td>{name}</td><td>{row["median_px"]:.2f}</td><td>{row["p90_px"]:.2f}</td><td>{100*row["PCK10"]:.1f}%</td><td>{100*row["PCK20"]:.1f}%</td></tr>')
    head=original.split('<h1>',1)[0].replace('CAD18 자기 가림 점만 PnP 대체','선택 8장 · 자기 가림 PnP 전후')
    intro='<h1>선택한 8장 · R0 / N3 / Replay · 자기 가림 PnP 전후</h1><p>원래 갤러리 번호: 2 · 6 · 7 · 8 · 9 · 10 · 11 · 12. 새로 추론하거나 좌표를 바꾸지 않고 기존 결과만 모았습니다.</p><p><b>초록 = GT · 주황 원 = PnP 대체 전 · 파랑 = 대체 후 · 흰 선 = 실제 이동.</b> 각 방법 아래에 가려진 코너 확대가 있습니다. 보이는 점과 중심점은 그대로 유지했습니다.</p><p>64개 코너 기준. 자기 가림은 직육면체 근사이며, 기존 숨은 점 GT 일부가 PnP 기반일 수 있어 독립적 물리 정확도 검증은 아닙니다.</p>'
    nav='<p>바로 이동: '+' · '.join(f'<a href="#frame-{i}">{i}번</a>' for i in SELECTED)+'</p>'
    table='<table><tr><th>방법</th><th>중앙값 px ↓</th><th>P90 px ↓</th><th>PCK10 ↑</th><th>PCK20 ↑</th></tr>'+''.join(table)+'</table>'
    body=''.join(cards[i].replace(f'{i:02}/18 ·',f'원래 {i}번 ·') for i in SELECTED)
    page=S.OUT/'selected8.html'
    S.G.write(page,head+intro+table+nav+body)
    S.G.write(S.OUT/'SELECTED8_METRICS.json',dict(indices=SELECTED,frames=8,corners=64,rows=rows,source=S.G.binding(S.OUT/'METRICS.json')))
    print(page)
    if open_page:
        from scripts.research.pallet_cad_refiner_comparison_v1.check_self_occlusion import open_browser
        open_browser(page,8,'selected8_')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--open',action='store_true');args=p.parse_args();main(args.open)

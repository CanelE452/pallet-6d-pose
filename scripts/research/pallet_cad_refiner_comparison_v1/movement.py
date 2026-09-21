"""Exact vector visualization of cached corrections; no inference or training."""
from pathlib import Path
import hashlib
import json
import os

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'outputs/pallet_cad_refiner_comparison_v1'
OUT=SOURCE/'movement'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    scores=json.loads((SOURCE/'METRICS.json').read_text())
    inputs=json.loads((SOURCE/'INPUTS.json').read_text())
    arms=['N3_DIM_SYM_seed1','REPLAY_RAW']
    maps={a:{r['id']:r for r in scores['rows'][a]} for a in ['R0',*arms]}
    rows=[]
    for i,r in enumerate(inputs,1):
        assert sha(ROOT/r['image']['path'])==r['image']['sha256']
        before=maps['R0'][r['id']]
        changes=[]
        for a in arms:
            q=maps[a][r['id']]
            assert before['gt']==q['gt'] and before['valid']==q['valid']
            assert before['prediction'][8]==q['prediction'][8]
            changes.append(dict(arm=a,after=q['prediction'],errors=q['errors_px'],movement=q['movement_px']))
        rows.append(dict(number=i,id=r['id'],image=os.path.relpath(ROOT/r['image']['path'],OUT),
            before=before['prediction'],gt=before['gt'],valid=before['valid'],errors_before=before['errors_px'],changes=changes))
    assert len(rows)==18
    OUT.mkdir(exist_ok=True)
    template=(HERE/'movement.html').read_text()
    page=template.replace('__DATA__',json.dumps(rows,ensure_ascii=False).replace('</','<\\/'))
    for path,content in [(OUT/'index.html',page),(OUT/'DATA.json',json.dumps(rows,ensure_ascii=False,indent=2)+'\n')]:
        if path.exists():assert path.read_text()==content
        else:path.write_text(content)
    manifest=dict(frames=18,arms=arms,no_new_inference=True,no_new_training=True,coordinates_changed=False,
        arrows='Start at actual cached R0 coordinate; end at actual cached refiner output. No displacement exaggeration.',
        zoom='The whole RGB patch and all coordinates are scaled together using an SVG viewBox, not amplified arrows.',
        default_corner='Largest displacement among supervised corners with all three points inside the RGB frame; fallback to all corners',
        source_metrics_sha256=sha(SOURCE/'METRICS.json'),source_inputs_sha256=sha(SOURCE/'INPUTS.json'),
        source_code_sha256=sha(Path(__file__)),template_sha256=sha(HERE/'movement.html'))
    (OUT/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(OUT/'index.html')


if __name__=='__main__':main()

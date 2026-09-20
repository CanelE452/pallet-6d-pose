"""Standalone CPU-only browser for frozen non-green predictions; no inference."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/pallet_posefix_other_audit_v1'


def main():
    data=json.loads((OUT/'DATA.json').read_text())
    template=(HERE/'review_template.html').read_text()
    # Share presentation CSS only; keep the previous diagnostic intact.
    old=(HERE.parent/'pallet_posefix_green_audit_v1/review_template.html').read_text()
    style=old.split('<style>',1)[1].split('</style>',1)[0]
    payload=json.dumps(data,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    assert template.count('__AUDIT_DATA__')==template.count('__BASE_STYLE__')==1
    html=template.replace('__BASE_STYLE__',style).replace('__AUDIT_DATA__',payload)
    target=OUT/'NON_GREEN_REFINER_REVIEW.html'
    target.write_text(html)
    (OUT/'HTML_BUILD.json').write_text(json.dumps(dict(
        data_sha256=hashlib.sha256((OUT/'DATA.json').read_bytes()).hexdigest(),
        html_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        images=len(data['records']),inference=False,training=False,
        note='All images and saved outputs embedded; GT is display/scoring only.'),indent=2)+'\n')
    print(target)
    print(f'{target.stat().st_size/1024/1024:.1f} MiB; {len(data["records"])} frozen images')


if __name__=='__main__':main()

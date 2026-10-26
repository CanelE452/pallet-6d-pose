"""Exact original RGB/M4 bytes for the six previously fixed SYN diagnostics."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).absolute().parent
SELECTED=(776,768,770,769,792,900)


def safe(p):
    p=Path(p)
    assert p.is_absolute()and not any(x in str(p).lower()for x in('v112','cache','.pyc','..'))
    for a in reversed((p,*p.parents)):assert not a.is_symlink()
    return p


def main():
    rows=[json.loads(x)for x in safe(HERE/'syn_refine_inputs_v1/FRAME_ORDER.jsonl').read_bytes().splitlines()]
    out=safe(HERE/'syn_selected_raw_inputs_v1');assert not out.exists();out.mkdir()
    files=[]
    for i in SELECTED:
        r=rows[i];assert r['frame_index']==i and r['split']=='val'
        for key in('RGB','M4'):
            b=r[key];raw=safe(Path(b['path'])).read_bytes()
            assert len(raw)==b['size_bytes']and hashlib.sha256(raw).hexdigest()==b['sha256']
            name=f'RAW_{key}_{i:04d}.png';safe(out/name).write_bytes(raw)
            files.append(dict(frame_index=i,source=b,output=name,sha256=b['sha256'],size_bytes=len(raw)))
    with safe(out/'SOURCE_RGB_M4_PAIRS.json').open('x')as f:
        json.dump(dict(completed=True,PASS=True,selection='Samefixed6SYNDEVcasesdefinedbeforeRGBinMETRIC_AND_VISUAL_PLAN_BEFORE_M4_OR_RGB.json',
            original_encoded_bytes_copied_exactly=True,files=files,new_fits_or_inference_or_selection=0),f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps(dict(completed=True,pairs=6,files=len(files))))


if __name__=='__main__':main()

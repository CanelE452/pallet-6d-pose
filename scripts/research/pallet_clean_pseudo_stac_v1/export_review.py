"""Portable review HTML with JPEG previews; no changes to original evidence."""
import base64
import hashlib
import json
from pathlib import Path
import re
import sys
import cv2

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.research.pallet_cad_r0_gt_gallery_v1 import build as G
OUT=ROOT/'_docs/experiments/pallet_clean_pseudo_stac_v1'


def main():
    OUT.mkdir(exist_ok=True,parents=True);entries=[]
    targets=[('outputs/pallet_cad_refiner_comparison_v1/self_occlusion/selected8.html','clean8_portable.html','image'),
             ('outputs/pallet_cad8_occlusion_v1/index.html','occlusion_training_portable.html','img')]
    for source,name,tag in targets:
        src=ROOT/source;content=src.read_text();attribute='href' if tag=='image' else 'src'
        pattern=rf'<{tag} {attribute}="([^"]+)"'
        refs=sorted(set(re.findall(pattern,content)));assert len(refs)==8
        embedded={};sources=[]
        for i,ref in enumerate(refs):
            path=(src.parent/ref).resolve();pixels=cv2.imread(str(path));assert pixels is not None
            if pixels.shape[1]>1000:pixels=cv2.resize(pixels,(1000,round(pixels.shape[0]*1000/pixels.shape[1])),interpolation=cv2.INTER_AREA)
            ok,encoded=cv2.imencode('.jpg',pixels,[cv2.IMWRITE_JPEG_QUALITY,80]);assert ok
            embedded[str(i)]='data:image/jpeg;base64,'+base64.b64encode(encoded).decode('ascii')
            sources.append(dict(key=str(i),**G.binding(path),preview_width=int(pixels.shape[1]),preview_height=int(pixels.shape[0])))
        keys={ref:str(i) for i,ref in enumerate(refs)}
        content=re.sub(pattern,lambda m:f'<{tag} data-preview="{keys[m[1]]}"',content)
        loader='<script>const PREVIEWS='+json.dumps(embedded)+';document.querySelectorAll("[data-preview]").forEach(el=>el.setAttribute(el.tagName.toLowerCase()==="image"?"href":"src",PREVIEWS[el.dataset.preview]));</script>'
        if '</html>' in content:content=content.replace('</html>',loader+'</html>')
        else:content+=loader
        G.write(OUT/name,content)
        entries.append(dict(source=G.binding(src),output=G.binding(OUT/name),images=sources,embedding='JPEG quality80, maximum width1000; display only, original predictions and coordinates unchanged'))
    G.write(OUT/'PORTABLE_MANIFEST.json',dict(exports=entries,code=G.binding(Path(__file__))))
    print(json.dumps({p['output']['path']: (ROOT/p['output']['path']).stat().st_size for p in entries},indent=2))


if __name__=='__main__':main()

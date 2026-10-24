"""Independently inspect source mask files without training or pose code."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest', action='append', required=True)
    p.add_argument('--output', default='YCB_MASK_AUDIT.json')
    args=p.parse_args()
    output=HERE/args.output
    assert output.resolve().is_relative_to(HERE)
    deny=set(json.loads((HERE/'EVALUATION_SHA256_DENYLIST.json').read_text())['image_sha256'])
    totals=Counter();splits={};hashes={}; seen=set();maskbindings=[];metadiffs=[]
    for path in args.manifest:
        rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        for row in rows:
            key=(row['scene_id'],row['frame_id'],row['instance_id'])
            assert key not in seen, ('Duplicate instance',key)
            seen.add(key)
            image=row['image'];split=row['split']
            rgb_hash=hashes.setdefault(image,sha(image))
            assert rgb_hash==row['image_sha256'] and rgb_hash not in deny
            assert rgb_hash not in splits or splits[rgb_hash]==split
            splits[rgb_hash]=split
            vis_raw=np.asarray(Image.open(row['mask_vis']))
            full_raw=np.asarray(Image.open(row['mask_full']))
            assert set(np.unique(vis_raw)) <= {0,255} and set(np.unique(full_raw)) <= {0,255}
            full,vis=full_raw>0,vis_raw>0
            assert full.shape==vis.shape==(row['height'],row['width'])
            assert not (vis&~full).any(), ('Visible not subset full',key)
            v,f,e=int(vis.sum()),int(full.sum()),int((full&~vis).sum())
            assert v==row['vis_pixels'] and f==row['full_pixels'] and e==row['external_proxy_pixels']
            totals['instances']+=1;totals['visible_target_pixels']+=v;totals['external_support_pixels']+=e
            totals['outside_target_support_pixels']+=int((~full).sum())
            info=row.get('bop_gt_info',{})
            if info and (v!=info['px_count_visib'] or f>info['px_count_all']):
                metadiffs.append(dict(scene=key[0],frame=key[1],instance=key[2],visible_mask_minus_metadata=v-info['px_count_visib'],full_inframe_mask_minus_all_metadata=f-info['px_count_all']))
            maskbindings.append(dict(scene=key[0],frame=key[1],instance=key[2],split=split,visible_path=row['mask_vis'],visible_sha256=sha(row['mask_vis']),full_path=row['mask_full'],full_sha256=sha(row['mask_full'])))
    result=dict(schema='YCB_INDEPENDENT_SOURCE_MASK_FILE_AUDIT_V1',source_manifests=[dict(path=str(Path(z).resolve()),sha256=sha(z)) for z in args.manifest],
                totals=dict(totals),unique_images=len(hashes),image_split_counts=dict(Counter(splits.values())),current_eval_sha_overlap=0,
                mask_shape_binary_subset_all_pass=True,source_record_pixel_counts_all_match=True,
                mask_vs_original_metadata_differences=metadiffs,
                target_rule=dict(visible='full1vis1',external='full1vis0',outside='full0vis0',semantic_self='UNKNOWN: not inferred'),
                limitations=['BOP masks are rendered/depth-derived and can misalign with RGB boundaries; original metadata count differences preserved.','Full silhouette does not provide semantic self corner labels.'],
                no_training_inference_PnP=True,mask_file_bindings=maskbindings)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(instances=totals['instances'],unique_images=len(hashes),out=str(output))))


if __name__=='__main__':main()

"""Offline compact replay: 540 count blocks and 50 actual binary/cutout masks."""
from pathlib import Path
import argparse, ast, collections, hashlib, json
import numpy as np
from PIL import Image
W=Path(__file__).absolute().parent
def safe(p):
    p=Path(p)
    assert p.is_absolute() and not any(x in str(p).lower() for x in ('v112','cache','.pyc','..'))
    for a in reversed((p,*p.parents)):assert not a.is_symlink(),str(a)
    return p
def j(rel):return json.loads(safe(W/rel).read_text())
def main():
    arg=argparse.ArgumentParser();arg.add_argument('--out',required=True);args=arg.parse_args();out=safe(W/args.out);assert not out.exists();out.mkdir()
    points=j('kp_comparison/POINT_MODEL_COMPARISON.json')['rows'];metrics=j('kp_comparison/METRICS_OVERALL_MATERIAL_NATIVE.json')['rows']
    for m in metrics:
        rr=[r for r in points if r['truth']in('D','E')and(m['cohort']=='ALL_REGISTERED_D_E'or r['primary'])and(m['group']=='ALL'or r['group']==m['group'])and(str(m['native_slot'])=='ALL'or r['native_slot']==int(m['native_slot']))]
        c=collections.Counter(r['truth']+'_'+r[m['method']]for r in rr);assert len(rr)==m['n']
        for k in ('D_D','D_E','D_U','E_D','E_E','E_U'):assert c[k]==m[k]
        for label in ('D','E'):
            for suffix,den in [('recall',sum(r['truth']==label for r in rr)),('precision',sum(r[m['method']]==label for r in rr))]:
                assert (c[label+'_'+label]/den if den else None)==m[label+'_'+suffix]
    # Compile only the already-inspected saved-image resize function, not the renderer entry point.
    source=safe(W/'render_segmentation_atlas_v271.py').read_text();tree=ast.parse(source);function=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='upsample_raw_logits');namespace={'np':np};exec(compile(ast.Module(body=[function],type_ignores=[]),'<saved-bilinear-function>','exec'),namespace)
    cases=j('segmentation_atlas/CASES.json');checked=[]
    with np.load(safe(W/'segmentation_atlas/SELECTED_RAW32_LOGITS.npz'),allow_pickle=False)as z:
        ids=z['frame_index'].tolist()
        for case in cases:
            at=ids.index(case['frame_index']);rgb=np.asarray(Image.open(safe(W/'segmentation_atlas'/case['rgb_png'])).convert('RGB'));assert list(rgb.shape[:2])==case['raw_hw']
            for model,paths in case['models'].items():
                mask=namespace['upsample_raw_logits'](z[model+'_logit'][at],case['raw_hw'])>=0
                saved=np.asarray(Image.open(safe(W/'segmentation_atlas'/paths['binary_png'])).convert('L'));assert np.array_equal(saved,mask.astype('uint8')*255)
                cut=np.asarray(Image.open(safe(W/'segmentation_atlas'/paths['cutout_png'])).convert('RGB'));assert np.array_equal(cut[mask],rgb[mask]);assert np.unique(cut[~mask].reshape(-1,3),axis=0).shape[0]<=1
                checked.append(dict(frame_index=case['frame_index'],model=model,hard_mask_pixel_sha256=hashlib.sha256(saved.tobytes()).hexdigest(),target_pixels=int(mask.sum())))
    assert len(metrics)==540 and len(checked)==50
    result=dict(PASS=True,completed=True,classification_metric_blocks_exact=540,native_rows=1784,mask_cutout_pixel_checks_exact=50,masks=checked,models_loaded=0,original_RGB_paths_read=0,new_training=0,new_model_forward=0,new_PnP=0,threshold_change=0)
    (out/'REPLAY_RESULT.json').write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='masks'}))
if __name__=='__main__':main()

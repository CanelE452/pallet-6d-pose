"""Apply one approved label version to all frozen backbone scores."""
from collections import Counter
import numpy as np
from .compute import C,R,M,DOC,RAW,SOURCE,write,bind
from scripts.research.pallet_n3_completion_v3 import evaluation as E

def main():
    truth,_=R.load_dev_context(SOURCE,include_pose=False);by={r['id']:r for r in truth}
    labels=C.read(DOC/'STATIC_LABEL_AUDIT.json');vis={(r['id'],r['corner_id']):r['status'] for r in labels['visibility_reviewed_points'] if r['status'] in ['DIRECT_VISIBLE','EXTERNAL_OCCLUDED']}
    output={}
    for backbone in ['dope','resnet18']:
        path=C.RAW/f'evaluation/{backbone}.json';old=C.read(path);raw=C.read(C.RAW/f'predictions/{backbone}_DEV319.json');preds=E.normalize_prediction_payload(raw)['methods'];methods={};visibility={}
        for method,v in old['methods'].items():
            result=v['result']
            for key in ['corner_rows','pose_rows']:
                for r in result[key]:r['material']=by[r['id']]['material'];r['occlusion']=by[r['id']]['occlusion']
            result['subgroups']={field:M._group_summaries(result['corner_rows'],field) for field in ['material','occlusion']}
            result['subgroups']['material_x_occlusion']=M._cross_group_summaries(result['corner_rows'])
            result['pose_subgroups']={field:M._pose_groups(result['pose_rows'],field,M._pose_contract()) for field in ['material','occlusion']}
            methods[method]=result
            groups={k:[] for k in ['DIRECT_VISIBLE','EXTERNAL_OCCLUDED','UNKNOWN']}
            for r in result['corner_rows']:
                if not r['evaluable']:continue
                pred=preds[method][r['id']];points=np.full((9,2),np.nan) if pred['points'] is None else np.array(pred['points']);pv=np.isfinite(points).all(-1)&~(points==-1).all(-1)&r['matched']
                perm=np.array(by[r['id']]['permutations'])[r['branch']];inverse=np.argsort(perm)
                for ci,ok in enumerate(r['canonical_valid']):
                    if ok:groups[vis.get((r['id'],ci),'UNKNOWN')].append((r['canonical_errors'][ci],bool(pv[inverse[ci]])))
            visibility[method]={}
            for group,p in groups.items():
                allv=np.array([x[0] for x in p]);obs=np.array([x[0] for x in p if x[1]])
                visibility[method][group]={'corners':len(p),'observed_corners':len(obs),'median_px':float(np.median(obs)) if len(obs) else None,'P90_px':float(np.quantile(obs,.9)) if len(obs) else None,'PCK10_fraction':float(np.mean(allv<=10)) if len(allv) else None}
        output[backbone]={'base':R._single_panel(methods['base']),'N3':R._seed_panel(methods,'n3'),'visibility':visibility,'source':bind(path),
            'overall_unchanged':True,'label_change':'Prior backbone evaluator had all319 unclassified; now joined to approved direct128+unknown191, no new human labels.'}
    write(DOC/'CROSS_BACKBONE_SUBGROUPS.json',output)

if __name__=='__main__':main()

"""Predefined descriptive group tables, never a rule-selection input."""
import numpy as np
import env as E
from metrics import summarize
def main():
    synth=E.read(E.RAW/'B/SYNTH_FULL_METRICS.json');manifest={r['frame_id']:r for r in E.read(E.EXPORT/'synth_val.json')['records']}
    baseline={r['frame_id']:r for r in synth['B0_P']['1']}
    cuts=np.quantile([r['frame_mean_px'] for r in baseline.values()],[.25,.5,.75])
    groups=[]
    for field in ['symmetry_order','source','asset','GT_assisted_point_error_quartile']:
        values=range(4) if field=='GT_assisted_point_error_quartile' else sorted({manifest[k][field] for k in baseline})
        for value in values:
            ids={k for k in baseline if (int(np.searchsorted(cuts,baseline[k]['frame_mean_px'],side='right')) if field=='GT_assisted_point_error_quartile' else manifest[k][field])==value}
            methods={arm:{seed:summarize([r for r in rows if r['frame_id'] in ids]) for seed,rows in seeds.items()} for arm,seeds in synth.items()}
            groups.append(dict(population='synth_val',group=field,value=value,frames=len(ids),methods=methods))
    dev=E.read(E.RAW/'B/DEV_FULL_METRICS.json');pe=E.C.old('paper_evaluation');meta={r['frame_id']:r for r in E.read(pe.POS)['items']}
    for value in sorted({r.get('object_type','plastic_standard_110x130x11') for r in meta.values()}):
        ids={k for k,r in meta.items() if r.get('object_type','plastic_standard_110x130x11')==value}
        groups.append(dict(population='DEV319',group='object_type_GT_assisted_reporting_only',value=value,frames=len(ids),
          methods={arm:{seed:summarize([r for r in rows if r['frame_id'] in ids]) for seed,rows in seeds.items()} for arm,seeds in dev.items()}))
    E.write(E.DOC/'B/SUBGROUP_RESULTS.json',dict(groups=groups,point_error_quartile_cutpoints_px=cuts.tolist(),
      only_descriptive=True,no_subgroup_tuning=True,not_independent_discoveries=True,unknown_visibility_not_imputed=True))
if __name__=='__main__':main()

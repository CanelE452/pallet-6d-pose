"""Descriptive fixed-population decomposition; never changes the primary estimand."""
import numpy as np
import env as E
def main():
    data=E.read(E.RAW/'A/EVALUATION_FULL_METRICS.json');out={}
    for split,arms in data.items():
        out[split]={}
        for seed in [1,2,3]:
            b=arms[f'INDEXED_seed{seed}'];n=arms[f'EQUIV_seed{seed}'];assert [r['id'] for r in b]==[r['id'] for r in n]
            paired=[(x,y) for x,y in zip(b,n) if x['evaluable'] and y['evaluable']];N=len(paired)
            subsets={
              'both_matched':[(x,y) for x,y in paired if x['matched'] and y['matched']],
              'INDEXED_miss_EQUIV_match':[(x,y) for x,y in paired if not x['matched'] and y['matched']],
              'INDEXED_match_EQUIV_miss':[(x,y) for x,y in paired if x['matched'] and not y['matched']],
              'both_missed':[(x,y) for x,y in paired if not x['matched'] and not y['matched']]}
            value={}
            for name,rows in subsets.items():
                value[name]=dict(frames=len(rows),delta_E_sym_contribution_to_full_mean=sum(y['E_sym']-x['E_sym'] for x,y in rows)/N,
                  conditional_delta_frame_mean_px=float(np.mean([y['frame_mean_px']-x['frame_mean_px'] for x,y in rows])) if rows else None)
            assert abs(sum(v['delta_E_sym_contribution_to_full_mean'] for v in value.values())-sum(y['E_sym']-x['E_sym'] for x,y in paired)/N)<1e-10
            out[split][seed]=value
    E.write(E.DOC/'A/MATCH_AND_GEOMETRY_DECOMPOSITION.json',dict(results=out,
      primary_changed=False,posthoc_descriptive_only=True,denominator='original full evaluated frames; contributions sum to original primary delta',
      conditional_subset='both-matched subset is diagnostic, not a replacement primary or a claim of unconditional improvement'))
    print('A matched/missed contribution decomposition complete',flush=True)
if __name__=='__main__':main()

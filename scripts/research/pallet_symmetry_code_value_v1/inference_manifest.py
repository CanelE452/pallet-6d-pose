"""Extract a GT-free ID/shape manifest once; no predictions or scores are read."""
import numpy as np
import cv_env as E

def main():
    source=E.read(E.D.LINE/'SOURCE_MANIFEST.json')['records'];manifest=E.read(E.D.LINE/'cache/CACHE_MANIFEST.json');indices=np.asarray(manifest['record_indices'])
    rows=[i for i,j in enumerate(indices) if source[j]['partition']=='heldout']
    records=[{k:source[indices[i]][k] for k in ['id','raw_shape_hw','prepared_shape_hw','scenario_id']} for i in rows]
    paper=dict(rows=rows,records=records,source=E.bound(E.D.LINE/'SOURCE_MANIFEST.json'),cache_manifest=E.bound(E.D.LINE/'cache/CACHE_MANIFEST.json'))
    member=E.read(E.D.SYM_RAW/'A/square_membership.json');offset=len(member['train']['records'])
    square=dict(rows=list(range(offset,offset+len(member['val']['records']))),records=[{k:r[k] for k in ['id','raw_hw']} for r in member['val']['records']],source=E.bound(E.D.SYM_RAW/'A/square_membership.json'))
    E.freeze(E.DOC/'INFERENCE_METADATA_ONLY.json',dict(complete=True,populations=dict(SYNTH=paper,SQUARE=square),GT_arrays=False,GT_target_coordinates=False,prediction_scores_used=False))
if __name__=='__main__':main()

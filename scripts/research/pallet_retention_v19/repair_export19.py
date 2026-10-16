"""Add omitted original YOLO coordinates required by shared score_file; preserve predictions."""
from context19 import *

def main():
    source=KIT/'context19.py';old=source.read_text();snapshot=RUN/'context19_at_training.py';snapshot.write_text(old)
    needle="    for k in ('points','base','proposal_delta','alpha','coverage'):"
    replacement="    out=dict(out,original_points=O['points'])\n    for k in ('original_points','points','base','proposal_delta','alpha','coverage'):"
    assert old.count(needle)==1;source.write_text(old.replace(needle,replacement));rows=[]
    for split in ('synth_val','real'):
        for condition in CONDITIONS:
            for p in (RUN/'predictions'/split/condition).glob('*.npz'):
                data=dict(np.load(p));assert 'original_points' not in data;previous=sha(p);seed=p.stem.rsplit('_seed',1)[1];parent=np.load(Q.RUN/'predictions'/split/condition/f'line_vector_seed{seed}.npz');assert np.array_equal(data['ids'],parent['ids']);assert np.array_equal(data['detected'],parent['detected']);data['original_points']=parent['original_points'];np.savez_compressed(p,**data);reloaded=np.load(p)
                assert all(np.array_equal(reloaded[k],v,equal_nan=True) if v.dtype.kind=='f' else np.array_equal(reloaded[k],v) for k,v in data.items());rows.append(dict(path=str(p.relative_to(ROOT)),before_sha256=previous,after_sha256=sha(p)))
    write(RUN/'EXPORT_METADATA_REPAIR.json',dict(reason='Shared scorer requires original_points for relative-to-YOLO metrics. Initial export omitted this metadata; initial scoring stopped before SUMMARY. No learned weights, coordinates, IDs, detections, alpha or GT changed.',source=str(source.relative_to(ROOT)),snapshot=str(snapshot.relative_to(ROOT)),before_source_sha256=sha(snapshot),after_source_sha256=sha(source),needle=needle,replacement=replacement,rows=rows));print('EXPORT METADATA REPAIR',len(rows),flush=True)

if __name__=='__main__':main()

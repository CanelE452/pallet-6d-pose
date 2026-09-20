"""Raw-image review proposals only; never remove evaluation data or score models."""
from collections import Counter
import json
from pathlib import Path
import shutil
from . import common as C
from scripts.evaluation.eval_workspace import (
    canonical_frame_tag_identity, load_frame_tag_overrides, resolve_effective_frame_tags,
)

DOC=C.DOC/'large_corner_recovery_v1/elevation_review'
OUT=C.OUT/'large_corner_recovery_v1/elevation_review'


def export_records(rows, decisions):
    """Validate a proposal; unreviewed rows remain explicit, never auto-dropped."""
    ids={r['id'] for r in rows}
    if set(decisions)-ids:raise ValueError('decision for unknown image')
    result=[]
    for row in rows:
        choice=decisions.get(row['id'],'unreviewed')
        if choice not in {'keep','exclude','unsure','unreviewed'}:raise ValueError('unknown decision')
        result.append(dict(id=row['id'],image_sha256=row['image']['sha256'],decision=choice,
            reason='proposed_extreme_low_elevation' if choice=='exclude' else None))
    return result


def main():
    base=C.DOC/'EVAL_PROTOCOL.json';records=[r for r in C.read(base)['records'] if r['kind']=='PLASTIC']
    assert len(records)==194
    rows=[];sources={};(OUT/'images').mkdir(parents=True,exist_ok=True)
    for i,r in enumerate(sorted(records,key=lambda r:(r['session'],r['id']))):
        C.verify(r['image']);root=(C.ROOT/r['image']['path']).parent.parent
        meta_path=root/'session.json';tag_path=root/'frame_tags.csv'
        meta=C.read(meta_path) if meta_path.exists() else {}
        overrides=load_frame_tag_overrides(root)
        identity=canonical_frame_tag_identity(Path(r['image']['path']).name,session_id=root.name)
        # Elevation is explicit frame/session metadata; do not read GT coordinates.
        tags,provenance=resolve_effective_frame_tags(meta,overrides.get(identity))
        for p in [meta_path,tag_path]:
            if p.exists():sources[str(p)]=C.bound(p)
        relative=f'images/{i:03d}'+Path(r['image']['path']).suffix.lower()
        dest=OUT/relative
        if not dest.exists():shutil.copyfile(C.ROOT/r['image']['path'],dest)
        assert C.sha(dest)==r['image']['sha256']
        rows.append(dict(id=r['id'],session=r['session'],image=r['image'],preview=relative,
            elevation=tags['elevation_bin'],elevation_source=provenance['elevation_bin']))
    counts=dict(Counter(r['elevation'] for r in rows))
    assert counts==dict(low=103,mid=78,high=12,unknown=1)
    plan=dict(status='AWAITING_SCOPE_AND_EXCLUSION_CONFIRMATION',scope='ordinary plastic194 only; GREEN150 and WOOD125 untouched',
        criterion='PROPOSED ONLY: upper face nearly invisible, corresponding corners cannot reliably be distinguished. LOW tag alone never excludes. No model-error threshold.',
        method='Display original RGB and existing elevation tags, order by session/id. No prediction/GT overlays or performance values. Prior human/model development exposure is NOT erased by this view.',
        excluded_count=0,active_dataset_changed=False,model_or_training_changed=False,
        counts=counts,excluded_evaluation_images_remain_prohibited_for_training=True,
        downstream='Only after user-confirmed exclusions: freeze ID list before scoring, apply identical IDs to all arms, report full194 and subset side by side with posthoc selection disclosure. A smaller test is not large-error recovery.',
        sources=[C.bound(base),C.bound(__file__),C.bound(Path(__file__).with_suffix('.html'))]+list(sources.values()))
    C.freeze(DOC/'REVIEW_PLAN.json',plan);C.freeze(OUT/'FRAMES.json',rows)
    payload=json.dumps(dict(rows=rows,plan_sha=C.sha(DOC/'REVIEW_PLAN.json')),ensure_ascii=False).replace('</','<\\/')
    page=Path(__file__).with_suffix('.html').read_text().replace('__PAYLOAD__',payload)
    C.write_text(OUT/'index.html',page)
    C.freeze(DOC/'PREPARED.json',dict(frames=194,default_review=104,exclusions_applied=0,
        images_exact_copies=True,sources=[C.bound(OUT/'FRAMES.json'),C.bound(OUT/'index.html'),C.bound(DOC/'REVIEW_PLAN.json')]))
    print(json.dumps(dict(counts=counts,page=str(OUT/'index.html'),exclusions_applied=0),ensure_ascii=False))


if __name__=='__main__':main()

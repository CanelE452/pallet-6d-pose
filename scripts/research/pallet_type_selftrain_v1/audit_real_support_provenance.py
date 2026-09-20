"""Audit existing annotation provenance only; never manufacture supervision."""
from collections import Counter
import json
from . import real_support_review_v2 as V


def main():
    records=[];counts=Counter()
    for session in V.V.QUOTAS:
        directory=V.C.ROOT/'challenge/data/01_real/manual_gt'/f'{session}_manual_gt'
        for path in sorted(directory.glob('*.json')):
            doc=V.C.read(path);obj=doc['objects'][0]
            manual=obj.get('manual_kps',[])[:8]
            annotations=obj.get('keypoint_annotations',[])[:8]
            source_counts=Counter(a.get('source','unknown') for a in annotations)
            row=dict(annotation=V.C.bound(path),session=session,
                manual_nonnull=sum(p is not None for p in manual),
                manual_null=sum(p is None for p in manual),
                has_point_provenance=bool(annotations),sources=dict(source_counts),
                explicit_manual_click=sum(a.get('source')=='manual_click' for a in annotations),
                legacy_split=obj.get('split'),not_converted_to_training=True)
            records.append(row);counts['files']+=1
            for key in ['manual_nonnull','manual_null','explicit_manual_click']:
                counts[key]+=row[key]
            counts['files_with_point_provenance']+=int(row['has_point_provenance'])
    result=dict(complete=True,scope='Legacy annotations in the7 proposed support/validation sessions only',
        counts=dict(counts),records=records,source=V.C.bound(__file__),
        status='HUMAN_PROVENANCE_AND_CORNER_REVIEW_REQUIRED',
        interpretation='Nonnull manual_kps is not proof of a human click. Do not infer manual provenance from integer coordinates, pose residual, or visual plausibility. No label coordinates changed, no masks inferred, no training.',
        broader_claim='This scoped audit does not claim every annotation elsewhere in the repository is unusable.')
    V.C.freeze(V.DOC/'EXISTING_PROVENANCE_AUDIT.json',result)
    print(json.dumps(result['counts'],indent=2))


if __name__=='__main__':main()

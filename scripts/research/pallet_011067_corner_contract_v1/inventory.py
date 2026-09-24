from . import common as C

def main():
    rows=[
      dict(source='annotation tool',file='scripts/annotate/annotate_pnp.py',definition='near/front 0..3, local -Z; X right/Y down; LR image x, TB image y, FR camera depth',limit='LR/TB/FR can admit two front-role candidates; diagnostic area call may clip to image'),
      dict(source='historical converter',file='scripts/annotate/convert_to_camera_facing_v4.py',definition='physical top/bottom; parallel side-pair max projected-area difference; larger side front; image-x LR',limit='Not proven to be current renderer writer; no independent origin3D in real011067'),
      dict(source='paper physical/CF frame',file='_docs/archive/paper_support_20260830/real_gt_v2/FRAME_CONVENTION.md',definition='canonical physical XYZ vs camera-facing frame; proper yaw role transforms; W/D swap at90/270',limit='Rectangular physical yaw180 equivalence is not arbitrary C4 scoring invariance'),
      dict(source='semantics audit',file='scripts/research/pallet_dht_structured_v2/SEMANTICS.md',definition='DLT/PnP alone may absorb semantic C4; role and physical symmetry differ',limit='Older two-sample renderer discussion superseded in coverage by later60k audit; not contradiction'),
      dict(source='renderer evidence',file='_docs/experiments/pallet_type_selftrain_v1/selftrain_recovery_v1/renderer_front_visibility_audit/INTERPRETATION_KO.md',definition='60000/60000 stored front = max side-normal dot object-center-to-camera ray',limit='Source evidence, not real011067 authority; writer implementation unavailable')]
    for r in rows:r['binding']=C.bind(C.ROOT/r['file'])
    C.put(C.DOC/'CONVENTION_INVENTORY.json',dict(rows=rows,single_identical_definition_established=False))
    text='# Convention inventory\n\n|출처|실제 정의|한계|\n|---|---|---|\n'+'\n'.join(f'|{r["source"]}|{r["definition"]}|{r["limit"]}|' for r in rows)+'\n'
    C.put(C.DOC/'CONVENTION_INVENTORY.md',text)
    directory=C.ROOT/'_docs/experiments/pallet_type_selftrain_v1/selftrain_recovery_v1/renderer_front_visibility_audit'
    audit=C.read(directory/'COMPLETION_AUDIT.json');b=audit['summary']
    assert C.sha(C.ROOT/b['path'])==b['sha256']
    result=C.read(C.ROOT/b['path'])
    for b in result['evidence']:assert C.sha(C.ROOT/b['path'])==b['sha256']
    r0=C.read(C.ROOT/'_docs/experiments/pallet_sensors_submission_v1/R0_PRETRAINING_PROVENANCE.json')
    for k in ('baseline','run_args','initialization_checkpoint'):
        b=r0[k];assert C.sha(C.ROOT/b['path'])==b['sha256']
    rows=[]
    for source,m in result['by_source'].items():
        c=result['center_by_source'][source]
        rows.append(dict(source=source,stored_convention='camera_dynamic_0123_v4',
            perm_v4_provenance='stored in original renderer annotation; builder consumes, does not choose front',
            total=m['total'],renderer_center_front_max=c['center_front_is_max'],
            historical_area_identity=m['legacy_identity'],historical_area_eligible=m['legacy_complete'],
            stored_front_visibility_cos_count=m['stored_cos_count'],stored_cos_max_error=m['stored_cos_max_error'],
            writer_implementation_available=False,real_GT_authority=False))
    C.put(C.DOC/'TRAINING_CONVENTION_PROVENANCE.json',dict(rows=rows,source_audit_binding=C.bind(directory/'RESULTS.json'),
        source_evidence_hashes_verified=True,R0_provenance=r0,
        manifest_binding=C.bind(C.ROOT/'data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json'),
        convention_mix_signal='POSSIBLE_RULE_MIX_NOT_PROVEN_REAL_LABEL_ERROR',
        limit='Existing60k audit reused and bound; raw60000 annotations not independently rescanned in this task; real GT never auto-edited'))
    C.figure_table(C.FIG/'06_training_convention_provenance.png','Source convention: renderer evidence is NOT real-image GT authority',
        ['Source','Count','Center-normal front match','Historical area match / eligible'],
        [[r['source'],r['total'],r['renderer_center_front_max'],f'{r["historical_area_identity"]}/{r["historical_area_eligible"]}'] for r in rows])
    C.put(C.DOC/'VERIFIED_ANCHOR_QA_FINAL.json',C.read(C.ANCHOR/'METADATA_QA_FINAL.json'))
    scores=C.read(C.ANCHOR/'VERIFIED_RESULTS_FINAL.json')['groups']['ALL']
    teacher=C.read(C.ANCHOR/'TEACHER_SUPPLEMENT_FINAL.json')['groups']['ALL']
    C.figure_table(C.FIG/'01_verified_anchor_qa_before_after.png','Human QA complete2/2 | direct-visible66 unchanged | teacher supplementary only',
        ['Arm','Provisional PCK10','Final PCK10','Final median px','Final >20'],
        [[a,f'{m["PCK"]["10"]["correct"]}/{m["n"]}' if a!='TEACHER' else 'not scored',
          f'{m["PCK"]["10"]["correct"]}/{m["n"]}',f'{m["median_px"]:.3f}',m['gt20']]
         for a,m in {**scores,'TEACHER':teacher}.items()])
    print('Inventory source/R0 bindings verified')

if __name__=='__main__':main()

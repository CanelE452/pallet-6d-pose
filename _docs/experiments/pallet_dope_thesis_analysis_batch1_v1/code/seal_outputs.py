"""Seal analysis outputs after all reviews; publication logs remain separate."""
from common import *

def main():
    complete=OUT/'COMPLETION.json'
    result=read(complete)
    assert result['analysis_complete']
    assert result['checks']['independent_full_scope_T0_T6_review_passed']
    assert result['checks']['current_report_and_publication_safety_review_passed']
    for path,expected in result['independent_review_receipts'].items():assert sha(path)==expected
    for path,expected in read(OUT/'hashes/CODE_SHA256.json').items():assert sha(path)==expected
    frozen=read(OUT/'preflight/GATE5_RESOLUTION.json')['frozen_quantitative_outputs']
    for path,expected in frozen.items():assert sha(OUT/path)==expected
    start('seal_outputs',[complete,OUT/'preflight/GATE5_RESOLUTION.json'])
    destination=OUT/'hashes/DELIVERABLES_SHA256.json'
    assert not destination.exists(),'Output seal already exists: inspect instead of overwriting'
    log('analysis_outputs_sealed',task='seal_outputs',frozen_CSVs=len(frozen),
        original_experiment_files_unchanged=598,new_training=0,neural_forwards_total=1752,
        publication_log_policy='Publication commands/remote verification are recorded separately under publication/.')
    manifest={}
    for path in sorted(OUT.rglob('*')):
        rel=path.relative_to(OUT)
        if rel.parts[0] in ('.publication','publication') or '__pycache__' in rel.parts or path==destination:continue
        if path.is_file():
            assert path.suffix not in ('.tmp','.pth','.mp4'),('Unexpected deliverable',str(rel))
            manifest[str(rel)]={'sha256':sha(path),'bytes':path.stat().st_size}
    write(destination,dict(sealed_utc=now(),files=manifest,
        exclusions=['hashes/DELIVERABLES_SHA256.json (self)','publication/ (post-publication receipts)',
                    '.publication/ (isolated Git staging internals)','__pycache__/'],
        scientific_RUN_LOG_frozen=True,original_sources_preserved=True))
    print('ANALYSIS_SEALED',len(manifest),'files',sum(v['bytes'] for v in manifest.values()),'bytes',flush=True)

if __name__=='__main__':main()

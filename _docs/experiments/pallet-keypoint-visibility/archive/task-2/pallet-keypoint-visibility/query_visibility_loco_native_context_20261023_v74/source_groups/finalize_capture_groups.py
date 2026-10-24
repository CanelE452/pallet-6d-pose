"""Additive group/runtime interpretation of immutable completed source audit."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib
import json
import sys
import numpy as np
import PIL
from PIL import Image,features,_imaging

HERE=Path(__file__).resolve().parent
V73=HERE.parents[1]/'query_visibility_public_pallet_loco_20261023_v73/data/intake'


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]


def main():
    assert not (HERE/'SOURCE_GROUP_FREEZE.json').exists(),'No overwrite'
    audit=json.loads((HERE/'CAPTURE_AUDIT_RESULT.json').read_text());assert audit['completed']
    records=rows(HERE/'RGB_TEMPORAL_HASH_MANIFEST.jsonl');by_path={r['archive_member']:r for r in records}
    original20=rows(V73/'FIXED_20_RGB_MANIFEST.jsonl')
    assert all(r['RGB_pixels_sha256']==by_path[r['archive_member']]['decoded_RGB_SHA256'] and r['raw_hw']==by_path[r['archive_member']]['raw_hw'] for r in original20)
    runtime=dict(schema='V74_EXACT_PUBLIC_RGB_DECODER_RUNTIME',executable=str(Path(sys.executable).resolve()),python=sys.version,
        Pillow=PIL.__version__,numpy=np.__version__,PIL_package=bind(PIL.__file__),PIL_Image=bind(Image.__file__),PIL_imaging_binary=bind(_imaging.__file__),
        jpeglib_version=Image.core.jpeglib_version,libjpeg_turbo=features.check_feature('libjpeg_turbo'),
        decode_recipe='Image.open(BytesIO(originalJPEG)).convert(RGB); no EXIF transpose, no resize before decodedSHA.',
        V73_fixed20_decoded_RGB_SHA_parity=dict(queries=20,bit_hash_mismatches=0,reference=bind(V73/'FIXED_20_RGB_MANIFEST.jsonl')),
        warning='Decoded-pixel hashes are decoder/runtime-bound. A different legacy Pillow/JPEG build can alter decoded RGB without a different capture. JPEG-file SHA remains the source-byte identity; never infer different capture or allow tolerance merely from cross-runtime decoded differences.',
        actual_legacy_decode_repeated_here=False)
    write(HERE/'DECODER_RUNTIME_AND_PARITY.json',runtime)
    sessions=json.loads((HERE/'PATH_CAPTURE_SESSION_TABLE.json').read_text())['sessions'];overlap=[]
    for i,a in enumerate(sessions):
        for b in sessions[i+1:]:
            if a['subsets']!=b['subsets'] or a['min_epoch'] is None or b['min_epoch'] is None:continue
            lo=max(a['min_epoch'],b['min_epoch']);hi=min(a['max_epoch'],b['max_epoch'])
            if lo<=hi:overlap.append(dict(pathfamilyA=a['path_capture_family'],pathfamilyB=b['path_capture_family'],overlap_epoch_interval=[lo,hi]))
    streams=defaultdict(list)
    for r in records:
        clock='UNIX_NORMALIZED_SECONDS' if r['timestamp_epoch_seconds'] is not None else r['timestamp_kind']
        streams[(r['path_stream_key'],clock)].append(r)
    sequence=[]
    for (stream,clock),rr in sorted(streams.items()):
        timed=sorted((r['timestamp_epoch_seconds'] if clock=='UNIX_NORMALIZED_SECONDS' else r['timestamp_filename_number'],r['index']) for r in rr if r['timestamp_filename_number'] is not None)
        values=np.array([x[0] for x in timed],float);delta=np.diff(values)
        sequence.append(dict(path_stream=stream,clock=clock,n=len(rr),ordered_indices=[i for t,i in timed],first_timestamp=float(values[0]) if len(values) else None,last_timestamp=float(values[-1]) if len(values) else None,
            adjacent_gap_q50=float(np.quantile(delta,.5)) if len(delta) else None,adjacent_gap_q95=float(np.quantile(delta,.95)) if len(delta) else None,gap_unit='seconds' if clock=='UNIX_NORMALIZED_SECONDS' else 'UNVERIFIED_SENSOR_UNIT'))
    write(HERE/'TEMPORAL_SEQUENCE_AND_OVERLAP_AUDIT.json',dict(completed=True,path_metadata_families=len(sessions),clock_stream_groups=len(sequence),same_subset_path_family_time_overlaps=overlap,
        interpretation='Overlapping timestamp spans across date_take directories show they cannot be assumed independent capture families. Whole official subset is the conservative split unit.',sequences=sequence))
    groups=[]
    for r in records:
        groups.append(dict(r,source_family='LOCO_AUTHOR_ENVIRONMENT_SUBSET:'+str(r['subset']),grouping_basis='Entire official subset; preserve all camera/date_take/sequence members; author-defined logistics environment.',source_site_identity='Author environment partition; universal physical-site independence not independently proved'))
    (HERE/'CONSERVATIVE_SOURCE_GROUP_MANIFEST.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in groups))
    vis=dict(schema='V74_ACTUAL_FIXED_SOURCE_CAPTURE_REVIEW_V1',completed=True,figures=audit['figures'],all3_figures_opened_and_inspected=True,
        fixed_crosssplit_pairs=12,fixed_subset1_5_pairs=2,selection='Predefined perceptual-hash top2 per source-subset pair, no model/semantic-performance selection.',
        observations_KO=[
            'subset1/5 고정 두 쌍은 창고 출입구/랙 통로의 일반 구조가 유사하지만, 같은 물품 배치나 고유 표식의 일치를 확인하지 못했습니다. 2019-07-31과2020-03-05로 날짜도 다릅니다. 같은 물리 site 재방문 가능성을 해시로 배제하지 않습니다.',
            'subset4/5의 가장 비슷한 후보도 yellow rack post/aisle 관점이 닮지만 화물과 시점이 다르며 dHash15/16,RGB_MAE.130/.138로 고정 strongnear 조건에 들지 않습니다. 이는 유사한 물류 장면이며 실제 동일 capture로 확정하지 않습니다.',
            '다른 고정 후보 역시 통로/밝은 출입구/스택 형태가 공통이나 사람/물체/벽의 고유 일치나 동일 프레임 재사용은 확인하지 못했습니다. 이 제한된 그림은 모든 가능한 동일-site 관계 검증을 대체하지 않습니다.'
        ],no_candidate_is_declared_independent_from_visual_resemblance_alone=True,no_new_model_forward_training_current_PnP=0)
    write(HERE/'ACTUAL_FIXED_CAPTURE_VISUAL_REVIEW.json',vis)
    # Short primary-source paraphrase, not a replacement for local actual counts.
    provenance=dict(official_README=bind(V73/'README_OFFICIAL.md'),paper_URL='https://mediatum.ub.tum.de/doc/1578845/1578845.pdf',
        paper_title='LOCO: Logistics Objects in Context',section='IV-B training/evaluation partition',
        source_claim='Authors state each subset represents one particular logistics environment and designate2/3/5 training,1/4 evaluation.',
        retrieval='Official TUM PDF indexed text was returned by web search. Direct PDF opens returnedHTTP429; no local full-PDF copy or complete-paper verification claimed.',
        use='Primary-source support for the official environment partition, scoped separately from our actual archive/hash/temporal observations.',
        our_inference='Current observations do not establish subset1 and5 are the same capture; generic racks are insufficient grounds for a new merge. Nor does zero hash duplication prove universal physical-site independence.')
    write(HERE/'OFFICIAL_ENVIRONMENT_SPLIT_EVIDENCE.json',provenance)
    freeze=dict(schema='V74_CONSERVATIVE_LOCO_SOURCE_GROUP_FREEZE_V1',completed=True,metadata_and_actual_RGB_audit_complete=True,
        input_count=5593,source_family_count=5,source_family_unit='Entire official subset, never image-level or date_take random split.',
        TRAIN_subsets=[2,3,5],VALIDATION_subsets=[1,4],source_counts_by_subset=dict(Counter(str(r['subset']) for r in records)),
        temporal_pathfamilies_19_are_not_independent_holdout_groups=True,overlapping_path_family_time_intervals=len(overlap),
        exact_decoded_duplicates=0,fixed_cross_subset_strongnear_pairs=0,official_split_crossing_detected_evidence_components=0,
        subset1_vs5='Kept as separate author-defined source environments; no unique same-capture evidence in fixed candidates. Physical-site revisit remains unruled-out.',
        split_use='Support for a source experiment using the pre-existing official environment partition, not a universal unseen-site or semantic-visibility generalization guarantee.',
        unannotated496_remain_U=True,head_or_semantic_label_admission=False,current_data_used=False,new_model_forward_training_PnP=0,
        manifest=bind(HERE/'CONSERVATIVE_SOURCE_GROUP_MANIFEST.jsonl'),audit=bind(HERE/'CAPTURE_AUDIT_RESULT.json'),
        runtime=bind(HERE/'DECODER_RUNTIME_AND_PARITY.json'),temporal=bind(HERE/'TEMPORAL_SEQUENCE_AND_OVERLAP_AUDIT.json'),
        visual=bind(HERE/'ACTUAL_FIXED_CAPTURE_VISUAL_REVIEW.json'),primary_source_evidence=bind(HERE/'OFFICIAL_ENVIRONMENT_SPLIT_EVIDENCE.json'),code=bind(__file__))
    write(HERE/'SOURCE_GROUP_FREEZE.json',freeze)
    files=sorted(p for p in HERE.rglob('*') if p.is_file() and p.name!='FINAL_SOURCE_GROUP_AUDIT_SEAL.json' and '__pycache__' not in p.parts)
    write(HERE/'FINAL_SOURCE_GROUP_AUDIT_SEAL.json',dict(completed=True,PASS=True,scope='Public source data grouping only; no model/GT-current/fit/PnP.',source_group_freeze=bind(HERE/'SOURCE_GROUP_FREEZE.json'),files=[bind(p) for p in files]))
    print(json.dumps(dict(freeze=bind(HERE/'SOURCE_GROUP_FREEZE.json'),seal=bind(HERE/'FINAL_SOURCE_GROUP_AUDIT_SEAL.json'),overlapping_path_intervals=len(overlap),runtime=runtime['Pillow'])))


if __name__=='__main__':main()

"""Copy an explicit experiment inventory, without traversing the project."""
from pathlib import Path
import gzip
import hashlib
import json
import os

R = Path(__file__).parent
OUT = R / "V236_PUBLICATION_PACKAGE"
TOPS = ['V234_MODEL_INPUT_AUDIT_A', 'V234_PRIOR_CF_AUDIT_A', 'V234_YCB_SHIFT_AUDIT_B', 'V234_ROOT_LABEL_DIAG', 'V234_SOURCE_PROVENANCE_C', 'V234_SELECTED_GT110_LINEAGE_C', 'V234_RENDERER_LINEAGE_C', 'V234_AUDIT_REPORT_C', 'V235_ROOT_PUBLIC_COUNTERFACTUAL', 'V235_CF_COMPOSITOR_A', 'V235_DONOR_MANIFEST_B', 'V235_CF_DATA_QA_B', 'V235_CONTINUOUS_REF_QA_B', 'V235_DATA_ONLY_TRAINER_A', 'V235_TRAIN_COMPLETION_QA_B', 'V235_SOURCE_REPORT_C', 'V235_SOURCE_GALLERY_A', 'V235_PORTABLE_CF_VERIFY_A', 'V236_SOURCE_CONTEXT_DIAGNOSTIC', 'V236_CONTEXT_FRAMES_B', 'V236_CONTEXT_ROUTE_C']
EXCLUDED = {'V235_ROOT_PUBLIC_COUNTERFACTUAL/pairs_v1/PAIR_RGB96.npy'}


def lexical(p):
    s = str(p)
    if "v112" in s.lower() or "__pycache__" in Path(p).parts or s.lower().endswith(".pyc"):
        raise ValueError("forbidden lexical path")
    return Path(p)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def inventory(p):
    lexical(p)
    if p.is_symlink():
        raise ValueError("symlink")
    for name in sorted(os.listdir(p)):
        # Filtering occurs before any filesystem metadata call on an entry.
        if "v112" in name.lower() or name == "__pycache__" or name.lower().endswith(".pyc"):
            continue
        f = lexical(p / name)
        rel = f.relative_to(R).as_posix()
        if rel in EXCLUDED:
            continue
        if f.is_symlink():
            raise ValueError("symlink")
        if f.is_dir():
            yield from inventory(f)
        else:
            yield f


def main():
    OUT.mkdir(exist_ok=False)
    sources = []
    for top in TOPS:
        sources.extend(inventory(R / top))
    # Executed pure helper code; no old mixed datasets or original RGB is copied.
    crop_complete = json.loads((R / "V233_ROOT_CROP_ADAPTATION/crop_samples_v1/COMPLETE.json").read_text())
    helpers = []
    for key in ("point_contract", "condition_contract", "crop_code"):
        binding = crop_complete["inputs"][key]
        p = lexical(binding["path"])
        data = p.read_bytes()
        assert digest(data) == binding["sha256"]
        helpers.append((f"artifacts/contracts/{key}.py", data, {"path": str(p), "sha256": digest(data), "bytes": len(data)}))
    source_map = []
    for p in sources:
        rel = p.relative_to(R).as_posix()
        data = p.read_bytes()
        assert len(data) < 160_000_000, rel
        compression = p.suffix == ".jsonl" and len(data) > 900_000
        storage = "artifacts/" + rel + (".gz" if compression else "")
        d = OUT / storage
        d.parent.mkdir(parents=True, exist_ok=True)
        d.write_bytes(gzip.compress(data, compresslevel=9, mtime=0) if compression else data)
        source_map.append({"path": storage, "source_relative_to_preparation": rel, "source_sha256": digest(data), "source_bytes": len(data), "encoding": "gzip-lossless" if compression else "identity"})
    for rel, data, binding in helpers:
        p = OUT / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        source_map.append({"path": rel, "source_binding": binding, "source_sha256": digest(data), "source_bytes": len(data), "encoding": "identity"})
    verifier = R / "V233_PUBLICATION_VERIFY_A/verify_package.py"
    vb = verifier.read_bytes()
    assert digest(vb) == "bd34cada69b56591171a3ee0686bb4a6eb8c16c3bdf8ebfe994db13b7dac30bc"
    (OUT / "verify_package.py").write_bytes(vb)
    (OUT / "build_v236_publication.py").write_bytes(Path(__file__).read_bytes())
    # Extra immutable portable aliases and parent SOURCE metadata, not raw images.
    aliases = [
        ("DIAGNOSIS_V234_KO.md", R / "V234_AUDIT_REPORT_C/PORTABLE_REPORT_KO.md", False),
        ("verify_cf_teacher.py", R / "V235_PORTABLE_CF_VERIFY_A/verify_cf_teacher.py", False),
        ("artifacts/parent_SOURCE/SAMPLE_DIAGNOSTICS.jsonl.gz", R / "V233_ROOT_CROP_ADAPTATION/crop_samples_v1/SAMPLE_DIAGNOSTICS.jsonl", True),
    ]
    for storage, source, compressed in aliases:
        lexical(source)
        data = source.read_bytes()
        target = OUT / storage
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(gzip.compress(data, compresslevel=9, mtime=0) if compressed else data)
        source_map.append({"path":storage,"source_relative_to_preparation":source.relative_to(R).as_posix(),"source_sha256":digest(data),"source_bytes":len(data),"encoding":"gzip-lossless" if compressed else "identity"})
    (OUT / "THIRD_PARTY_NOTICES.md").write_text(NOTICE)
    write_json(OUT / "PUBLICATION_SOURCE_MAP.json", {"source_root": str(R), "files": source_map, "non_payload_exclusions": sorted(EXCLUDED), "exclusion_note_KO": "신규 합성RGB299MB는Git payload에서제외했다. 기존원rawRGB/valid/prefix도복제하지않고원SHA·크기·구성코드·저장결과를보존한다. 신규private/current/held payload는없다."})
    (OUT / "README_KO.md").write_text(README)
    # Only known output package is inventoried; no external paths are followed.
    manifest_files = []
    for p in package_files(OUT):
        b = p.read_bytes()
        manifest_files.append({"path": p.relative_to(OUT).as_posix(), "sha256": digest(b), "bytes": len(b)})
    write_json(OUT / "MANIFEST.json", {"schema": "V236_PORTABLE_PUBLICATION_V1", "files": manifest_files, "file_count": len(manifest_files), "total_bytes": sum(f["bytes"] for f in manifest_files), "scope": "V234 source diagnosis, V235 completed counterfactual data-only experiment, V236 context diagnostics; gate failed, no current-real success claim", "current_admission": False, "source_gate_pass": False})
    print(json.dumps({"package": str(OUT), "files": len(manifest_files), "bytes": sum(f['bytes'] for f in manifest_files), "manifest_sha256": digest((OUT / 'MANIFEST.json').read_bytes())}))


def package_files(p):
    for n in sorted(os.listdir(p)):
        f = lexical(p / n)
        assert not f.is_symlink()
        if f.is_dir():
            yield from package_files(f)
        else:
            yield f


README = '# V234–V236: 공개 실사 실패 진단과 같은 점 가림 대조 실험\n\n**대조쌍 학습은 완료했으나 실사 가림 판별을 개선하지 못했다. 모델을 채택하지 않았다.** 현재223/held는 새로 읽거나 추론·학습·문턱 선택에 사용하지 않았다. PnP와 원 q/L/native도 변경하지 않았다.\n\n|원 SOURCE 내부 DEV|고정 encoder/기존 head D·E recall|같은 구조 + 새 가림 대조쌍 D·E recall|\n|---|---|---|\n|SYN main N1344|98.84% · 91.07%|98.68% · 92.86%|\n|YCB AUX N397|97.94% · 0/57|96.76% · 0/57|\n\n사전 기준은 두 가족 평균 균형 recall +3pp 이상, 각 D/E recall 하락2pp 이하였다. 관측 gain **+0.113491pp**로 실패했다. SYN witness0와 YCB associated-r surface(native UNKNOWN)는 서로 다른 referent로 따로 평가한다. SOURCE DEV는 이미 노출된 내부 진단 자료이며 blind 평가나 현재 실사 성공이 아니다. 원7,617행 입력은 전부 available로 모델 U율0%; target−1인7,423행을 모델 보류로 세지 않는다.\n\n## 결과와 실제 영상\n\n- [V234 진단 통합 보고서](DIAGNOSIS_V234_KO.md): 코드 오류·좌표·라벨·물체×표면점 구성 차이 확인\n- [V235 전체 CM/precision/recall/보류율/paired 변화](artifacts/V235_SOURCE_REPORT_C/results/PORTABLE_REPORT_KO.md) · [전체7617행 비교 CSV](artifacts/V235_SOURCE_REPORT_C/results/PAIRED_ALL7617.csv)\n- [같은 점에 가림물을 덮은 경우와 옆에 둔 경우](artifacts/V235_ROOT_PUBLIC_COUNTERFACTUAL/fixed2_prefit_gallery_v2/PREFIT_COUNTERFACTUALS.png)\n- [실제 DEV 성공·실패·회복·악화 이미지](artifacts/V235_SOURCE_GALLERY_A/gallery_v1/FIXED12_SOURCE_DEV.png) · [로컬 HTML gallery](artifacts/V235_SOURCE_GALLERY_A/gallery_v1/PORTABLE_GALLERY_KO.html)\n- [작은 crop과 원 공개 장면을 나란히 보기](artifacts/V236_CONTEXT_FRAMES_B/three_source_context_v1/THREE_SOURCE_CONTEXT_BOARD.png) · [로컬 HTML](artifacts/V236_CONTEXT_FRAMES_B/three_source_context_v1/PORTABLE_CONTEXT_GALLERY_KO.html)\n- [crop 밖 다른 query의 대상 픽셀 존재 진단](artifacts/V236_SOURCE_CONTEXT_DIAGNOSTIC/OTHER_QUERY_OWNERSHIP_SUMMARY.json) · [원 q8–실제 RGB 연결 경로](artifacts/V236_CONTEXT_ROUTE_C/KO.md)\n- [사전 고정 실험 조건](artifacts/V235_ROOT_PUBLIC_COUNTERFACTUAL/PROTOCOL_BEFORE_DONOR_PIXELS_OR_TREATMENT.json) · [실제 학습 완료](artifacts/V235_ROOT_PUBLIC_COUNTERFACTUAL/data_only_source_fit_v1/COMPLETE.json)\n- [합성 쌍 독립 QA](artifacts/V235_CF_DATA_QA_B/actual_review_v2/REVIEW.json) · [continuous reference 추가 검산](artifacts/V235_CONTINUOUS_REF_QA_B/CONTINUOUS_REFERENCE_4TAP_QA.json) · [학습·저장 출력 독립 QA](artifacts/V235_TRAIN_COMPLETION_QA_B/ACTUAL_QA_COMPLETE.json)\n\n## 이번에 실제로 한 일\n\nV234에서는 YCB DEV E57에 같은 object×aux slot의 FIT E 지원이0/57임을 확인했다. 같은 조합의 FIT D는 있었다. q 자체도56/57은 다른 물체 픽셀 위였고, q–r≤3px인 E30도 모두 그랬다. 좌표 차이만으로 전체 실패를 설명할 수 없다는 근거다. loss·출력 indexing·행 pairing의 명백한 코드 오류는 허용된 코드/저장값 감사에서 발견하지 못했다. 식별 불가능이나 원인 확정의 증거는 아니다.\n\nV235는 모델 구조를 고정한 데이터 대조다. 원 SOURCE FIT D5,406개에 공개 YCB FIT의64개 donor(물체별최대16, scene0..4)만 사용해 COVER와 SHAM을 만들었다. 같은 q/L에서 원 기준점 주변3×3의 모든 raw96 tap을 덮은 COVER는 E, 같은 donor를 옆에 두고 그 support를 보존한 SHAM은 D다. 성공5,367쌍(10,734영상), 실패39개는 masked로 기록하고 원 D를 유지했다. continuous r 자체의 네 tap도 전수 검산했다. donor가 없거나 S/U인 자료를 E로 만들지 않았다. 합성 쌍은 새 자연 관측이나 human gold가 아니다.\n\n기존 CONTROL의 완료1000회 checkpoint/log/CSV를 bytes 그대로 재사용했다(새CONTROL update0). 같은 DINO 고정 encoder와1186→64→2 head(76,098개), 같은 초기 head·원 PCG17 ledger·AdamW·문턱0으로 새 head만1000회 학습했다. 각 가족8D8E 중 D절반은 성공 SHAM, E절반은 별도 PCG235의 성공 COVER로 대체했다. 새 head를 고정한 뒤 원7,617행을 출력했고, 기존 CONTROL의두logit을7,617행 전부 byte까지 재현했다. LoRA는 사용하지 않았다. SOURCE/증강 자료와 초기·최종 조건이 바뀐 부분을 로그와 ledger로 확인할 수 있다.\n\nV236에서는 실패57개 중56개가 다른 원 query에 대상 물체 픽셀이 crop밖에 있음을 저장 SOURCE ownership 진단으로 확인했다. 세 전체장면에는 비슷한 상자가 여러 개 보인다. 이는 crop 밖 정보의 존재 증거이며 새 모델의 성공·인과·대상 정체의 새 human label을 뜻하지 않는다. 다음은 원 q8로 정의한 전체 영역을 실제 source RGB와 연결하는 입력 검증이다. C11 fullcontext는 이미 시도된 방법이므로 첫 context 사용이라고 주장하지 않는다. [다음 입력 검증 범위](artifacts/V236_SOURCE_CONTEXT_DIAGNOSTIC/ROOT_NEXT_INPUT_DECISION_AFTER_SOURCE_FAILURE.json)를 별도로 고정했다. 새 context 모델은 아직 학습하지 않았다.\n\n## 다른 컴퓨터에서 재검산\n\n이 폴더에서 Python3 표준 라이브러리만으로 실행한다.\n\n```bash\npython3 -B verify_package.py --package . --manifest MANIFEST.json\npython3 -B artifacts/V235_SOURCE_REPORT_C/summarize_pair.py --verify-saved\npython3 -B verify_cf_teacher.py --metadata artifacts/V235_ROOT_PUBLIC_COUNTERFACTUAL/pairs_v1/PAIR_METADATA.jsonl.gz --alpha artifacts/V235_ROOT_PUBLIC_COUNTERFACTUAL/pairs_v1/PAIR_ALPHA_PACKBITS.npy --parent-diagnostics artifacts/parent_SOURCE/SAMPLE_DIAGNOSTICS.jsonl.gz --ycb-reference artifacts/V234_ROOT_LABEL_DIAG/YCB_KNOWN1521_LABEL_QUERY_MODEL_DIAGNOSTIC.csv\n```\n\n순서대로 전체 manifest 파일의 bytes, 저장7617행 CM/P/R/U/gate 산술, 저장 packed alpha와 원 q/L/reference의 가림 teacher support를 검증한다. 세 번째 도구는 원 RGB/valid/depth/donor나 모델을 재실행하지 않으며 그 한계를 결과에 표시한다. 절대 경로 pointer를 따라가지 않는다. 신경망 전체 학습을 단일 명령으로 재현하는 패키지는 아니다.\n\n전체 신규 합성 RGB299MB와 기존 raw RGB/valid/prefix cache는 Git에 복제하지 않았다. 원 SHA·크기·조립 코드·전행 metadata·전 packed alpha·작은 모델·훈련로그·행별 예측·실제 사례는 포함한다. [원본과 배포 경로/SHA 대응](PUBLICATION_SOURCE_MAP.json)을 사용한다. 큰 JSONL은 gzip으로 무손실 저장했다. metadata 안의 당시 절대 경로는 실행 출처 기록이다. 원 context용SYN RGB440 buffer는 아직 열지 않은 다음 단계의 경로이며 현재 패키지가 그 픽셀을 검증한 것은 아니다.\n\n공개 사진과 기존 encoder 관련 출처는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에 기록했다. 이전 v61/기존 가림 모델의 현재223 고정 비교는 [V229 기록](../V229_FIXED_CONTEXT_REAL/PORTABLE_REPORT_KO.md)에 남아 있다. 이번 source gate가 실패했으므로 현재223에 새 후보 비교를 추가하지 않았다.\n'


NOTICE = '# Third-party data and model notices\n\nYCB-Video imagery/annotations: Yu Xiang, Tanner Schmidt, Venkatraman Narayanan and Dieter Fox, PoseCNN (RSS 2018). Original project: https://rse-lab.cs.washington.edu/projects/posecnn/ . Author dataset/toolbox README: https://github.com/yuxng/YCB_Video_toolbox/blob/master/README.md . Official BOP YCB-V format/source/license: https://bop.felk.cvut.cz/datasets/ . Both author README and BOP YCB-V entry identify MIT licensing (checked2026-10-25). Author LICENSE: https://github.com/yuxng/YCB_Video_toolbox/blob/master/LICENSE .\n\nMIT License\n\nCopyright (c) 2017 Yu Xiang\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the "Software"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all\ncopies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\nSOFTWARE.\n\nThe three clean public SOURCE RGB examples retain their original encoded bytes. Crops, overlays and opaque cutout composites are derivative experiment visualizations; masks/coordinates and SHA bindings preserve the source links. No new private/current223 image is included.\n\nDINOv2 fixed base: Meta DINOv2, https://github.com/facebookresearch/dinov2 ; pinned vendor7764ea0f912e53c92e82eb78a2a1631e92725fc8 and official checkpoint SHA are recorded in experiment entries. Official base weights and vendor source tree are not redistributed in this package. The small experiment head checkpoints are new trained artifacts, not copies of the official base checkpoint. The rendered pallet SOURCE provenance is documented in the V234 renderer audit. This notice does not relicense unrelated files in the repository.\n'

if __name__ == "__main__":
    main()

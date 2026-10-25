"""Copy an explicit experiment inventory, without traversing the project."""
from pathlib import Path
import gzip
import hashlib
import json
import os

R = Path(__file__).parent
OUT = R / "V233_PUBLICATION_PACKAGE"
TOPS = [
    "V232_YCB_TRAIN_READINESS_B", "V232_SYN_TRAIN_ROUTE_C",
    "V232_ADAPTER_PREPARATION_A", "V232_MODULE_QA_B", "V232_PREFIX_TRAIN_CODE_A",
    "V232_ROOT_SOURCE_ADAPTATION", "V232_MANIFEST_QA_B",
    "V233_ROOT_CROP_ADAPTATION", "V233_CROSS_FAMILY_PROVENANCE_B",
    "V233_CROP_DATA_PREPARATION_C", "V233_CROP_SAVED_QA_B",
    "V233_PREFIX_CODE_QA_B", "V233_CROP_ADAPT_TRAINER_A",
    "V233_DETERMINISTIC_CROP_FIX_A", "V233_TRAIN_COMPLETION_QA_B",
    "V233_SOURCE_REPORT_C", "V233_PUBLICATION_VERIFY_A", "V233_GALLERY_QA_B",
]
EXCLUDED = {
    "V233_ROOT_CROP_ADAPTATION/crop_samples_v1/raw_rgb96.npy",
    "V233_ROOT_CROP_ADAPTATION/crop_samples_v1/raw_valid96.npy",
    "V233_ROOT_CROP_ADAPTATION/prefix_cache_v1/prefix.npy",
}


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
        assert len(data) < 40_000_000, rel
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
    (OUT / "build_v233_publication.py").write_bytes(Path(__file__).read_bytes())
    write_json(OUT / "PUBLICATION_SOURCE_MAP.json", {"source_root": str(R), "files": source_map, "non_payload_exclusions": sorted(EXCLUDED), "exclusion_note_KO": "전체 raw crop280.8MB와 prefix6.107GB는 Git payload에 넣지 않았다. 원본 SHA/크기/구성 코드와 11개의 사전/사후 사례 crop은 포함한다. 원 평가/held payload를 새로 복사하지 않았다."})
    (OUT / "README_KO.md").write_text(README)
    # Only known output package is inventoried; no external paths are followed.
    manifest_files = []
    for p in package_files(OUT):
        b = p.read_bytes()
        manifest_files.append({"path": p.relative_to(OUT).as_posix(), "sha256": digest(b), "bytes": len(b)})
    write_json(OUT / "MANIFEST.json", {"schema": "V233_PORTABLE_PUBLICATION_V1", "files": manifest_files, "file_count": len(manifest_files), "total_bytes": sum(f["bytes"] for f in manifest_files), "scope": "V232 preparation + V233 actual completed and failed attempts, saved SOURCE results and review evidence; no claim of current-real success", "current_admission": False, "source_gate_pass": False})
    print(json.dumps({"package": str(OUT), "files": len(manifest_files), "bytes": sum(f['bytes'] for f in manifest_files), "manifest_sha256": digest((OUT / 'MANIFEST.json').read_bytes())}))


def package_files(p):
    for n in sorted(os.listdir(p)):
        f = lexical(p / n)
        assert not f.is_symlink()
        if f.is_dir():
            yield from package_files(f)
        else:
            yield f


README = """# V232–V233: 공개 실사·합성 crop, 작은 head와 encoder 적응 대조

**실제 학습·SOURCE 비교와 검산까지 완료했지만, 실사 가림 판별 문제는 아직 해결되지 않았다.** 이번 encoder 적응의 SOURCE 순개선은 0pp여서 채택하지 않았다. 현재223장에 새 추론·학습·문턱 선택을 하지 않았고 PnP도 변경하지 않았다. 과거 현재223 비교는 [V229 기록](../V229_FIXED_CONTEXT_REAL/PORTABLE_REPORT_KO.md)에서 별도로 확인할 수 있다.

|SOURCE 내부 DEV|고정 encoder D recall / E recall|qkv LoRA D recall / E recall|균형 recall 변화|
|---|---|---|---|
|SYN main, N1344|98.84% / 91.07%|98.84% / 91.07%|0pp|
|YCB AUX, N397|97.94% / 0%|97.94% / 0%|0pp|

D=보임, E=외부 가림. SYN의 intended-native witness0 associated-point와 YCB의 associated-r surface는 다른 라벨 의미다. YCB native는 UNKNOWN이고 두 가족을 하나의 동일 과업으로 합치지 않는다. 정답 D/E 7617행의 입력은 전부 사용 가능해 U율은0%다. 전체15040행 중 target−1인7423행(S/U/제외)은 forward 대상이 아니며 모델 보류가 아니다. 실제 배포 보류 정책의 성능을 검증한 결과도 아니다.

## 바로 확인하기

- [CM·precision/recall·보류율·paired 변화·3px strata 보고서](artifacts/V233_SOURCE_REPORT_C/results/PORTABLE_REPORT_KO.md)
- [전체 수치 JSON](artifacts/V233_SOURCE_REPORT_C/results/SUMMARY.json) · [표 CSV](artifacts/V233_SOURCE_REPORT_C/results/METRICS.csv) · [7617행 비교 CSV](artifacts/V233_SOURCE_REPORT_C/results/PAIRED_ALL7617.csv)
- [실제 성공/실패 crop 이미지](artifacts/V233_ROOT_CROP_ADAPTATION/fixed8_source_dev_gallery_v2/SOURCE_DEV_FIXED8.png) · [로컬에서 여는 사례 HTML](artifacts/V233_ROOT_CROP_ADAPTATION/fixed8_source_dev_gallery_v2/ROOT_READOUT_KO.html) · [좌표·라벨·점수 JSON](artifacts/V233_ROOT_CROP_ADAPTATION/fixed8_source_dev_gallery_v2/CASE_DATA.json)
- [사전 고정 프로토콜](artifacts/V233_ROOT_CROP_ADAPTATION/PROTOCOL_BEFORE_SOURCE_PACKET_RGB_WEIGHT_FIT.json) · [학습 실행 완료](artifacts/V233_ROOT_CROP_ADAPTATION/common_local_crop_pair_v2/COMPLETE.json)
- [학습·체크포인트·출력 독립 QA](artifacts/V233_TRAIN_COMPLETION_QA_B/ACTUAL_QA_COMPLETE.json) · [crop/label 보존 QA](artifacts/V233_CROP_SAVED_QA_B/REVIEW.json) · [교차 계보 한계](artifacts/V233_CROSS_FAMILY_PROVENANCE_B/ROUTE.json)

## 무엇을 어떻게 실행했나

원 q/L/native를 유지한 raw96 crop에서 전체 crop CLS, 원 q 위치 patch, 중앙32 crop CLS, 입력 유효성, 조건32를 읽어 1186→64→2 head(76,098개)를 학습했다. 출력0은 SYN main, 출력1은 YCB AUX로 loss를 분리했다. 비교 팔만 DINOv2 마지막 block qkv에 rank4 LoRA(6,144개)를 추가했다. 같은 초기 head, 같은 draw1000, 가족마다8D8E batch, 고정 AdamW와 문턱0으로 비교했다. 좌표 회귀와 self classifier는 없다. historical full-context C11을 재현한 실험도 아니다.

원 SYN7040의 known6096과 YCB8000의 known1521에서 총7617 crop을 조립했다. FIT는 SYN4752/YCB1124, 내부 DEV는 SYN1344/YCB397이다. YCB는 원 TRAIN의 scene000000–000004 FIT,000006/000007 DEV로 정했다. SYN은 사전 component hash 규칙으로 나눴다. loss에는 FIT만 사용했고 두 최종1000 체크포인트를 고정한 다음 DEV 출력을 생성했다. SOURCE DEV는 이전에 노출된 내부 진단 자료이므로 fresh blind 평가가 아니다. SYN 배경·donor와 YCB DEV 사이 계보 독립성은 UNKNOWN이다.

원 CONTROL은1000회 완료했다. TREAT 첫 backward에서 결정적 CUDA grid_sample backward 미지원으로 중단돼 TREAT update0/DEV0이었다. 실패 기록을 보존하고 q-center의 같은4token 가중합으로 계산만 교체했다. SOURCEFIT5876 전부에서 old/new 특징과 CONTROL logits의 FP32 bytes가 같음을 확인했다. V2는 완료 CONTROL checkpoint/log를 byte 그대로 복사(새 CONTROL update0)했고 TREAT만1000회 실행했다. 최종 checkpoint2개·로그·공통 draw·freeze 기록·원 실패와 수리 코드를 모두 포함한다.

수치 gate는 DEV 가족 평균 균형 recall +3pp 이상, 각 가족 D/E recall 하락2pp 이하였다. 관측0pp로 실패했다. 두 팔의 집계 CM은 같지만 가족마다 D에서1개 회복·1개 악화가 있었다. YCB FIT E recall100%에 비해 DEV E recall0/57로, 새 공개 실사 조건으로의 가림 일반화가 해결되지 않았다. 작은 crop만으로 오답의 원인을 확정하지 않는다. 교차 계보 UNKNOWN 때문에 수치 통과 여부와 별도로 현재 평가 후보 admission은 false다.

## 다른 컴퓨터에서 확인

이 폴더에서 Python3 표준 라이브러리만으로 실행한다.

```bash
python3 -B verify_package.py --package . --manifest MANIFEST.json
python3 -B artifacts/V233_SOURCE_REPORT_C/summarize_pair.py --verify-saved
```

첫 명령은 manifest에 기재된 모든 파일의 SHA256/크기, 두 번째는 저장7617 paired 행에서 고정 문턱·CM/P/R/U·gate 계산을 재검산한다. 원래 로컬 절대 경로를 따라가지 않는다. 신경망 forward와 학습을 재실행하는 검증은 아니다. Manifest 자체의 SHA는 Git commit으로 고정된다.

원본 metadata/entry/protocol의 절대 경로는 당시 실행 출처 기록이다. portable 저장 위치와 원본 SHA의 대응은 [PUBLICATION_SOURCE_MAP.json](PUBLICATION_SOURCE_MAP.json)에 있다. 큰 JSONL은 무손실 gzip이며 `gzip -dc FILE.jsonl.gz` 또는 Python gzip으로 볼 수 있다. 나머지 원본 파일은 bytes를 유지했다. 코드 실행 entry를 새 환경에 맞춰 rebase하거나 새 학습할 권한을 암묵적으로 만들지 않는다.

전체 raw crop280.8MB와 prefix6.107GB, 외부 공식 DINO base와 YCB 원본 전체는 Git에 복제하지 않았다. 대신 원 SHA·조립 코드·전행 metadata·저장 예측·작은 학습 checkpoint·사전4/사후7 실제 crop을 담았다. 따라서 이 패키지는 과정과 저장 결과 감사용이며 모든 원본을 포함한 단일 명령 재학습 배포물은 아니다. PyTorch 학습 환경은 Python3.10/Torch2.1.1+cu118, CUDA RTX3080이었다. 표준 라이브러리 검산에는 이 GPU 환경이 필요 없다.

이번 push는 실사 문제 해결 완료 선언이 아니다. 공개 실사에서 가림의 일반화와 native 라벨 대응, 교차 계보 독립성은 아직 남아 있다. 시도와 실패를 같은 main 기록에서 확인할 수 있게 공개한다.
"""


if __name__ == "__main__":
    main()

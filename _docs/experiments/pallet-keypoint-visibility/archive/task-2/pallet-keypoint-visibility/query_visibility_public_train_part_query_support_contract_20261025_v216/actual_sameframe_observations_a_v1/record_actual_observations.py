"""Record A's actually tool-viewed images; no other reader or original source IO.

The literals below are A's independent visual judgments. This is not an image
classifier, target generator, or training-admission program.
"""
import copy
import csv
import hashlib
import importlib.util
import io
import json
import os
import stat
from collections import Counter
from datetime import datetime, timezone

P = "/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_public_train_part_query_support_contract_20261025_v216"
H = P + "/actual_sameframe_observations_a_v1"
ENTRY = P + "/ROOT_ACTUAL_SAMEFRAME_TWO_READER_ANNOTATION_ENTRY.json"
READS = {}


def read(p, png=False):
    assert "v112" not in p.lower() and "\x00" not in p
    assert not p.lower().endswith((".npz", ".npy", ".jpg", ".jpeg", ".pt", ".pth"))
    assert not p.lower().endswith(".png") or png
    ss = p.strip("/").split("/"); assert all(s not in ("", ".", "..") for s in ss)
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for s in ss[:-1]:
            n = os.open(s, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = n
        n = os.open(ss[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
        with os.fdopen(n, "rb") as f:
            assert stat.S_ISREG(os.fstat(f.fileno()).st_mode)
            b = f.read()
    finally: os.close(fd)
    READS[p] = {"path": p, "sha256": hashlib.sha256(b).hexdigest()}
    return b


def checked(b, png=False):
    x = read(b["path"], png=png)
    assert hashlib.sha256(x).hexdigest() == b["sha256"]
    return x


def put(name, obj):
    p = H + "/" + name
    b = (json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, "wb") as f: f.write(b)
    return {"path": p, "sha256": hashlib.sha256(b).hexdigest()}


# Each tuple: inclusive raw96 ROI, feature kind, independently observed feature,
# query relation to THAT feature. None means no sufficiently distinct witness.
# U slots are retained by the original input mask, not these image judgments.
F = {
 0: [None, None, None, None,
     ([30,59,68,77],"EDGE_JUNCTION","전경 목재 상판의 위쪽 긴 윤곽과 바닥 경계", "OFF"),
     ([38,54,57,81],"END_CORNER","전경 팔레트 긴 상판의 오른쪽 끝에서 윗면과 끝면이 만나는 코너", "NEAR"),
     ([44,26,65,70],"END_CORNER","같은 오른쪽 상판 끝의 수직 윤곽과 목재 면", "ON"),
     ([15,28,72,55],"EDGE_JUNCTION","전경 상판 윗면과 그 위쪽 긴 바깥 윤곽", "ON")],
 1: [([10,48,55,77],"EDGE_JUNCTION","회색 상자 아래 검은 프레임의 두 막대가 만나는 부위", "ON"),
     ([10,1,50,38],"END_CORNER","회색 플라스틱 상자의 아래쪽 둥근 코너", "OFF"),
     ([0,0,27,20],"END_CORNER","crop 왼쪽 위에 일부 남은 회색 상자 아래 코너", "OFF"),
     ([20,49,70,70],"EDGE_JUNCTION","검은 받침의 가로 막대와 사선 테두리", "ON"),
     ([10,45,39,72],"END_CORNER","회색 상자 뒷쪽 왼편 윗 테두리의 꺾이는 코너", "NEAR"),
     ([8,42,34,72],"END_CORNER","회색 상자의 오른편 윗 테두리 끝과 벽의 경계", "OFF"),
     ([10,7,42,80],"EDGE_JUNCTION","회색 상자 오른쪽 수직 테두리와 작은 홈", "NEAR"),
     ([31,27,61,63],"EDGE_JUNCTION","회색 상자 안쪽 수직 모서리와 윗 테두리", "ON")],
 2: [([51,40,76,75],"END_CORNER","대각선 팔레트 왼쪽 앞 상판 끝의 윗면과 끝면 코너", "NEAR"), None, None,
     ([37,34,68,74],"END_CORNER","같은 팔레트 왼쪽 앞 지지블록의 끝면·옆면 코너", "ON"),
     ([36,47,64,76],"END_CORNER","대각선 팔레트 먼쪽 왼편 상판 끝의 코너", "NEAR"),
     ([0,43,30,74],"END_CORNER","crop 왼쪽에 남은 먼쪽 목재 판 끝과 바닥의 경계", "OFF"),
     ([2,29,28,65],"EDGE_JUNCTION","오른쪽 목재 판 끝의 바깥 윤곽과 바닥 경계", "OFF"),
     ([22,29,60,57],"EDGE_JUNCTION","먼쪽 왼편 상판 끝의 아래 윤곽과 판 사이 틈", "ON")],
 3: [([16,0,43,52],"EDGE_JUNCTION","뒤 나무상자의 세로 finger joint와 위쪽 가로 테두리", "NEAR"),
     ([41,20,65,58],"END_CORNER","뒤 나무상자의 오른편 윗 테두리 코너", "NEAR"),
     ([38,4,60,57],"EDGE_JUNCTION","나무상자 오른쪽 세로 테두리와 아래를 지나는 회색 물체 경계", "NEAR"),
     ([14,4,41,83],"EDGE_JUNCTION","나무상자 앞면의 세로 finger joint", "NEAR"),
     ([10,59,67,82],"EDGE_JUNCTION","나무상자 윗 테두리와 검은 손가락 홈 주변", "OFF"),
     ([7,51,49,75],"END_CORNER","나무상자의 오른쪽 위 꺾이는 윤곽", "NEAR"),
     ([30,19,62,60],"END_CORNER","나무상자 오른편 상단 모서리", "ON"),
     ([8,33,76,76],"EDGE_JUNCTION","나무상자 윗 테두리와 검은 반원형 홈", "ON")],
 4: [([63,44,87,71],"END_CORNER","전경 EPAL 팔레트 왼쪽 앞 상판 끝 코너", "OFF"),
     ([56,23,83,61],"BLOCK_FACE","오른편 EPAL 지지블록 앞면과 상판의 접합부", "ON"),
     ([34,18,76,47],"EDGE_JUNCTION","오른편 지지블록 아래의 긴 목재 레일 윤곽", "OFF"),
     ([60,9,84,40],"END_CORNER","왼쪽 EPAL 지지블록과 아래 레일의 바깥 끝 코너", "OFF"),
     ([10,78,62,92],"EDGE_JUNCTION","crop 아래쪽에 보이는 팔레트 상판의 긴 가장자리", "OFF"),
     ([4,1,29,39],"EDGE_JUNCTION","회색 통의 왼쪽 위 굽은 테두리와 주름", "OFF"),
     ([16,46,77,69],"EDGE_JUNCTION","회색 통의 아래 테두리와 목재 상판이 맞닿아 보이는 경계", "NEAR"),
     ([9,43,74,65],"EDGE_JUNCTION","파란 상자 아래의 팔레트 상판 긴 가장자리", "ON")],
 5: [None,
     ([0,8,31,43],"END_CORNER","crop 왼편 일부의 팔레트 목재 끝과 회색 통 옆 경계", "OFF"),
     ([0,0,20,25],"END_CORNER","crop 왼쪽 위에 일부 남은 전경 목재 끝", "OFF"), None,
     ([46,44,74,76],"END_CORNER","팔레트 먼쪽 왼편 바깥 상판 끝의 코너", "NEAR"),
     ([51,23,77,77],"EDGE_JUNCTION","파란 상자의 오른편 리브와 옆 나무상자 모서리", "NEAR"),
     ([35,40,70,84],"END_CORNER","파란 상자의 아래 오른쪽 코너와 나무상자 면의 경계", "ON"),
     ([42,16,68,46],"EDGE_JUNCTION","같은 팔레트 왼편 상판 끝의 아래쪽 윤곽", "NEAR")],
 6: [([39,21,71,60],"END_CORNER","앞에 놓인 작은 나무상자의 아래 코너", "NEAR"),
     ([36,22,62,61],"END_CORNER","열린 나무상자의 윗 모서리와 흰 연결부", "ON"),
     ([32,9,61,81],"EDGE_JUNCTION","열린 나무상자의 세로 모서리와 가로 판 접합", "ON"),
     ([6,0,45,23],"END_CORNER","crop 위쪽 일부의 작은 나무상자 아래 코너", "OFF"),
     ([51,40,79,70],"END_CORNER","뒤에 세운 팔레트 맨 위 가로판의 왼쪽 끝", "NEAR"),
     ([65,54,83,84],"END_CORNER","뒤에 세운 팔레트 맨 위 가로판의 오른쪽 끝", "OFF"),
     ([47,30,79,62],"END_CORNER","같은 맨 위 가로판 오른쪽 끝의 앞면과 바깥 윤곽", "ON"),
     ([35,18,63,57],"EDGE_JUNCTION","같은 맨 위 가로판 왼쪽 끝의 아래 윤곽과 틈", "ON")],
 7: [([9,0,39,30],"END_CORNER","뒤 작은 나무상자의 아래 코너", "OFF"),
     ([13,25,68,55],"EDGE_JUNCTION","앞 열린 나무상자의 밝게 칠해진 가로판 윗 경계", "ON"),
     ([28,38,69,70],"EDGE_JUNCTION","앞 나무상자 가로판 사이의 틈과 작은 흰 연결부", "ON"), None,
     ([43,2,68,45],"END_CORNER","뒤 작은 나무상자의 밝은 윗 코너", "NEAR"),
     ([22,27,72,53],"EDGE_JUNCTION","열린 나무상자 뒤쪽의 가로 테두리", "ON"),
     ([17,41,62,70],"EDGE_JUNCTION","나무판 사이 검은 손잡이 모양 절개부와 그 테두리", "ON"),
     ([28,8,54,69],"EDGE_JUNCTION","뒤 나무상자 왼편 세로 모서리와 홈", "NEAR")],
 8: [([45,8,74,40],"END_CORNER","앞 열린 나무상자 아래쪽 목재 코너", "NEAR"),
     ([0,0,16,46],"EDGE_JUNCTION","crop 왼쪽에 일부 보이는 앞 나무상자의 세로 끝", "OFF"),
     ([0,18,14,61],"EDGE_JUNCTION","crop 왼쪽 목재 면의 짧은 바깥 윤곽", "OFF"),
     ([18,0,69,22],"END_CORNER","crop 위쪽의 앞 나무상자 아래 코너", "OFF"),
     ([7,22,54,68],"EDGE_JUNCTION","나무판 접합과 작은 검은 홈", "ON"),
     ([13,38,80,58],"EDGE_JUNCTION","밝게 칠한 앞 가로판의 위쪽 윤곽", "ON"),
     ([7,64,38,87],"OTHER","가로판 아래쪽의 작은 은색 연결단추와 판 경계", "OFF"),
     ([19,15,70,63],"EDGE_JUNCTION","앞 나무상자의 가로판과 세로 모서리 접합", "ON")],
 9: [([44,26,81,76],"EDGE_JUNCTION","검은 카트 손잡이 왼쪽 아래 둥근 연결부", "NEAR"),
     ([26,26,66,72],"EDGE_JUNCTION","검은 손잡이 오른쪽 아래 굽은 테두리", "ON"),
     ([33,24,69,70],"EDGE_JUNCTION","검은 손잡이 오른편 아래 바깥 윤곽", "NEAR"),
     ([41,27,77,74],"EDGE_JUNCTION","검은 손잡이 왼쪽 아래 가로·세로 연결부", "ON"),
     ([38,44,70,79],"EDGE_JUNCTION","검은 손잡이 왼쪽 위에서 굽어지는 윤곽", "NEAR"),
     ([37,35,71,67],"EDGE_JUNCTION","검은 손잡이 오른쪽 위 굽은 코너", "ON"),
     ([28,29,67,78],"EDGE_JUNCTION","검은 손잡이 오른쪽 안쪽 곡선과 세로 지지부", "ON"),
     ([38,32,70,71],"EDGE_JUNCTION","검은 손잡이 왼쪽 위 세로·가로 연결부", "NEAR")],
 10:[([33,21,68,71],"EDGE_JUNCTION","작은 검은 손잡이 왼쪽 아래 굽은 연결부", "ON"),
     ([24,24,65,69],"EDGE_JUNCTION","작은 검은 손잡이 오른쪽 아래 코너", "ON"),
     ([35,32,67,77],"EDGE_JUNCTION","검은 손잡이 오른쪽 바깥 아래 윤곽", "NEAR"),
     ([33,27,73,74],"EDGE_JUNCTION","작은 손잡이 아래 가로부와 왼쪽 지지부", "ON"),
     ([56,57,88,88],"EDGE_JUNCTION","crop 오른쪽 아래 일부의 검은 손잡이 굽은 윤곽", "OFF"),
     ([32,28,70,63],"EDGE_JUNCTION","작은 손잡이 오른쪽 위 곡선", "NEAR"),
     ([21,26,68,75],"EDGE_JUNCTION","작은 손잡이 안쪽 곡선과 오른편 세로부", "ON"),
     ([52,27,82,70],"EDGE_JUNCTION","작은 검은 손잡이 왼쪽 위 세로 연결부", "NEAR")],
 11:[None,None,None,None,
     ([25,30,70,67],"END_CORNER","회색 플라스틱 상자 앞 윗 테두리와 코너", "ON"),
     ([25,19,50,58],"END_CORNER","회색 상자 아래 드러난 목재 지지판 오른쪽 끝 코너", "NEAR"),
     ([23,19,50,84],"EDGE_JUNCTION","목재 지지판 오른편의 긴 바깥 모서리", "NEAR"),
     ([31,18,59,75],"EDGE_JUNCTION","회색 상자 앞면 세로 리브와 윗 테두리", "ON")],
}

# D is assigned only where A can positively connect a concrete visible pallet
# end/corner to the original proposed native/q8 topology, not merely a material.
NATIVE_PARTS = {
 (0,5): "전경 팔레트 먼쪽 오른편 상판 끝의 위 코너",
 (0,6): "같은 먼쪽 오른편 상판 끝의 아래쪽 코너/끝면",
 (2,0): "대각선 팔레트 왼쪽 앞 상판 끝 코너",
 (2,3): "같은 왼쪽 앞 끝의 아래 지지블록 코너",
 (2,4): "같은 팔레트 먼쪽 왼편 상판 끝 코너",
 (2,7): "같은 먼쪽 왼편 상판 끝의 아래 윤곽/끝면",
 (4,0): "전경 EPAL 팔레트 왼쪽 앞 상판 끝",
 (4,3): "같은 왼쪽 앞 지지블록과 아래 레일의 끝 코너",
 (5,4): "상자를 받친 팔레트 먼쪽 왼편 상판의 바깥 끝",
 (5,7): "같은 왼편 상판 끝의 아래쪽 바깥 윤곽",
 (6,4): "뒤에 세운 팔레트 맨 위 판의 왼쪽 끝 위 코너",
 (6,5): "같은 맨 위 판의 오른쪽 끝 위 코너",
 (6,6): "같은 맨 위 판 오른쪽 끝의 아래 윤곽/끝면",
 (6,7): "같은 맨 위 판 왼쪽 끝의 아래 윤곽/끝면",
}
INSTANCE = {
 0: "full-frame 전경의 평평한 목재 팔레트와 crop의 긴 상판이 연속된다.",
 2: "full-frame 가운데 대각선으로 놓인 목재 팔레트의 상판과 앞 지지블록을 crop에서 연결할 수 있다.",
 3: "full-frame 아래 전경의 EPAL 팔레트는 식별된다. 다만 q8 crop은 뒤 나무상자 쪽이다.",
 4: "full-frame 아래 전경 EPAL 팔레트의 앞 레일·블록과 crop의 목재 면이 연결된다.",
 5: "full-frame에서 상자·통을 받친 한 목재 팔레트의 바깥 상판 윤곽을 식별할 수 있다.",
 6: "full-frame 뒤에 세워진 목재 팔레트의 맨 위 판과 crop의 끝을 연결할 수 있다. 앞 나무상자는 별개다.",
 7: "full-frame 뒤에 세워진 목재 팔레트는 식별되지만 q8의 국소 crop은 앞 나무상자 쪽이다.",
 8: "full-frame 뒤의 목재 팔레트와 앞 나무상자는 구분된다. 원 q8은 주로 앞 나무상자 쪽이다.",
}
OPTIONAL_CLEAN = [(0,5),(0,6),(2,0),(2,3),(6,4),(6,5),(11,5),(11,6)]


def main():
    eb = read(ENTRY); e = json.loads(eb)
    assert hashlib.sha256(eb).hexdigest() == "9566b0b2419bdb7819d9e824ab281c17b2a733d903f6c9862d76d2507ba80e40"
    assert e["approved"] and e["observation_array_key"] == "rows96" and not e["training_admission"]
    t = json.loads(checked(e["templates"]))
    for key in ("contract", "protocol", "schema", "root_actual_view_scope", "exact_pixel_union_addendum"):
        checked(e[key])
    wh = json.loads(checked(e["generated_image_whitelist_exact"]))
    allowed = {(x["binding"]["path"], x["binding"]["sha256"]) for x in wh["images"]}
    code = checked(e["helper"]).decode()
    pure = {}; exec(compile(code, e["helper"]["path"], "exec"), pure)
    views = []
    for i, row in enumerate(t["rows"][::8]):
        b = row["display_bindings"]["contact"]
        assert (b["path"], b["sha256"]) in allowed; checked(b, png=True)
        views.append({"binding": b, "position": i, "kind": "FULLFRAME_AND_ALL8_CONTACT", "tool": "tools.view_image via functions.exec", "detail": "original", "actually_viewed": True,
                      "actual_batch": "positions_%d_to_%d" % (i//3*3, i//3*3+2), "fullframe_and_original_all8_crop_metadata_seen": True})
    for i, n in OPTIONAL_CLEAN:
        b = t["rows"][8*i+n]["display_bindings"]["clean96"]
        assert (b["path"], b["sha256"]) in allowed; checked(b, png=True)
        views.append({"binding": b, "position": i, "native_slot": n, "kind": "OPTIONAL_CLEAN96", "tool": "tools.view_image via functions.exec", "detail": "original", "actually_viewed": True})
    rows = copy.deepcopy(t["rows"])
    for row in rows:
        m = row["input"]; i, n = m["position"], m["native_slot"]
        row["reader_id"] = "A"; row["observation_complete"] = True
        row["actually_viewed_image_bindings"] = [row["display_bindings"]["contact"]]
        if (i,n) in OPTIONAL_CLEAN: row["actually_viewed_image_bindings"].append(row["display_bindings"]["clean96"])
        if not m["original_query_available"]:
            row["ambiguity_note"] = "원 입력 U다. finite q나 주변 물체가 보여도 원 availability를 보충하지 않고 모든 의미축 UNKNOWN을 유지한다."
            pure["validate_observation"](row, t["rows"][8*i+n]); continue
        row["instance"] = {"status": "YES" if i in INSTANCE else "UNKNOWN", "evidence": INSTANCE.get(i, "목재 support나 다른 물체는 보이지만 요청된 pallet instance를 이 입력에서 확정하지 못한다.")}
        row["part"] = {"status": "UNKNOWN", "description": "", "evidence": "원 native가 뜻하는 pallet 부위와 이 국소 feature의 대응은 미확정이다."}
        row["visibility"] = {"state": "UNKNOWN", "evidence": "의도 native 부위의 대응을 확정하지 못하므로 국소 물체나 배경을 D/E로 변환하지 않는다.", "occlusion_evidence": "", "evidence_roi": None}
        row["query_support"] = {"state": "UNKNOWN", "evidence": "의도 부위의 support를 확정하지 못한다."}
        row["addressability"] = {"status": "UNKNOWN", "evidence": ""}
        row["witness"] = {"status": "UNKNOWN", "kind": "UNKNOWN", "roi": None, "description": "", "evidence": ""}
        item = F[i][n]
        if item is not None:
            box, kind, desc, q = item
            row["witness"] = {"status": "VISIBLE", "kind": kind, "roi": box, "description": desc,
                "evidence": "실제 원96 표시에서 " + desc + "의 색·면 또는 꺾이는 윤곽을 볼 수 있다. ROI는 보이는 feature의 약영역이며 3D 점 좌표가 아니다."}
            row["addressability"] = {"status": "EXPLICIT_ROI_ONLY", "evidence": "이 feature는 원crop에서 보이지만 원 q/native만으로 요청된 pallet 부위라고 설명하기 어려워, 추가 ROI로 지정한 local-feature 관찰로만 남긴다."}
            qe = {"ON": "원47.5 중심은 여기 기술한 local-feature의 면/경계 부근에 놓인다. 이는 native visibility 판정이 아니다.",
                  "NEAR": "원47.5 중심은 기술한 경계 가까이에 보이나 정밀 점일치나 수치거리 threshold를 새로 판정하지 않는다.",
                  "OFF": "원47.5 중심은 기술한 feature와 떨어진 바닥/배경 또는 다른 면 쪽이다. 이 관찰을 외부가림 E로 바꾸지 않는다."}[q]
            row["query_support"] = {"state": q, "evidence": qe}
        if (i,n) in NATIVE_PARTS:
            part = NATIVE_PARTS[i,n]
            row["part"] = {"status": "YES", "description": part,
                "evidence": "full-frame의 같은 팔레트에서 연속된 상판/끝면/블록 윤곽을 따라 " + part + "를 원crop의 구체 feature와 연결할 수 있다."}
            row["addressability"] = {"status": "QUERY_NATIVE_SUPPORTED",
                "evidence": "원 native/q8의 좌우 끝 및 위·아래 제안 배치와 같은 팔레트의 연속된 판/블록 맥락으로 " + part + "를 묻는 약한 대응 근거가 있다. q를 옮기거나 실제 3D vertex를 확정한 것은 아니다."}
            row["visibility"] = {"state": "D", "evidence": part + "의 구체적인 목재 끝/윤곽이 원crop 안에서 실제 보인다. q support가 OFF/NEAR여도 이를 가림 E로 바꾸지 않는다.", "occlusion_evidence": "", "evidence_roi": None}
        row["ambiguity_note"] = ("SOURCE assistant weak 판단이며 정확한 canonical 3D점·사람 gold·학습 admission은 아니다. "
            + ("이 intended-part 대응은 원 q8/topology의 약근거로만 남기며 다른 reader와 불일치할 수 있다." if (i,n) in NATIVE_PARTS else
               "보이는 feature를 intended native로 자동복사하지 않는다. 다른 object/목재/padding만으로 external E나 SELF를 부여하지 않는다."))
        if item is None:
            row["ambiguity_note"] += " 이 crop에서는 구별되는 intended/local feature를 충분히 고정하지 못해 ROI도 null로 남긴다."
        pure["validate_observation"](row, t["rows"][8*i+n])
    assert len(rows) == 96 and len({r["input"]["row_id"] for r in rows}) == 96
    assert sum(not r["input"]["original_query_available"] for r in rows) == 11
    assert Counter(r["input"]["internal_source_role"] for r in rows) == {"SOURCE_FIT":64,"SOURCE_DEV":32}
    actual = {pure["binding_identity"](x["binding"]) for x in views}
    assert all(pure["binding_identity"](b) in actual for r in rows for b in r["actually_viewed_image_bindings"])
    counts = {
        "frames":12,"rows":96,"original_inputU":11,"SOURCE_FIT":64,"SOURCE_DEV":32,
        "visibility":dict(Counter(r["visibility"]["state"] for r in rows)),
        "instance":dict(Counter(r["instance"]["status"] for r in rows)),
        "part":dict(Counter(r["part"]["status"] for r in rows)),
        "witness":dict(Counter(r["witness"]["status"] for r in rows)),
        "addressability":dict(Counter(r["addressability"]["status"] for r in rows)),
        "query_support":dict(Counter(r["query_support"]["state"] for r in rows)),
        "actual_toolviews":20,"fullframe_all8_contacts":12,"optional_clean96":8,"training_admitted":0,
    }
    entry_binding = {"path":ENTRY,"sha256":hashlib.sha256(eb).hexdigest()}
    receipt = put("ACTUAL_IMAGE_VIEW_RECEIPT.json", {"completed":True,"PASS":True,"reader_id":"A","entry":entry_binding,
        "views":views,"own_views_only":True,"actual_toolviews":20,"all12_fullframe_and_native8_seen":True,
        "generated_PNG_only":True,"original_source_array_JPEG_GT_teacher_oldlabels_models_scores_opens":0,
        "historical_V215_appearance_exposure_disclosed":True,"other_reader_new_observation_values_read":0,
        "preflight_adapter_note":"First local flat-whitelist adapter failed672be5, nested binding corrected f43cc8. The first3 exact root-preflighted contacts were viewed before local rehash; all12 and optional8 exact NOFOLLOW SHA now verified. Failure/correction receipt preserved.",
        "HTML_browser_rendered":False,"UTC":datetime.now(timezone.utc).isoformat()})
    observations = put("ALL96_OBSERVATIONS.json", {"completed":True,"PASS":True,"reader_id":"A","entry":entry_binding,
        "contract":e["contract"],"protocol":e["protocol"],"template":e["templates"],"split":e["split"],
        "exact_pixel_union_addendum":e["exact_pixel_union_addendum"],"rows96":rows,"counts":counts,
        "training_admission":False,"semantic_authority":"Independent assistant SOURCE weak observation; not blinded human/native/physical gold."})
    csvout = io.StringIO(); writer = csv.writer(csvout)
    writer.writerow(["row_id","frame_key","native_slot","query_xy","L","original_query_available","role","internal_source_role","instance","part","witness","witness_kind","ROI_inclusive_raw96","addressability","query_support","visibility","positive_feature_evidence","ambiguity_note","training_admission"])
    for r in rows:
        m=r["input"];writer.writerow([m["row_id"],m["frame_key"],m["native_slot"],json.dumps(m["query_xy"]),m["L"],m["original_query_available"],m["role"],m["internal_source_role"],r["instance"]["status"],r["part"]["status"],r["witness"]["status"],r["witness"]["kind"],json.dumps(r["witness"]["roi"]),r["addressability"]["status"],r["query_support"]["state"],r["visibility"]["state"],r["witness"]["evidence"],r["ambiguity_note"],False])
    cb=csvout.getvalue().encode(); cp=H+"/ALL96_OBSERVATIONS.csv"
    with open(cp,"xb") as f:f.write(cb)
    csvbinding={"path":cp,"sha256":hashlib.sha256(cb).hexdigest()}
    qa=put("SAVED_OBSERVATION_SCHEMA_QA.json", {"completed":True,"PASS":True,"reader_id":"A","entry":entry_binding,
        "observations":observations,"view_receipt":receipt,"template":e["templates"],"pure_helper":e["helper"],
        "production_validate_observation_all96_executed":True,"all96_input_display_identity_exact":True,
        "originalU11_all_six_axes_UNKNOWN_and_null_ROI":True,"all96_actual_view_bindings_in_own_receipt":True,
        "sourceFIT64_DEV32_and_originalTRAIN_preserved":True,"no_UNREVIEWED_in_completed_records":True,
        "same_reader_semantic_truth_certification":False,"training_admission":False,"counts":counts})
    independence=put("INDEPENDENCE_RECEIPT.json", {"completed":True,"PASS":True,"reader_id":"A", "other_reader_new_observation_files_read":0,
        "other_reader_new_values_received":0,"shared_schema_only":True,"historical_V215_appearance_exposure_disclosed":True,
        "entry":entry_binding,"own_observations_binding":observations})
    code_binding={"path":__file__,"sha256":hashlib.sha256(read(__file__)).hexdigest()}
    complete=put("COMPLETE.json", {"completed":True,"PASS":True,"reader_id":"A","entry":entry_binding,"contract":e["contract"],"protocol":e["protocol"],
        "template":e["templates"],"split":e["split"],"exact_pixel_union_addendum":e["exact_pixel_union_addendum"],
        "observations":observations,"CSV":csvbinding,"view_receipt":receipt,"schema_QA":qa,"independence_receipt":independence,
        "recording_code":code_binding,"counts":counts,"other_reader_new_values_received_or_read":0,
        "native_weak_and_explicit_ROI_auxiliary_authorities_separate":True,"training_admission":False,
        "new_control_generation_or_fit_or_Git":False,"own_actual_folder_final_frozen_no_more_writes":True})
    print(json.dumps({"PASS":True,"COMPLETE":complete,"observations":observations,"counts":counts},ensure_ascii=False))


if __name__ == "__main__": main()

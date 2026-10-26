"""Render the completed SYN-label audit without changing any targets."""
from pathlib import Path
import hashlib
import html
import json

HERE=Path(__file__).absolute().parent


def safe(value):
    p=Path(value)
    if not p.is_absolute() or any(x in str(p).lower() for x in ('v112','cache','.pyc','..')):raise ValueError(str(p))
    for a in reversed((p,*p.parents)):
        if a.is_symlink():raise ValueError(str(a))
    return p


def binding(path):
    p=safe(path);b=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}


def main():
    native=json.loads(safe(HERE/'syn_joint_targets_v1/TARGET_AUDIT_SUMMARY.json').read_text())
    pixels=json.loads(safe(HERE/'syn_label_visual_audit_v1/PIXEL_PROXY_AUDIT.json').read_text())
    columns=[('frames','영상'),('points','8점 전체'),('loc_valid','위치 감독 가능'),('D_clear0','D_proxy=0'),('E_ge_half','E_proxy≥0.5'),
       ('unknown_partial_between0_half','0&lt;fraction&lt;0.5 제외'),('self_inside','자기 가림 제외'),('outside_or_nonpositive_depth','범위 밖/음의 depth 제외')]
    rows=[]
    for split,c in native['counts'].items():
        rows.append('<tr><th>'+html.escape(split)+'</th>'+''.join('<td>'+str(c[k])+'</td>' for k,_ in columns)+'</tr>')
    table='<table><tr><th>분할</th>'+''.join('<th>'+label+'</th>' for _,label in columns)+'</tr>'+''.join(rows)+'</table>'
    pixelrows=[]
    for x in pixels['census']:
        pixelrows.append('<tr>'+''.join('<td>'+str(x[k])+'</td>' for k in ['split','visibility_y','N','M0_center_target','M4_center_target','M0_patch3_any_target','M4_patch3_any_target','raster_center_unsampleable'])+'</tr>')
    cases=[
      {'file':'CASE_007_F2_K1_3.png','finding':'F2 같은 이미지: k1 ray0와 k3 ray0.5556 비교. k1은 팔레트 외곽에 있고 k3은 전경 구조물이 맞닿는 쪽이다. 가시성은 주변 ray 차단의 proxy이며 pixel 중심 마스크와 일치할 필요가 없다.'},
      {'file':'CASE_029_F60_K7.png','finding':'F60/k7: ray0.5556이라 E_proxy지만 중앙 M4 픽셀에는 작은 팔레트 영역이 남는다. 9ray의 다수 차단과 한 중앙점/사람 native 보임 정의의 차이를 보여준다. 물리 가시성 오라벨로 확정하지 않았다.'},
      {'file':'CASE_030_F0_K0.png','finding':'F0/k0: ray0이고 외부 차단은 없으나 중앙 픽셀은 M0에서도 target이 아니다. 실제 영상에서는 팔레트의 외곽/가상 cuboid 코너 주변이다. foreground-at-q를 D/E 교사로 대신 쓸 수 없음을 보여준다.'},
      {'file':'CASE_020_F768_K1_3.png','finding':'DEV F768: k1 ray0의 외곽과 k3 ray0.7778의 외부 구조물 가림이 같은 전체 이미지에 있다. 위치 표시와 마스크/원RGB 정렬을 확인했다.'},
    ]
    for x in cases:x['binding']=binding(HERE/'syn_label_visual_audit_v1'/x['file'])
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>V270 합성 공동학습 라벨 감사</title>
<style>body{font:17px sans-serif;line-height:1.65;max-width:1500px;margin:32px auto;padding:12px;background:#111820;color:#e6edf5}table{border-collapse:collapse;font-size:15px}th,td{border:1px solid #526071;padding:8px}a{color:#8ed0ff}img{width:100%;max-width:1320px}strong{color:#ffcf87}section{margin:30px 0}</style>
<h1>V270 초기 전체 이미지 키포인트·가림 공동학습: 합성 교사 감사</h1>
<p><strong>기존 합성1003장으로 빠른 실험을 할 수 있다. 다만 가시성 교사는 물리 native D/E 정답이 아니라 저장된 virtual cuboid 주변 외부 ray 차단 proxy다.</strong> 실사223장은 이 라벨 준비·진단에 사용하지 않았다. 실제 모델 성능은 별도 평가 결과를 봐야 한다.</p>
<p>위치는 원 renderer의 projected_cuboid8+centroid9를 그대로 감독한다. 가려진 점도 기하상 투영좌표가 있으므로 위치 branch는 amodal 좌표를 학습한다. 가림 branch는 native8에만 적용하며 centroid는 제외한다.</p>
<h2>고정한 감독과 제외 수</h2>'''+table+'''
<p>D_proxy는 source convex-box self-visible·positive depth·화면 내부이면서 fraction=0인 점, E_proxy는 같은 조건에서 fraction≥0.5인 점이다. 0&lt;fraction&lt;0.5는 애매한 점으로 제외하고 자기 가림과 화면 밖도 외부가림 loss에서 제외했다. 실사 성적을 보고 기준을 고르지 않았다. 전체 이미지 위치 loss에는 화면 안의 가상 좌표를 별도로 사용한다.</p>
<p>원 RGB는 TRAIN768/DEV235 분할을 유지한다. DEV는 기존 실험에 반복 노출된 개발셋이며 새 독립 검증셋이 아니다. 모델은 전체 이미지와 초기 모델 학습 규약을 사용하고 아래 crop은 설명용이다.</p>
<h2>순서·대응 검산</h2>
<p>원 annotation의 cuboid/projected_cuboid/fraction은 이미 camera_dynamic_0123_v4 같은 순서다. saved projected_GT_native_xy와1003장 모두 정확히 같으며, 원 xyz·R/t·K 재투영 오차는 최대0.000357px다. stored perm_v4는 raw model 채널→고정 asset native의 대응으로만 보존했다. canonical_xyz[perm_v4]와 saved native_GT_xyz가1003장 모두 정확히 같다. 좌표나 가림 라벨에 permutation을 다시 적용하지 않았다. source self+fraction≥0.5 집계도1003장 모두 저장 V_actual/V_vis/ext_occ와 일치한다.</p>
<h2>M0/M4 대조: 물체 픽셀과 native 가시성은 같은 목표가 아니다</h2>
<table><tr><th>분할</th><th>D=0/E=1 proxy</th><th>N</th><th>M0 중앙 target</th><th>M4 중앙 target</th><th>M0 3×3 내 target</th><th>M4 3×3 내 target</th><th>반올림 중앙 범위 밖</th></tr>'''+''.join(pixelrows)+'''
</table><p>M0=팔레트 단독 silhouette, M4=최종 보이는 팔레트 silhouette. D_proxy 중앙이 M0에서 target이 아닌 경우가 많다. 가상 cuboid 끝점, 실제 재료 지지 여부, 경계 raster/alias 차이가 섞여 있어 이를 모두 물리 오라벨이라고 확정할 수 없다. D_proxy에서 M0 중앙은 target인데 M4에서 사라진 사례는0이다. E_proxy에서도 TRAIN3점은 중앙 M4 target이 남으며 주변9ray 다수 차단과 중앙 한 점의 차이가 있다. <strong>이 진단으로 frozen 라벨을 수정하거나 M4를 외부가림 교사로 쓰지 않았다.</strong></p>
<h2>실제 확인한 합성 이미지</h2>'''
    for x in cases:
        page+='<section><p>'+html.escape(x['finding'])+'</p><img src="syn_label_visual_audit_v1/'+x['file']+'"></section>'
    page+='''<h2>가용성과 한계</h2><p>전체 이미지의9좌표+8가림 모델을 원래 출력채널에 맞게 학습할 준비는 되어 있다. 이전 crop 후처리 실험과 다른 구조다. 그러나 물리 표면에서의 native 보임·외부가림이 인증된 합성 교사는 아니다. 과거 source 전체의 바이트 동일성 및 각9ray의 hit-owner/offset은 저장자료만으로 재인증하지 못한다. 기존 로컬 mesh 감사에서 가상 bbox 코너가 실제 표면에 놓이지 않을 수 있음도 확인됐다. 이 한계를 실험 성공으로 포장하지 않는다.</p>
<p><a href="syn_label_visual_audit_v1/INDEX_KO.html">전체33개 라벨 예제</a> · <a href="syn_joint_targets_v1/TARGET_AUDIT_SUMMARY.json">native8 감사 원표</a> · <a href="raw_model_targets9_v1/RAW_CHANNEL_CONTRACT.json">raw9 모델 채널 규약</a></p></html>'''
    path=safe(HERE/'LABEL_AUDIT_READOUT_KO.html');assert not path.exists(),path;path.write_text(page+'\n')
    qa={'completed':True,'PASS':True,'actual_PNGs_inspected':cases,'targets_modified':0,'current223_reads':0,
      'model_forwards':0,'optimizer_updates':0,'visibility_criteria_unchanged':'D0,E>=0.5,partial/self/outside masked',
      'readout':binding(path),'source_visual_complete':binding(HERE/'syn_label_visual_audit_v1/COMPLETE.json')}
    p=safe(HERE/'LABEL_VISUAL_QA.json');assert not p.exists();p.write_text(json.dumps(qa,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'readout':binding(path),'QA':binding(p)}),flush=True)


if __name__=='__main__':main()

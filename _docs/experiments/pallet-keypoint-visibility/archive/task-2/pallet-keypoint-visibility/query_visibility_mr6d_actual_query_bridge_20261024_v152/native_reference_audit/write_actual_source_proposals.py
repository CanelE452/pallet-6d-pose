"""Record actual visual assistant proposals; never overwrite original all-U reference output."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,collections,html
V=Path(__file__).resolve().parents[1];H=Path(__file__).resolve().parent;R=H/'provided_reference_v1'
def rd(p):return json.loads(p.read_text())
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def write(p,x):
 with p.open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
refpath=R/'ALL104_DYNAMIC_NATIVE_REFERENCE_GT_ONLY.jsonl';refs=[json.loads(x)for x in refpath.read_text().splitlines()];frames=rd(R/'FRAME10_INSTANCE13_GEOMETRY_AND_OVERLAYS.json')['frames']
states={
 (0,3):['D','D','D','D','D','E','U','D'],
 (1,3):['OOF','D','D','OOF','OOF','E','U','OOF'],
 (2,0):['OOF']*8,(2,1):['OOF']*8,
 (3,0):['D','D','D','OOF','D','D','S','D'],
 (3,1):['OOF','D','OOF','OOF','E','E','U','U'],
 (4,0):['D','D','D','D','D','D','S','D'],
 (4,1):['D','D','D','OOF','E','E','U','U'],
 (5,0):['D','OOF','OOF','D','U','U','U','U'],
 (6,0):['D','D','D','OOF','U','E','U','U'],
 (7,0):['OOF','U','OOF','OOF','E','D','D','U'],
 (8,0):['D','D','D','D','D','E','U','U'],
 (9,0):['OOF','OOF','OOF','OOF','E','D','D','U']}
reasons={}
def put(fi,ii,ks,text,blocker=None,tier='MODERATE_STRUCTURAL'):
 for k in ks:reasons[(fi,ii,k)]=dict(reason_KO=text,visual_blocker=blocker,evidence_tier=tier)
put(0,3,[0,3],'벽에 세운 target의 오른쪽 위 끝에서 deck와 runner 두께를 이루는 외곽 구조가 드러난다. 원 +Z deck 역할을 유지하며 화면 위아래로 다시 정렬하지 않았다.')
put(0,3,[1,2],'세운 target의 오른쪽 아래 끝과 짧은 두께 방향 끝 구조가 파란 통 오른쪽으로 노출된다. 명목 투영의 한 픽셀 일치가 아니라 실제 보이는 우측 끝 역할을 근거로 D를 제안한다.')
put(0,3,[4,7],'세운 target의 왼쪽 위 외곽 plank/runner 끝이 노출된다. 뒤쪽 깊이 순서는 제공 pose와 대응 수직변을 유지했으며 정확 좌표 gold는 아니다.')
put(0,3,[5],'target의 왼쪽 아래 deck 끝 역할 앞을 별도 목재 수납물이 가린다. 해당 면은 카메라를 향하는 구조이고 노출된 target 가장자리가 수납물 뒤로 이어져 외부 가림 후보로 판단한다.',blocker='앞쪽 별도 목재 수납/상자 물체; 공식 GT index는 이 검토에서 미연결')
put(0,3,[6],'같은 왼쪽 아래의 runner/하부 역할은 target 자체 두께와 앞 목재 수납물이 함께 영향을 준다. 외부 물체가 먼저 보인다는 사실만으로 E 또는 S를 결정할 수 없어 U.',blocker='앞 목재 수납물과 target 자체의 혼합')
put(1,3,[1,2],'우측 하단 target 끝은 수납물/파란 통의 오른쪽에서 드러난다. 세운 팔레트의 deck와 runner 끝 역할은 유지하되 cm급 명목 pose 오차를 별도 둔다.')
put(1,3,[5],'세운 target의 왼쪽 아래 deck 끝은 앞의 별도 목재 수납물 뒤에 들어가 있다. frame15의 노출된 상부·우측 구조와 동일 target 배치를 함께 보아 외부 가림 후보로 제안한다.',blocker='앞쪽 별도 목재 수납/상자 물체; 공식 GT index 미연결')
put(1,3,[6],'하부 뒤 역할에는 target 자체와 앞 수납물이 겹쳐 외부/자기 원인을 확정하기 어렵다.',blocker='목재 수납물과 target 자체의 혼합')
for fi in[3,4]:
 put(fi,0,[0,1,2,3,4,5,7],'위 팔레트의 노출된 deck 외곽과 보이는 두께/끝 구조를 역할별로 확인했다. 아래 팔레트와 분리된 상단 target이며, 정확 투영 픽셀의 목재 소속을 D의 유일 근거로 삼지 않았다.',tier='CLEAR_STRUCTURAL_WEAK')
 put(fi,0,[6],'위 팔레트의 뒤쪽 아래 corner 역할은 보이는 상단 deck/자체 bearer 뒤에 숨는다. 같은 수직변의 위 역할은 보이고 그 아래는 자신의 불투명 구조가 가리는 배치이며, 이 끝에는 별도 적재물이 없다. convex proxy와 별도로 실제 구조에 근거한 S 후보.',tier='MODERATE_STRUCTURAL')
 put(fi,1,[0,1,2,3],'아래 팔레트의 앞쪽 외곽 deck/끝 및 하단 지지 역할이 노출된다. 위 팔레트의 native 역할을 복사하지 않고 아래 target의 제공 pose·구조를 유지했다.',tier='CLEAR_STRUCTURAL_WEAK')
 put(fi,1,[4,5],'아래 팔레트의 뒤쪽 위 deck 끝 역할은 위의 별도 팔레트 아래로 들어간다. target의 앞쪽 노출 영역과 두 palette의 층 구조가 분리되며 위 팔레트가 외부 blocker인 E 후보다. 겹친 visible masks는 판정 근거로 쓰지 않았다.',blocker='같은 프레임의 별도 위 팔레트 GT instance0',tier='CLEAR_STRUCTURAL_WEAK')
 put(fi,1,[6,7],'아래 target의 뒤쪽 하부는 target 자체의 숨김과 위 팔레트의 덮임을 구분하기 어렵다. source visible masks도 비배타적이므로 자동 owner/E/S를 만들지 않는다.',blocker='위 팔레트와 target 자체의 혼합')
put(5,0,[0,3],'낮은 정면 시점에서 팔레트 왼쪽 앞 deck 끝과 하부 runner/지지 끝이 뚜렷하게 보인다. 적재물은 이 앞 끝 역할을 덮지 않는다.',tier='CLEAR_STRUCTURAL_WEAK')
put(5,0,[4,5],'낮은 시점에서 rear top 명목 참조는 얕은 deck 경계 및 파란/회색 적재물 경계 근처에 놓인다. cm급 pose·치수 오차로 D/E/self 관계가 달라질 수 있어 한 픽셀 mask로 결정하지 않고 U.',blocker='적재물 경계/자기 구조 원인 불확실')
put(5,0,[6,7],'뒤쪽 하부는 slot 개구부·runner와 자체 가림이 섞인다. 실제 corner 의미가 직접 보이는지, 자체에 숨는지 제공 투영만으로 확정하지 못해 U.')
put(6,0,[0,1,2],'target 앞쪽 deck 외곽과 우측 끝의 하단 지지가 노출된다. 빈 pallet 틈을 background라고 해서 사라진 corner로 판단하지 않았다.',tier='CLEAR_STRUCTURAL_WEAK')
put(6,0,[4],'뒤쪽 왼쪽 위 역할은 파란 KLT의 가장자리와 target deck 경계가 가깝다. 역할 지지 위치와 nominal projection의 오차 범위가 정해지지 않아 U.',blocker='파란 KLT 경계 불확실')
put(6,0,[5],'뒤쪽 오른쪽 deck 끝 역할을 가로놓인 회색 통이 덮는다. 위쪽 deck 면이 카메라 쪽에서 보이는 배치이며, 회색 적재물의 물체 범위가 기대 뒤쪽 끝을 가리는 외부 가림 후보.',blocker='target 위 가로놓인 회색 둥근 통')
put(6,0,[6,7],'뒤쪽 아래 역할은 팔레트 자체 두께·개구부와 적재물의 영향이 섞인다. proxy false를 semanticS로 복사하지 않고 U.')
put(7,0,[1],'명목 앞쪽 오른쪽 위 참조 y≈719.8은 영상 하단 바로 옆이다. 실제 구조와 cm급 pose 오차를 고려하면 보임/화면밖을 확정하기 어려워 U; nominal inframe flag는 그대로 보존한다.')
put(7,0,[4],'뒤쪽 왼쪽 위 deck 역할은 파란 KLT 뒤에 놓인다. target의 노출된 deck와 KLT가 차지한 적재 범위로부터 외부 blocker를 확인한 E 후보이며 KLT 픽셀 색만으로 붙인 라벨이 아니다.',blocker='파란 KLT 적재물')
put(7,0,[5,6],'팔레트 뒤쪽 오른쪽 끝은 금속/회색 적재물 오른쪽에서 짧은 끝면·외곽으로 노출된다. 명목 참조의 주변 구조 역할을 보아 D를 제안하되 exact physical xy는 인증하지 않는다.')
put(7,0,[7],'뒤쪽 왼쪽 하부는 KLT와 자체 구조가 함께 영향을 주며 직접 보임/자기/외부 원인을 구분하기 어렵다.',blocker='KLT와 target 자체의 혼합')
put(8,0,[0,1,2,3,4],'앞쪽 좌우 deck·하단 끝과 왼쪽 뒤 deck 끝의 노출된 목재 구조를 확인했다. 흰 적재물과 KLT가 있어도 이 외곽 끝 역할들은 보인다.',tier='CLEAR_STRUCTURAL_WEAK')
put(8,0,[5],'target 뒤쪽 오른쪽 위 deck 역할을 회색 KLT가 덮는다. 팔레트 deck가 카메라를 향하는 배치와 적재물의 범위로 외부 가림 후보를 제안한다.',blocker='회색 KLT 적재물')
put(8,0,[6,7],'뒤쪽 아래 runner 역할은 팔레트 자체와 적재물·명목 pose 불확실성이 겹친다. 실제 숨김 원인 확정이 어려워 U.')
put(9,0,[4],'target 뒤쪽 왼쪽 위 deck 끝 역할은 회색 KLT 아래로 들어간다. KLT 내부의 금속 통 픽셀이 있다는 이유가 아니라 KLT 전체가 target 끝을 덮는 구조로 E를 제안한다.',blocker='회색 KLT 적재물; 내부 금속 통은 별도物체')
put(9,0,[5,6],'뒤쪽 오른쪽 끝의 노출된 목재 외곽과 짧은 두께 끝을 확인했다. 명목 점이 인접 바닥 쪽으로 어긋나 있어도 보이는 구조 역할은 유지하며 E로 바꾸지 않는다.')
put(9,0,[7],'뒤쪽 왼쪽 아래는 KLT가 겹친 위치이면서 target 자체의 가림 가능성도 있다. semanticE/S를 단정하지 않고 U.',blocker='회색 KLT와 target 자체 혼합')
rows=[]
for r in refs:
 fi=r['frame_index'];ii=r['gt_instance_index'];k=r['native_slot'];state=states[(fi,ii)][k]
 if state=='OOF':
  assert not r['continuous_inframe'];why={'reason_KO':'제공 pose/K의 명목 reference가 원 영상 밖이다. 원 reference를 삭제하거나 안쪽으로 보정하지 않는다. 실제 corner의 cm급 오차가 남으므로 exact physical OOF gold는 아니며, inframe일 때의 D/E/S는 미판정이다.','visual_blocker':None,'evidence_tier':'NOMINAL_GEOMETRY_OOF_WITH_STRUCTURAL_TRUNCATION'}
 else:why=reasons[(fi,ii,k)]
 assert state=='OOF'or r['continuous_inframe']
 if state in['E','S']:assert r['convex_self_visible_proxy_only']if state=='E'else True
 if state=='E':assert why['visual_blocker']
 row=dict(r,proposal_state=state,proposal_scope='SOURCE_ASSISTANT_WEAK_ROLE_VISIBILITY_ONLY',proposal_human_gold=False,
  train_ready=False,query_target_association_reviewed=False,source_only=True,**why)
 row['semantic_visibility_given_inframe']='U'if state=='OOF'else state
 row['original_reference_semantic_state_unchanged']=r['semantic_state']
 row['reference_coordinate_precision']='Nominal supplied pose/CAD; low-centimeter actual uncertainty, no certified pixel radius or <=3px cohort.'
 rows.append(row)
assert len(rows)==104 and len({(x['frame_index'],x['gt_instance_index'],x['native_slot'])for x in rows})==104
out=H/'SOURCE104_NATIVE_SEMANTIC_PROPOSALS_AGENT.jsonl'
with out.open('x')as f:
 for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
counts=dict(collections.Counter(x['proposal_state']for x in rows));assert counts=={'D':39,'E':10,'U':18,'OOF':35,'S':2}
actually_viewed=[]
for f in frames:
 # For the two stacked frames, both per-instance overlays were personally viewed; otherwise all-instance overlay.
 if f['frame_index']in[3,4]:actually_viewed.extend(f['single_instance_overlays'])
 else:actually_viewed.append(f['all_instance_overlay'])
raw_extra=[Path(frames[i]['raw_RGB']['path'])for i in[7,9]]
summary={'completed':True,'source_only':True,'UTC':datetime.now(timezone.utc).isoformat(),'projection_seal':bind(R/'REFERENCE_PROJECTION_COMPLETE.json'),'proposals':bind(out),'proposal_code':bind(Path(__file__).resolve()),
 'actual_view_method':'functions.view_image:10frame overlays covering13GTinstance observations; stacked2frames viewed per-instance separately, plus unmarked fullRGB frames7/9. No actual R0 output read.',
 'personally_viewed_overlays':actually_viewed,'extra_raw_views':[bind(p)for p in raw_extra],
 'counts':counts,'frames':10,'GT_instance_observations':13,'not_13_distinct_physical_pallets':True,'native_rows':104,
 'per_instance':[{'frame_index':fi,'gt_instance_index':ii,'source_id':next(x['source_id']for x in rows if x['frame_index']==fi),'proposal_states':st,'counts':dict(collections.Counter(st))}for(fi,ii),st in states.items()],
 'findings_KO':['원래normal-facing/앞 top imageX/paired rear 규칙을 그대로 적용한 참조에서 실제 보이는 front 구조D 후보와 cargo/다른pallet 외부가림E 후보를 구분했다.','upright frame0/1의 왼쪽 아래deck 역할E와 하부뒤 혼합U를 분리했다.','stacked frame3/4에서 upper i0의 직접 보임과 자체 아래뒤 S후보, lower i1의 rear top을 upperpallet이 가리는 E후보를 구분했다.','낮은 정면 frame5의 rear top4/5는 cargo·얕은 경계와 cm급 pose불확실성 때문에 U로 남겼다.','nominal OOF35를 모두 보존했다. boundary-inside frame7n1도 실제 inframe의 불확실성이 커 U로 남긴다.','39D/10E는 query target association 전 source-only 역할 제안이다. 실제 첫 추론이 다른 물체를 선택하면 해당 라벨을 복사하지 않는다.'],
 'source_mask_overlap_not_used_to_autolabel':True,'convex_proxy_preserved_104_not_copied_to_semanticS':True,'original_all_U_references_unchanged':bind(refpath),
 'provided_T_K_R_pixel_or_reference_correction':False,'human_gold':False,'train_ready':False,'model_fit_download_private_access':0}
write(H/'SOURCE104_PROPOSAL_SUMMARY_KO.json',summary)
body='<!doctype html><html lang="ko"><meta charset="utf-8"><title>MR6D 원 native 참조·source 역할 제안</title><style>body{font:17px/1.65 sans-serif;max-width:1500px;margin:30px auto;padding:0 20px;color:#203247}img{max-width:100%;height:auto}article{margin:38px 0}table{border-collapse:collapse}td,th{border:1px solid #bdcbd9;padding:6px 12px}</style><h1>MR6D 10프레임 · 13개 instance 관측 · 원 native104 참조</h1><p>제공 pose/K와 원 V74 규칙만 사용했습니다. 원본 RGB와 별도 표시 복사본을 나란히 보여줍니다. source 역할 후보 D39/E10/S2/OOF35/U18이며 사람 gold·전체 학습 준비 완료를 뜻하지 않습니다. q 또는 가림 모델 출력은 읽지 않았습니다.</p><p>명목 pose/치수의 cm급 불확실성, nonexclusive visible masks, 실제 query target 대응 미확정을 유지합니다. OOF는 명목 참조 기준이며 exact pixel truth가 아닙니다.</p>'
for f in frames:
 rs=[x for x in summary['per_instance']if x['frame_index']==f['frame_index']];label='; '.join('i'+str(x['gt_instance_index'])+': '+','.join(x['proposal_states'])for x in rs)
 rel=f['raw_and_reference_panel']['path'].split('/native_reference_audit/')[1]
 body+='<article><h2>'+html.escape(f"{f['frame_index']:02d} {f['source_id']}")+'</h2><p>'+html.escape(label)+'</p><img src="'+html.escape(rel)+'" alt="원 RGB와 모든 instance의 원 native 참조"></article>'
body+='<p><a href="SOURCE104_NATIVE_SEMANTIC_PROPOSALS_AGENT.jsonl">104개 제안·근거</a> · <a href="SOURCE104_PROPOSAL_SUMMARY_KO.json">검토 요약</a> · <a href="provided_reference_v1/REFERENCE_PROJECTION_COMPLETE.json">원 참조 seal</a></p></html>'
with(H/'INDEX.html').open('x')as f:f.write(body)
write(H/'ACTUAL_VISUAL_REVIEW_COMPLETE.json',{'completed':True,'PASS':True,'summary':bind(H/'SOURCE104_PROPOSAL_SUMMARY_KO.json'),'proposals':bind(out),'index':bind(H/'INDEX.html'),'projection':bind(R/'REFERENCE_PROJECTION_COMPLETE.json'),'counts':counts,'all104_retained':True,'actual_all10_frame_pixels_viewed':True,'human_gold':False,'train_ready':False,'no_query_outputs_or_models_read':True,'new_model_fit_private_access':0})
print(json.dumps({'complete':bind(H/'ACTUAL_VISUAL_REVIEW_COMPLETE.json'),'counts':counts,'summary':bind(H/'SOURCE104_PROPOSAL_SUMMARY_KO.json')}))

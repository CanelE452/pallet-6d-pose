from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,json
R=Path(__file__).resolve().parent;V=R.parent
bind=lambda p:{'path':str(p),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest(),'bytes':Path(p).stat().st_size}
rd=lambda p:json.loads(Path(p).read_text())
def save(p,o):
 with p.open('x')as f:json.dump(o,f,ensure_ascii=False,indent=2);f.write('\n')
packet=rd(V/'SECONDARY5_REVIEW_PACKET_WITHOUT_PRIMARY_AFTER.json')
remarks={
'1.92m_lab:0000_0001':[
(True,'큰 가까운 앞wall의 가로 윗윤곽이 왼쪽 세로외곽으로 꺾이는 영역이 frozen범위에서 보인다. 뒤왼쪽 wall과 앞wall junction이 구분된다.'),
(True,'앞wall 오른쪽 윗윤곽/끝면의 junction이 frozen범위에서 보인다. 옆wall 연결과 겹치지만 principal앞wall endpoint는 방어가능하다.'),None,None,None,None],
'0.55m_lab:0016_0017':[
(True,'앞wall topplank의 왼쪽 윗선과 바깥 세로끝이 만나는 외곽을 실제로 볼 수 있다.'),
(True,'topplank 오른쪽 윗선/세로끝 junction이 box안에서 보인다.'),
(True,'가장 아래 넓은 frontplank의 왼쪽 아래lip/세로끝이 support위에서 드러난다.'),
(True,'가장 아래 넓은 plank 오른쪽 아래lip/세로끝이 support위에서 드러난다.'),
(True,'좁은 base runner 왼쪽 돌출부의 윗윤곽과 짧은 외곽이 보이며 floor선과 구분된다.'),
(True,'좁은 base runner 오른쪽 끝 상부/외곽이 보인다. 곡면·blur는 원불확실범위 안에 보존한다.')],
'0.55m_hall:0004_0005':[
(True,'왼쪽 foregroundcardboard 뒤로 드러난 woodenwall의 왼쪽 topcorner가 y188근처에 보인다. 상단과 세로끝 일부가 같이 드러난다.'),None,None,None,None,None],
'1.23m_hall:0032_0033':[
(True,'작은 frontwall 왼쪽 위 corner가 rear/insidewall보다 앞쪽으로 구분된다. frozen8px범위 수준의 region은 보인다.'),
(True,'frontwall 오른쪽 윗선/외곽끝이 보인다. 작고blurred하여 exactpixel이 아닌 range교차만 지지한다.'),
(True,'lowestwideboard 왼쪽 하단lip이 darksupport 위로 끝나는 영역은 보인다. 작은해상도에 따른 불확실성은 남는다.'),
(True,'lowestwideboard 오른쪽 아래lip과 외곽이 support상부에서 분리된다. exactjunction은 blurred하다.'),None,None],
'1.23m_hall:0024_0025':[
(True,'가까운 frontwall 왼쪽 상단 junction이 rearwall/cargo와 별개로 보인다.'),
(True,'frontwall 오른쪽 topcorner가 같은principalwall endpoint로 식별된다.'),
(True,'lowestwideplank 왼쪽 끝/하단lip이 어두운 foot위에서 끝나는 것을 볼 수 있다.'),
(True,'lowestwideplank 오른쪽 끝/하단lip이 어두운 foot와 구분된다.'),
(None,'frozenbase범위는 darksupport·좁은runner·바닥경계에 걸친다. 작은해상도/낮은대비 때문에 명명된 upperrunner-end junction을 독립적으로 확정하기 어렵다. 위치는 바꾸지 않는다.'),
(None,'오른쪽 base범위도 darksupport/runner/floor가 섞여 개별 upperouterjunction인지 불분명하다. namedcue 근거가 약하므로 before지원은null로 둔다.')]
}
records=[]
for row in packet['rows']:
 before=rd(row['before_annotation']['path']);slots=before['slots']
 for k,s in enumerate(slots):
  n=remarks[row['pair_id']][k]
  if s['reference_xy']is None:
   assert n is None;support=None;evidence='Primary before reference가 null이다. fullbefore에서 불가/가림/화면밖 상태를 보존하며 새 위치를 만들지 않는다.'
  else:assert n is not None;support,evidence=n
  records.append({'pair_id':row['pair_id'],'source_split':row['source_split'],'part_slot':s['part_slot'],'reference_xy':s['reference_xy'],
   'uncertainty_bbox_xyxy_half_open':s['uncertainty_bbox_xyxy_half_open'],'primary_before_state':s['before_state'],
   'before_supported_by_RGB':support,'independent_before_evidence_KO':evidence,
   'reference_or_uncertainty_changed':False,'after_RGB_seen_before_this_record':False,'native8_semantic_id':None,'human_gold':False,'training_admitted':False})
counts=Counter('true'if x['before_supported_by_RGB']is True else'false'if x['before_supported_by_RGB']is False else'null'for x in records)
report={'schema':'V85_SECONDARY5_BEFORE_SUPPORT_SEALED_BEFORE_AFTER','completed':True,'created_UTC':datetime.now(timezone.utc).isoformat(),
 'fixed_pairs':5,'slots':30,'source_RGB_before_full_and6detail_panels_actually_viewed':True,'primary_after_files_or_rawafter_read':0,
 'summary':{'before_supported':dict(counts),'primary_null_references':sum(x['reference_xy']is None for x in records)},'records':records,
 'bindings':[bind(R/'REVIEW_POLICY_BEFORE_RGB.json'),bind(R/'BEFORE_RENDER_RECEIPT.json'),bind(V/'SECONDARY5_REVIEW_PACKET_WITHOUT_PRIMARY_AFTER.json'),bind(Path(__file__))],
 'ML_current_forward_fit_PnP':0,'original_geometry_preserved':True}
o=R/'SECONDARY_BEFORE_SUPPORT.json';save(o,report);s=R/'SECONDARY_BEFORE_SUPPORT_SEAL.json';save(s,{'completed':True,'PASS':True,'before_support':bind(o),'policy':bind(R/'REVIEW_POLICY_BEFORE_RGB.json'),'render_receipt':bind(R/'BEFORE_RENDER_RECEIPT.json'),'no_after_read_before_this_seal':True,'primary_after_artifacts_read':0})
print(json.dumps({'report':bind(o),'seal':bind(s),'summary':report['summary']},ensure_ascii=False))

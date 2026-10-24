"""Render a local readout from completed source-only evidence; no training."""
from pathlib import Path
import hashlib, html, json
from PIL import Image

V=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def bind(p):
 p=Path(p).resolve();assert '_v112' not in str(p).lower()
 return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,obj):
 with Path(p).open('x')as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
 r=read(V/'ROOT_RESULTS_KO.json');g=V/'source_gallery_v1'
 assert (g/'GALLERY_COMPLETE.json').exists()
 pngs=list(g.glob('*.png'))
 assert pngs
 # Select the already-rendered contact sheet by its actual dimensions, never by case outcome.
 sizes=[(Image.open(p).width*Image.open(p).height,p)for p in pngs]
 contact=max(sizes,key=lambda z:z[0])[1]
 states=list(r['owner_states'])
 table=''.join('<tr><th>'+lab+'</th>'+''.join('<td>'+str(r['final_counts'][key][s])+'</td>'for s in states)+'</tr>'for key,lab in [('ROLE_D','원래 보임 D · 341'),('ROLE_E','원래 외부 가림 E · 49'),('ALL','합계 · 390')])
 interpretations=''.join('<li>'+html.escape(x)+'</li>'for x in r['interpretation_KO'])
 limits=r['progress_and_limits_KO']
 doc='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>공개 TRAIN crop 중심 감사 · V158</title>
<style>body{margin:0;background:#f4f6fa;color:#172b42;font:17px/1.7 system-ui,sans-serif}main{max-width:1150px;margin:38px auto;padding:0 24px 70px}h1{font-size:34px;line-height:1.35;margin:12px 0 20px}h2{font-size:25px;margin:36px 0 12px}.tag{color:#a24420;font-weight:700}.lead{font-size:21px}section{background:white;border:1px solid #dce3ec;padding:24px 28px;border-radius:12px;margin:24px 0}img{width:100%;height:auto;display:block}a{color:#176aab}table{border-collapse:collapse;width:100%;font-size:16px}td,th{padding:12px 10px;border-bottom:1px solid #d8e1ed;text-align:right}th:first-child{text-align:left}thead{background:#eef3f9}li{margin:10px 0}.note{font-size:15px;color:#4f6074}.status{font-weight:700;color:#a44623}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.grid p{margin:0}code{font-size:14px}@media(max-width:750px){main{padding:0 14px}section{padding:18px}.grid{display:block}h1{font-size:28px}table{font-size:13px}td,th{padding:8px 3px}}</style>
<main><p class="tag">V158 · 공개 학습자료 감사 완료 · 전체 목표는 미해결</p><h1>학습 라벨이 묻는 것과<br>초기 crop 중심이 보여 주는 것을 분리했습니다</h1><p class="lead">공개 TRAIN 390개를 실제로 확인했습니다. 원래 모서리의 보임 역할과, 고정된 초기 좌표의 중심에 어떤 표면이 보이는지는 서로 다른 정보입니다. 이번 결과는 실패 원인을 좁히는 근거이며 새 모델의 성능 향상은 아닙니다.</p>
<section><h2>현재 확인한 내용</h2><div class="grid"><p><b>입력 범위</b><br>공개 MR6D 목재 팔레트 88프레임 · 390 query. 원래 D341/E49, 좌표·crop 크기·native 순서·분할을 유지했습니다.</p><p><b>검토 방식</b><br>기존 역할 라벨과 모델 점수를 가리고 195개씩 검토했습니다. 고정한 40개는 서로 교차 검토했고 root의 별도 관찰도 대조했습니다.</p></div><p class="note">원래 D/E도 공개 프레임에서 앞선 단계에 만든 약한 구조적 역할 라벨이며, 공식 사람 키포인트 가림 정답이 아닙니다. 관찰 대상은 저장 RGB96의 중심 2×2 표본입니다. 주변32 crop과 전체 장면을 같이 보았습니다. 표본 혼합·물체 경계·소유 불명·padding은 미확정입니다. 사람의 정밀한 키포인트 정답이 아닙니다.</p></section>
<section><h2>원래 역할 라벨과 실제 중심 관찰</h2><img src="SOURCE_ROLE_VS_QUERY_OWNER_KO.png" alt="D341 E49 안의 중심 소유 관찰 분포"><table><thead><tr><th>원래 라벨</th><th>대상 표면</th><th>다른 물체</th><th>배경</th><th>미확정</th></tr></thead><tbody>'''+table+'''</tbody></table><p class="note">모델 혼동행렬이 아닙니다. 원래 D/E를 바꾸지 않고 별도 관찰을 붙인 교차표입니다. 배경146개는 D 라벨 오류146개라는 뜻이 아닙니다.</p></section>
<section><h2>실제 crop에서 확인할 차이</h2><p>다음 여섯 예시는 공개 TRAIN만 사용합니다. 각 범주 안에서 고정한 해시 규칙으로 골랐으며, 일치한 교차 검토 사례를 우선했습니다. 단일 검토 사례는 카드에 표시했습니다.</p><a href="'''+html.escape(str(contact.relative_to(V)))+'''"><img src="'''+html.escape(str(contact.relative_to(V)))+'''" alt="공개자료의 실제 crop 6개"></a><p class="note">이미지를 누르면 원본 크기로 볼 수 있습니다. 원래 역할 D/E와 초기 중심의 표면 소유를 같은 정답으로 취급하지 않습니다.</p></section>
<section><h2>이 결과를 어떻게 해석해야 하나</h2><ul>'''+interpretations+'''</ul><p><b>검토 일치:</b> 두 검토 34/40, root까지 세 관찰 30/40입니다. 40개 중 최종 미확정은18개이며, 그중8개는 모두 미확정으로 일치했습니다. 나머지10개는 의견 불일치 정책 적용 사례이고, 주검토의 확정 판단에서 추가로 미확정이 된 것은3개입니다. 관찰 반복성이며 정확도는 아닙니다. 교차 검토 밖의 나머지 350개는 한 검토자의 약한 관찰이며, 40개의 일치 결과를 350개에 일반화하지 않습니다.</p></section>
<section><h2>진전과 남은 문제</h2><p><b>확인한 진전:</b> '''+html.escape(limits['confirmed'])+'''</p><p><b>아직 확인하지 못한 것:</b> '''+html.escape(limits['not_confirmed'])+'''</p><p><b>남은 일:</b> '''+html.escape(limits['still_open'])+'''</p><p><b>성공 가능성:</b> '''+html.escape(limits['feasibility'])+'''</p><p class="status">새 모델 채택 없음 · 이번 단계 새 학습/추론/PnP/좌표 변경 없음</p><p>다음에는 공개 TRAIN에서 해당 native 부위가 실제로 보이는 근거를 추가로 확인합니다. 중심에 목재가 보인다는 이유만으로 정확한 키포인트라 하거나, 중심을 덮었다는 이유만으로 native 외부 가림 정답을 만들지 않습니다. 원래 목재·정사각형·native8 목표는 유지합니다.</p></section>
<section><h2>기록</h2><p><a href="ROOT_RESULTS_KO.json">전체 결과 JSON</a> · <a href="joined_owner_witness_v1/ALL390_SEPARATE_QUERY_VISIBLE_OWNER_WITNESS.jsonl">390개 별도 관찰</a> · <a href="joined_owner_witness_v1/ALL40_BLIND_CROSSREVIEW_AND_ROOT_PROPOSALS.jsonl">교차 검토40개</a> · <a href="independent_join_review/REVIEW.json">독립 결합 검증</a> · <a href="ROOT_NEXT_SOURCE_ACTION_AFTER_OWNER_WITNESS.json">다음 공개자료 확인 범위</a></p><p class="note">이 페이지는 공개 TRAIN 의미 감사입니다. 현재223장을 학습·문턱·모델 선택에 사용하지 않았고, 새 private 이미지·라벨·query별 출력을 읽지 않았습니다. 사용자 상태 응답을 위해 기존 부모 상태의 저장된 집계 성능을 확인한 사실은 결과 JSON에 따로 기록했습니다. 최근 실제 모델 학습은 V155, 최근 현재 평가 결과는 V156에 보존했습니다.</p></section></main></html>'''
 with (V/'ROOT_READOUT_KO.html').open('x')as f:f.write(doc)
 write(V/'ROOT_READOUT_SOURCE_SELF_CHECK.json',dict(completed=True,HTML=bind(V/'ROOT_READOUT_KO.html'),results=bind(V/'ROOT_RESULTS_KO.json'),contact=bind(contact),plot=bind(V/'SOURCE_ROLE_VS_QUERY_OWNER_KO.png'),all390_total=sum(r['final_counts']['ALL'].values()),no_model_accuracy_claim=True,no_role_relabel=True,not_browser_render_validation=True))
 print(json.dumps(dict(completed=True,report=bind(V/'ROOT_READOUT_KO.html'),contact=bind(contact)),ensure_ascii=False))
if __name__=='__main__':main()

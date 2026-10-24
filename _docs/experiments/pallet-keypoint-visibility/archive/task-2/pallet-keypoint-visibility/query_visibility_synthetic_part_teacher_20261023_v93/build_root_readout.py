from pathlib import Path
import json,hashlib,html,datetime
P=Path(__file__).resolve().parent
BASE=P.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):p=Path(p);return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def read(p):return json.loads(Path(p).read_text())
def write(p,o):
 assert not p.exists(),str(p)
 p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def link(p,label=None):
 p=Path(p);return '<a href="'+html.escape(str(p.relative_to(P)) if p.is_relative_to(P) else str(p),quote=True)+'">'+html.escape(label or p.name)+'</a>'
def main():
 actual=P/'raster_prototype/FIXED_PART152_SIX_CAMERAS'
 a=read(actual/'GEOMETRY_DIAGNOSTIC_RESULTS.json');seal=actual/'RASTER_OUTPUT_SEAL.json'
 assert a['completed'] and a['PASS'] and len(a['results'])==6
 for v in read(seal)['files']:assert sha(v['path'])==v['sha256']
 geo=P/'independent_geometry_review/GEOMETRY_AUDIT.json'
 pre=P/'independent_raster_preparation_review/REVIEW.json'
 post=P/'independent_actual_raster_audit/REVIEW.json'
 assert read(geo)['PASS'] and read(pre)['PASS'] and read(post)['PASS']
 visual=P/'independent_fixed6_visual_review/VISUAL_REVIEW_KO.json'
 assert visual.exists()
 crop=P/'saved_aux_query_crop_diagnosis/FINAL_SAVED_AUX_CROP_DIAGNOSIS_SEAL.json'
 assert crop.exists()
 cr=read(crop)
 cue=P/'semantic_contract_draft/EXPECTED_PART_CUE_MINIMUM_SEMANTIC_CONTRACT_DRAFT_V2_KO.md'
 total={k:sum(r['counts'][k]for r in a['results'])for k in ['A','S','V','SELF','FOREIGN','U']}
 assert total['A']==total['V']+total['SELF']+total['FOREIGN']+total['U']
 out=dict(schema='V93_FIXED_PART_GEOMETRY_RASTER_AND_MATCHED_AUX_QUERY_OPERATOR_DIAGNOSTIC_COMPLETE',completed=True,created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),goal_status='active',fully_solved=False,
  user_capture_provenance='직접 촬영했고 KU Leuven 자료와 무관합니다',
  root_entry=bind(P/'ROOT_ACTUAL_FIXED6_RASTER_ENTRY.json'),geometry_audit=bind(geo),raster_preparation_review=bind(pre),actual_raster_seal=bind(seal),actual_saved_buffer_audit=bind(post),actual_visual_review=bind(visual),root_actual_visual_observations=bind(P/'ROOT_ACTUAL_FIXED6_VISUAL_OBSERVATIONS_KO.json'),auxiliary_crop_diagnostic=bind(crop),semantic_contract_draft=bind(cue),
  actual_counts_by_camera=[dict(camera=r['camera'],**r['counts'])for r in a['results']],total_diagnostic_pixel_areas=total,
  resource=dict(seconds=a['seconds'],peak_RSS_bytes=a['peak_RSS_bytes'],actual_render_output_bytes=a['output_bytes_before_JSON']),
  finding='A long deck-like member can be hidden by other assembly members while the pallet-like assembly remains visible. Same-camera immutable-layer ownership distinguishes own-assembly hiding from additional foreign blocking. This is wholepart geometric evidence, not semantic keypoint classification.',
  admissibility='DIAGNOSTIC_ONLY; no training corpus, nativecorner D/E/S gold, new classifier or deployment admission.',
  limitations=['Separate auxiliary scene3 asset; native8 correspondence UNKNOWN, no q/ref read or snapping.','Part152 wholeboard coverage does not define a specific corner or junction cue.','Missing MDL references; flat opaque diagnostic palette and selectedgreen/foreignred highlight must never be training inputs.','All308 assembly ownership is a modelling assumption consistent with six views, not individual hiddenmesh semantic certification.','Declared metres retained; physical scale/author/license UNKNOWN.','U0 is numerical ownership result on six views, not a semantic abstention model.','Actual scalar-ray verification used 29 predeclared selected stratum rows / 15 unique rays, with19 empty strata unreplaced. It is a diagnostic sample, not an exhaustive independent pixel raster.','Per-part layers were not saved; their before/after immutability is a producer-bound digest certificate, while saved PS summaries can be replayed directly.','Aux query crop is explicitly posthoc geometry-centre emulation, not actual initializer q or a native8 test.','Current223 remains evaluation-only; latest current evaluation V91 unchanged, already reused development evaluation.'],
  new_source_training_updates=0,new_learned_model_forward=0,new_current223_data_access_or_forward=0,new_native_or_human_DES_labels=0,new_PnP_or_pose_optimization=0,
  latest_current_evaluation=dict(version='V91',report=str(BASE/'query_visibility_physical_cue_aux_20261023_v91/ROOT_READOUT_KO.html'),state='UNCHANGED'),
  next_required=['Define a cue-specific auxiliary support on the verified board/assembly without mapping virtual native8 to nearest physical surface.','Use same original-q-centred scalar/operator for train and sourceVAL; public-real cue/ownership and synthetic expectedpart evidence remain separate supervision tiers.','Create source before/foreign/sham/own-assembly negatives with GT-free expectedidentity; freeze source gates before any fit.','No current223 training/threshold/checkpoint selection; do not claim solved until frozen semantic transfer works.'])
 write(P/'ROOT_PHASE_RESULTS_KO.json',out)
 rows=''.join('<tr><td>'+html.escape(r['camera'])+'</td>'+''.join('<td>'+str(r['counts'][k])+'</td>'for k in ['A','S','V','SELF','FOREIGN','U'])+'</tr>'for r in a['results'])
 table='<table><thead><tr><th>고정 시점</th><th>A 단독</th><th>S 조립체에서 보임</th><th>V 외부 추가 후 보임</th><th>다른 부재에 숨음</th><th>추가 foreign에 숨음</th><th>U</th></tr></thead><tbody>'+rows+'</tbody></table>'
 figs=''.join('<details><summary>'+html.escape(r['camera'])+' — A '+str(r['counts']['A'])+' / S '+str(r['counts']['S'])+' / V '+str(r['counts']['V'])+'</summary><img loading="lazy" src="'+str(Path(r['PNG']['path']).relative_to(P))+'" alt="'+html.escape(r['camera'])+' 고정 렌더와 owner 마스크"></details>'for r in a['results'])
 crop_image=P/'saved_aux_query_crop_diagnosis/ACTUAL_SAVED_AUX_CROPS/ALL6_AUXILIARY_QUERY_CROP_CONTACT.png'
 assert crop_image.exists()
 cropfig='<img src="'+str(crop_image.relative_to(P))+'" alt="여섯 시점 auxiliary query 동일 위치 crop">'
 body=f"""<!doctype html><html lang="ko"><meta charset="utf-8"><title>V93 — 기대 부품 가림의 합성 감독 진단</title>
 <style>body{{font:17px/1.7 'Noto Sans CJK KR','NanumGothic',sans-serif;color:#15263c;background:#f3f6fa;margin:0}}main{{max-width:1160px;margin:auto;padding:32px}}h1{{font-size:31px;line-height:1.35}}h2{{font-size:23px;margin-top:32px}}p{{margin:12px 0}}.box{{background:white;border:1px solid #d9e1eb;border-radius:12px;padding:22px;margin:18px 0}}.status{{color:#8b4310;font-weight:bold}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #ccd6e2;padding:9px;text-align:right}}th:first-child,td:first-child{{text-align:left}}img{{width:100%;height:auto;background:white}}details{{background:white;padding:12px;margin:12px 0;border:1px solid #d9e1eb;border-radius:8px}}summary{{cursor:pointer;font-weight:bold}}a{{color:#1663a2}}.small{{font-size:14px;color:#43546c}}code{{word-break:break-all}}</style><main>
 <h1>기대 부품이 숨겨져도 팔레트는 보일 수 있다</h1>
 <p class="status">V93: 합성 geometry·가림 분리·같은 위치 crop 진단 완료. 분류 문제는 아직 미해결.</p>
 <div class="box"><p>한 팔레트 모양 조립체의 긴 상판 하나를 고정하고, 여섯 카메라에서 <b>부품 단독 A → 전체 조립체 S → 외부 가림 추가 V</b>를 비교했다. 다른 부재 뒤에 상판이 완전히 숨는 시점에서도 조립체의 다른 표면은 남았다. 따라서 전체 팔레트의 존재나 재질만으로 기대 부품의 가시성을 판정할 수 없다.</p>
 <p>이번 결과는 <b>부품 전체의 opaque depth/owner 진단</b>이다. 키포인트의 특정 모서리·접합선 가시성, 사람 D/E/S 정답, 새 모델 정확도를 측정한 결과가 아니다.</p></div>
 <h2>고정한 여섯 시점의 실제 렌더</h2>
 <p>수치는 분류 개수가 아니라 256×256 영상의 projected pixel 면적이다. A의 각 pixel을 V / 다른 부재에 숨음 / 추가 foreign에 숨음 / U로 나눈다. S와 V는 같은 원본 target raster layer를 재사용했다.</p>{table}
 <p class="small">A 밖은 물리적 부재 정답이 아니다. Background·동률 owner·입력 누락을 SELF로 바꾸지 않는다. 이번 여섯 시점에서 수치 U=0이었으며, 보류 판별 능력을 학습한 것은 아니다.</p>
 <div class="box"><p><b>실제 영상 관찰:</b> parallel deck·가로 받침·블록·포크 통로가 한 직사각 팔레트 모양 조립체를 이룬다. 선택 part152는 긴 상부 deck board로 보인다. 정확한 corner cue와 native8 대응은 UNKNOWN이다.</p>
 <p class="small">308개 mesh를 한 target assembly로 보는 것은 모델링 가정이다. 여섯 영상과 일치하지만, 개별 hidden mesh의 의미·ownership이나 native identity를 인증하지는 않는다.</p>
 <p>±X에서는 선택 부품이 전부 다른 부재에 숨는다. 아래 시점 −Y에서는 가로 받침 때문에 선택 부품이 여러 조각으로 나뉘어 보이고, 외부 물체가 추가로 다른 영역을 가린다.</p></div>{figs}
 <h2>같은 위치·같은 크기의 crop</h2>
 <p>모든 시점에서 보조 query를 부품 AABB 중심의 투영으로 정의하고 L=64px, RGB96을 사용했다. A/S/V의 q·L은 같다. 이는 저장 영상의 <b>사후 연산자 예시</b>이며, 실제 초기 추론 좌표·native8·모서리 cue가 아니다.</p>{cropfig}
 <p class="small">초록 selected / 빨강 foreign은 진단용 강조색이다. 이 영상이나 ID 색을 학습 입력으로 쓰면 정답이 새므로 훈련 데이터로 사용하지 않는다.</p>
 <h2>검산과 남은 조건</h2>
 <p>원본 USD↔export의 전체 좌표·triangle·part/material ID를 별도 코드로 검산했다. 렌더 전 생성 fixture에서 perspective 깊이, pixel centre, culling, shared edge, tie/U를 확인했고, 실제 저장 buffer는 사전에 고정한 광선 표본 정책으로 검산했다.</p>
 <p class="small">실제 광선 검산은 사전 정책으로 뽑은 29개 stratum 행(고유 광선 15개), 비어 있던 19개 stratum은 대체 없이 남겼다. 최대 깊이 차이는 1.07×10⁻¹⁴m였다. 전체 pixel을 별도 렌더러로 다시 만든 검사는 아니다. 저장하지 않은 per-part layer의 불변성은 producer가 남긴 before/after digest 증명에 한정하며, 저장된 PS 요약은 직접 검산했다.</p>
 <p>렌더 {a['seconds']:.2f}초, 최대 RSS {a['peak_RSS_bytes']/1e6:.1f}MB. 새 학습·학습모델 forward·현재223 접근/추론·PnP는 모두 0이다.</p>
 <p>기존 {link(BASE/'query_visibility_physical_cue_aux_20261023_v91/ROOT_READOUT_KO.html','V91 현재 평가 결과')}는 그대로다. 외부 가림 전이와 작은 좌표오차에서도 남는 실패는 이번 geometry 진단으로 해결됐다고 주장할 수 없다.</p>
 <p>다음은 <b>특정 구조 단서의 support와 기대 identity를 정하고, 공개 실사·합성에서 같은 q 중심 scalar를 학습·검증하는 것</b>이다. 부품 가림 감독은 보조 경로 하나이며, native virtual reference를 mesh 표면에 옮기거나 PnP를 최적화할 조건으로 삼지 않는다.</p>
 <p>원본 material MDL을 찾지 못해 flat opaque palette를 사용했다. 실제 재질·물리 크기·asset 저작자/license는 인증되지 않았다. 공개 실사 AI 관찰과 합성 proxy를 사람 native D/E/S gold로 승격하지 않는다. 평가223은 직접 촬영 자료이며 KU Leuven 자료와 무관하다는 사용자 확인을 보존했다.</p>
 <h2>근거 파일</h2><p>{link(P/'ROOT_PHASE_RESULTS_KO.json','기계 판독 결과')} · {link(seal,'실제 렌더 seal')} · {link(geo,'USD↔NPZ 검산')} · {link(pre,'렌더 코드 검토')} · {link(post,'실제 buffer 검산')} · {link(visual,'여섯 영상 별도 관찰')} · {link(crop,'crop 진단 seal')} · {link(cue,'감독 의미 계약 V2 초안')}</p>
 <p class="small">Goal active / fully_solved=false. 새 threshold·checkpoint·모델 선택에 현재 평가셋을 사용하지 않는다.</p></main></html>"""
 report=P/'ROOT_READOUT_KO.html';assert not report.exists();report.write_text(body)
 print(json.dumps(dict(report=bind(report),results=bind(P/'ROOT_PHASE_RESULTS_KO.json')),ensure_ascii=False,indent=2))
if __name__=='__main__':main()

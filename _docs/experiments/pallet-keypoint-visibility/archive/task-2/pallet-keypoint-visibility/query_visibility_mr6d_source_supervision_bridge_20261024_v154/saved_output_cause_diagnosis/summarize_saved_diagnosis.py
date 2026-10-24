from pathlib import Path
import hashlib,json,html
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def write(p,x):
    with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
    r=json.loads((H/'RESULTS.json').read_text());s=json.loads((H/'FIXED8_CASE_SELECTION_BEFORE_RGB.json').read_text());render=json.loads((H/'CROP_RENDER_COMPLETE.json').read_text())
    assert r['completed'] and r['PASS'] and render['PASS']
    for b in r['inputs']:assert sha(b['path'])==b['sha256']
    labels=[json.loads(x) for x in (V/'ROOT_ALL1840_SEPARATE_QUERY_SUPERVISION.jsonl').read_text().splitlines()]
    path=next((V/'runs/C35_C36_SEPARATE_QUERY_FIXED2000').glob('C36*/FINAL_MR1840_OUTPUTS.npz'))
    with np.load(path,allow_pickle=False)as f:z=f['raw_three_logits'];av=f['query_available']
    y=np.array([x['role_target']for x in labels]);e=np.array([x['entity_target']for x in labels]);tr=np.array([x['split']=='SOURCE_DEVELOPMENT'for x in labels])
    unknown={}
    for fold,fm in [('TRAIN',tr),('VAL',~tr)]:
        unknown[fold]={}
        for en in(-1,0,1):
            m=fm&av&(y<0)&(e==en)
            unknown[fold]['entity_truth_'+str(en)]=dict(N=int(m.sum()),entity_negative=int((m&(z[:,1]<0)).sum()),quality_negative=int((m&(z[:,2]<0)).sum()))
    observations={
        1764:'실제96은 회색 상자와 흰 윗면이 대부분이다. 명목 같은역할 참조와182px/1.69L 떨어져 품질음수가 위치품질을 반영할 가능성이 있다. 이 crop만으로 whole-pallet entity 정답을 재판정하지 않는다.',
        1213:'실제96은 회색 원형/사각 부품과 검은 틀 및 물체 가장자리이다. 명목거리65.8px/.37L. 가려진 팔레트와 다른물체 patch의 유사성은 보이지만 원 entity양성 약한 라벨을 그대로 둔다.',
        789:'실제96에 목재판 끝과 뒤의 회색/검정 구조가 함께 보인다. 명목20.2px/.12L로 >100px 사례와 구분된다. primaryE는강한양수인데 entity음수로만U가 된다.',
        741:'실제96에 목재 판재와 회색/검정 둥근 구조가 함께 보인다. 명목27.1px/.13L. entity거절이 단순 초대형 좌표편차 사례에만 한정되지는 않는다.',
        922:'실제96에 목재판/블록과 넓은 검은 invalid영역이 있다. 명목192.8px/1.46L로 역할D와 정확한query위치 지지는 별개다. 원D를E로 재주석하지 않는다.',
        905:'실제96에 목재 블록과 어두운 수직 구조가 있다. 명목144.3px/.98L. primaryD→E가 발생하지만 이 위치의role정확성은 실측gold가 아니다.',
        1816:'실제96은 회색/검정 흐린 구조다. 저장라벨은 firstargmax wrongentity이고 entity음수는 그라벨과 일치한다. crop만으로 독립 whole-object 정체성 검증을 주장하지 않는다.',
        587:'실제96은 거의 검정이고 일부 회색 경계다. 저장라벨은 검은 metalstand wrongentity. primary visibility의E출력을 wrongentity의E정답으로 취급하지 않는다.'}
    view=dict(completed=True,actual_board_viewed=True,board=render['board'],cases=[dict(global_index=c['global_index'],original96=render['cases'][i]['original96'],observation_KO=observations[c['global_index']],weak_labels_preserved=True)for i,c in enumerate(s['cases'])],view_scope='fixed board shows all8 original96 and exact central32 nearest displays; no additional whole-frame/RGB image read',independent_groundtruth_readjudication=False)
    write(H/'ACTUAL_FIXED8_VIEW_RECORD.json',view)
    conclusion={
        'completed':True,'PASS':True,'source_only':True,'control_complete':r['control_complete'],
        'key_findings_KO':[
            'C36의 세 출력은 분리돼 있지만64차원 hidden은 공유한다. 보조 두 손실을 함께 넣은 처치로 MR VAL E13→31/52가 늘고 D89→64/93이 줄었다. source35도22→17개로 감소하여 공동 목표 개선은 아니다.',
            'C36 entity gate는 TRAIN에서 맞게E인49개 중14개, VAL에서31개 중16개를U로 바꾼다. TRAIN D341개 entity거절0과 비교해 가림E에 편향된 거절이 학습자료에서도 남는다.',
            '모든2000 draw를 재생했다. entity양성 E49개는 총4865회, 각80–117회 노출되었고 거절14개도81–117회였다. 누락이나미추출은 아니지만 E는 entity양성638개 중49개로 작은하위집단이며 class내 의미적 균형은 보장되지 않는다.',
            'teacher D/E 각각에서도 visibility와entity logit은 거의 반대축(Pearson 약−.993~−.999), entity와quality는 같은축(약+.934~+.980)을 따른다. 출력의 강한 결합은 관측되지만 어느aux의gradient가 원인인지 또는 sharedhidden만이 원인인지는 이번 두aux동시처치로 식별되지 않는다.',
            'MR quality음수를 전부 visibility shortcut으로 단정하면 안 된다. 명목 같은역할 거리 중앙값은 TRAIN D19.1px/E68.8px, VAL D21.2px/E44.5px이며 E의 더 큰 query편차가 가능하다. 실제 real pointquality gold는 전혀 없다.',
            'VAL entity거절 E16개의 명목거리는 전부20–100px, 중앙39.8px/.202L이고 >100px는0개다. TRAIN거절E14중4개는>100px, 고정사례하나는182.3px/1.69L다. cm오차의providedpose라20px근처를 실측정답/오답으로 판정하거나 학습필터로 쓰지 않는다.',
            '독립 auxiliary hiddenbranch를 같은고정feature에서 학습하는 다음 source-only control은 근거가 있다. visibility hidden에 auxgradient가 들어가지 않게 하고 원C35visibility학습/primary결과를 그대로 유지하는 진단으로, entity가 trueE 양성을 구별할수있는지 먼저 확인한다. 현재 결과로 새설계의 성공이나 채택을 주장하지 않는다.'
        ],
        'all_source_rows_preserved':28976,'MR_original_rows':1840,'MR_ROLE_unknown_by_entity_truth':unknown,
        'draw_exposure':r['exposure'],'draw_hash':r['draw_replays'][0]['draw_sha'],
        'real_point_quality_gold':False,'nominal_distance_purpose':'cm-uncertain source-only diagnosis, never3pxgold/visibilityfilter/relabel/GTnearest',
        'causal_limit':'matched total effect of adding both entity and quality losses; separate aux effects and shared-latent causal mechanism not independently identified',
        'next_design_status':'recommendation only; no new fit/protocol/threshold/model choice',
        'original_bound_inputs_unchanged':True,'new_model_feature_weight_private_or_fit_calls':0,'selected_model':None,'goal_status':'active_unsolved',
        'results':bind(H/'RESULTS.json'),'view_record':bind(H/'ACTUAL_FIXED8_VIEW_RECORD.json'),'board':render['board'],
        'initial_failed_attempt':'provenance/INITIAL_SYNTAX_ONLY_FAILURE; parser failure before source reads, repaired closing metadata dict only'
    }
    write(H/'CAUSE_ASSESSMENT_KO.json',conclusion)
    rows=[]
    for fold in('SOURCE_DEVELOPMENT','SOURCE_VALIDATION'):
        for state in('D','E','ROLE_UNKNOWN_COMBINED'):
            a=r['conditions']['MR1840'][fold][state]['C36'];rows.append([fold,state,a['entity']['N'],a['entity']['negative'],a['quality']['negative'],f"{a['available_visibility_entity_Pearson']:.5f}",f"{a['available_entity_quality_Pearson']:.5f}"])
    th=['분할','역할','원input유효N','entity음수','quality음수','vis/entity Pearson','entity/quality Pearson']
    table='<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>'for x in th)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>'for x in row)+'</tr>'for row in rows)+'</table>'
    body='<!doctype html><html lang="ko"><meta charset="utf-8"><title>V154 source saved-output diagnosis</title><style>body{font:17px sans-serif;max-width:1280px;margin:32px auto;line-height:1.6;background:#f7f8fa;color:#172333}table{border-collapse:collapse;background:white;width:100%}th,td{padding:8px;border:1px solid #ccd3da}img{width:100%}p{max-width:1100px}</style><h1>세 출력 분리 후에도 함께 움직이는 점수</h1>'
    body+=''.join('<p>'+html.escape(p)+'</p>'for p in conclusion['key_findings_KO'])+table
    body+='<p>원본라벨·query·L·원96은 변경하지 않았다. 아래8개는 저장점수로 고정한 사후 진단 사례이며 전체평가분모의 대체가 아니다.</p><img src="FIXED8_SAVED_CROP_DIAGNOSIS.png" alt="fixed8 source crops">'
    body+='<ul>'+''.join('<li>'+html.escape(c['category']+' / '+c['query_id'])+' <a href="'+html.escape(c['original_query_evidence']['path'])+'">기존 whole-query board</a></li>'for c in s['cases'])+'</ul>'
    body+='<p><a href="RESULTS.json">전체 조건표</a> · <a href="FIXED8_CASE_SELECTION_BEFORE_RGB.json">선정manifest</a> · <a href="ACTUAL_FIXED8_VIEW_RECORD.json">실제view기록</a></p></html>'
    with (H/'CAUSE_ASSESSMENT_KO.html').open('x')as f:f.write(body)
    write(H/'CAUSE_ASSESSMENT_COMPLETE.json',dict(completed=True,PASS=True,assessment=bind(H/'CAUSE_ASSESSMENT_KO.json'),HTML=bind(H/'CAUSE_ASSESSMENT_KO.html'),view_record=bind(H/'ACTUAL_FIXED8_VIEW_RECORD.json'),crop_render=bind(H/'CROP_RENDER_COMPLETE.json'),diagnosis=bind(H/'DIAGNOSIS_COMPLETE.json'),code=bind(__file__),selected_model=None,new_model_fit_private_calls=0,original_labels_and_scores_unchanged=True))
    print(json.dumps(bind(H/'CAUSE_ASSESSMENT_COMPLETE.json')))
if __name__=='__main__':main()

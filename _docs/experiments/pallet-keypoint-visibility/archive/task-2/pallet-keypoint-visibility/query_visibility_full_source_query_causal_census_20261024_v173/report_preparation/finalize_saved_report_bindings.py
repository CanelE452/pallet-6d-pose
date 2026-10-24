"""Bind completed saved-result audit and actual v2 visual QA; no arrays/source reads."""
from pathlib import Path
import argparse,hashlib,json,shutil
from html.parser import HTMLParser
V=Path(__file__).resolve().parent.parent
def bind(p):return dict(path=str(Path(p).resolve()),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('w') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main(rootqa):
    ap=V/'actual_saved_census_review_a/REVIEW.json';a=read(ap);assert a['PASS'] and bind(ap)['sha256']=='d0ab3b768272560946622f61d40fdb8971593b1aee7271c23d7d73841986c73e'
    terminal=V/'ROOT_ACTUAL_FULL_SOURCE_CENSUS_EXECUTION_TERMINAL.json';assert bind(terminal)['sha256']=='5dd140a35335486fd1f86a9564450cd638b3a5a5067bae7759e4dc9474124e87'
    figure=V/'scientific_readout_v2/FULL6144_SOFT_OWNER_CAUSAL_SEMANTIC_CENSUS.png';assert bind(figure)['sha256']=='fb9be42905426c3367b61687abf87a69da8260be7ecb2bed35dba7d70070d166'
    rootqa=Path(rootqa).resolve();assert rootqa.is_relative_to(V);qr=read(rootqa);assert bind(figure)['sha256'] in rootqa.read_text()
    ownqa=V/'scientific_readout_v2/ACTUAL_VISUAL_QA.json';assert read(ownqa)['PASS']
    prior=V/'provenance/BEFORE_FINAL_AUDIT_AND_ROOT_V2_QA_BINDING';prior.mkdir(parents=True)
    for n in ('ROOT_RESULTS_KO.json','ROOT_READOUT_KO.html','ROOT_PHASE_RESULT.json'):shutil.copy2(V/n,prior/n)
    rp=V/'ROOT_RESULTS_KO.json';r=read(rp);r['publication_ready']=True;r['pending_final_bindings']=[];r['bindings']['figure']=bind(figure)
    r['bindings'].update(independent_actual_saved_census_review=bind(ap),root_v2_actual_visual_QA=bind(rootqa),author_v2_actual_visual_QA=bind(ownqa),root_actual_execution_terminal=bind(terminal),preserved_v1_figure=bind(V/'scientific_readout_v1/FULL6144_SOFT_OWNER_CAUSAL_SEMANTIC_CENSUS.png'),report_finalizer=bind(__file__))
    r['independent_audit_scope']=dict(PASS=True,coverage=a['coverage'],limitations=a['limitations'],method=a['method'])
    r['interpretation_KO'].append('독립검산은 원768 bufferSHA·6144metadata와6088 유한 q의 nearest/4-source-pixel point owner/joint를 직접 재계산했다. 9soft배열의 NPZ↔JSONL 일치와189집계도 확인했다. central4/32는 저장값 일치·집계 검산이며 독립 재샘플링하지 않았다. 원 fullframe joint/reason pixelcount도 독립 재계산 범위 밖이다.')
    write(rp,r)
    hp=V/'ROOT_READOUT_KO.html';page=hp.read_text().replace('scientific_readout_v1/FULL6144_SOFT_OWNER_CAUSAL_SEMANTIC_CENSUS.png','scientific_readout_v2/FULL6144_SOFT_OWNER_CAUSAL_SEMANTIC_CENSUS.png')
    page=page.replace('독립 실제검산과 root의 PNG 실제 시각 QA 최종 바인딩은 대기 중입니다.','독립 실제검산 PASS와 root/작성자의 v2 PNG 실제 시각 QA를 최종 바인딩했습니다. 원 v1 그림과 보고서는 보존했습니다. 독립검산은 원768 buffer SHA,6144 metadata,null56/U88 및6088 q point의 nearest/4-source-pixel owner/joint를 직접 재계산했습니다. 9soft 저장배열 일치와189집계도 검산했습니다. central4/32는 저장값/집계 확인이며 독립 재샘플링하지 않았고 fullframe joint/reason count도 독립 재계산하지 않았습니다.')
    page=page.replace('</small>','</small><p><a href="actual_saved_census_review_a/REVIEW.json">독립 실제검산</a> · <a href="'+str(rootqa.relative_to(V))+'">root v2 실제 시각 QA</a> · <a href="scientific_readout_v2/ACTUAL_VISUAL_QA.json">작성자 v2 실제 시각 QA</a> · <a href="scientific_readout_v1/FULL6144_SOFT_OWNER_CAUSAL_SEMANTIC_CENSUS.png">보존된 v1 그림</a></p>')
    hp.write_text(page)
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.refs=[]
        def handle_starttag(self,t,attrs):self.refs.extend(x for k,x in attrs if k in ('href','src'))
    parser=Links();parser.feed(page);assert all((V/x).is_file() for x in parser.refs)
    pp=V/'ROOT_PHASE_RESULT.json';p=read(pp);p.update(publication_ready=True,pending_final_bindings=[],results=bind(rp),readout=bind(hp),figure=bind(figure),independent_actual_saved_census_review=bind(ap),root_v2_actual_visual_QA=bind(rootqa),author_v2_actual_visual_QA=bind(ownqa),root_actual_execution_terminal=bind(terminal),preserved_v1_figure=bind(V/'scientific_readout_v1/FULL6144_SOFT_OWNER_CAUSAL_SEMANTIC_CENSUS.png'),independent_central4_central32_resampling_performed=False,local_links_and_image_refs_checked=len(parser.refs),report_finalizer=bind(__file__))
    write(pp,p);print(json.dumps(dict(results=bind(rp),readout=bind(hp),phase=bind(pp),figure=bind(figure))),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root-qa',type=Path,required=True);args=ap.parse_args();main(args.root_qa)

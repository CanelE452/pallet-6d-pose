"""Number IDs, precise JSON pointers, transformations and manuscript locations."""
from env import *
def run(source,rtpath,values):
    pop=OLD_DOC/'POPULATION_AND_METRIC_LOCK.json';means=OLD_DOC/'SEED_MEAN_RESULTS.json';pr=OLD_DOC/'P_VS_R0_PAIRED.json';pd=OLD_DOC/'P_VS_D_PAIRED.json';rows=[]
    mapping={
        'Pparams':(rtpath,'params.P1.refiner_trainable','parameters','integer with separators'),
        'Dparams':(rtpath,'params.D1.refiner_trainable','parameters','integer with separators'),
        'DevN':(pop,'positive','images','integer'),'NegN':(pop,'negative','images','integer'),
        'SessionN':(pop,'sessions','sessions','length'),'MatchedN':(pop,'matched','images','integer'),
        'MatchedPoints':(pop,'matched_supervised','points','integer with separators'),'AllPoints':(pop,'total_supervised','points','integer with separators'),
        'TrainN':(pop,'synthetic_partitions.train','images','integer with separators'),'UsableTrainN':(pop,'synthetic_matched_train','rows','integer with separators'),
        'CalN':(pop,'synthetic_partitions.calibration','images','integer with separators'),'SelectionN':(pop,'synthetic_partitions.selection','images','integer with separators'),'HeldoutN':(pop,'synthetic_partitions.heldout','images','integer with separators'),
        'Rmedian':(source,'methods.R0.median_px','original px','3 decimals'),'Pmedian':(means,'P.median_px','original px','3 decimals'),
        'Improve':(pr,'session.delta','original px','negate;3 decimals'),'ImproveLow':(pr,'session.high','original px','negate;3 decimals'),'ImproveHigh':(pr,'session.low','original px','negate;3 decimals'),
        'RPCK':(source,'methods.R0.ALL_GT_PCK.10','percent','fraction x100;3 decimals'),'PPCK':(means,'P.ALL_GT_PCK.10','percent','fraction x100;3 decimals'),
        'PDdelta':(pd,'session.delta','original px','3 decimals'),'PDlow':(pd,'session.low','original px','3 decimals'),'PDhigh':(pd,'session.high','original px','3 decimals'),
        'Ptime':(rtpath,'summary.P1.image_to_2d_ms.median','ms','3 decimals'),'Pfulltime':(rtpath,'summary.P1.end_to_end_ms.median','ms','3 decimals'),
        'Rtime':(rtpath,'summary.R0.image_to_2d_ms.median','ms','3 decimals'),'Rfulltime':(rtpath,'summary.R0.end_to_end_ms.median','ms','3 decimals')}
    texts={p.name:p.read_text() for p in (PAPER/'manuscript.tex',PAPER/'supplementary.tex')}
    for name,(p,key,unit,transform) in mapping.items():
        value=read(p)
        for k in key.split('.'):value=value[k]
        rows.append(dict(id=name,source=bound(p),json_key=key,raw_value=value,unit=unit,displayed=values[name],transformation=transform,population='source partitions' if p==pop and name in ('TrainN','UsableTrainN','CalN','SelectionN','HeldoutN') else 'paired13sessions' if p in (pr,pd) else 'DEV319 / negative2689 / fullGT2818 as named; timing26images x5',locations=[f'{n}:{i}' for n,t in texts.items() for i,line in enumerate(t.splitlines(),1) if '\\'+name in line]))
    rows.append(dict(id='TranslationImprove',sources=[bound(source),bound(means)],json_keys=['methods.R0.pose.translation_median_cm','P.pose.translation_median_cm'],transformation='(R0-Pseedmean)*10;3 decimals',unit='mm change of geometry-reference error statistic, not independent physical measurement',displayed=values['TranslationImprove'],locations=['manuscript.tex:Results geometry subsection']))
    # Every numeric result/table cell gets a pointer, including supplementary rows.
    def leaves(obj,prefix=''):
        if isinstance(obj,dict):
            for k,v in obj.items():yield from leaves(v,prefix+'.'+str(k) if prefix else str(k))
        elif isinstance(obj,list):
            for i,v in enumerate(obj):yield from leaves(v,prefix+'.'+str(i))
        elif isinstance(obj,(int,float)) and not isinstance(obj,bool):yield prefix,obj
    tables={'methods':(source,read(source)['methods'],['accuracy.tex','full_accuracy.tex','full_pose.tex']), 'runtime':(rtpath,read(rtpath)['summary'],['runtime.tex','full_runtime.tex']), 'resources':(DOC/'RESOURCE_COMPARISON.json',read(DOC/'RESOURCE_COMPARISON.json'),['resources.tex'])}
    for group,(p,obj,loc) in tables.items():
        for key,value in leaves(obj):rows.append(dict(id=group+'.'+key,source=bound(p),json_key=('methods.' if group=='methods' else 'summary.' if group=='runtime' else '')+key,raw_value=value,locations=['generated_tables/'+n for n in loc],transformation='see deterministic manuscript.py table generator: seed mean only for main family rows; PCK/gross x100; all supplemental rows per-model',rounding='main3 decimals except PCK2; supplementary3 except rates2'))
    for p in (DOC/'P_VS_PRIOR_PAIRED.json',DOC/'CONFIRMATION_RESULTS.json',DOC/'MEASUREMENT_INPUT_AUDIT.json'):
        if p.exists():
            for key,value in leaves(read(p)):rows.append(dict(id=p.stem+'.'+key,source=bound(p),json_key=key,raw_value=value,locations=['generated_tables/measurement_inputs.tex' if 'MEASUREMENT' in p.name else 'generated_tables/prior_results.tex' if 'PRIOR' in p.name else 'generated_tables/confirmation_results.tex','generated_tables/numbers.tex'],rounding='dimensions2, intrinsics3 decimals; integer counts unchanged'))
    write(PAPER/'NUMBERS_MANIFEST.json',dict(numbers=rows,source_selection='No manual substitution; all outputs regenerated from specified JSON',implementation=bound(HERE/'manuscript.py'),asset_implementation=bound(HERE/'paper_assets.py')))

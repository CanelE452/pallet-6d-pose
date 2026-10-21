"""Stored-pose C2 coverage aggregation only; never fit or infer a head."""
import pathlib,importlib.util,time,json,resource
R=pathlib.Path(__file__).resolve().parent;start=time.monotonic();cpu=time.process_time();s=importlib.util.spec_from_file_location('leave2_saved_coverage',R/'experiment_v2.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.targets()
cost=dict(status='COMPLETE',seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,new_training_forward_RGB_PnP=0,scope='read sealed feature poses and existing synthetic labels; fixed C2 metric and eligible bank coverage only; excludes candidate export time')
with(R/'COVERAGE_COST.json').open('x')as h:json.dump(cost,h,indent=2);h.write('\n')
print(cost)

"""Resume runtime without repeating completed training, inference or bootstraps."""
from env import *
def run():
    if complete('EVALUATE_COMPLETE'): verify(); return
    import dev_completion
    if complete('DEV_COMPLETE'):
        from submission_runtime import run as runtime
        start=now();verify();runtime()
        receipt('EVALUATE_COMPLETE',[DOC/'DEV_COMPLETE.json',HERE/'evaluate_pipeline.py',HERE/'submission_runtime.py'],
            [DOC/'UNIFIED_DEV_RESULTS.json',DOC/'P_VS_PRIOR_PAIRED.json',DOC/'RUNTIME_PANEL.json'],start)
    else:
        import evaluate_compatible
        try:evaluate_compatible.run()
        finally:
            if all(complete(f'PRIOR{s}{suffix}_SCORED') for s in (1,2,3) for suffix in ('','_raw')) and (DOC/'P_VS_PRIOR_PAIRED.json').exists():
                dev_completion.run()

"""Unchanged V16 saved-clock/counter arithmetic with READY input authority."""
import runpy
from pathlib import Path
_entry_name=__name__
_env=runpy.run_path(str(Path(__file__).with_name('ready_dispatch.py')))['derive']('runtime_check',Path(__file__))
globals().update(_env)
if _entry_name=='__main__':main()

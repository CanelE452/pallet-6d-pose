"""Initialize CPU library control before the unchanged guarded audit runs.

The first direct invocation stopped when threadpoolctl opened /dev/null after
the audit's write guard was installed. No per-model audit or output existed.
This entry point changes initialization order only; run(), guards, mathematics,
input bindings, masks, precision and all diagnostic criteria remain unchanged.
"""
from . import representation_audit as A
from threadpoolctl import threadpool_limits


if __name__=='__main__':
    with threadpool_limits(limits=1):
        A.run()

# Symmetry-aware DHT local fusion v2

Final bounded experiment correcting exactly four v1 equation-level mismatches:
anchor-vs-point loss scaling, structural support vs correction utility, hard-bin
vs continuous line supervision, and simultaneous fusion of alternative modes.
It reuses the immutable v1 1,792/256/512 export and does not alter v1 results.

Run modules from the repository root with
`python -m scripts.research.pallet_symmetry_dht_local_v2.symdht_local_v2.<module>`.
CUDA requests never fall back silently to CPU.

"""Train one arm/seed on the exported ROI features. The backbone stays frozen."""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
from plpose_local.heads import PoseNet, ARMS          # noqa: E402
from plpose_local import losses, solver               # noqa: E402

CFG = dict(thetas=36, rhos=65, modes=3, starts=4, width=48, iters=5, temperature=.05,
           energy_temperature=1.,
           w_point=1., w_line=.2, w_pose=1., w_seed=.2, w_rank=.1, channels=192)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def state_sha(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Split:
    def __init__(self, manifest):
        self.path = Path(manifest).resolve(); self.root = self.path.parent
        self.doc = json.loads(self.path.read_text())
        self.records = self.doc['records']

    def batch(self, idx, device, targets=True):
        obs, sup, orders = [], [], []
        for i in idx:
            r = self.records[int(i)]
            o = torch.load(self.root / r['observation'], map_location='cpu', weights_only=True)
            obs.append(o); orders.append(r['symmetry']['order'])
            if targets:
                sup.append(torch.load(self.root / r['supervision'], map_location='cpu', weights_only=True))
        O = {k: torch.cat([o[k] for o in obs], 0).to(device) for k in obs[0]}
        b = O['box'].shape[0]
        O['point_mask'] = torch.tensor([[True] * 8 + [False]], device=device).expand(b, 9).contiguous()
        O['line_mask'] = torch.ones(b, 12, dtype=torch.bool, device=device)
        orders = torch.tensor(orders, device=device)
        if not targets:
            return O, None, orders
        S = {k: torch.cat([s[k] for s in sup], 0).to(device) for k in sup[0]}
        return O, S, orders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True); ap.add_argument('--output', required=True)
    ap.add_argument('--arm', choices=ARMS, required=True); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--steps', type=int, default=2000); ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'cell exists, refusing to overwrite: {out}')
    out.mkdir(parents=True, exist_ok=True)
    data = Split(args.manifest)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    model = PoseNet(arm=args.arm, channels=CFG['channels'], width=CFG['width'],
                    starts=CFG['starts'], thetas=CFG['thetas'], rhos=CFG['rhos'],
                    modes=CFG['modes']).to(args.device)
    initial = state_sha(model.state_dict())
    shared = state_sha({k: v for k, v in model.state_dict().items() if not k.startswith('line.')})
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    plan = []
    while len(plan) < args.steps * args.batch:
        plan.extend(rng.permutation(len(data.records)).tolist())
    plan = np.asarray(plan[:args.steps * args.batch], np.int64).reshape(args.steps, args.batch)
    (out / 'START.json').write_text(json.dumps(dict(
        arm=args.arm, seed=args.seed, steps=args.steps, batch=args.batch, lr=args.lr,
        config=CFG, manifest_sha256=sha(args.manifest), records=len(data.records),
        initial_state_sha256=initial, shared_state_sha256=shared,
        plan_sha256=hashlib.sha256(plan.tobytes()).hexdigest(),
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        trainable_line_branch=sum(p.numel() for n, p in model.named_parameters() if n.startswith('line.')),
        torch=torch.__version__), ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    started, done = time.monotonic(), 0
    try:
        model.train()
        for step in range(args.steps):
            O, S, orders = data.batch(plan[step], args.device)
            out_m = model(O)
            loss, parts, _ = losses.symmetric_loss(out_m, O, S, orders, solver, CFG)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True))
            opt.step(); done += 1
            with (out / 'TRACE.jsonl').open('a', encoding='utf8') as f:
                f.write(json.dumps(dict(step=done, loss=float(loss.detach()), grad_norm=gn,
                                        indices=plan[step].tolist(), **parts)) + '\n')
            if done % 200 == 0 or done == args.steps:
                print(f'{args.arm} seed={args.seed} {done}/{args.steps} loss={float(loss.detach()):.5f}', flush=True)
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(dict(schema='plpose_local_head_1', arm=args.arm, seed=args.seed, config=CFG,
                        state_dict=state, steps=done, initial_state_sha256=initial),
                   out / 'checkpoint_final.pt')
        (out / 'COMPLETION.json').write_text(json.dumps(dict(
            training_completed=True, optimizer_steps=done,
            checkpoint_sha256=sha(out / 'checkpoint_final.pt'),
            final_state_sha256=state_sha(state), initial_state_sha256=initial,
            elapsed_seconds=time.monotonic() - started), ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    except BaseException as exc:
        (out / 'FAILURE.json').write_text(json.dumps(dict(completed=False, steps=done, error=repr(exc)),
                                                     ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        raise


if __name__ == '__main__':
    main()

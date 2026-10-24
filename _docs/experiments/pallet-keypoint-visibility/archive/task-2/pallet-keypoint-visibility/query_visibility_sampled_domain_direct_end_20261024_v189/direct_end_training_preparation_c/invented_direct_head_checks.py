"""Root-addendum-authorized invented CPU arithmetic only; never save toy weights."""
import copy
import hashlib
import numpy as np
import torch


def digest(value):
    h = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            x = x.detach().cpu().contiguous()
            h.update(str((str(x.dtype), tuple(x.shape))).encode()); h.update(x.numpy().tobytes())
        elif isinstance(x, dict):
            for key in sorted(x, key=str): h.update(str(key).encode()); visit(x[key])
        elif isinstance(x, (list, tuple)):
            for item in x: visit(item)
        else: h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def run_checks(head, candidate, five_losses):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checks = {}
    new_class = candidate.make_candidate_head_class(head)

    def initialize(extended):
        torch.manual_seed(1890)
        model = new_class() if extended else head.Raw96SupportHead()
        return model.float(), torch.get_rng_state().clone()

    baseline, base_rng = initialize(False)
    extended, ext_rng = initialize(True)
    names = list(baseline.state_dict())
    checks['same_original_initial_parameters_and_RNG_with_zero_extra65'] = (
        all(torch.equal(baseline.state_dict()[k], extended.state_dict()[k]) for k in names)
        and torch.equal(base_rng, ext_rng)
        and torch.count_nonzero(extended.direct_end_head.weight).item() == 0
        and torch.count_nonzero(extended.direct_end_head.bias).item() == 0)
    checks['exact115157_plus65_parameters'] = (
        sum(p.numel() for p in baseline.parameters()) == 115157
        and sum(p.numel() for p in extended.parameters()) == 115222)
    # Exact actual candidate forward exercises encoder -> pool -> shared64.
    generator = torch.Generator().manual_seed(1891)
    x = torch.randn((2, 8, 96, 96), generator=generator, dtype=torch.float32)
    old = baseline(x); new = extended(x)
    checks['actual_CPU_forward_pool_shape_and_old_logits_exact'] = (
        new['direct_end_external_logit'].shape == (2,)
        and all(torch.equal(old[k], new[k]) for k in old))
    del baseline, extended, old, new

    def branch(extended):
        model, initial_rng = initialize(extended)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        rng = np.random.default_rng(1892)
        history = []
        for step in range(3):
            # Newly invented input, no source cache or production forward.
            inp = torch.tensor(rng.standard_normal((2,8,96,96)), dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            out = model(inp)
            # Existing exact five-loss function still receives its fixed192
            # output rows, tiled from the two invented inputs for CPU economy.
            tiled = {k: out[k].repeat((96,1)) for k in ('support_logits','owner_logits')}
            loss, components = five_losses(tiled)
            if extended:
                before = loss
                loss = candidate.add_direct_objective(
                    loss, out['direct_end_external_logit'],
                    torch.tensor([0,1], dtype=torch.int8), torch.ones(2,dtype=torch.bool), 0)
                candidate.require(loss is before, 'Lambda0 exact original loss object')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            opt.step()
            params = dict(model.named_parameters())
            original = {k: model.state_dict()[k].clone() for k in names}
            old_opt = {k: copy.deepcopy(opt.state[params[k]]) for k in names}
            history.append(dict(step=step+1, parameters=digest(original),
                optimizer=digest(old_opt), RNG=digest(torch.get_rng_state()),
                draw=digest(rng.bit_generator.state), loss=digest(loss.detach()),
                components=digest([v.detach() for v in components]), norm=digest(norm)))
            if extended:
                candidate.require(all(params[k].grad is None for k in ('direct_end_head.weight','direct_end_head.bias')), 'Lambda0 direct branch no gradient')
                candidate.require(all(params[k] not in opt.state for k in ('direct_end_head.weight','direct_end_head.bias')), 'Lambda0 no extra optimizer state')
        return history

    old_history, new_history = branch(False), branch(True)
    checks['three_invented_AdamW_steps_old_parameters_optimizer_RNG_draw_loss_bytes_exact_lambda0'] = old_history == new_history

    # Nonzero invented direct weights expose the trunk gradient; no optimizer
    # update is performed here. At zero initialization, trunk gradient is zero
    # until the new last layer learns, which is expected.
    extended, _ = initialize(True)
    with torch.no_grad(): extended.direct_end_head.weight.fill_(0.125)
    generator = torch.Generator().manual_seed(1893)
    out = extended(torch.randn((4,8,96,96),generator=generator,dtype=torch.float32))
    logits = out['direct_end_external_logit']; logits.retain_grad()
    targets = torch.tensor([0,1,-1,-1],dtype=torch.int8)
    available = torch.tensor([True,True,True,False])
    loss = candidate.balanced_direct_end_loss(logits,targets,available); loss.backward()
    checks['direct_only_gradient_trunk_and_direct_not_broad_or_owner_unknown_U_zero'] = (
        bool(torch.isfinite(logits.grad).all()) and torch.equal(logits.grad[2:],torch.zeros(2))
        and extended.support_head.weight.grad is None and extended.owner_head.weight.grad is None
        and bool((extended.encoder[0].weight.grad != 0).any())
        and bool((extended.direct_end_head.weight.grad != 0).any()))
    checks['masked_direct_analytic_classmean_gradient'] = torch.allclose(
        logits.grad[:2], (torch.sigmoid(logits.detach()[:2])-torch.tensor([0.,1.]))/2, rtol=0,atol=1e-7)
    candidate.require(all(checks.values()), 'Invented candidate forward/gradient/parity checks')
    return dict(completed=True,PASS=True,invented_only=True,checks=checks,
        paired_AdamW_steps=3,three_step_byte_history=old_history,
        toy_weights_saved=False,toy_weights_never_production_initialization=True,
        source_assets_cache_checkpoints_features_or_evaluation_reads=0,
        CUDA_initialized=torch.cuda.is_initialized())

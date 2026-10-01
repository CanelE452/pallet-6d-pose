"""Independent TRAIN-only four-edge target graph audit; no scorer or fit.

The clarified total order retains the old whole-pose winner but resolves
lower-rank exact-cost ties globally before making any pair label.
"""
from scripts.research.pallet_pose_union_selection_20261001_v1 import train as OLD
from scripts.research.pallet_pose_union_selection_20261001_v1 import common as U
from pathlib import Path
from collections import Counter
import json
import time
import math
import numpy as np

DOC = U.ROOT / '_docs/experiments/pallet_pose_selector_pairwise_20261001_v1'
EDGES = ((0, 1), (2, 3), (0, 2), (1, 3))
EDGE_NAMES = ('WD_R0', 'WD_DIVERSE', 'EXPERT_LONG', 'EXPERT_SHORT')
ALL_EDGES = EDGES + ((0, 3), (1, 2))


def global_order(errors, valid, scale, names):
    """Independent reference implementation, not the trainer implementation."""
    errors = np.asarray(errors, np.float64)
    ranks = np.full(valid.shape, -1, np.int64)
    costs = np.max(errors / np.asarray(scale, np.float64), axis=-1)
    for j in range(len(errors)):
        nodes = np.flatnonzero(valid[j]).tolist()
        ordered = []
        for cost in sorted({float(costs[j, k]) for k in nodes}):
            remaining = [k for k in nodes if costs[j, k] == cost]
            while remaining:
                front = [a for a in remaining if not any(
                    np.all(errors[j, b] <= errors[j, a]) and np.any(errors[j, b] < errors[j, a])
                    for b in remaining)]
                assert front
                ordered.extend(sorted(front, key=lambda k: OLD.tie_key(names[k])))
                remaining = [k for k in remaining if k not in front]
        assert len(ordered) == len(nodes) and set(ordered) == set(nodes)
        for rank, k in enumerate(ordered):
            ranks[j, k] = rank
    return ranks, costs


def labels_for(errors, valid, scale, names, ranks, edges):
    local = np.full((len(errors), len(edges)), -1, np.int64)
    global_labels = np.full_like(local, -1)
    active = np.zeros(local.shape, bool)
    for k, (a, b) in enumerate(edges):
        active[:, k] = valid[:, a] & valid[:, b]
        pair = OLD.targets(errors[:, [a, b]], valid[:, [a, b]], scale, [names[a], names[b]])
        use = active[:, k]
        local[use, k] = np.where(pair[use] == 0, a, b)
        global_labels[use, k] = np.where(ranks[use, a] < ranks[use, b], a, b)
    return local, global_labels, active


def graph_summary(winner, active, edges, target, valid, ids):
    adjacency = np.zeros((len(valid), valid.shape[1], valid.shape[1]), bool)
    for k, (a, b) in enumerate(edges):
        for good, bad in ((a, b), (b, a)):
            use = active[:, k] & (winner[:, k] == good)
            adjacency[use, good, bad] = True
    closure = adjacency.copy()
    for middle in range(valid.shape[1]):
        closure |= closure[:, :, middle, None] & closure[:, None, middle, :]
    cycle = np.diagonal(closure, axis1=1, axis2=2).any(1)
    present = target >= 0
    loses = np.zeros(len(valid), bool)
    reaches = np.zeros(len(valid), bool)
    for j in np.flatnonzero(present):
        loses[j] = adjacency[j, :, target[j]].any()
        required = valid[j].copy(); required[target[j]] = False
        reaches[j] = np.all(closure[j, target[j], required])
    indegree = adjacency.sum(1)
    roots = ((indegree == 0) & valid).sum(1)
    return dict(cycle_frames=int(cycle.sum()), cycle_ids=np.asarray(ids)[cycle].tolist(),
        whole_best_loses_incident_edge=int(loses.sum()), whole_best_loses_ids=np.asarray(ids)[loses].tolist(),
        whole_best_reaches_all_other_valid=int(reaches[present].sum()), present_frames=int(present.sum()),
        whole_best_not_reachable_to_all_ids=np.asarray(ids)[present & ~reaches].tolist(),
        zero_indegree_valid_nodes_histogram={str(k): v for k, v in sorted(Counter(roots.tolist()).items())})


def describe(values):
    values = np.asarray(values, np.float64)
    assert np.isfinite(values).all()
    return dict(count=int(values.size), mean=float(values.mean()) if values.size else None,
        quantiles={str(q): float(np.quantile(values, q / 100)) if values.size else None for q in (0,10,50,90,95,99,100)})


def count_bands(values):
    x = np.asarray(values)
    return dict(exact_zero=int((x == 0).sum()), positive_to_0_1=int(((x > 0) & (x <= .1)).sum()),
                above_0_1_to_1=int(((x > .1) & (x <= 1)).sum()),
                above_1_to_10=int(((x > 1) & (x <= 10)).sum()), above10=int((x > 10).sum()))


def invented_counterexample():
    names = OLD.candidate_names('UNION', 1)
    errors = np.array([[[.9,1.], [1.,.5], [.8,1.], [.7,1.]]])
    valid = np.ones((1,4), bool)
    ranks, costs = global_order(errors, valid, [1.,1.], names)
    target = OLD.targets(errors, valid, [1.,1.], names)
    local, new, active = labels_for(errors, valid, [1.,1.], names, ranks, EDGES)
    oldgraph = graph_summary(local, active, EDGES, target, valid, ['invented'])
    newgraph = graph_summary(new, active, EDGES, target, valid, ['invented'])
    assert costs.tolist() == [[1.,1.,1.,1.]]
    assert local.tolist() == [[0,3,2,1]] and new.tolist() == [[1,3,2,1]]
    assert target.tolist() == [1] and np.argsort(ranks[0]).tolist() == [1,3,2,0]
    assert oldgraph['cycle_frames'] == oldgraph['whole_best_loses_incident_edge'] == 1
    assert newgraph['cycle_frames'] == newgraph['whole_best_loses_incident_edge'] == 0
    return dict(source='Invented analytic fixture only; not a sampled image.', errors=errors[0].tolist(),
        scale=[1.,1.], cost=costs[0].tolist(), names=names, edges=[list(e) for e in EDGES],
        old_pair_winners=local[0].tolist(), old_cycle=[0,1,3,2,0],
        global_order=np.argsort(ranks[0]).tolist(), global_pair_winners=new[0].tolist(),
        old_whole_best=int(target[0]), changed_edge_indices=np.flatnonzero(local[0] != new[0]).tolist())


def audit_seed(seed, data):
    names = OLD.candidate_names('UNION', seed)
    parents = ['R0', f'DIVERSE251_s{seed}']
    errors = np.concatenate([data['errors'][p] for p in parents], axis=1)
    valid = np.concatenate([data['valid'][p] for p in parents], axis=1)
    ids, scale = data['ids'], data['scale']
    ranks, costs = global_order(errors, valid, scale, names)
    target = OLD.targets(errors, valid, scale, names)
    rank_target = np.where(valid.any(1), np.argmax(ranks == 0, axis=1), -1)
    np.testing.assert_array_equal(rank_target, target)
    local, new, active = labels_for(errors, valid, scale, names, ranks, ALL_EDGES)
    oldgraph = graph_summary(local[:, :4], active[:, :4], EDGES, target, valid, ids)
    graph = graph_summary(new[:, :4], active[:, :4], EDGES, target, valid, ids)
    oldfull = graph_summary(local, active, ALL_EDGES, target, valid, ids)
    full = graph_summary(new, active, ALL_EDGES, target, valid, ids)
    assert graph['cycle_frames'] == graph['whole_best_loses_incident_edge'] == 0
    assert full['cycle_frames'] == full['whole_best_loses_incident_edge'] == 0
    assert full['whole_best_reaches_all_other_valid'] == full['present_frames']
    stats = {}
    differences = []
    for k, ((a, b), label) in enumerate(zip(EDGES, EDGE_NAMES)):
        use = active[:, k]
        gap = np.abs(costs[use,a] - costs[use,b])
        tied = use & (costs[:,a] == costs[:,b])
        left_dom = np.zeros(len(ids), bool); right_dom = left_dom.copy()
        left_dom[use] = np.all(errors[use,a] <= errors[use,b], axis=1) & np.any(errors[use,a] < errors[use,b], axis=1)
        right_dom[use] = np.all(errors[use,b] <= errors[use,a], axis=1) & np.any(errors[use,b] < errors[use,a], axis=1)
        winner_dominated = use & (((new[:,k] == a) & right_dom) | ((new[:,k] == b) & left_dom))
        assert not winner_dominated.any()
        changed = use & (local[:,k] != new[:,k])
        assert not np.any(changed & ~tied)
        stats[label] = dict(pair=[a,b], names=[names[a],names[b]], active=int(use.sum()),
            inactive_zero_valid=int((~valid[:,a] & ~valid[:,b]).sum()),
            inactive_one_valid=int((valid[:,a] ^ valid[:,b]).sum()),
            global_left_wins=int((use & (new[:,k]==a)).sum()), global_right_wins=int((use & (new[:,k]==b)).sum()),
            exact_cost_ties=int(tied.sum()), exact_cost_ties_with_dominance=int((tied & (left_dom|right_dom)).sum()),
            identical_T_R=int((use & np.all(errors[:,a] == errors[:,b],axis=1)).sum()),
            tradeoff_or_equal=int((use & ~left_dom & ~right_dom).sum()),
            whole_best_incident=int((use & ((target==a)|(target==b))).sum()),
            whole_best_absent=int((use & (target!=a) & (target!=b)).sum()),
            old_pair_vs_global_label_changes=int(changed.sum()),
            cost_gap=describe(gap), cost_gap_bands=count_bands(gap),
            absolute_T_gap_cm=describe(np.abs(errors[use,a,0]-errors[use,b,0])),
            absolute_R_gap_deg=describe(np.abs(errors[use,a,1]-errors[use,b,1])))
        for j in np.flatnonzero(changed):
            differences.append(dict(id=str(ids[j]), edge=label, errors=errors[j].tolist(),
                costs=costs[j].tolist(), old_winner=int(local[j,k]), global_winner=int(new[j,k]),
                global_order=np.argsort(np.where(ranks[j]>=0,ranks[j],99)).tolist(), whole_best=int(target[j])))
    groups = {}
    for group, edge_indices in [('WD',[0,1]),('EXPERT',[2,3])]:
        gap = np.concatenate([np.abs(costs[active[:,k],EDGES[k][0]]-costs[active[:,k],EDGES[k][1]]) for k in edge_indices])
        counts = active[:,edge_indices].sum(1)
        groups[group] = dict(active_edges=int(counts.sum()), possible_edges=len(ids)*2,
            row_active_edge_histogram={str(k):v for k,v in sorted(Counter(counts.tolist()).items())},
            loss_formula='0.5 * (loss_edge0+loss_edge1)/2, each inactive edge0; then mean over all2598 rows.',
            fixed_coefficient_per_active_edge=.25, effective_coefficient_sum_all_rows=float(.25*counts.sum()/len(ids)),
            zero_score_logistic_contribution_all_rows=float(.25*counts.sum()/len(ids)*math.log(2)),
            cost_gap=describe(gap), cost_gap_bands=count_bands(gap))
    strict_cost_violations=0
    for a,b in ALL_EDGES:
        use=valid[:,a]&valid[:,b]&(costs[:,a]!=costs[:,b])
        strict_cost_violations+=int(np.sum((ranks[use,a]<ranks[use,b]) != (costs[use,a]<costs[use,b])))
    assert strict_cost_violations==0
    return dict(seed=seed,frames=len(ids),candidate_names=names,
        valid_count_histogram={str(k):v for k,v in sorted(Counter(valid.sum(1).tolist()).items())},
        invalid_ids=np.asarray(ids)[~valid.all(1)].tolist(),
        whole_best_distribution=dict(Counter(names[t] if t>=0 else 'NO_VALID' for t in target)),
        global_top_equals_original_whole_best=True,global_top_parity_rows=len(ids),
        old_pair_local_four_edge_graph=oldgraph,global_four_edge_graph=graph,
        old_pair_local_full_six_edge_diagnostic=oldfull,global_full_six_edge_diagnostic=full,
        full_six_edge_exact_cost_ties={f'{a}:{b}':int(np.sum(active[:,k]&(costs[:,a]==costs[:,b])))for k,(a,b)in enumerate(ALL_EDGES)},
        full_six_edge_old_pair_vs_global_label_changes=int(np.sum(active&(local!=new))),
        strict_cost_order_violations=strict_cost_violations,edge_statistics=stats,loss_groups=groups,
        old_pair_vs_global_label_changes=sum(v['old_pair_vs_global_label_changes']for v in stats.values()),
        label_change_examples=differences[:30],label_change_ids=sorted({v['id']for v in differences}),
        hashes=dict(global_rank=OLD.array_sha(ranks),original_whole_best=OLD.array_sha(target),
            edge_winners=OLD.array_sha(new[:,:4]),edge_active=OLD.array_sha(active[:,:4]),
            original_pair_winners=OLD.array_sha(local[:,:4])),
        zero_weight_four_way_CE_all_rows=float(np.log(np.maximum(valid.sum(1),1)).mean()),
        zero_weight_factorized_pairwise_all_rows=sum(g['zero_score_logistic_contribution_all_rows']for g in groups.values()))


def main():
    start=time.monotonic()
    OLD.install_training_guard()
    fixture=invented_counterexample()
    data=OLD.load_training_inputs()
    assert len(data['ids'])==2598
    seeds={str(s):audit_seed(s,data)for s in (1,2,3)}
    r0rank,_=global_order(data['errors']['R0'],data['valid']['R0'],data['scale'],OLD.candidate_names('R0_ONLY',1))
    r0target=OLD.targets(data['errors']['R0'],data['valid']['R0'],data['scale'],OLD.candidate_names('R0_ONLY',1))
    np.testing.assert_array_equal(np.where(data['valid']['R0'].any(1),np.argmax(r0rank==0,axis=1),-1),r0target)
    result=dict(complete=True,PASS=True,status='PASS_GLOBAL_ORDER_TRAIN_CONSISTENCY',created_at=U.now(),
        scope='Original2598 declared-C2 proper-rigid source TRAIN only; no VAL label values or real GT.',
        protocol_clarification_before_fit='Cost ascending; exact-cost group Pareto front layers; within each layer OLD.tie_key. Choose pair winner by earlier global rank.',
        whole_pose_cost='max(T_cm/sT_cm,R_deg/sR_deg)',scale=data['scale'].tolist(),
        ties_are_exact=True,pairwise_does_not_create_separate_T_R_targets=True,
        edges=[dict(name=n,pair=list(e),group='WD'if k<2 else'EXPERT')for k,(e,n)in enumerate(zip(EDGES,EDGE_NAMES))],
        excluded_diagonal_edges=list(map(list,ALL_EDGES[4:])),excluded_edges_are_diagnostic_only=True,
        invalid_policy='Both candidates must be valid. Otherwise edge loss0; denominator2598 and fixedgroupweights unchanged.',
        invented_counterexample=fixture,seeds=seeds,R0_ONLY_top_parity_rows=2598,
        R0_ONLY_loss='Single active binaryedge coefficient1; same old2candidate CE. This audit checks labels, not frozen-scorer objective/gradient equivalence.',
        training_input_ids_sha=OLD.array_sha(data['ids']),normalization_sha=data['normalization_sha'],
        inputs=dict(parent_train_protocol=data['binding'],**data['protocol']['inputs']),
        codes=[U.bind(Path(__file__)),U.bind(Path(OLD.__file__)),U.bind(Path(U.__file__))],
        fits=0,optimizer_calls=0,scorer_forwards=0,image_forwards=0,new_PnP_solves=0,
        VAL_quality_reads=0,real_reference_reads=0,
        caution='Per-frame graph consistency is necessary label hygiene, not proof one shared linear94 can satisfy allframes, nor proof of sourceVAL or realT/R gains.',
        sparse_graph_limit='Four edges can leave two zero-indegree candidates:4/5/2 frames for refinerseeds1/2/3. No cyclic contradiction, but pair supervision alone does not identify the unique wholebest there. The six-edge check is diagnostic only; no diagonal is added.',
        former_pair_local_general_acyclicity_claim=False,goal_complete=False,
        wall_seconds=time.monotonic()-start)
    DOC.mkdir(parents=True,exist_ok=True)
    path=DOC/'TARGET_CONTRACT_AUDIT.json'
    with path.open('x')as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=result['status'],scale=result['scale'],seeds={s:{k:v[k]for k in ('valid_count_histogram','global_top_parity_rows','old_pair_vs_global_label_changes','old_pair_local_four_edge_graph','global_four_edge_graph','loss_groups')}for s,v in seeds.items()}),ensure_ascii=False,indent=2),flush=True)
    print('TARGET_AUDIT_COMPLETE',U.bind(path),flush=True)


if __name__=='__main__':main()

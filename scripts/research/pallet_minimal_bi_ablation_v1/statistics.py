"""Saved-row summaries and paired whole-recording bootstrap; no inference IO."""
from __future__ import annotations

from collections import Counter
import hashlib
import numpy as np

ARMS = ('R0', 'R1', 'N0', 'N1')
CONTRASTS = (('R1', 'R0'), ('N1', 'N0'), ('R1', 'N1'), ('R1', 'N0'))
METRICS = ('translation_cm', 'rotation_deg', 'ADDsym_cm')
RESAMPLES = 10000
SEED = 20260917
P_VALUE_REASON = ('No verified historical cluster-aware paired success-rate hypothesis-test '
                  'implementation was found in the reviewed research/paper sources. '
                  'No p value is inferred from bootstrap intervals; Holm p values remain NA.')


def distribution(values):
    """All finite observed errors, with the unavailable denominator explicit."""
    values = list(values)
    available = np.asarray([float(x) for x in values if x is not None], dtype=float)
    if available.ndim != 1 or not np.isfinite(available).all():
        raise ValueError('Nonfinite error is a schema failure, not an omitted observation')
    n = len(available)
    return dict(n=n, denominator=len(values), missing=len(values)-n,
                mean=float(available.mean()) if n else None,
                sample_std=float(available.std(ddof=1)) if n > 1 else None,
                max=float(available.max()) if n else None, ddof=1,
                population='available-only; missing poses are retained separately',
                finite_tail_trimming=False)


class PairedClusterBootstrap:
    """New draws for this population, shared across all arms, metrics and masks.

    A sampled recording contributes all its original frames. Each bootstrap
    mean divides the pooled total by the pooled eligible frame count; recordings
    are not given equal weight in the reported frame-level estimand.
    """
    def __init__(self, rows):
        self.rows = list(rows)
        ids = [r['id'] for r in self.rows]
        clusters = [r['cluster_id'] for r in self.rows]
        if not ids or len(set(ids)) != len(ids) or any(not isinstance(c, str) or not c for c in clusters):
            raise ValueError('Unique paired IDs and explicit recording cluster IDs required')
        self.units, self.inverse = np.unique(clusters, return_inverse=True)
        n = len(self.units)
        self.draws = np.random.default_rng(SEED).multinomial(
            n, np.full(n, 1/n), size=RESAMPLES).astype('uint16')
        self.weights = self.draws.astype(float)

    def metadata(self):
        return dict(seed=SEED, resamples=RESAMPLES, units=len(self.units),
                    clusters=self.units.tolist(),
                    cluster_sizes=dict(Counter(r['cluster_id'] for r in self.rows)),
                    session_sizes=dict(Counter(r['session'] for r in self.rows)),
                    draw_sha256=hashlib.sha256(self.draws.tobytes()).hexdigest(),
                    method='paired whole-recording multinomial percentile bootstrap',
                    interval='np.quantile(valid resample means, [.025,.975], method=linear)',
                    estimand='original frame-weighted mean/rate difference; new minus reference',
                    same_draws_for_every_arm_metric_and_availability_mask=True,
                    historical_draw_array_reused=False,
                    limitation='Few source-recording clusters and repeatedly developed data; '
                    'this is not a confidence interval adjusted for all past model selection.')

    def contrast(self, values):
        if len(values) != len(self.rows):
            raise ValueError('Pairing/availability mask must preserve all population rows')
        mask = np.asarray([x is not None for x in values], dtype=bool)
        x = np.asarray([0. if x is None else float(x) for x in values])
        if not np.isfinite(x).all():
            raise ValueError('Finite paired differences required')
        counts = np.bincount(self.inverse, weights=mask, minlength=len(self.units))
        totals = np.bincount(self.inverse, weights=x, minlength=len(self.units))
        denominators = self.weights @ counts
        keep = denominators > 0
        replicates = (self.weights @ totals)[keep]/denominators[keep]
        eligible_units = int(np.count_nonzero(counts))
        point = float(x[mask].mean()) if mask.any() else None
        return dict(mean_difference=point,
                    CI95=np.quantile(replicates, [.025, .975]).tolist()
                    if eligible_units > 1 and len(replicates) else None,
                    CI_unavailable_reason=None if eligible_units > 1 else
                    'Fewer than two eligible source-recording clusters',
                    common_available_n=int(mask.sum()), full_denominator=len(values),
                    excluded_unavailable_n=int((~mask).sum()), eligible_clusters=eligible_units,
                    valid_resamples=int(keep.sum()), empty_resamples=int((~keep).sum()),
                    improved=int((x[mask] < 0).sum()), worsened=int((x[mask] > 0).sum()),
                    tied=int((x[mask] == 0).sum()))


def transitions(rows, new, before):
    groups = {k: [] for k in ('failure_to_success', 'success_to_failure',
                              'both_success', 'both_failure', 'reference_unresolved')}
    for r in rows:
        a, b = (r['metrics'][arm]['success_le5'] for arm in (new, before))
        category = ('reference_unresolved' if a is None or b is None else
                    'failure_to_success' if a and not b else
                    'success_to_failure' if b and not a else
                    'both_success' if a else 'both_failure')
        groups[category].append(r['id'])
    return dict(counts={k: len(v) for k, v in groups.items()}, ids=groups,
                full_denominator=len(rows), all_rows_kept=True)


def summarize_population(rows):
    """Main 4-arm table and the four prespecified paired contrasts."""
    rows = list(rows)
    bootstrap = PairedClusterBootstrap(rows)
    summaries = {}
    common_masks = {metric: [all(r['metrics'][a][metric] is not None for a in ARMS) for r in rows]
                    for metric in METRICS}
    for arm in ARMS:
        values = [r['metrics'][arm] for r in rows]
        statuses = [r['prediction_status'][arm] for r in rows]
        accepted = [s['accepted'] for s in statuses]
        fallbacks = [s['fallback'] for s in statuses]
        unresolved = sum(v['success_le5'] is None for v in values)
        summaries[arm] = dict(denominator=len(rows),
            successes_le5=sum(v['success_le5'] is True for v in values),
            success_rate_le5=sum(v['success_le5'] is True for v in values)/len(rows),
            reference_unresolved_n=unresolved,
            success_rate_is_confirmed_lower_bound=bool(unresolved),
            successes_strict_lt5=sum(v['success_strict_lt5'] is True for v in values),
            translation_exactly_5_n=sum(v['translation_cm'] == 5. for v in values),
            rotation_exactly_5_n=sum(v['rotation_deg'] == 5. for v in values),
            strict_vs_inclusive_changed_ids=[r['id'] for r in rows
                if r['metrics'][arm]['success_le5'] is True and
                r['metrics'][arm]['success_strict_lt5'] is False],
            pose_available_n=sum(v['pose_available'] for v in values),
            no_pose_n=sum(not v['pose_available'] for v in values),
            accepted_n=sum(x is True for x in accepted),
            accepted_available_n=sum(x is not None for x in accepted),
            accepted_status='NA' if all(x is None for x in accepted) else 'recorded constraint acceptance, not accuracy',
            fallback_n=sum(x is True for x in fallbacks),
            fallback_available_n=sum(x is not None for x in fallbacks),
            fallback_status='NA' if all(x is None for x in fallbacks) else 'recorded',
            errors={m: distribution(v[m] for v in values) for m in METRICS},
            common_all_four_available_errors={m: distribution(
                r['metrics'][arm][m] for r, keep in zip(rows, common_masks[m]) if keep) for m in METRICS},
            status_counts=dict(Counter(str(s['status']) for s in statuses)))
    comparisons = {}
    for new, before in CONTRASTS:
        binary = [None if any(r['metrics'][a]['success_le5'] is None for a in (new, before))
                  else float(r['metrics'][new]['success_le5'])-float(r['metrics'][before]['success_le5']) for r in rows]
        success = bootstrap.contrast(binary)
        success['gains'] = success.pop('worsened')
        success['losses'] = success.pop('improved')
        success['unit'] = 'rate difference; multiply by100 for percentage points'
        if any(x is None for x in binary):
            success.update(conditional_known_label_mean_difference=success['mean_difference'],
                           mean_difference=None, CI95=None,
                           CI_unavailable_reason='Unresolved reference accuracy labels; no full-denominator contrast imputed')
        comparisons[new+'-'+before] = dict(new=new, reference=before,
            transition=transitions(rows, new, before), success_rate=success,
            continuous={m: bootstrap.contrast([None if r['metrics'][new][m] is None or r['metrics'][before][m] is None
                else r['metrics'][new][m]-r['metrics'][before][m] for r in rows]) for m in METRICS},
            p_raw=None, p_Holm=None, p_value_reason=P_VALUE_REASON,
            not_equivalence_or_noninferiority=True)
    return dict(rows=len(rows), arms=summaries, comparisons=comparisons,
                bootstrap=bootstrap.metadata(),
                common_all_four_available_n={m: sum(mask) for m, mask in common_masks.items()},
                Holm_family=dict(hypotheses=[a+'-'+b for a,b in CONTRASTS],
                    population='PRIMARY192 only; subgroup and CAL tables descriptive',
                    raw_p_values_available=False, adjusted_p_values_available=False, reason=P_VALUE_REASON))

"""Root-fixed three-rule SYN DEV selector. No I/O or current/held selection."""
import torch


def select_on_syn_dev(spatial, output, q0, side, reference_xy, gt_fixed_loc_eligible, rules):
    if gt_fixed_loc_eligible.dtype != torch.bool:
        raise ValueError('Fixed GT eligibility bool, never predicted VIS')
    total=len(q0)
    if not total:
        raise ValueError('SYN DEV denominator must be nonempty')
    table=[]; chosen=None
    for name in ('OPEN','MODERATE','STRICT'):
        accepted,diagnostic=spatial.acceptance(output,rules[name])
        points=spatial.image_points(output,q0,side,accepted)
        error=torch.linalg.vector_norm(points-reference_xy,dim=1)
        correct=accepted & gt_fixed_loc_eligible & torch.isfinite(error) & (error<=3.)
        count=int(accepted.sum()); numerator=int(correct.sum())
        known=accepted & gt_fixed_loc_eligible
        row={'rule':name,'all_queries':total,'accepted':count,'verified_correct':numerator,
             'unverified_accepted':int((accepted & ~gt_fixed_loc_eligible).sum()),
             'known_scorable_accepted':int(known.sum()),
             'conservative_verified_precision':numerator/count if count else None,
             'known_scorable_precision':numerator/int(known.sum()) if bool(known.any()) else None,
             'all_query_coverage':count/total,
             'eligible_query_coverage':int(known.sum())/int(gt_fixed_loc_eligible.sum()) if bool(gt_fixed_loc_eligible.any()) else None,
             'qualifies':bool(count and numerator/count>=.90)}
        table.append(row)
        if row['qualifies'] and (chosen is None or count>chosen['accepted']):
            chosen=row
    return {'selected_rule':chosen['rule'] if chosen else 'ALL_ABSTAIN',
            'table':table,'selection_partition':'SYN DEV only',
            'precision_denominator':'ALL accepted, including unverified',
            'current_or_held_rows_used':0}

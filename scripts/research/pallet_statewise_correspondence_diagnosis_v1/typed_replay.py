"""Replay the original hierarchical e consumer; log endpoints without new math.

No file, reference, RGB, detector, or metric access occurs here.  MULTI uses
the old G10 family choice, then the d same-family solution.  SINGLE freezes
that original choice and substitutes E0 for the backend's start list.  The
original coarse prior, scalar/analytic backend, objective, guards, limits,
fallback readout, and all feasible sample lists are retained.

AST clones add endpoint logging only. SINGLE additionally replaces the
start list and omits the separate BI seed fit, so exactly one joint SLSQP
call is made when the old backend admits an optimizer. Unsupported-outline
G10 delegates exactly as before and performs no new optimization.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import importlib
import inspect
import textwrap
from types import SimpleNamespace

import numpy as np

PKG = 'scripts.research.pallet_visibility_curriculum_20261013_v1.'
START_NAME = 'SINGLE_ORIGINAL_E_FINAL_POSE_ONLY'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _module(name):
    return importlib.import_module(PKG + name)


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def original_stage_inputs(packet, family_name):
    """Undo the *later* V15 archive's added d/G10 winner starts, exactly.

    V15 order is [d winner, original WD search, G10 winner, (original P1)].
    Only WD and original P1 were starts of original G10/d. The swapped seed
    has the exact original multiplication order and original start name.
    """
    bank = packet['original_e_stage1_family_seed_bank']
    require(set(bank) == {'ORIGINAL_WD', 'SWAPPED_WD'}, 'Original two-family bank required')
    original, family = bank['ORIGINAL_WD'], bank[family_name]
    require(len(original['initial_solutions']) == 4, 'V15 original archive slot contract')
    require(len(bank['SWAPPED_WD']['initial_solutions']) == 3, 'V15 swapped archive slot contract')
    p1 = original['initial_solutions'][3]
    solutions, starts = [deepcopy(family['initial_solutions'][1])], []
    if family_name == 'ORIGINAL_WD':
        solutions.append(deepcopy(p1))
    else:
        R, t, _ = choose_readout(p1, original['family']['coarse_pose'])
        starts.append(dict(R=(np.asarray(R) @ _module('statewise_two_wd_family_v4_9').RY90.T).tolist(),
                           t=deepcopy(t), name='PRIOR_PHYSICAL_POSE_TRANSFORMED_SEED_ONLY'))
    return solutions, starts


def choose_readout(solution, coarse_pose):
    """Exact statewise_joint_api_v4.finish_observations numeric fallback rule."""
    prior = bool(solution.get('prior_audit', {}).get('enabled', False))
    seed = solution.get('BI_prior_seed') or solution.get('corrected_BI_seed') or {}
    if solution['accepted']:
        return solution['R'], solution['t'], ('SOFT_BI_COARSE_PRIOR_ALL_OT_JOINT_SE3' if prior
                                               else 'SOFT_BI_ALL_OT_JOINT_SE3_NO_PRIOR')
    if seed.get('visible_support_guards_pass') and not seed.get('contains_underconstrained_pose_family'):
        return seed['R'], seed['t'], ('EXPLICIT_SOFT_BI_PRIOR_SEED_FALLBACK_S_ONLY' if prior
                                        else 'EXPLICIT_SOFT_BI_SEED_FALLBACK_S_ONLY')
    return coarse_pose['R'], coarse_pose['t'], 'EXPLICIT_REGISTERED_COARSE_FALLBACK_S_ONLY'


def actual_pose(solution, family):
    R, t, source = choose_readout(solution, family['coarse_pose'])
    if solution['accepted']:
        # Exact final metadata assignment in evaluate_global_outline_pose_v4_10._run,
        # also used by d's caller-bound typed backend.
        source = 'CONTINUOUS_GLOBAL_OUTLINE_BI_PARTIAL_OT_U_JOINT_SE3'
    R, t = np.asarray(R), np.asarray(t)
    xyz, Q = np.asarray(family['native_xyz']), np.asarray(family['physical_registration_Q'])
    require(np.max(np.abs(xyz.mean(0))) < 1e-12, 'Original centered native model required')
    return dict(available=True, R_cf=R.tolist(), centroid=t.tolist(),
                R_physical=(R @ Q).tolist(), physical_registration_Q=Q.tolist(),
                cf_extents=deepcopy(family['cf_extents']), source=source,
                joint_feasible=bool(solution['accepted']), fallback=not solution['accepted'],
                fallback_pose_does_not_solve_OT=not solution['accepted'])


class _Clone(ast.NodeTransformer):
    """Only append logging statements; SINGLE's additional edits are audited."""
    def __init__(self, *, single=False, partial=False):
        self.single, self.partial = single, partial
        self.function_stack = []
        self.edits = []

    @staticmethod
    def statements(code):
        return ast.parse(code).body

    def visit_FunctionDef(self, node):
        self.function_stack.append(node.name)
        node = self.generic_visit(node)
        self.function_stack.pop()
        return node

    def visit_Expr(self, node):
        node = self.generic_visit(node)
        # R/t already exist here. Logging invokes no objective/constraint.
        if (self.function_stack[-1:] == ['consider'] and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == 'records.append'):
            self.edits.append('log_consider_Rt')
            return [node] + self.statements('_endpoint_log(R, t, record, "outline_consider")')
        return node

    def visit_Return(self, node):
        node = self.generic_visit(node)
        if self.partial and self.function_stack[-1:] == ['fit'] and isinstance(node.value, ast.Tuple):
            self.edits.append('log_partial_fit_Rt')
            # Evaluate the original return expression exactly once.
            assignment = ast.Assign(targets=[ast.Name(id='_logged_return', ctx=ast.Store())], value=node.value)
            return [assignment] + self.statements(
                '_endpoint_log(_logged_return[1]["R"], _logged_return[1]["t"], '
                '_logged_return[1], "partial_joint_fit" if use_boundary else "partial_BI_seed_fit")\n'
                'return _logged_return')
        return node

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if not self.single:
            return node
        targets = [ast.unparse(t) for t in node.targets]
        if not self.partial and targets == ['choices']:
            self.edits.append('single_outline_start')
            return self.statements('unique = [deepcopy(_single_start)]') + [node]
        if self.partial and targets == ['(starts, search)']:
            require(isinstance(node.value, ast.Call) and '_initial_poses' in ast.unparse(node.value.func),
                    'Exact original partial initial-pose assignment required')
            self.edits.append('single_partial_start_no_new_PnP')
            return self.statements('starts = [deepcopy(_single_start)]\n'
                                   'search = {"single_E0_start_only": True, "new_P3P_starts": 0}')
        return node

    def visit_For(self, node):
        node = self.generic_visit(node)
        if self.single and self.partial and ast.unparse(node.target) == 'start' and ast.unparse(node.iter) == 'starts':
            # Only the pre-joint seed loop has fit(... use_boundary=False).
            if 'use_boundary=False' in ast.unparse(node):
                self.edits.append('single_skip_BI_seed_fit')
                node.iter = ast.List(elts=[], ctx=ast.Load())
        return node


def _clone(function, globals_updates, logger, *, single_start=None, partial=False):
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    transformer = _Clone(single=single_start is not None, partial=partial)
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    env = dict(function.__globals__)
    env.update(globals_updates)
    env.update(_endpoint_log=logger, _single_start=deepcopy(single_start))
    exec(compile(tree, function.__code__.co_filename + '::statewise_audited_clone', 'exec'), env)
    result = env[function.__name__]
    result.__replay_clone_audit__ = dict(source_module=function.__module__, source_function=function.__name__,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(), edits=transformer.edits,
        original_arithmetic_and_statement_order_preserved_except_declared_single_starts=True)
    return result


def _facade(module, **changes):
    values = dict(vars(module)); values.update(changes)
    return SimpleNamespace(**values)


def _backend(name, logger, *, single_start=None):
    """Clone without monkeypatching shared imported modules."""
    module = _module(name)
    updates = {}
    if name == 'statewise_partial_BI_outline_solver_v4_13':
        old = _backend('statewise_global_outline_solver_v4_11', logger, single_start=single_start)
        partial_module = module.P
        partial_fn = _clone(partial_module.solve_partial_BI_u_region_joint_ot_pnp, {}, logger,
                            single_start=single_start, partial=True)
        updates.update(OLD=old, P=_facade(partial_module, solve_partial_BI_u_region_joint_ot_pnp=partial_fn))
    fn = _clone(module.refine_global_outline_joint_pose, updates, logger, single_start=single_start)
    return _facade(module, refine_global_outline_joint_pose=fn)


def _ledger_sink(ledger, stage, family):
    def log(R, t, record, kind):
        ledger.append(dict(stage=stage, family=family, kind=kind,
                           R=_plain(deepcopy(R)), t=_plain(deepcopy(t)),
                           original_endpoint_record=_plain(deepcopy(record))))
    return log


def _family_run(packet, stage, family_name, ledger, *, single=False):
    bank = packet['original_e_stage1_family_seed_bank']
    family = bank[family_name]['family']
    spec = packet['solver_input']
    solutions, initializations = original_stage_inputs(packet, family_name)
    typed = stage == 'G13'
    backend_name = ('statewise_partial_BI_outline_solver_v4_13' if typed
                    else 'statewise_global_outline_solver_v4_10')
    start = None
    if single:
        pose = packet['e0_pose']
        require(packet['registered_family']['name'] == family_name, 'SINGLE fixes original e family')
        require(np.allclose(pose['physical_registration_Q'], family['physical_registration_Q'], atol=1e-12, rtol=0),
                'E0 start native registration differs')
        start = dict(R=deepcopy(pose['R_cf']), t=deepcopy(pose['centroid']), name=START_NAME)
    backend = _backend(backend_name, _ledger_sink(ledger, stage, family_name), single_start=start)
    boundary = (spec['boundary_observations'] if typed else packet['original_e_legacy_boundary_observations'])
    answer = backend.refine_global_outline_joint_pose(family['native_xyz'], spec['K'],
        deepcopy(spec['states']), deepcopy(spec['visible']), deepcopy(spec['domains']), spec['image_hw'],
        deepcopy(family['coarse_pose']), outline_observation=deepcopy(spec['outline_observation']),
        initial_solutions=solutions, initializations=initializations, u_regions=deepcopy(spec['u_regions']),
        boundary_observations=deepcopy(boundary), config=deepcopy(spec['config']))
    return answer, deepcopy(backend.refine_global_outline_joint_pose.__replay_clone_audit__)


def _two_family_stage(packet, stage, ledger):
    family_module = _module('statewise_two_wd_family_v4_9')
    spec = packet['solver_input']; original = packet['original_e_stage1_family_seed_bank']['ORIGINAL_WD']['family']
    audits = {}
    def solve(xyz, K, states, visible, domains, image_hw, coarse_pose, **kwargs):
        matches = [name for name, value in packet['original_e_stage1_family_seed_bank'].items()
                   if np.allclose(value['family']['native_xyz'], xyz, atol=1e-12, rtol=0)]
        require(len(matches) == 1, 'Exact original family-native model binding')
        answer, audit = _family_run(packet, stage, matches[0], ledger)
        audits[matches[0]] = audit
        return answer
    result = family_module.solve_two_wd_families(original['native_xyz'], spec['K'], spec['states'],
        spec['visible'], spec['domains'], spec['image_hw'], original['coarse_pose'],
        config=spec['config'], boundary_observations=(spec['boundary_observations'] if stage == 'G13'
            else packet['original_e_legacy_boundary_observations']), u_regions=spec['u_regions'], solver_callable=solve)
    return result, audits


def _has_new_typed(packet):
    return any(b.get('explicit_complete_BI_likelihood_revocation') is True
               for b in packet['solver_input']['boundary_observations'])


def run_multi(packet):
    """Original e numerical replay plus all visited optimizer endpoint poses."""
    require(packet['original_T_union']['branch_count'] == 1, 'Frozen population has one original T-union branch')
    ledger = []
    g10, ga = _two_family_stage(packet, 'G10', ledger)
    typed = _has_new_typed(packet)
    if typed:
        g13, da = _two_family_stage(packet, 'G13', ledger)
    else:
        g13, da = deepcopy(g10), {}
    name = g10['selected_family']
    family = next(f for f in g13['families'] if f['name'] == name)
    solution = deepcopy(family['solution'])
    initial_ambiguous = bool(g10['family_ambiguity'].get('detected'))
    if initial_ambiguous:
        solution.setdefault('ambiguity', {})['detected'] = True
        solution['ambiguity']['initial_family_ambiguity_preserved'] = True
        solution['OT_solved'] = False
        solution['OT_unique_solution_claimed'] = False
    return _plain(dict(id=packet['id'], method='TYPED_MULTI', accepted=bool(solution['accepted']),
        fallback=not solution['accepted'], solution=solution, actual_pose=actual_pose(solution, family),
        selected_family=name, stage_results=dict(G10=g10, G13=g13), generated_candidates=ledger,
        audit=dict(exact_original_hierarchical_family_choice=True, no_typed_exact_G10_cache_reuse=not typed,
            original_e_selected_family=packet['registered_family']['name'],
            selected_family_parity=name == packet['registered_family']['name'], clone_audits=dict(G10=ga, G13=da),
            candidate_scope='All original replay outline consider and typed partial-fit endpoints; optimizer internal iterates and saved historical WD/P1 generation are not regenerated',
            labels_or_metrics_read=False)))


def run_single(packet):
    """One E0-start joint fit in the original e selected registered family."""
    require(packet['original_T_union']['branch_count'] == 1, 'Strict single requires one original union branch')
    ledger = []; name = packet['registered_family']['name']
    stage = 'G13' if _has_new_typed(packet) else 'G10'
    solution, clone = _family_run(packet, stage, name, ledger, single=True)
    family = packet['original_e_stage1_family_seed_bank'][name]['family']
    initial_ambiguous = bool((packet.get('original_e_hierarchical_correspondence_audit') or {})
                            .get('initial_family_ambiguity', {}).get('detected'))
    if initial_ambiguous:
        solution.setdefault('ambiguity', {})['detected'] = True
        solution['ambiguity']['initial_family_ambiguity_preserved'] = True
        solution['OT_solved'] = False
        solution['OT_unique_solution_claimed'] = False
    return _plain(dict(id=packet['id'], method='TYPED_SINGLE', accepted=bool(solution['accepted']),
        fallback=not solution['accepted'], solution=solution, actual_pose=actual_pose(solution, family),
        selected_family=name, generated_candidates=ledger,
        audit=dict(E0_start_only=True, original_coarse_prior_unchanged=True, family_fixed=True,
            original_union_branch_count=packet['original_T_union']['branch_count'],
            clone_audit=clone, optimizer_endpoint_count=sum(x['kind'] != 'outline_consider' or
                x['original_endpoint_record'].get('optimizer_success') is not None for x in ledger),
            delegated_without_new_single_fit=not ledger,
            start_policy_comparison_NOT_ISOLATED=True,
            reason='MULTI chooses family using G10 full/legacy factors before d refinement; SINGLE fixes the already chosen e family. Separate BI seed fitting is omitted to enforce one joint optimization.',
            global_uniqueness_proved=False, labels_or_metrics_read=False)))

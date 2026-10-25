"""V227 root entry 이후 한 번 완결하는 고정 2-arm SOURCE 비교.

실제 실행: python -B train_compare.py --entry ROOT_ENTRY.json
가상 검사: python -B train_compare.py --self-test  (optimizer step/실제 자료 접근 없음)
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from head import ContextReadProbe, balanced_bce

HERE = Path(__file__).absolute().parent
PROTOCOL = Path("/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation/V227_ROOT_CONTEXT_PROBE/PROTOCOL_BEFORE_CONTEXT_FEATURES_OR_PROBE_FIT.json")
PROTOCOL_SHA = "8f92b3da3d6e66df4e1bac2c7121ef975d2eb5f6b2c19ebf9215977881399462"


def checked(path):
    path = Path(os.path.abspath(os.fspath(path)))
    assert "v112" not in str(path).casefold(), "금지 경로"
    for item in reversed((path, *path.parents)):
        assert not item.is_symlink(), f"Symlink 거부: {item}"
    return path


def sha(path):
    digest = hashlib.sha256()
    with checked(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path):
    path = checked(path)
    return {"path": str(path), "sha256": sha(path)}


def bound(item):
    assert isinstance(item, dict) and isinstance(item.get("sha256"), str)
    path = checked(item["path"])
    assert sha(path) == item["sha256"], f"SHA 불일치: {path}"
    return path


def write_json(path, obj):
    with checked(path).open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def configure():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False


def metadata(packet):
    required = {"contextCLS", "tokens", "localCLS", "fixedQ", "condition32", "target", "fit_bool", "row_id", "native", "normalized_active_q", "frame_index", "context_index", "context_frame_index"}
    assert required.issubset(packet.files), required - set(packet.files)
    data = {key: packet[key] for key in ("target", "fit_bool", "row_id", "native", "frame_index", "context_index", "context_frame_index")}
    for key in ("target", "fit_bool", "row_id", "native", "frame_index", "context_index"):
        assert data[key].shape == (368,), (key, data[key].shape)
    assert data["context_frame_index"].shape == (151,)
    assert data["row_id"].dtype.kind in "US" and len(set(data["row_id"].tolist())) == 368
    assert data["fit_bool"].dtype == np.bool_
    assert data["target"].dtype.kind in "iuf"
    for key in ("native", "frame_index", "context_index", "context_frame_index"):
        assert data[key].dtype.kind in "iu", (key, data[key].dtype)
    assert np.isin(data["target"], [0, 1]).all()
    assert ((data["native"] >= 0) & (data["native"] < 8)).all()
    assert ((data["context_index"] >= 0) & (data["context_index"] < 151)).all()
    assert len(set(data["context_frame_index"].tolist())) == 151
    assert np.array_equal(data["frame_index"], data["context_frame_index"][data["context_index"]])
    fit = data["fit_bool"]
    assert int(fit.sum()) == 65 and int((~fit).sum()) == 303
    assert [int(((data["target"] == k) & fit).sum()) for k in (0, 1)] == [57, 8]
    assert [int(((data["target"] == k) & ~fit).sum()) for k in (0, 1)] == [267, 36]
    return data


def feature_batch(packet, data, indices):
    """FIT 호출 또는 양 final 저장 후 DEV 호출만. 전체 통계/정규화 fit 없음.

    NPZ 멤버는 물리적으로 전체 배열을 decode할 수 있지만 선택 row만 모델 tensor로
    만든다. DEV descriptor normalization/read/logit 호출은 final barrier 뒤에만 한다.
    """
    frame_take = data["context_index"][indices]
    shapes = {"contextCLS": (151, 384), "tokens": (151, 256, 384), "localCLS": (368, 384), "fixedQ": (368, 384), "condition32": (368, 32), "normalized_active_q": (368, 2)}
    result = []
    for key in ("localCLS", "fixedQ", "contextCLS", "tokens", "condition32", "normalized_active_q"):
        original = packet[key]
        assert original.shape == shapes[key] and original.dtype == np.float32, (key, original.shape, original.dtype)
        selected = np.ascontiguousarray(original[frame_take if key in ("contextCLS", "tokens") else indices])
        assert np.isfinite(selected).all(), key
        result.append(torch.from_numpy(selected))
    return tuple(result)


def fit_model(model, features, target):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
    logs = []
    model.train()
    for step in range(1, 301):
        optimizer.zero_grad(set_to_none=True)
        logits = model(*features)
        bce = balanced_bce(logits, target)
        l2 = model.active_l2()
        loss = bce + 0.001 * l2
        assert torch.isfinite(loss)
        loss.backward()
        if not model.use_attention:
            assert all(p.grad is None for p in list(model.query.parameters()) + list(model.key.parameters()))
        optimizer.step()
        logs.append({"update": step, "balanced_BCE": float(bce.detach()), "L2_sum": float(l2.detach()), "total_loss": float(loss.detach())})
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())
    return optimizer, logs


def metrics(target, logits):
    pred = (logits >= 0).astype(np.int64)
    cm = [[int(((target == truth) & (pred == guess)).sum()) for guess in (0, 1)] + [0] for truth in (0, 1)]
    result = {"rows": int(len(target)), "CM_D_E_by_D_E_U": cm, "forced_binary_U": 0}
    for label, k in (("D", 0), ("E", 1)):
        tp = cm[k][k]
        den_truth = sum(cm[k])
        den_pred = cm[0][k] + cm[1][k]
        result[label] = {"truth_count": den_truth, "prediction_count": den_pred, "precision": tp / den_pred if den_pred else None, "recall": tp / den_truth if den_truth else None}
    result["balanced_recall"] = (result["D"]["recall"] + result["E"]["recall"]) / 2
    return result


def self_test():
    configure()
    generator = torch.Generator().manual_seed(1717)
    rand = lambda *shape: torch.randn(*shape, generator=generator)
    features = (rand(4, 384), rand(4, 384), rand(4, 384), rand(4, 256, 384), rand(4, 32), rand(4, 2))
    control = ContextReadProbe(False)
    treat = copy.deepcopy(control)
    treat.use_attention = True
    assert state_hash(control) == state_hash(treat)
    assert sum(p.numel() for p in control.parameters()) == 4561
    assert torch.equal(control(*features), torch.zeros(4)) and torch.equal(treat(*features), torch.zeros(4))
    _, weights = treat.attention_read(features[3], features[4], features[5])
    assert weights.shape == (4, 256) and torch.allclose(weights.sum(1), torch.ones(4), atol=1e-6)
    _, changed = treat.attention_read(features[3], features[4] + 1.0, features[5])
    assert not torch.equal(weights, changed)
    _, moved = treat.attention_read(features[3], features[4], features[5] + 0.25)
    assert not torch.equal(weights, moved)
    with torch.no_grad():
        pattern = torch.linspace(-0.1, 0.1, 1184).reshape(1, 1184)
        control.classifier.weight.copy_(pattern)
        treat.classifier.weight.copy_(pattern)
    y = torch.tensor([0., 1., 0., 1.])
    balanced_bce(control(*features), y).backward()
    balanced_bce(treat(*features), y).backward()
    assert all(p.grad is None for p in list(control.query.parameters()) + list(control.key.parameters()))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in treat.parameters())
    assert treat.query.weight.grad.abs().sum() > 0 and treat.key.weight.grad.abs().sum() > 0
    control.zero_grad(set_to_none=True)
    control.active_l2().backward()
    assert all(p.grad is None for p in list(control.query.parameters()) + list(control.key.parameters()))
    toy = metrics(np.array([0, 0, 1, 1]), np.array([-1., 0., -1., 2.]))
    assert toy["CM_D_E_by_D_E_U"] == [[1, 1, 0], [1, 1, 0]]
    print(json.dumps({"completed": True, "PASS": True, "random_only": True, "actual_data_reads": 0, "optimizer_steps": 0, "backward_only": True, "parameters_each": 4561, "checks": ["same_state_zero_initial_logits", "softmax_sum", "condition_change", "query_position_change", "attention_gradient_path", "CONTROL_unused_attention_and_L2_exclusion", "fixed_zero_threshold_CM"]}, ensure_ascii=False))


def run(entry_path):
    configure()
    entry_path = checked(entry_path)
    entry = json.loads(entry_path.read_text())
    assert entry["approved_context_probe_fit"] is True
    assert entry["protocol"] == {"path": str(PROTOCOL), "sha256": PROTOCOL_SHA}
    bound(entry["protocol"])
    assert bound(entry["code"]) == checked(__file__)
    assert bound(entry["head"]) == checked(HERE / "head.py")
    input_path = bound(entry["feature_packet"])
    assert input_path.suffix == ".npz"
    out = checked(entry["output_directory"])
    assert not out.exists(), "기존 실행 산출물은 덮어쓰지 않습니다"
    out.mkdir(parents=True)
    write_json(out / "ENTRY_USED.json", {"entry": binding(entry_path), "protocol": entry["protocol"], "code": entry["code"], "head": entry["head"], "feature_packet": entry["feature_packet"], "completed": False})
    with np.load(input_path, allow_pickle=False) as packet:
        data = metadata(packet)
        fit_indices = np.flatnonzero(data["fit_bool"])
        dev_indices = np.flatnonzero(~data["fit_bool"])
        fit_features = feature_batch(packet, data, fit_indices)
        fit_targets = torch.from_numpy(data["target"][fit_indices].astype(np.float32))
        control = ContextReadProbe(False)
        treat = copy.deepcopy(control)
        treat.use_attention = True
        initial_hash = state_hash(control)
        assert initial_hash == state_hash(treat)
        models = {"CONTROL_FIXED_Q": control, "TREAT_ATTENTION": treat}
        checkpoints = {}
        logs_bindings = {}
        for arm, model in models.items():
            optimizer, logs = fit_model(model, fit_features, fit_targets)
            model.eval().requires_grad_(False)
            path = out / f"{arm}_FINAL300.pt"
            torch.save({"state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "arm": arm, "updates": 300, "seed": 17, "use_attention": model.use_attention, "parameters": 4561, "initial_state_sha256": initial_hash, "protocol": entry["protocol"], "feature_packet": entry["feature_packet"]}, checked(path))
            checkpoints[arm] = binding(path)
            log_path = out / f"{arm}_TRAIN300_LOG.json"
            write_json(log_path, {"completed": True, "rows": logs, "updates": 300, "fit_rows": 65, "DEV_used_for_fit_or_normalization": False})
            logs_bindings[arm] = binding(log_path)
        barrier_path = out / "ALL2_FINAL300_FROZEN_BEFORE_DEV_DESCRIPTOR_OR_LOGIT.json"
        write_json(barrier_path, {"completed": True, "all2_frozen": True, "updates_each": 300, "checkpoints": checkpoints, "initial_state_sha256": initial_hash, "DEV_descriptor_or_logit_calls_before_barrier": 0, "feature_packet": entry["feature_packet"]})
        # 이 지점 이전에는 DEV descriptor를 만들거나 DEV 모델 출력을 계산하지 않는다.
        dev_features = feature_batch(packet, data, dev_indices)
        reductions, row_outputs = {}, []
        with torch.no_grad():
            for arm, model in models.items():
                reductions[arm] = {}
                for role, indices, features in (("FIT", fit_indices, fit_features), ("DEV", dev_indices, dev_features)):
                    logits = model(*features).cpu().numpy()
                    assert np.isfinite(logits).all()
                    reductions[arm][role] = metrics(data["target"][indices], logits)
                    for ordinal, index in enumerate(indices):
                        row_outputs.append({"arm": arm, "row_id": str(data["row_id"][index]), "frame_index": int(data["frame_index"][index]), "native": int(data["native"][index]), "role": role, "truth_D0_E1": int(data["target"][index]), "logit": float(logits[ordinal]), "pred_D0_E1": int(logits[ordinal] >= 0)})
    csv_path = out / "ALL736_ARM_QUERY_LOGITS.csv"
    with checked(csv_path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row_outputs[0]))
        writer.writeheader()
        writer.writerows(row_outputs)
    c, t = reductions["CONTROL_FIXED_Q"]["DEV"], reductions["TREAT_ATTENTION"]["DEV"]
    gates = {"D_recall_at_least_0.8": t["D"]["recall"] >= 0.8, "E_recall_at_least_0.8": t["E"]["recall"] >= 0.8, "balanced_recall_gain_at_least_0.05": t["balanced_recall"] - c["balanced_recall"] >= 0.05, "neither_class_recall_declines": all(t[k]["recall"] >= c[k]["recall"] for k in ("D", "E"))}
    summary_path = out / "SUMMARY.json"
    write_json(summary_path, {"completed": True, "all_arms_reported": True, "criterion_evaluated_on": "DEV303_REUSED_SOURCE_NOT_FRESH_BLIND", "metrics": reductions, "source_branch_gates": gates, "source_branch_pass": all(gates.values()), "gain_is_not_goal_success_or_practical_performance": True, "no_abstention_head_forced_binary_U0": True, "new_current_or_protected_inference": 0})
    complete = {"completed": True, "PASS": True, "code": entry["code"], "head": entry["head"], "protocol": entry["protocol"], "entry": binding(entry_path), "feature_packet": entry["feature_packet"], "checkpoints": checkpoints, "logs": logs_bindings, "all2_final_barrier": binding(barrier_path), "summary": binding(summary_path), "row_logits": binding(csv_path), "steps_each": 300, "parameters_each": 4561, "source_branch_pass": all(gates.values()), "further_epoch_seed_or_nearby_head_search_authorized": False, "DEV_used_for_fit_or_normalization": False, "source_weak_labels_not_physical_coordinate_gold": True}
    write_json(out / "COMPLETE.json", complete)
    print(json.dumps({"completed": True, "PASS": True, "COMPLETE": binding(out / "COMPLETE.json"), "source_branch_pass": all(gates.values())}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--entry")
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--entry-template", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.entry_template:
        print(json.dumps({"approved_context_probe_fit": False, "protocol": {"path": str(PROTOCOL), "sha256": PROTOCOL_SHA}, "code": binding(__file__), "head": binding(HERE / "head.py"), "feature_packet": {"path": None, "sha256": None}, "output_directory": None}, ensure_ascii=False, indent=2))
    else:
        run(args.entry)


if __name__ == "__main__":
    main()

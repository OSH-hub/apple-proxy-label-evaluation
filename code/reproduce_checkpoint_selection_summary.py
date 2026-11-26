"""Recalculate aggregate tables/checkpoint_selection_sensitivity.csv from authorized, private replay evidence.

Only aggregate CSV/JSON outputs are written. No archive extraction, model fitting,
source-record export or checkpoint-tensor loading is performed. Tensor agreement
fields are supported by archived comparison logs, not rechecked tensors here.
Requires Python and NumPy; the original calculation used NumPy 2.4.4.
"""
import argparse
import csv
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np


class InvalidEvidence(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidEvidence(message)


def load_json(archive, member):
    return json.loads(archive.read(member).decode("utf-8-sig"))


def history_by_epoch(records):
    require(isinstance(records, list) and len(records) == 40,
            "Each history must contain exactly 40 epochs.")
    require([r["epoch"] for r in records] == list(range(40)),
            "History epochs must be ordered, unique and zero-based (0 to 39).")
    return records


def calculate(archive_path, histories_dir):
    summaries, checks = [], []
    array_count = prediction_count = 0
    reference_labels = reference_keys = None
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Archive contains duplicate members.")
        prefixes = {}
        for member in names:
            match = re.search(r"(?:^|/)zone_seed([0-2])/history\.json$", member)
            if match:
                seed = int(match.group(1))
                require(seed not in prefixes, "Ambiguous replay history for a seed.")
                prefixes[seed] = member[:-len("history.json")]
        require(set(prefixes) == {0, 1, 2}, "Replay histories for seeds 0, 1 and 2 are required.")
        logged = load_json(archive, "checkpoint_final_remote_verification.json")
        require(isinstance(logged, list) and len(logged) == 3,
                "Three archived tensor-comparison records are required.")
        remote = {r["seed"]: r for r in logged}
        require(set(remote) == {0, 1, 2}, "Archived comparison seeds are invalid.")
        for seed in range(3):
            prefix = prefixes[seed]
            history = history_by_epoch(load_json(archive, prefix + "history.json"))
            original = history_by_epoch(json.loads(
                (histories_dir / f"zone_seed{seed}_history.json").read_text(encoding="utf-8-sig")))
            completed = load_json(archive, prefix + "completed.json")
            require(completed["seed"] == seed, "Completion record has the wrong seed.")
            npz_members = {n for n in names if n.startswith(prefix + "selection_epoch_") and n.endswith(".npz")}
            require(npz_members == {prefix + f"selection_epoch_{i:02d}.npz" for i in range(40)},
                    "Each seed must have exactly 40 epoch prediction arrays.")
            full_values, omitted_values, loss_differences, mae_differences = [], [], [], []
            for epoch in range(40):
                # Reading only required members avoids extracting private log/config content.
                with np.load(io.BytesIO(archive.read(prefix + f"selection_epoch_{epoch:02d}.npz")),
                             allow_pickle=False) as arrays:
                    pred, labels, keys = arrays["pred"], arrays["nir"], arrays["objkeys"]
                require(pred.ndim == labels.ndim == keys.ndim == 1 and
                        pred.shape == labels.shape == keys.shape and len(labels) > 1,
                        "Prediction, label and identity arrays have incompatible shapes.")
                require(np.isfinite(pred).all() and np.isfinite(labels).all(),
                        "Non-finite predictions or labels are not allowed.")
                require(len(np.unique(keys)) == len(keys), "Selection identities must be unique.")
                zero = np.flatnonzero(labels == 0)
                require(len(zero) == 1, "Exactly one NIR value of zero is required.")
                if reference_labels is None:
                    reference_labels, reference_keys = labels.copy(), keys.copy()
                require(np.array_equal(labels, reference_labels) and np.array_equal(keys, reference_keys),
                        "Selection labels or ordering changed across epochs or seeds.")
                errors = np.abs(pred - labels)
                full, omitted = float(errors.mean()), float(np.delete(errors, zero).mean())
                h, old = history[epoch], original[epoch]
                require(full == h["selection_nir_mae"] and omitted == h["selection_omit_zero_mae"],
                        "Recalculated MAE differs from the replay history.")
                require(np.isfinite(h["loss"]) and np.isfinite(old["loss"]) and
                        np.isfinite(old["selection_nir_mae"]), "History values must be finite.")
                full_values.append(full)
                omitted_values.append(omitted)
                loss_differences.append(abs(h["loss"] - old["loss"]))
                mae_differences.append(abs(full - old["selection_nir_mae"]))
                array_count += 1
                prediction_count += len(labels)
            # The original trainer accepts epochs 6--40 and retains the first minimum.
            full_best = min(range(5, 40), key=full_values.__getitem__)
            omitted_best = min(range(5, 40), key=omitted_values.__getitem__)
            original_best = min(range(5, 40), key=lambda i: original[i]["selection_nir_mae"])
            require(full_best == original_best == completed["original_selected_epoch_zero_based"] ==
                    completed["replay_selected_epoch_zero_based"], "Full-pool selected epochs disagree.")
            require(omitted_best == completed["omit_zero_selected_epoch_zero_based"],
                    "Omitted-row selected epoch differs from the archived record.")
            require(max(loss_differences) == completed["max_abs_history_loss_delta"] and
                    max(mae_differences) == completed["max_abs_history_mae_delta"],
                    "History differences disagree with the archived comparison.")
            log = remote[seed]
            require(completed["checkpoint_tensor_count"] == log["tensor_count"] > 0,
                    "Archived tensor counts disagree.")
            require(type(completed["original_checkpoint_tensors_exact"]) is bool and
                    type(log["full_vs_omit_checkpoint_tensors_exact"]) is bool,
                    "Archived tensor agreement fields must be booleans.")
            replay_match = completed["original_checkpoint_tensors_exact"]
            full_omit_match = log["full_vs_omit_checkpoint_tensors_exact"]
            require(replay_match == (completed["checkpoint_unequal_tensor_count"] == 0) and
                    full_omit_match == (log["unequal_tensor_count"] == 0),
                    "Archived tensor agreement flags and counts disagree.")
            require(log["epoch_array_count"] == 40, "Archived array count is inconsistent.")
            summaries.append({
                "seed": seed,
                "selection_records_including_zero": len(reference_labels),
                "selection_records_omitting_zero": len(reference_labels) - 1,
                "selected_epoch_including_zero": full_best + 1,
                "selected_epoch_omitting_zero": omitted_best + 1,
                "replayed_epochs": 40,
                "eligible_selection_epochs": 35,
                "maximum_absolute_training_loss_difference": max(loss_differences),
                "maximum_absolute_selection_mae_difference": max(mae_differences),
                "checkpoint_tensors_compared": completed["checkpoint_tensor_count"],
                "replayed_checkpoint_matches_original": replay_match,
                "omitted_checkpoint_matches_original": replay_match and full_omit_match,
            })
            checks.append({"seed": seed, "all_epoch_training_losses_exact": max(loss_differences) == 0,
                           "all_epoch_selection_maes_exact": max(mae_differences) == 0,
                           "selected_epoch_unchanged": full_best == omitted_best})
    audit = {
        "scope": "Aggregate recomputation from authorized private replay predictions and original histories",
        "epoch_prediction_arrays_checked": array_count,
        "total_predictions_recalculated": prediction_count,
        "zero_record_identification": "Exactly one NIR == 0; no source identity is hardcoded or exported",
        "eligible_epochs_one_based": [6, 40],
        "checkpoint_tensor_status_source": "Archived completed and final comparison logs; actual checkpoint files were not loaded",
        "checkpoint_tensors_recompared_in_this_run": False,
        "training_losses_recomputed_from_training": False,
        "training_loss_comparison": "Archived replay history versus separately supplied original history",
        "private_records_written": False,
        "numpy_version": np.__version__,
        "seed_checks": checks,
    }
    return summaries, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-evidence", type=Path, required=True, metavar="PRIVATE_ZIP")
    parser.add_argument("--original-histories", type=Path, required=True, metavar="PRIVATE_HISTORY_DIRECTORY")
    parser.add_argument("--output", type=Path, required=True, metavar="NEW_CSV")
    parser.add_argument("--audit-output", type=Path, required=True, metavar="NEW_AGGREGATE_JSON")
    parser.add_argument("--expected", type=Path, metavar="PUBLISHED_DATA_S16_CSV")
    args = parser.parse_args()
    try:
        require(args.output.resolve() != args.audit_output.resolve(), "Outputs must be different files.")
        require(not args.output.exists() and not args.audit_output.exists(), "Output exists; choose new output files.")
        rows, audit = calculate(args.replay_evidence, args.original_histories)
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        if args.expected:
            with args.expected.open(encoding="utf-8-sig", newline="") as stream:
                expected = list(csv.DictReader(stream))
            calculated = list(csv.DictReader(io.StringIO(buffer.getvalue())))
            require(expected == calculated, "Calculated aggregate rows differ from the expected table.")
            audit["published_table_matches"] = True
        # Exclusive creation protects both the supplied evidence and existing outputs.
        with args.output.open("x", encoding="utf-8", newline="") as stream:
            stream.write(buffer.getvalue())
        with args.audit_output.open("x", encoding="utf-8", newline="") as stream:
            json.dump(audit, stream, indent=2)
            stream.write("\n")
        print(json.dumps({"status": "PASS", "seeds": len(rows), "arrays_checked": audit["epoch_prediction_arrays_checked"],
                          "predictions_recalculated": audit["total_predictions_recalculated"],
                          "published_table_matches": audit.get("published_table_matches"),
                          "tensor_comparison_source": "archived logs; not rerun"}))
        return 0
    except InvalidEvidence as error:
        print("Validation failed: " + str(error), file=sys.stderr)
    except (OSError, KeyError, ValueError, TypeError, zipfile.BadZipFile):
        # Do not echo private input paths, record identities or archive contents.
        print("Input or output failure; check permissions and the documented evidence format.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

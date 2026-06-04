#!/usr/bin/env python
"""
Run graph-based EEG classification on PhysioNet EEG Motor Movement/Imagery.

Example:
  # If you already extracted the Kaggle dataset:
  python physionet_graph_ml.py --data_dir ./data/physionet-eeg-motor-movement-imagery --subjects 1 2 3 4 5 --task mi_lr

  # Or let MNE download from PhysioNet:
  python physionet_graph_ml.py --subjects 1 2 3 4 5 --task mi_lr

Tasks:
  mi_lr     : imagined left fist vs imagined right fist, runs 4,8,12
  me_lr     : real left fist vs real right fist, runs 3,7,11
  mi_hf     : imagined both fists vs imagined both feet, runs 6,10,14
  me_hf     : real both fists vs real both feet, runs 5,9,13
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import mne
import networkx as nx
import numpy as np
import pandas as pd
from mne.datasets import eegbci
from mne.io import read_raw_edf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TASKS = {
    "mi_lr": {"runs": [4, 8, 12], "labels": {"T1": "left_fist_imagery", "T2": "right_fist_imagery"}},
    "me_lr": {"runs": [3, 7, 11], "labels": {"T1": "left_fist_movement", "T2": "right_fist_movement"}},
    "mi_hf": {"runs": [6, 10, 14], "labels": {"T1": "both_fists_imagery", "T2": "both_feet_imagery"}},
    "me_hf": {"runs": [5, 9, 13], "labels": {"T1": "both_fists_movement", "T2": "both_feet_movement"}},
}

BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _find_edf(data_dir: Path, subject: int, run: int) -> Path:
    name = f"S{subject:03d}R{run:02d}.edf"
    hits = list(data_dir.rglob(name))
    if not hits:
        raise FileNotFoundError(f"Could not find {name} under {data_dir}")
    return hits[0]


def _prepare_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Standardize PhysioNet channel names, filter, and attach a 10-05 montage."""
    try:
        eegbci.standardize(raw)
    except Exception:
        raw.rename_channels(lambda ch: ch.strip().rstrip(".").upper())

    raw.pick_types(eeg=True, stim=False)
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="ignore")
    raw.filter(1.0, 45.0, fir_design="firwin", verbose="ERROR")
    return raw


def load_subject_epochs(subject: int, runs: list[int], data_dir: Path | None) -> mne.Epochs:
    if data_dir is None:
        files = eegbci.load_data(subject, runs, verbose="ERROR")
    else:
        files = [_find_edf(data_dir, subject, run) for run in runs]

    raws = []
    for f in files:
        raw = read_raw_edf(str(f), preload=True, verbose="ERROR")
        raws.append(_prepare_raw(raw))

    raw_all = mne.concatenate_raws(raws, verbose="ERROR")

    # PhysioNet EEGBCI annotations: T0=rest, T1/T2=task-specific class labels.
    events, _ = mne.events_from_annotations(raw_all, event_id={"T1": 1, "T2": 2}, verbose="ERROR")
    if len(events) == 0:
        raise RuntimeError(f"No T1/T2 events found for subject {subject}")

    epochs = mne.Epochs(
        raw_all,
        events,
        event_id={"T1": 1, "T2": 2},
        tmin=0.0,
        tmax=4.0,
        baseline=None,
        preload=True,
        reject_by_annotation=True,
        verbose="ERROR",
    )
    return epochs


def graph_features(matrix: np.ndarray, ch_names: list[str], band: str, density: float) -> dict[str, float]:
    """Create graph-theoretical features from a channel connectivity matrix."""
    mat = np.nan_to_num(np.abs(matrix), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(mat, 0.0)

    upper = mat[np.triu_indices_from(mat, k=1)]
    if not np.any(upper):
        threshold = 1.0
    else:
        threshold = np.quantile(upper, max(0.0, min(1.0, 1.0 - density)))

    adj = np.where(mat >= threshold, mat, 0.0)
    np.fill_diagonal(adj, 0.0)

    graph_weighted = nx.from_numpy_array(adj)
    graph_binary = nx.from_numpy_array((adj > 0).astype(int))

    feats: dict[str, float] = {
        f"{band}_density": nx.density(graph_binary),
        f"{band}_global_efficiency": nx.global_efficiency(graph_binary),
    }

    clustering = nx.clustering(graph_weighted, weight="weight")
    feats[f"{band}_mean_clustering"] = float(np.mean(list(clustering.values()))) if clustering else 0.0

    strengths = adj.sum(axis=1)
    for idx, ch in enumerate(ch_names):
        safe_ch = ch.replace(" ", "_")
        feats[f"{band}_{safe_ch}_strength"] = float(strengths[idx])

    return feats


def extract_features(epochs: mne.Epochs, subject: int, class_names: dict[str, str], density: float) -> pd.DataFrame:
    labels = epochs.events[:, 2]
    y = np.where(labels == 1, 0, 1)
    y_name = [class_names["T1"] if yy == 0 else class_names["T2"] for yy in y]

    rows: list[dict[str, float | int | str]] = [
        {"subject": subject, "trial": i, "label": int(y[i]), "label_name": y_name[i]}
        for i in range(len(epochs))
    ]

    for band, (fmin, fmax) in BANDS.items():
        ep_band = epochs.copy().filter(fmin, fmax, fir_design="firwin", verbose="ERROR")
        data = ep_band.get_data(copy=True)  # shape: n_epochs x n_channels x n_times
        ch_names = ep_band.ch_names

        for i in range(data.shape[0]):
            # Lightweight functional connectivity proxy: channel-wise Pearson correlation per epoch.
            # For exact coherence, replace this matrix with mne-connectivity spectral_connectivity_epochs.
            corr = np.corrcoef(data[i])
            rows[i].update(graph_features(corr, ch_names, band, density))

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None, help="Root of extracted Kaggle/PhysioNet dataset. If omitted, MNE downloads the files.")
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 110)), help="Subject numbers, e.g. --subjects 1 2 3")
    parser.add_argument("--task", choices=TASKS.keys(), default="mi_lr")
    parser.add_argument("--density", type=float, default=0.20, help="Keep top density of connectivity edges, e.g. 0.20")
    parser.add_argument("--out", type=str, default="physionet_graph_features.csv")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else None
    task = TASKS[args.task]
    all_rows = []

    mne.set_log_level("WARNING")
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print(f"Task: {args.task} | runs: {task['runs']} | labels: {task['labels']}")
    print(f"Subjects: {args.subjects}")

    for subject in args.subjects:
        try:
            epochs = load_subject_epochs(subject, task["runs"], data_dir)
            df_sub = extract_features(epochs, subject, task["labels"], args.density)
            all_rows.append(df_sub)
            print(f"Subject {subject:03d}: {len(df_sub)} epochs")
        except Exception as exc:
            print(f"Subject {subject:03d}: skipped ({exc})")

    if not all_rows:
        raise RuntimeError("No features were extracted. Check --data_dir and subject/run files.")

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved features to: {args.out}")
    print(df["label_name"].value_counts())

    feature_cols = [c for c in df.columns if c not in {"subject", "trial", "label", "label_name"}]
    X = df[feature_cols].to_numpy()
    y = df["label"].to_numpy()
    groups = df["subject"].to_numpy()

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced")),
        ]
    )

    # Prefer subject-wise CV to avoid mixing trials from the same subject across train/test.
    if df["subject"].nunique() >= 5:
        cv = GroupKFold(n_splits=5)
        scores = cross_val_score(clf, X, y, groups=groups, cv=cv, scoring="accuracy")
        print(f"\nGroupKFold subject-wise accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        print(f"\nStratified trial-wise accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20, random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(f"\nHoldout accuracy: {accuracy_score(y_test, pred):.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, pred))
    print("\nClassification report:")
    print(classification_report(y_test, pred, target_names=[task["labels"]["T1"], task["labels"]["T2"]]))


if __name__ == "__main__":
    main()

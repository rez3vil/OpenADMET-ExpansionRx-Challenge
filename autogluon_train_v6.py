import os
import sys
import time
import json
import argparse

import numpy as np
import pandas as pd

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem import rdFingerprintGenerator

from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor


# Smiles to Mol
def smiles_to_mol(smi: str):
    if pd.isna(smi):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return mol
    except Exception:
        return None
    

# Calculate Descriptors
def compute_rdkit_descriptors(mol_list):
    desc_funcs = [(name, func) for name, func in Descriptors._descList]
    desc_names = [name for name, _ in desc_funcs]

    records = []

    for mol in mol_list:
        if mol is None:
            records.append([np.nan] * len(desc_funcs))
            continue

        vals = []
        for _, func in desc_funcs:
            try:
                vals.append(func(mol))
            except Exception:
                vals.append(np.nan)
        records.append(vals)

    df = pd.DataFrame(records, columns=[f"desc_{n}" for n in desc_names])
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


# Converts rdkit explicitbitvect to python list of 0/1
def explicit_bits_to_list(fp) -> list:
    return list(fp)


# Compute fingerprints and returns concatenated dataframe
def compute_fingerprints(
        mol_list,
        use_morgan=False,
        morgan_radius=2,
        use_rdkitfp=False,
        use_atompair=False,
        use_toptors=False,
        fp_nbits=2048,
        use_maccs=False
):
    mats = []

    if use_morgan:
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=morgan_radius,
            fpSize=fp_nbits,
            includeChirality=True
            )
        morgan_bits = []
        for mol in mol_list:
            if mol is None:
                morgan_bits.append([np.nan] *fp_nbits)
            else:
                fp = morgan_gen.GetFingerprint(mol)
                morgan_bits.append(explicit_bits_to_list(fp))
        morgan_df = pd.DataFrame(
            morgan_bits,
            columns=[f"fp_morgan_r{morgan_radius}_{i}" for i in range(fp_nbits)]
            )
        mats.append(morgan_df)

    if use_rdkitfp:
        rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(
            fpSize=fp_nbits
            )
        rdkit_bits = []
        for mol in mol_list:
            if mol is None:
                rdkit_bits.append([np.nan] * fp_nbits)
            else:
                fp = rdkit_gen.GetFingerprint(mol)
                rdkit_bits.append(explicit_bits_to_list(fp))
        rdkit_df = pd.DataFrame(
            rdkit_bits,
            columns=[f"fp_rdkit_{i}" for i in range(fp_nbits)]
            )
        mats.append(rdkit_df)
    
    if use_atompair:
        ap_gen = rdFingerprintGenerator.GetAtomPairGenerator(
            fpSize=fp_nbits,
            includeChirality=True
            )
        ap_bits = []
        for mol in mol_list:
            if mol is None:
                ap_bits.append([np.nan] * fp_nbits)
            else:
                fp = ap_gen.GetFingerprint(mol)
                ap_bits.append(explicit_bits_to_list(fp))
        ap_df = pd.DataFrame(
            ap_bits,
            columns=[f"fp_atompair_{i}" for i in range(fp_nbits)]
            )
        mats.append(ap_df)

    if use_toptors:
        tt_gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
            fpSize=fp_nbits,
            includeChirality=True
        )
        tt_bits = []
        for mol in mol_list:
            if mol is None:
                tt_bits.append([np.nan] * fp_nbits)
            else:
                fp = tt_gen.GetFingerprint(mol)
                tt_bits.append(explicit_bits_to_list(fp))
        tt_df = pd.DataFrame(
            tt_bits, 
            columns=[f"fp_toptors_{i}" for i in range(fp_nbits)])
        mats.append(tt_df)

    if use_maccs:
        # MACCS is 167 bits, index 0 is typically unused, we keep all for consistancy
        maccs_bits = []
        for mol in mol_list:
            if mol is None:
                maccs_bits.append([np.nan] * 167)
            else:
                fp = MACCSkeys._pyGenMACCSKeys(mol)
                maccs_bits.append(explicit_bits_to_list(fp))
            maccs_df = pd.DataFrame(
                maccs_bits, 
                columns=[f"fp_maccs_{i}" for i in range(167)]
                )
            mats.append(maccs_df)

    if not mats:
        return pd.DataFrame(index=range(len(mol_list)))
    
    fp_df = pd.concat(mats, axis=1)
    # Ensure integer dtype (AutoGluon handles ints fine); NaNs will remain floats
    for c in fp_df.columns:
        if not fp_df[c].isna().any():
            fp_df[c] = fp_df[c].astype(np.int8)
    return fp_df


# Build Feature Matrix
def build_features(df, smiles_col, include_desc=True, fp_cfg=None):
    """
    df: DataFrame with SMILES column
    fp_cfg: dict of fingerprints configs
    """
    # Convert SMILES to molecules
    mols = [smiles_to_mol(s) for s in df[smiles_col].values]
    valid_mask = [m is not None for m in mols]

    # Drop invalid SMILES
    df_valid = df.loc[valid_mask].copy()
    mols = [m for m in mols if m is not None]
    df_valid.reset_index(drop=True, inplace=True)

    parts = []

    if include_desc:
        desc_df = compute_rdkit_descriptors(mols)
        parts.append(desc_df)

    if fp_cfg is None:
        fp_cfg = {}

    fp_df = compute_fingerprints(
        mols,
        use_morgan=fp_cfg.get("use_morgan", False),
        morgan_radius=fp_cfg.get("morgan_radius", 2),
        use_rdkitfp=fp_cfg.get("use_rdkitfp", False),
        use_atompair=fp_cfg.get("use_atompair", False),
        use_toptors=fp_cfg.get("use_toptors", False),
        fp_nbits=fp_cfg.get("fp_nbits", 2048),
        use_maccs=fp_cfg.get("use_maccs", False)
    )

    if fp_df.shape[1] > 0:
        parts.append(fp_df)

    if not parts:
        raise ValueError("No features were generated. Enable descriptors and/or at least one fingerprint.")
    
    X = pd.concat(parts, axis=1)
    return df_valid, X


# Training with AutoGluon
def train_autogluon_regression(
        train_df,
        test_df,
        label,
        outdir,
        time_limit=None,
        presets="best_quality",
        hyperparameters=None,
        num_stack_levels=1,
        # num_bag_folds=5,
        compute_feature_importance=True,
        keep_only_best=True,
        save_space=True,
        # auto_stack=True,
        verbosity=2
):
    os.makedirs(outdir, exist_ok=True)

    predictor = TabularPredictor(
        label=label,
        eval_metric="mean_absolute_error",
        path=str(Path(outdir) / "autogluon_predictor"),
        log_to_file=True
    )

    fit_kwargs = dict(
        presets=presets,
        time_limit=time_limit,
        hyperparameters=hyperparameters,
        num_stack_levels=num_stack_levels,
        #num_bag_folds=num_bag_folds,
        keep_only_best=keep_only_best,
        save_space=save_space,
        # auto_stack=auto_stack,
        verbosity=verbosity
    )

    predictor.fit(
        train_df, 
        **fit_kwargs, 
        num_gpus=0, 
        # dynamic_stacking=True, 
        ag_args_fit={"ag.max_memory_usage_ratio": 1.5}, 
        ds_args={
            "validation_procedure": 'cv', 
            "n_folds": 5, 
            "n_repeats": 5, 
            "enable_ray_logging": True
            }
            )

    leaderboard = predictor.leaderboard(test_df, extra_info=False, extra_metrics=["r2", "root_mean_squared_error"], silent=True)
    leaderboard.to_csv(Path(outdir) / "leaderboard.csv", index=False)

    perf = predictor.evaluate(test_df, detailed_report=False, silent=True)

    with open(Path(outdir) / "test_metrics.json", "w") as f:
        json.dump(perf, f, indent=2)

    if compute_feature_importance:
    # Feature importance can be expensive on huge feature sets; limit rows
        try:
            fi = predictor.feature_importance(test_df.sample(min(500, len(test_df)), random_state=0))
            fi.to_csv(Path(outdir) / "feature_importance.csv")
        except Exception as e:
            print(f"Feature importance failed: {e}", file=sys.stderr)

    return predictor, leaderboard, perf

# Main
def main():
    parser = argparse.ArgumentParser(description="AutoGluon Regression Pipeline for Training Models")
    parser.add_argument("--csv", required=True, help="Path to input .csv file with SMILES and pIC50")
    parser.add_argument("--smiles_col", default="SMILES", help="SMILES column name (Default: SMILES)")
    parser.add_argument("--target_col", default="pIC50", help="Target column name (Default: pIC50)")
    parser.add_argument("--outdir", default="run_001", help="Output directory (Default: run_001)")
    parser.add_argument("--no_descriptors", action="store_true", help="Disables RDKIT descriptors, which are enabled by default")
    parser.add_argument("--use_morgan", action="store_true", help="Enable Morgan Fingerprints")
    parser.add_argument("--morgan_radius", type=int, default=2, help="Morgan Radius (Default: 2)")
    parser.add_argument("--use_rdkitfp", action="store_true", help="Enable RDKit Topological Fingerprints")
    parser.add_argument("--use_atompair", action="store_true", help="Enable AtomPair Fingerprints")
    parser.add_argument("--use_toptors", action="store_true", help="Enable Topological Torsional Fingerprints")
    parser.add_argument("--use_maccs", action="store_true", help="Enable MACCS Keys Fingerprints")
    parser.add_argument("--fp_nbits", type=int, default=2048, help="Fingerprints Bit Length (Default: 2048)")
    parser.add_argument("--no_feat_imp", action="store_true", help="Disables computing feature importance")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size fraction (Default: 0.2)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed (Default: 42)")
    parser.add_argument("--time_limit", type=int, default=None, help="Time limit in seconds for training (e.g, 3600) (Default: None)")
    parser.add_argument("--presets", type=str, default="best_quality",choices=["extreme_quality", "best_quality", "high_quality", "good_quality", "medium_quality", "experimental_quality", "optimize_for_deployment", "interpretable", "ignore_text"], help="AutoGluon Presets (Default: best_quality)")
    parser.add_argument("--num_stack_levels", type=int, default=1, choices=[0, 1, 2], help="Stacking levels (Default: 1)")
    parser.add_argument("--verbosity", type=int, default=2, choices=[0, 1, 2, 3, 4], help="AutoGluon verbosity (Default: 2)")

    args = parser.parse_args()

    t0 = time.time()
    os.makedirs(args.outdir, exist_ok=True)

    # Read Data
    df = pd.read_csv(args.csv)

    if args.smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{args.smiles_col}' not found in .csv file")
    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' is not found in .csv file")
    
    # Ensure numeric target
    df[args.target_col] = pd.to_numeric(df[args.target_col], errors="coerce")
    df = df.dropna(subset=[args.target_col]).reset_index(drop=True)

    # Building features
    fp_cfg = dict(
        use_morgan=args.use_morgan,
        morgan_radius=args.morgan_radius,
        use_rdkitfp=args.use_rdkitfp,
        use_atompair=args.use_atompair,
        use_toptors=args.use_toptors,
        fp_nbits=args.fp_nbits,
        use_maccs=args.use_maccs
        )
    
    include_desc = not args.no_descriptors

    print("Generating RDKit features...")

    df_valid, X = build_features(df, args.smiles_col, include_desc=include_desc, fp_cfg=fp_cfg)
    y = df_valid[args.target_col].astype(float)

    # Combine features and labels for AutoGluon
    data = X.copy()
    data[args.target_col] = y.values

    # optional: drop constant columns (all 0/1)
    nunique = data.drop(columns=[args.target_col]).nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        data = data.drop(columns=constant_cols)
        print(f"Dropped {len(constant_cols)} constant features.")

    # Train/Test split
    train_df, test_df = train_test_split(data, test_size=args.test_size, random_state=args.random_state)

    # Save preprocessed features for reproducibility
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    train_df.to_csv(Path(args.outdir) / "train_features.csv", index=False)
    test_df.to_csv(Path(args.outdir) / "test_features.csv", index=False)

    # AutoGluon hyperparameters (broad model zoo)
    hyperparameters = {
        "LR": {},
        "GBM": {},
        "XGB": {},
        "CAT": {},
        "RF": {},
        "XT": {},
        "KNN": {},
        "LR": {},
        "NN_TORCH": {}
    }
    
    # Compute feature importance
    compute_feature_importance = not args.no_feat_imp

    # Training
    predictor, leaderboard, perf = train_autogluon_regression(
        train_df=train_df,
        test_df=test_df,
        label=args.target_col,
        outdir=args.outdir,
        time_limit=args.time_limit,
        presets=args.presets,
        hyperparameters=hyperparameters,
        num_stack_levels=args.num_stack_levels,
        compute_feature_importance=compute_feature_importance,
        verbosity=args.verbosity
    )

    # Saving the run configuration
    run_config = {
        "csv": args.csv,
        "smiles_col": args.smiles_col,
        "target_col": args.target_col,
        "outdir": args.outdir,
        "include_descriptors": include_desc,
        "fingerprints": fp_cfg,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "presets": args.presets,
        "time_limit": args.time_limit,
        "num_stack_levels": args.num_stack_levels,
        "models_trained": list(leaderboard["model"]),
        "compute_feature_importance": compute_feature_importance,
        "elapsed_sec": round(time.time() - t0, 2)
        }
    with open(Path(args.outdir) / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    print("\nDone.")
    print(f"- Output dir: {args.outdir}")
    print(f"- Test R^2: {perf.get('r2', 'n/a'):.4f}")
    print(f"- Test MAE: {perf.get('mean_absolute_error', 'n/a'):.4f}")
    print(f"- Test RMSE: {perf.get('root_mean_squared_error', 'n/a'):.4f}")
    print(f"- Leaderboard saved to {Path(args.outdir) / 'leaderboard.csv'}")
    print(f"- Features saved to train_features.csv / test_features.csv")


if __name__ == "__main__":
    main()
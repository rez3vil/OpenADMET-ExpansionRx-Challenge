import sys
import json
import argparse

import numpy as np
import pandas as pd

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem import rdFingerprintGenerator

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
                morgan_bits.append([np.nan] * fp_nbits)
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
    # Where no NaNs exist, cast to int8 to reduce memory
    for c in fp_df.columns:
        if not fp_df[c].isna().any():
            fp_df[c] = fp_df[c].astype(np.int8)
    return fp_df


# Build Feature Matrix
def build_features_for_inference(df, smiles_col, include_desc=True, fp_cfg=None):
    """
    Returns:
        df_out: copy of df with 'mol_valid' boolean
        X: feature matrix for rows where mol_valid is True (aligned in order)
        valid_index: positions in original df corresponding to X rows
    """
    df = df.copy()
    mols = [smiles_to_mol(s) for s in df[smiles_col].values]
    valid_mask = [m is not None for m in mols]
    df["mol_valid"] = valid_mask

    valid_index = df.index[df["mol_valid"]].tolist()
    mols_valid = [m for m in mols if m is not None]

    parts = []

    if include_desc:
        desc_df = compute_rdkit_descriptors(mols_valid)
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
    return df, X, valid_index


# Align Features to Predictor
def align_to_predictor_features(X, predictor):
    expected = predictor.features() # list of feature names used by the model
    X_aligned = pd.DataFrame(index=X.index, columns=expected, dtype=float)
    common = [c for c in expected if c in X.columns]
    if common:
        X_aligned.loc[:, common] = X[common]
    # Any missing columns remain NaN; AutoGluon can handle NaNs via its internal imputers.
    # Any extra columns in X not used by predictor are ignored.
    return X_aligned


# Main
def main():
    parser = argparse.ArgumentParser(description="AutoGluon Regression Pipeline for Predicting on Trained Models")
    parser.add_argument("--csv", required=True, help="Path to input .csv file with SMILES")
    parser.add_argument("--smiles_col", default=None, help="SMILES column name. Defaults to the training run_config or 'SMILES'.")
    parser.add_argument("--model_dir", required=True, help="Training run directory containing autogluon_predictor/ and run_config.json.")
    parser.add_argument("--out", default="predictions.csv", help="Output .csv path (Default: predictions.csv)")
    parser.add_argument("--save_features", action="store_true", help="Save computed features for valid molecues.")

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    predictor_path = model_dir / "autogluon_predictor"
    run_cfg_path = model_dir / "run_config.json"

    if not predictor_path.exists():
        print(f"[ERROR] Predictor not found at: {predictor_path}", file=sys.stderr)
        sys.exit(1)
    if not run_cfg_path.exists():
        print(f"[WARN] run_config.json not found at: {run_cfg_path}. Will assume defaults for features.", file=sys.stderr)
        run_cfg = {}
    else:
        with open(run_cfg_path, "r") as f:
            run_cfg = json.load(f)

    # Determine SMILES column and feaure settings
    smiles_col = args.smiles_col or run_cfg.get("smiles_col", "SMILES")
    include_desc = run_cfg.get("include_descriptors", True)
    fp_cfg = run_cfg.get(
        "fingerprints", {
            "use_morgan": False,
            "morgan_radius": 2,
            "use_rdkitfp": False,
            "use_atompair": False,
            "use_toptors": False,
            "fp_nbits": 2048,
            "use_maccs": False
        })

    # Load Predictor
    print(f"[INFO] Loading predictor from: {predictor_path}")
    predictor = TabularPredictor.load(str(predictor_path))

    # Read Input
    print(f"[INFO] Reading input csv: {args.csv}")
    df_in = pd.read_csv(args.csv)
    if smiles_col not in df_in.columns:
        print(f"[ERROR] SMILES column '{smiles_col}' not found in input CSV. Available columns: {list(df_in.columns)}", file=sys.stderr)
        sys.exit(1)

    # Build features for valid molecules
    print("[INFO] Generating RDKit features for inference...")
    df_work, X, valid_index = build_features_for_inference(
        df_in, 
        smiles_col=smiles_col, 
        include_desc=include_desc, 
        fp_cfg=fp_cfg
    )

    # Align to predictor features
    X_aligned = align_to_predictor_features(X, predictor)

    # Predict on valid molecules
    print("[INFO] Running predictions...")
    preds_valid = predictor.predict(X_aligned)

    # Prepare output aligned to original rows
    out = df_work[["Molecule_Name", smiles_col, "mol_valid"]].copy()
    out[f"predicted_{run_cfg.get("target_col")}"] = preds_valid.values

    # Save predictions
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[OK] Predictions saved to: {out_path.resolve()}")
    
    # Optionally save the computed features for debugging/reproducibility
    if args.save_features:
        feats_path = out_path.with_suffix(".features.csv")
        X_save = X.copy()
        X_save.insert(0, "row_index_in_input", valid_index)
        X_save.to_csv(feats_path, index=False)
        print(f"[OK] Features (valid rows) saved to: {feats_path.resolve()}")

    # Small summary
    n_total = len(df_in)
    n_valid = int(df_work["mol_valid"].sum())
    n_invalid = n_total - n_valid
    used_feats = len(predictor.features())
    print(f"[SUMMARY] Total rows: {n_total} | Valid SMILES: {n_valid} | Invalid SMILES: {n_invalid} | Features expected by model: {used_feats}")


if __name__ == "__main__":
    main()
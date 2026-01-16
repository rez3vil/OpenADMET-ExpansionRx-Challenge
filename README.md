# OpenADMET-ExpansionRx-Challenge

This repository contains Training and Inference scripts for [OpenADMET + ExpansionRx Blind Challenge (2025)](https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge).<br>
**Author:** Piyush Sawner <br>
**Date:** January 16, 2026

---

#### Requirements
- autogluon
- scikit-learn
- rdkit
- pandas
- numpy
- python > 3.9

---

#### Training Usage
```
python autogluon_train_v6.py --csv data.csv --smiles_col SMILES --target_col pIC50 --outdir run_001
```
options:
|Arguments|Description|Default|
|---|---|---|
|`-h` / `--help`|Show this help message and exit|-|
|`--csv CSV`|Path to input .csv file with SMILES and pIC50|-|
|`--smiles_col SMILES_COL`|SMILES column name|SMILES|
|`--target_col TARGET_COL`|Target column name|pIC50|
|`--outdir OUTDIR`|Output directory|run_001|
|`--no_descriptors`|Disables RDKIT descriptors|Enabled|
|`--use_morgan`|Enable Morgan Fingerprints|-|
|`--morgan_radius MORGAN_RADIUS`|Morgan Radius|2|
|`--use_rdkitfp`|Enable RDKit Topological Fingerprints|-|
|`--use_atompair`|Enable AtomPair Fingerprints|-|
|`--use_toptors`|Enable Topological Torsional Fingerprints|-|
|`--use_maccs`|Enable MACCS Keys Fingerprints|-|
|`--fp_nbits FP_NBITS`|Fingerprints Bit Length|2048|
|`--no_feat_imp`|Disables computing feature importance|Enabled|
|`--test_size TEST_SIZE`|Test size fraction|0.2|
|`--random_state RANDOM_STATE`|Random seed|42|
|`--time_limit TIME_LIMIT`|Time limit in seconds for training (e.g, 3600)|None|
|`--presets best_quality,good_quality,medium_quality` |AutoGluon Presets|best_quality|
|`--num_stack_levels 0,1,2`|Stacking levels|1|
|`--verbosity 0,1,2,3,4`|AutoGluon verbosity|2|

---

#### Inference Usage
```
python autogluon_pred_v6.py --csv test.csv --model_dir run_001 --out predictions.csv
```

options:
|Arguments|Description|Default|
|---|---|---|
|`-h`, `--help`|Show this help message and exit|-|
|`--csv CSV`|Path to input .csv file with SMILES|-|
|`--smiles_col SMILES_COL`|SMILES column name.|SMILES|
|`--model_dir MODEL_DIR`|Training run directory containing autogluon_predictor/ and run_config.json.|-|
|`--out OUT`|Output .csv path|predictions.csv|-|
|`--save_features`|Save computed features for valid molecues.|-|

---

## Model Summary

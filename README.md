# OpenADMET-ExpansionRx-Challenge

This repository contains Training and Inference scripts for [OpenADMET + ExpansionRx Blind Challenge (2025)](https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge).

The scripts exclusively uses AutoGluon library to achieve it's predictive performance. It is different because it doesn’t rely on Hyper Paramaters Optimizations to achieve great performance, but rather it’s based on three main principles:  
(1) training a variety of different models,  
(2) using bagging when training those models, and  
(3) stack-ensembling those models to combine their predictive power into a “super” model.  
More info: [How it works](https://auto.gluon.ai/stable/tutorials/tabular/how-it-works.html)

**Author:** Piyush Sawner <br>
**Date:** January 16, 2026

---

#### Requirements
- autogluon == 1.4.0
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
|`--csv`|Path to input .csv file with SMILES and pIC50|-|
|`--smiles_col`|SMILES column name|SMILES|
|`--target_col`|Target column name|pIC50|
|`--outdir`|Output directory|run_001|
|`--no_descriptors`|Disables RDKIT descriptors|Enabled|
|`--use_morgan`|Enable Morgan Fingerprints|-|
|`--morgan_radius`|Morgan Radius|2|
|`--use_rdkitfp`|Enable RDKit Topological Fingerprints|-|
|`--use_atompair`|Enable AtomPair Fingerprints|-|
|`--use_toptors`|Enable Topological Torsional Fingerprints|-|
|`--use_maccs`|Enable MACCS Keys Fingerprints|-|
|`--fp_nbits`|Fingerprints Bit Length|2048|
|`--no_feat_imp`|Disables computing feature importance|Enabled|
|`--test_size`|Test size fraction|0.2|
|`--random_state`|Random seed|42|
|`--time_limit`|Time limit in seconds for training (e.g, 3600)|None|
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
|`--csv`|Path to input .csv file with SMILES|-|
|`--smiles_col`|SMILES column name.|SMILES|
|`--model_dir`|Training run directory containing autogluon_predictor/ and run_config.json.|-|
|`--out`|Output .csv path|predictions.csv|-|
|`--save_features`|Save computed features for valid molecues.|-|

---

## Model Summary

### Description of the Dataset
No additional dataset was used for training. All the endpoints were converted to log values except LogD by using [Pat Walters EDA Analysis Jupyter Notebook](https://github.com/PatWalters/practical_cheminformatics_posts/tree/main/expansion_data_exploration). No data augmentation was done.

|SN|Column|Records|Fraction of Total|
|---|---|---|---|
|1|LogD|5039|0.95|
|2|Log_KSOL|5128|0.96|
|3|Log_HLM CLint|3759|0.71|
|4|Log_MLM CLint|4522|0.85|
|5|Log_Caco-2 Permeability Papp A>B|2157|0.40|
|6|Log_Caco-2 Permeability Efflux|2161|0.41|
|7|Log_MPPB|1302|0.24|
|8|Log_MBPB|975|0.18|
|9|Log_MGMB|222|0.04|

### Description of the Training Features
- 2D RDKit descriptors
- 2048 bits Fingerprints: Morgan (Radius 2), RDKit, Atom Pairs, Torsional

### Description of the Training Model Parameters
- Random State: 42
- Preset: Best Quality
- Training / Test Set Ratio: 0.8 / 0.2
- Cross Validation: 5 Folds x 5 Repeats
- Time Limit: None
- Number of Stack Levels: 1
- Models Trained: LR, GBM, XGB, CAT, RF, XT, KNN, NN_TORCH
- Device: CPU

### Summary of the Models (only best ones)

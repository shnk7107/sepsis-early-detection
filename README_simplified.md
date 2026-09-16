# Sepsis Prediction: Ordering Behaviour vs Physiology (Simplified)

## Project Goal
This project studies whether a sepsis prediction model learns mainly from:
- clinician ordering behaviour, or
- the patient's physiological measurements.

It also examines whether standard feature ablation can separate these two information sources after proper statistical correction.

## Main Result
The model relies mainly on physiological laboratory values.

Ordering behaviour appears important under ordinary statistical testing, but this effect disappears after applying Nadeau–Bengio and Benjamini–Hochberg corrections.

## Pipeline
1. Load the PhysioNet 2019 dataset.
2. Create leakage-free six-hour pre-onset windows.
3. Freeze the train-test split.
4. Perform ablation under eight imbalance methods.
5. Carry out channel-level decomposition.
6. Perform significance testing.
7. Validate using Random Forest, XGBoost and LightGBM.
8. Perform cross-site and robustness analyses.

## Key Design Decisions
- Only data before sepsis onset were used.
- Repeated 5×5 cross-validation with Nadeau–Bengio correction.
- Benjamini–Hochberg correction for multiple testing.
- SOFA-system ablation removed both value and ordering channels.

## Running the Pipeline
Install the dependencies, obtain the PhysioNet 2019 dataset, and run the stages in numerical order using run_all.py.

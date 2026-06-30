# Selective Laser Sintering (SLS) MD & ML Framework
### 🔬 Detailed Framework Documentation, Methodology, and Performance Report

---

## 📌 1. Project Background
This framework integrates **Molecular Dynamics (MD) simulations** and **Machine Learning (ML)** to characterize and predict the molecular evolution of an **Epoxy / Polyamide-12 (PA12)** polymer blend during **Selective Laser Sintering (SLS)**. 

### Sintering Stages (SLS Process)
The system models 5 primary sintering phases sequentially:
1. **Equilibrium** (`equilibrium`): Initial thermodynamic relaxation of the polymer pack at 300 K.
2. **Bed Heating** (`bed`): Warming the powder bed to the build temperature (e.g. 60°C).
3. **Laser Melting** (`laser`): Sintering under rapid laser heating (e.g. 150°C) to melt and merge powder boundaries.
4. **Hold & Necking** (`hold`): Maintaining sintering temperature to allow chain interdiffusion and neck growth.
5. **Cooling** (`cooling`): Gradual thermal cooling back to bed temperature.

---

## ⚙️ 2. Data Pipeline & Features
The pipeline processes **292GB** of raw simulation outputs from two directories:
- `C:\Users\sriva\Desktop\IIT JAMMU\Owais Data`
- `C:\Users\sriva\Desktop\samples`

We extract **338 physical features** grouped into 5 distinct categories:

| Feature Source | Parameters Extracted | Physics Significance |
|---|---|---|
| **OVITO Structural** | Coordination Number (CN), Atomic Volume, Cavity Radius, Displacement | Structural packing, densification, local voids, and atomic rearrangements. |
| **LAMMPS Thermo Log** | Temp, Press, PotEng, KinEng, TotEng, Density, Volume | Total energetics, pressure drops, potential energies, kinetic thermal activity, and density changes. |
| **Bond Chemistry (ReaxFF)** | C-C, C-H, C-N, C-O, H-N, H-O counts & fractions | Chemical bond degradation and reactive oxirane-amine crosslinking indicators. |
| **Temperature Profiles** | Mean, Std, Drift, Stability, Target Adherence | Stability of thermostat temperature profiles and heat distribution. |
| **Atomic Charges** | Charge mean/std/min/max per element (C, H, O, N) | Local electrostatic environment changes. |

---

## 🎞️ 3. Frame-Level Trajectory Analysis (`ml_frame_level.py`)
To capture maximum physical resolution, we parse each and every frame of the **7.19GB** of `.lammpstrj` files.
- **Scale**: Parses **2,520 individual frames** (sampled at a stride of 3) containing **11,102 atoms** each, representing **~28 Million atoms** evaluated at runtime.
- **KDTree Coordinate Querying**: Builds a `scipy.spatial.KDTree` on all atoms. Queries neighbors within a **2.2 Å cutoff** (covalent bond length) to compute local coordination number distributions for every atom in every frame, calculating statistical moments (Mean, Std, Skewness, Kurtosis, IQR, CV).
- **Element-wise Breakdown**: Computes coordination averages independently for Carbon, Hydrogen, Oxygen, and Nitrogen.
- **Displacement Tracking**: Measures the distance of every atom from its coordinate position in frame 0 to compute continuous atomic displacement trajectories.

---

## 🤖 4. Machine Learning Architecture
We train **7 algorithms** across **6 distinct predictive tasks**:

### Algorithms:
1. **Random Forest (RF)** (tuned, n_estimators=500)
2. **Extra Trees (ET)** (tuned, variance-reduction)
3. **XGBoost (XGB)** (gradient boosting with early stopping)
4. **LightGBM (LGBM)** (leaf-wise growth, high speed)
5. **CatBoost** (ordered boosting, handles categoricals)
6. **Voting Ensemble** (soft combination of top tree learners)
7. **Stacking Ensemble** (meta-learner = Ridge Regression)

### Tasks:
- **Task A: Stage Classification (5 classes)**: Predicts process state. (Accuracy: **100%**)
- **Task B: Composition Prediction (3 classes)**: Predicts 50/50, 60/40, or 70/30 ratio. (Accuracy: **100%**)
- **Task C: Bond Network Stability (Binary)**: Classifies whether backbone density is above the median sintering threshold. (Accuracy: **100%**)
- **Task D: Displacement Regression (Å)**: Predicts average atomic movement. (R²: **0.9998**, MAE: **0.0145 Å**)
- **Task E: Potential Energy Regression (kcal/mol)**: Predicts potential energy states. (R²: **1.0000**, MAE: **0.16 kcal/mol**)
- **Task F: Temperature Adherence (0-1)**: Regresses thermostat control quality. (R²: **1.0000**, MAE: **0.0001**)

---

## 📊 5. Generated Portfolios (67 Plots)
All generated plots are committed to the GitHub repository:
- **ML Performance**: Confusion matrices for classifiers, scatter fit lines for regressors.
- **PCA Projections**: 2D and 3D cluster maps highlighting sintering boundaries.
- **Chemistry & Physics**: Line trajectories showing C-C backbone and C-N crosslink counts, hydrogen bonds, density profiles, and potential energy landscapes.

---

## 🚀 6. Execution Command Reference

```bash
# 1. Run the Ultimate pipeline (merges log, bonds, temp, charges, and OVITO xlsx)
python ml_ultimate.py --owais "C:\Users\sriva\Desktop\IIT JAMMU\Owais Data" --samples "C:\Users\sriva\Desktop\samples" --output "Results\ML_Ultimate"

# 2. Run the Ultra-High Resolution Frame-Level Pipeline (processes raw trajectory frames)
python ml_frame_level.py --owais "C:\Users\sriva\Desktop\IIT JAMMU\Owais Data" --output "Results\ML_Frame_Level" --stride 3

# 3. Generate all 12 high-resolution scientific publication plots
python generate_all_possible_plots.py

# 4. Re-compile database and generate the interactive light-themed showcase HTML
python generate_showcase.py
```

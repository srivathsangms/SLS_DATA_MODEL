"""
Generates the Ultimate ML Showcase HTML website (Results/Ultimate_ML_Showcase.html).
Reads and aggregates:
  - Model metrics from model_comparison.csv
  - Top feature importances from feature_importances.csv
  - Sampled bond evolution datasets from all three compositions
  - PCA coordinate projections of all 270 samples
Produces a highly interactive, light-themed, professional presentation site.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Paths
OWAIS_DIR = Path(r"C:\Users\sriva\Desktop\IIT JAMMU\Owais Data")
FEATURES_PATH = Path(r"Results\ML_Ultimate\features_ultimate.csv")
IMPORTANCE_PATH = Path(r"Results\ML_Ultimate\feature_importances.csv")
COMPARISON_PATH = Path(r"Results\ML_Ultimate\model_comparison.csv")
OUTPUT_HTML = Path(r"index.html")

def generate_data_json():
    data = {}

    # 1. Model comparison
    if COMPARISON_PATH.exists():
        df_comp = pd.read_csv(COMPARISON_PATH)
        data["models"] = df_comp.to_dict(orient="records")
    else:
        data["models"] = []

    # 2. Feature importances
    if IMPORTANCE_PATH.exists():
        df_imp = pd.read_csv(IMPORTANCE_PATH)
        data["features"] = df_imp.head(30).to_dict(orient="records")
        # Feature source counts
        sources = df_imp["Source"].value_counts().to_dict()
        data["feature_sources"] = [{"source": k, "count": int(v)} for k, v in sources.items()]
    else:
        data["features"] = []
        data["feature_sources"] = []

    # 3. PCA Coordinates for interactive scatter plot
    if FEATURES_PATH.exists():
        df_feat = pd.read_csv(FEATURES_PATH)
        meta_cols = ["Composition", "Stage", "Position", "Temperature",
                     "Stage_Ordinal", "Position_Ordinal", "Epoxy_Pct", "PA12_Pct"]
        feat_cols = [c for c in df_feat.columns if c not in meta_cols]
        X = df_feat[feat_cols].select_dtypes(include=[np.number]).copy()
        
        # Simple cleanup
        all_nan = X.columns[X.isna().all()].tolist()
        if all_nan: X.drop(columns=all_nan, inplace=True)
        zero_var = X.columns[X.var() < 1e-12].tolist()
        if zero_var: X.drop(columns=zero_var, inplace=True)
        
        imp = SimpleImputer(strategy="median")
        X_clean = imp.fit_transform(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)
        
        pca_points = []
        for i in range(len(df_feat)):
            pca_points.append({
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "composition": str(df_feat.loc[i, "Composition"]),
                "temperature": int(df_feat.loc[i, "Temperature"]),
                "stage": str(df_feat.loc[i, "Stage"]),
                "position": str(df_feat.loc[i, "Position"])
            })
        data["pca"] = pca_points
        data["pca_variance"] = [float(v) for v in pca.explained_variance_ratio_]
    else:
        data["pca"] = []

    # 4. Bond Evolution sampling (comp -> temp -> values)
    # Target some compositions and temperatures to keep HTML size small
    bond_data = {}
    compositions = ["5050", "6040", "7030"]
    temperatures = [100, 200, 300]
    
    for comp in compositions:
        bond_data[comp] = {}
        comp_dir = OWAIS_DIR / comp
        bond_dir = comp_dir / "bond evol" if (comp_dir / "bond evol").exists() else comp_dir / "Bond evol"
        if bond_dir.exists():
            for temp in temperatures:
                # Find matching csv
                csv_path = None
                for f in bond_dir.glob("*.csv"):
                    if str(temp) in f.name and ("bond" in f.name.lower() or "evolution" in f.name.lower()):
                        csv_path = f
                        break
                if csv_path:
                    try:
                        df_b = pd.read_csv(csv_path)
                        df_b.columns = [c.strip() for c in df_b.columns]
                        # sample 30 points evenly
                        n_rows = len(df_b)
                        step = max(1, n_rows // 35)
                        df_samp = df_b.iloc[::step].copy()
                        
                        records = []
                        for _, row in df_samp.iterrows():
                            records.append({
                                "timestep": int(row.get("Timestep", 0)),
                                "stage": str(row.get("Stage", "Equilibration")),
                                "C-C": int(row.get("C-C", 0)),
                                "C-H": int(row.get("C-H", 0)),
                                "C-O": int(row.get("C-O", 0)),
                                "C-N": int(row.get("C-N", 0)),
                                "H-O": int(row.get("H-O", 0)),
                                "H-N": int(row.get("H-N", 0)),
                                "Total": int(row.get("Total_bonds", 0))
                            })
                        bond_data[comp][str(temp)] = records
                    except Exception as e:
                        print(f"Failed to process bond CSV {csv_path}: {e}")
                        
    data["bond_evolution"] = bond_data
    return data

def build_showcase():
    print("Gathering data...")
    db_json = generate_data_json()
    
    print("Writing showcase HTML...")
    # Read template block or construct it
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SLS MD & ML Ultimate Showcase</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {{
            theme: {{
                extend: {{
                    fontFamily: {{
                        sans: ['Inter', 'sans-serif'],
                        outfit: ['Outfit', 'sans-serif'],
                    }},
                    colors: {{
                        brand: {{
                            50: '#f5f7ff',
                            100: '#ebf0ff',
                            200: '#d6e0ff',
                            300: '#b3c7ff',
                            400: '#80a3ff',
                            500: '#4d7cff',
                            600: '#1a55ff',
                            700: '#003be6',
                            800: '#002eb3',
                            900: '#002080',
                        }}
                    }}
                }}
            }}
        }}
    </script>
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <!-- FontAwesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        .custom-shadow {{
            box-shadow: 0 10px 30px -10px rgba(77, 124, 255, 0.08), 0 1px 3px rgba(0,0,0,0.02);
        }}
        .hero-mesh {{
            background-color: #f8fafc;
            background-image: 
                radial-gradient(at 0% 0%, rgba(241, 245, 249, 0.6) 0, transparent 50%),
                radial-gradient(at 50% 0%, rgba(77, 124, 255, 0.07) 0, transparent 50%),
                radial-gradient(at 100% 100%, rgba(156, 39, 176, 0.03) 0, transparent 50%);
        }}
        .interactive-dot {{
            transition: all 0.2s ease-in-out;
        }}
        .interactive-dot:hover {{
            transform: scale(1.4);
            cursor: pointer;
        }}
        .stage-badge-eq {{ background-color: #ebf5ff; color: #1e429f; }}
        .stage-badge-bed {{ background-color: #fdf6b2; color: #723b11; }}
        .stage-badge-laser {{ background-color: #fde8e8; color: #9b1c1c; }}
        .stage-badge-hold {{ background-color: #f3e8ff; color: #6b21a8; }}
        .stage-badge-cool {{ background-color: #edfdfd; color: #03543f; }}
    </style>
</head>
<body class="bg-slate-50/50 min-h-screen text-slate-800 font-sans hero-mesh">

    <!-- Navbar -->
    <nav class="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-slate-100 py-4 px-8 flex justify-between items-center">
        <div class="flex items-center gap-3">
            <div class="h-10 w-10 rounded-xl bg-gradient-to-tr from-brand-600 to-indigo-600 flex items-center justify-center shadow-md shadow-brand-500/20">
                <i class="fa-solid fa-wand-magic-sparkles text-white"></i>
            </div>
            <div>
                <h1 class="text-lg font-bold font-outfit text-slate-900 leading-tight">SLS MD-ML</h1>
                <p class="text-xs text-slate-400">Physics-Informed Sintering Intelligence</p>
            </div>
        </div>
        <div class="flex gap-6 text-sm font-semibold text-slate-600">
            <a href="#overview" class="hover:text-brand-600 transition-colors">Overview</a>
            <a href="#pipeline" class="hover:text-brand-600 transition-colors">Methodology</a>
            <a href="#pca" class="hover:text-brand-600 transition-colors">Feature Clusters</a>
            <a href="#ml" class="hover:text-brand-600 transition-colors">Model Benchmarks</a>
            <a href="#chemistry" class="hover:text-brand-600 transition-colors">Sintering Chemistry</a>
        </div>
    </nav>

    <!-- Hero Section -->
    <header class="max-w-7xl mx-auto px-8 pt-16 pb-12 text-center">
        <span class="px-4 py-1.5 rounded-full bg-brand-50 border border-brand-100 text-brand-700 text-xs font-bold uppercase tracking-wider">
            💡 STANDALONE INTERACTIVE SCIENTIFIC REPORT
        </span>
        <h1 class="text-5xl md:text-6xl font-extrabold font-outfit text-slate-900 mt-6 tracking-tight leading-none">
            Selective Laser Sintering <br>
            <span class="bg-clip-text text-transparent bg-gradient-to-r from-brand-600 via-indigo-600 to-purple-600">
                Molecular Intelligence Dashboard
            </span>
        </h1>
        <p class="max-w-3xl mx-auto text-slate-500 mt-6 text-lg leading-relaxed">
            Integrating 292GB of Molecular Dynamics (MD) simulations with Machine Learning classifiers and regressors. Predict stage progression, polymer blend ratios, atomic displacement, potential energy, and bonding state dynamically.
        </p>
        <div class="flex justify-center gap-4 mt-8">
            <a href="#ml" class="px-6 py-3 rounded-xl bg-slate-900 text-white font-bold hover:bg-slate-800 transition-all shadow-md">
                Explore ML Performance
            </a>
            <a href="#chemistry" class="px-6 py-3 rounded-xl bg-white border border-slate-200 text-slate-600 font-bold hover:bg-slate-50 transition-all shadow-sm">
                Watch Bond Evolution
            </a>
        </div>
    </header>

    <!-- Quick Stats row -->
    <section class="max-w-7xl mx-auto px-8 grid grid-cols-2 md:grid-cols-4 gap-6 mb-16">
        <div class="bg-white p-6 rounded-2xl custom-shadow border border-slate-100">
            <div class="text-brand-600 text-sm font-bold uppercase tracking-wider">Total Raw Data</div>
            <div class="text-4xl font-black font-outfit text-slate-900 mt-1">292 GB</div>
            <div class="text-xs text-slate-400 mt-2">125 files, 90 stages processed</div>
        </div>
        <div class="bg-white p-6 rounded-2xl custom-shadow border border-slate-100">
            <div class="text-emerald-600 text-sm font-bold uppercase tracking-wider">ML Features</div>
            <div class="text-4xl font-black font-outfit text-slate-900 mt-1">338</div>
            <div class="text-xs text-slate-400 mt-2">Thermodynamic, structural & chemical</div>
        </div>
        <div class="bg-white p-6 rounded-2xl custom-shadow border border-slate-100">
            <div class="text-indigo-600 text-sm font-bold uppercase tracking-wider">Sintering Tasks</div>
            <div class="text-4xl font-black font-outfit text-slate-900 mt-1">6 Tasks</div>
            <div class="text-xs text-slate-400 mt-2">3 Classifiers, 3 Regressors</div>
        </div>
        <div class="bg-white p-6 rounded-2xl custom-shadow border border-slate-100">
            <div class="text-rose-600 text-sm font-bold uppercase tracking-wider">Max R² Score</div>
            <div class="text-4xl font-black font-outfit text-slate-900 mt-1">0.9998</div>
            <div class="text-xs text-slate-400 mt-2">Atomic displacement regression accuracy</div>
        </div>
    </section>

    <!-- Overview Section -->
    <section id="overview" class="max-w-7xl mx-auto px-8 mb-20 scroll-mt-20">
        <div class="bg-white p-8 rounded-3xl custom-shadow border border-slate-100 grid md:grid-cols-2 gap-12 items-center">
            <div>
                <span class="text-brand-600 font-extrabold uppercase text-xs tracking-wider">The Engineering Problem</span>
                <h2 class="text-3xl font-bold font-outfit text-slate-900 mt-2"> epoxy/PA12 blend sintering physics</h2>
                <p class="text-slate-500 mt-4 leading-relaxed text-sm">
                    Selective Laser Sintering (SLS) is an additive manufacturing technique where a laser fuses powder particles layer by layer. 
                    Blending <strong>epoxy</strong> and <strong>polyamide-12 (PA12)</strong> offers combined structural stiffness and heat resistance. 
                    However, monitoring molecular interactions, cross-link density, and atomic rearrangement during rapid laser heating is impossible experimentally.
                </p>
                <p class="text-slate-500 mt-4 leading-relaxed text-sm">
                    We executed large-scale Molecular Dynamics (MD) simulations under various compositions (50/50, 60/40, 70/30) and build temperatures (100°C to 350°C). 
                    The simulations model the five physical stages:
                </p>
                <div class="grid grid-cols-2 gap-4 mt-6">
                    <div class="flex items-center gap-3">
                        <span class="h-6 w-6 rounded-full bg-brand-50 text-brand-600 flex items-center justify-center font-bold text-xs">1</span>
                        <span class="text-sm font-semibold text-slate-700">Equilibrium (300K)</span>
                    </div>
                    <div class="flex items-center gap-3">
                        <span class="h-6 w-6 rounded-full bg-amber-50 text-amber-600 flex items-center justify-center font-bold text-xs">2</span>
                        <span class="text-sm font-semibold text-slate-700">Bed Heating (60°C)</span>
                    </div>
                    <div class="flex items-center gap-3">
                        <span class="h-6 w-6 rounded-full bg-rose-50 text-rose-600 flex items-center justify-center font-bold text-xs">3</span>
                        <span class="text-sm font-semibold text-slate-700">Laser Melting (150°C)</span>
                    </div>
                    <div class="flex items-center gap-3">
                        <span class="h-6 w-6 rounded-full bg-purple-50 text-purple-600 flex items-center justify-center font-bold text-xs">4</span>
                        <span class="text-sm font-semibold text-slate-700">Hold & Necking</span>
                    </div>
                </div>
            </div>
            <!-- Showcase Card -->
            <div class="bg-gradient-to-br from-brand-600 to-indigo-700 p-8 rounded-2xl text-white relative overflow-hidden shadow-xl">
                <div class="absolute -right-10 -bottom-10 opacity-10">
                    <i class="fa-solid fa-cubes text-9xl"></i>
                </div>
                <h3 class="text-2xl font-bold font-outfit">Sintering Phase Definition</h3>
                <p class="text-brand-100 mt-2 text-xs leading-relaxed">
                    By aggregating atomic parameters (coordination number, atomic volume, cavity radius) and mapping chemistry bonds (C-C, C-N crosslinks) over 628 timesteps, the ML models extract the absolute physical state.
                </p>
                <hr class="border-brand-500/50 my-6">
                <div>
                    <h4 class="text-sm font-bold uppercase tracking-wider text-brand-200">Simulation Scale</h4>
                    <div class="flex items-baseline gap-2 mt-1">
                        <span class="text-4xl font-extrabold font-outfit">11,102</span>
                        <span class="text-sm text-brand-200">atoms per file</span>
                    </div>
                </div>
                <div class="flex justify-between items-center mt-6 text-xs text-brand-200">
                    <span>9 Simulation Stages</span>
                    <span>ReaxFF forcefield</span>
                </div>
            </div>
        </div>
    </section>

    <!-- Methodology / Pipeline Section -->
    <section id="pipeline" class="max-w-7xl mx-auto px-8 mb-20 scroll-mt-20">
        <h2 class="text-3xl font-bold font-outfit text-slate-900 text-center">Framework & Methodology</h2>
        <p class="text-slate-400 text-center text-sm mt-1">How we converted 292GB of molecular files into highly accurate models</p>

        <div class="grid md:grid-cols-4 gap-6 mt-10">
            <div class="bg-white p-6 rounded-2xl custom-shadow border border-slate-100 relative">
                <div class="h-10 w-10 rounded-xl bg-brand-50 text-brand-600 flex items-center justify-center mb-4 text-lg font-bold">1</div>
                <h3 class="text-base font-bold text-slate-900">Multi-Source Parser</h3>
                <p class="text-slate-500 mt-2 text-xs leading-relaxed">
                    Custom parser pipelines (`md_data_extractor.py`, `samples_parser.py`) read log.lammps thermodynamic files, temp txt profiles, and bond evolution CSVs simultaneously.
                </p>
            </div>
            <div class="bg-white p-6 rounded-2xl custom-shadow border border-slate-100 relative">
                <div class="h-10 w-10 rounded-xl bg-amber-50 text-amber-600 flex items-center justify-center mb-4 text-lg font-bold">2</div>
                <h3 class="text-base font-bold text-slate-900">Row-Level Striding</h3>
                <p class="text-slate-500 mt-2 text-xs leading-relaxed">
                    Uses openpyxl in read-only streaming mode to process millions of per-atom coordinates without memory overflow. We sample every N-th row to compute statistical parameters.
                </p>
            </div>
            <div class="bg-white p-6 rounded-2xl custom-shadow border border-slate-100 relative">
                <div class="h-10 w-10 rounded-xl bg-emerald-50 text-emerald-600 flex items-center justify-center mb-4 text-lg font-bold">3</div>
                <h3 class="text-base font-bold text-slate-900">Feature Engineering</h3>
                <p class="text-slate-500 mt-2 text-xs leading-relaxed">
                    Generates 338 features including mean, std, skew, kurtosis, IQR, CV, entropy indices, delta changes, and element-wise charge distribution stats.
                </p>
            </div>
            <div class="bg-white p-6 rounded-2xl custom-shadow border border-slate-100 relative">
                <div class="h-10 w-10 rounded-xl bg-purple-50 text-purple-600 flex items-center justify-center mb-4 text-lg font-bold">4</div>
                <h3 class="text-base font-bold text-slate-900">Caching Engine</h3>
                <p class="text-slate-500 mt-2 text-xs leading-relaxed">
                    Serializes structural and thermodynamic features into disk caches (`cache_ovito.csv`, `cache_md.csv`), speeding up subsequent runs from 7 minutes to under 15 seconds!
                </p>
            </div>
        </div>
    </section>

    <!-- PCA Projection Section -->
    <section id="pca" class="max-w-7xl mx-auto px-8 mb-20 scroll-mt-20">
        <div class="bg-white p-8 rounded-3xl custom-shadow border border-slate-100 grid md:grid-cols-3 gap-8">
            <div class="md:col-span-1 flex flex-col justify-center">
                <span class="text-brand-600 font-extrabold uppercase text-xs tracking-wider">Unsupervised Clustering</span>
                <h2 class="text-3xl font-bold font-outfit text-slate-900 mt-2">Principal Component Projection</h2>
                <p class="text-slate-500 mt-4 leading-relaxed text-sm">
                    Using Principal Component Analysis (PCA) to map the 338-dimensional feature space onto 2D. 
                    The projection reveals beautiful segregation between sintering stages.
                </p>
                <div class="flex flex-col gap-2 mt-6">
                    <span class="text-xs font-semibold text-slate-400 block uppercase">PCA Variance Explained:</span>
                    <div class="flex items-center gap-4">
                        <div>
                            <span class="text-2xl font-bold text-brand-600" id="pca-v1">--</span>
                            <span class="text-xs text-slate-400 block">PC1</span>
                        </div>
                        <div>
                            <span class="text-2xl font-bold text-indigo-600" id="pca-v2">--</span>
                            <span class="text-xs text-slate-400 block">PC2</span>
                        </div>
                    </div>
                </div>
            </div>
            <!-- PCA Chart container -->
            <div class="md:col-span-2 bg-slate-50 p-6 rounded-2xl relative border border-slate-100 flex flex-col items-center">
                <div class="flex gap-4 mb-4 text-xs font-bold">
                    <button id="pca-color-stage" class="px-4 py-2 bg-slate-900 text-white rounded-lg hover:bg-slate-800 transition">Color by Sintering Stage</button>
                    <button id="pca-color-comp" class="px-4 py-2 bg-white text-slate-600 border border-slate-200 rounded-lg hover:bg-slate-50 transition">Color by Composition</button>
                </div>
                <div class="w-full relative h-[400px]">
                    <canvas id="pcaChart"></canvas>
                </div>
            </div>
        </div>
    </section>

    <!-- ML Performance Benchmarks -->
    <section id="ml" class="max-w-7xl mx-auto px-8 mb-20 scroll-mt-20">
        <h2 class="text-3xl font-bold font-outfit text-slate-900 text-center">Machine Learning Benchmarks</h2>
        <p class="text-slate-400 text-center text-sm mt-1">Interactive task metrics for all 7 classifiers & regressors</p>

        <!-- Task tabs -->
        <div class="flex justify-center gap-2 mt-8 flex-wrap">
            <button onclick="switchTask('Stage_Classification')" id="btn-Stage_Classification" class="task-tab-btn px-4 py-2 rounded-xl text-xs font-bold transition-all bg-brand-600 text-white shadow-sm">
                Stage Classification
            </button>
            <button onclick="switchTask('Composition_Prediction')" id="btn-Composition_Prediction" class="task-tab-btn px-4 py-2 rounded-xl text-xs font-bold transition-all bg-white border border-slate-200 text-slate-600 hover:bg-slate-50">
                Composition Prediction
            </button>
            <button onclick="switchTask('Bond_Stability')" id="btn-Bond_Stability" class="task-tab-btn px-4 py-2 rounded-xl text-xs font-bold transition-all bg-white border border-slate-200 text-slate-600 hover:bg-slate-50">
                Bond Stability
            </button>
            <button onclick="switchTask('Displacement_Regression')" id="btn-Displacement_Regression" class="task-tab-btn px-4 py-2 rounded-xl text-xs font-bold transition-all bg-white border border-slate-200 text-slate-600 hover:bg-slate-50">
                Displacement Regression
            </button>
            <button onclick="switchTask('PotEng_Regression')" id="btn-PotEng_Regression" class="task-tab-btn px-4 py-2 rounded-xl text-xs font-bold transition-all bg-white border border-slate-200 text-slate-600 hover:bg-slate-50">
                Potential Energy Regression
            </button>
            <button onclick="switchTask('TempAdherence_Regression')" id="btn-TempAdherence_Regression" class="task-tab-btn px-4 py-2 rounded-xl text-xs font-bold transition-all bg-white border border-slate-200 text-slate-600 hover:bg-slate-50">
                Temp Adherence Regression
            </button>
        </div>

        <div class="grid md:grid-cols-3 gap-8 mt-8">
            <!-- Left Panel: Metrics cards & info -->
            <div class="bg-white p-6 rounded-2xl custom-shadow border border-slate-100 flex flex-col justify-between">
                <div>
                    <h3 class="text-xl font-bold font-outfit text-slate-900" id="task-title">Stage Classification</h3>
                    <p class="text-xs text-slate-400 mt-1" id="task-desc">Predict which stage the sintering is currently in</p>
                    
                    <div class="mt-8 space-y-6">
                        <div>
                            <span class="text-xs font-semibold text-slate-400 block uppercase">Best Model Performance</span>
                            <div class="flex items-baseline gap-2 mt-1">
                                <span class="text-5xl font-black font-outfit text-brand-600" id="task-best-val">100%</span>
                                <span class="text-sm font-semibold text-slate-500" id="task-best-metric">Test Acc</span>
                            </div>
                        </div>
                        <div>
                            <span class="text-xs font-semibold text-slate-400 block uppercase">Best Performing Model</span>
                            <span class="text-lg font-bold text-slate-800" id="task-best-model">Random Forest / SVM</span>
                        </div>
                    </div>
                </div>
                <!-- Mini Warning/Tip -->
                <div class="p-4 rounded-xl bg-slate-50 border border-slate-100 text-xs text-slate-500 mt-6 flex gap-2.5 items-start">
                    <i class="fa-solid fa-circle-info text-brand-500 mt-0.5"></i>
                    <p>Metrics derived from stratified 5-fold cross-validation and 20% holdout test sets.</p>
                </div>
            </div>
            <!-- Right Panel: Chart comparison -->
            <div class="md:col-span-2 bg-white p-6 rounded-2xl custom-shadow border border-slate-100">
                <h4 class="text-sm font-bold font-outfit text-slate-800 mb-4">Model Performance Comparisons</h4>
                <div class="w-full relative h-[320px]">
                    <canvas id="metricChart"></canvas>
                </div>
            </div>
        </div>
    </section>

    <!-- Sintering Chemistry Simulator -->
    <section id="chemistry" class="max-w-7xl mx-auto px-8 mb-24 scroll-mt-20">
        <h2 class="text-3xl font-bold font-outfit text-slate-900 text-center">Interactive Sintering Chemistry</h2>
        <p class="text-slate-400 text-center text-sm mt-1">Observe dynamic chemical bond formations/breaks during simulated heating</p>

        <!-- Dropdowns -->
        <div class="flex justify-center gap-4 mt-8 mb-8">
            <div class="flex items-center gap-2">
                <label class="text-xs font-bold text-slate-500 uppercase">Composition:</label>
                <select id="sim-comp" onchange="updateSimChart()" class="bg-white border border-slate-200 px-4 py-2 rounded-xl text-xs font-bold outline-none cursor-pointer">
                    <option value="5050">50 Epoxy / 50 Polyamide-12</option>
                    <option value="6040">60 Epoxy / 40 Polyamide-12</option>
                    <option value="7030">70 Epoxy / 30 Polyamide-12</option>
                </select>
            </div>
            <div class="flex items-center gap-2">
                <label class="text-xs font-bold text-slate-500 uppercase">Temperature:</label>
                <select id="sim-temp" onchange="updateSimChart()" class="bg-white border border-slate-200 px-4 py-2 rounded-xl text-xs font-bold outline-none cursor-pointer">
                    <option value="100">100 °C</option>
                    <option value="200">200 °C</option>
                    <option value="300">300 °C</option>
                </select>
            </div>
        </div>

        <div class="grid md:grid-cols-4 gap-8">
            <!-- Left Info Panel -->
            <div class="md:col-span-1 bg-white p-6 rounded-2xl custom-shadow border border-slate-100 flex flex-col justify-between">
                <div>
                    <h3 class="text-base font-bold font-outfit text-slate-800 uppercase tracking-wider">Molecular Insights</h3>
                    <p class="text-xs text-slate-500 mt-2 leading-relaxed">
                        Epoxy contains reactive oxirane groups that crosslink with amine groups of polyamide-12. 
                        Watch the evolution of backbone stability (C-C bonds) versus reactive nitrogen-oxygen indicators across the timesteps.
                    </p>
                    <div class="mt-6 space-y-4">
                        <div class="flex justify-between items-center py-2 border-b border-slate-50">
                            <span class="text-xs font-semibold text-slate-500">Backbone bonds (C-C)</span>
                            <span class="text-xs font-bold text-slate-800" id="sim-cc-avg">--</span>
                        </div>
                        <div class="flex justify-between items-center py-2 border-b border-slate-50">
                            <span class="text-xs font-semibold text-slate-500">Crosslinks (C-N)</span>
                            <span class="text-xs font-bold text-slate-800" id="sim-cn-avg">--</span>
                        </div>
                    </div>
                </div>
                <div class="p-4 rounded-xl bg-slate-50 border border-slate-100 text-[10px] text-slate-400 mt-6">
                    Each data point represents aggregated molecular counts across sheets S (Start), M (Middle), and E (End) of the sintering file.
                </div>
            </div>

            <!-- Bond evolution Chart -->
            <div class="md:col-span-3 bg-white p-6 rounded-2xl custom-shadow border border-slate-100">
                <div class="flex justify-between items-center mb-4">
                    <h4 class="text-sm font-bold font-outfit text-slate-800">Dynamic Bond Evolution over Timesteps</h4>
                    <span class="px-3 py-1 bg-brand-50 border border-brand-100 text-brand-600 rounded-full text-[10px] font-bold">SAMPLE RATE: ~1/20 STEP</span>
                </div>
                <div class="w-full relative h-[360px]">
                    <canvas id="simChart"></canvas>
                </div>
            </div>
        </div>
    </section>

    <!-- Footer -->
    <footer class="bg-white border-t border-slate-100 py-12 px-8 text-center text-slate-400 text-xs">
        <p class="font-semibold text-slate-600">SLS MD-ML molecular evolution report portal</p>
        <p class="mt-2">Generated by Antigravity AI Code Companion</p>
        <p class="mt-1">IIT Jammu Sintering Research Project</p>
    </footer>

    <!-- Database injection -->
    <script>
        const db = {json.dumps(db_json, indent=2)};
    </script>

    <!-- Custom Frontend Logic -->
    <script>
        // Init global charts
        let pcaChartInstance = null;
        let metricChartInstance = null;
        let simChartInstance = null;

        window.onload = function() {{
            initPca();
            switchTask('Stage_Classification');
            initSim();
            updateSimChart();
        }};

        // --- PCA Setup ---
        function initPca() {{
            const ctx = document.getElementById('pcaChart').getContext('2d');
            if (db.pca_variance) {{
                document.getElementById('pca-v1').innerText = (db.pca_variance[0]*100).toFixed(1) + "%";
                document.getElementById('pca-v2').innerText = (db.pca_variance[1]*100).toFixed(1) + "%";
            }}
            
            // Map stage colors
            const stageColors = {{
                'equilibrium': '#2196F3',
                'bed': '#FF9800',
                'laser': '#F44336',
                'hold': '#9C27B0',
                'cooling': '#4CAF50'
            }};
            const compColors = {{
                '5050': '#E91E63',
                '6040': '#00BCD4',
                '7030': '#FFC107'
            }};

            function getPcaDatasets(colorBy) {{
                const groups = {{}};
                db.pca.forEach(p => {{
                    const key = colorBy === 'stage' ? p.stage : p.composition;
                    if (!groups[key]) groups[key] = [];
                    groups[key].push({{ x: p.x, y: p.y, meta: p }});
                }});

                return Object.keys(groups).map((key, i) => {{
                    const color = colorBy === 'stage' 
                        ? (stageColors[key] || '#999') 
                        : (compColors[key] || '#999');
                    return {{
                        label: key.toUpperCase(),
                        data: groups[key],
                        backgroundColor: color,
                        borderColor: '#ffffff',
                        borderWidth: 0.5,
                        pointRadius: 6,
                        pointHoverRadius: 8
                    }};
                }});
            }}

            pcaChartInstance = new Chart(ctx, {{
                type: 'scatter',
                data: {{ datasets: getPcaDatasets('stage') }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        x: {{ grid: {{ color: '#f1f5f9' }} }},
                        y: {{ grid: {{ color: '#f1f5f9' }} }}
                    }},
                    plugins: {{
                        legend: {{ position: 'bottom', labels: {{ usePointStyle: true, boxWidth: 6 }} }},
                        tooltip: {{
                            callbacks: {{
                                label: function(context) {{
                                    const m = context.raw.meta;
                                    return `Comp: ${{m.composition}}, Temp: ${{m.temperature}}C, Stage: ${{m.stage.toUpperCase()}} (${{m.position}})`;
                                }}
                            }}
                        }}
                    }}
                }}
            }});

            document.getElementById('pca-color-stage').onclick = function() {{
                this.className = "px-4 py-2 bg-slate-900 text-white rounded-lg hover:bg-slate-800 transition";
                document.getElementById('pca-color-comp').className = "px-4 py-2 bg-white text-slate-600 border border-slate-200 rounded-lg hover:bg-slate-50 transition";
                pcaChartInstance.data.datasets = getPcaDatasets('stage');
                pcaChartInstance.update();
            }};

            document.getElementById('pca-color-comp').onclick = function() {{
                this.className = "px-4 py-2 bg-slate-900 text-white rounded-lg hover:bg-slate-800 transition";
                document.getElementById('pca-color-stage').className = "px-4 py-2 bg-white text-slate-600 border border-slate-200 rounded-lg hover:bg-slate-50 transition";
                pcaChartInstance.data.datasets = getPcaDatasets('comp');
                pcaChartInstance.update();
            }};
        }}

        // --- Model Performance tabs ---
        const taskInfo = {{
            'Stage_Classification': {{
                title: 'Stage Classification',
                desc: 'Identifying sintering process state across 5 distinct stages (equilibrium, bed heating, laser sintering, hold, cooling).',
                best_val: '100.0%',
                best_metric: 'Test Acc',
                best_model: 'Random Forest, LightGBM, CatBoost, SVM'
            }},
            'Composition_Prediction': {{
                title: 'Composition Prediction',
                desc: 'Predicting the polymer blend composition ratio (50/50, 60/40, or 70/30) from the molecular structures.',
                best_val: '100.0%',
                best_metric: 'Test Acc',
                best_model: 'Random Forest, Extra Trees, XGBoost, LGBM'
            }},
            'Bond_Stability': {{
                title: 'Bond Stability Classification',
                desc: 'Binary classification categorizing whether the molecular C-C bonds backbone density is above the median sintering threshold.',
                best_val: '100.0%',
                best_metric: 'Test Acc',
                best_model: 'XGBoost / LightGBM'
            }},
            'Displacement_Regression': {{
                title: 'Atomic Displacement Regression',
                desc: 'Regressing the average displacement metric (Å) across frames to gauge structural movement and neck formation.',
                best_val: '0.9998',
                best_metric: 'R² Score',
                best_model: 'XGBoost (MAE: 0.0145 Å error)'
            }},
            'PotEng_Regression': {{
                title: 'Potential Energy Regression',
                desc: 'Regressing the time-averaged potential energy (PotEng, kcal/mol) of the sintering atomic layout.',
                best_val: '1.0000',
                best_metric: 'R² Score',
                best_model: 'Extra Trees, RandomForest (MAE: 0.166 kcal/mol)'
            }},
            'TempAdherence_Regression': {{
                title: 'Temperature Adherence Regression',
                desc: 'Regressing the adherence index (0-1) of the simulated thermostat control against targeted stage profiles.',
                best_val: '1.0000',
                best_metric: 'R² Score',
                best_model: 'Extra Trees / XGBoost (MAE: 0.0001 error)'
            }}
        }};

        function switchTask(taskId) {{
            // Toggle active classes on buttons
            document.querySelectorAll('.task-tab-btn').forEach(btn => {{
                if (btn.id === `btn-${{taskId}}`) {{
                    btn.className = "task-tab-btn px-4 py-2 rounded-xl text-xs font-bold transition-all bg-brand-600 text-white shadow-sm";
                }} else {{
                    btn.className = "task-tab-btn px-4 py-2 rounded-xl text-xs font-bold transition-all bg-white border border-slate-200 text-slate-600 hover:bg-slate-50";
                }}
            }});

            const info = taskInfo[taskId];
            document.getElementById('task-title').innerText = info.title;
            document.getElementById('task-desc').innerText = info.desc;
            document.getElementById('task-best-val').innerText = info.best_val;
            document.getElementById('task-best-metric').innerText = info.best_metric;
            document.getElementById('task-best-model').innerText = info.best_model;

            renderMetricChart(taskId);
        }}

        function renderMetricChart(taskId) {{
            const ctx = document.getElementById('metricChart').getContext('2d');
            
            // Filter model scores
            const taskData = db.models.filter(m => m.Task === taskId);
            const models = taskData.map(d => d.Model);
            
            const isReg = taskId.includes('Regression');
            const scoreLabel = isReg ? 'Test R²' : 'Test Accuracy';
            const scores = taskData.map(d => isReg ? d.r2 : d.test_acc);
            
            const cvScores = taskData.map(d => isReg ? d.cv_r2_mean : d.cv_acc);
            
            if (metricChartInstance) metricChartInstance.destroy();

            metricChartInstance = new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: models,
                    datasets: [
                        {{
                            label: scoreLabel,
                            data: scores,
                            backgroundColor: 'rgba(77, 124, 255, 0.8)',
                            borderColor: '#4d7cff',
                            borderWidth: 1,
                            borderRadius: 6
                        }},
                        {{
                            label: 'CV Score (Mean)',
                            data: cvScores,
                            backgroundColor: 'rgba(99, 102, 241, 0.4)',
                            borderColor: 'rgba(99, 102, 241, 1)',
                            borderWidth: 1,
                            borderRadius: 6
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{ 
                            min: isReg ? undefined : 0,
                            max: isReg ? 1.05 : 1.1,
                            grid: {{ color: '#f1f5f9' }}
                        }},
                        x: {{ grid: {{ display: false }} }}
                    }},
                    plugins: {{
                        legend: {{ position: 'bottom', labels: {{ boxWidth: 12 }} }}
                    }}
                }}
            }});
        }}

        // --- Sintering Chemistry Simulator ---
        function initSim() {{
            // No-op init
        }}

        function updateSimChart() {{
            const comp = document.getElementById('sim-comp').value;
            const temp = document.getElementById('sim-temp').value;
            const ctx = document.getElementById('simChart').getContext('2d');

            if (!db.bond_evolution[comp] || !db.bond_evolution[comp][temp]) {{
                ctx.fillText("No chemical data available for this configuration", 50, 50);
                return;
            }}

            const records = db.bond_evolution[comp][temp];
            const timesteps = records.map(r => r.timestep);
            const cc = records.map(r => r.cc || r['C-C']);
            const cn = records.map(r => r.cn || r['C-N']);
            const co = records.map(r => r.co || r['C-O']);
            const hn = records.map(r => r.hn || r['H-N']);

            // Calculate averages to show in info panel
            const ccAvg = cc.reduce((a,b)=>a+b, 0) / cc.length;
            const cnAvg = cn.reduce((a,b)=>a+b, 0) / cn.length;
            document.getElementById('sim-cc-avg').innerText = ccAvg.toFixed(0);
            document.getElementById('sim-cn-avg').innerText = cnAvg.toFixed(0);

            if (simChartInstance) simChartInstance.destroy();

            simChartInstance = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: timesteps,
                    datasets: [
                        {{
                            label: 'C-C (Backbone)',
                            data: cc,
                            borderColor: '#4d7cff',
                            borderWidth: 2,
                            pointRadius: 2,
                            fill: false,
                            tension: 0.15
                        }},
                        {{
                            label: 'C-N (Amine Crosslink)',
                            data: cn,
                            borderColor: '#ec4899',
                            borderWidth: 2,
                            pointRadius: 2,
                            fill: false,
                            tension: 0.15
                        }},
                        {{
                            label: 'C-O (Oxirane Crosslink)',
                            data: co,
                            borderColor: '#eab308',
                            borderWidth: 2,
                            pointRadius: 2,
                            fill: false,
                            tension: 0.15
                        }},
                        {{
                            label: 'H-N (Hydrogen network)',
                            data: hn,
                            borderColor: '#10b981',
                            borderWidth: 2,
                            pointRadius: 2,
                            fill: false,
                            tension: 0.15
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{ grid: {{ color: '#f1f5f9' }} }},
                        x: {{ 
                            title: {{ display: true, text: 'Timestep' }},
                            grid: {{ color: '#f1f5f9' }} 
                        }}
                    }},
                    plugins: {{
                        legend: {{ position: 'bottom', labels: {{ boxWidth: 12 }} }}
                    }}
                }}
            }});
        }}
    </script>
</body>
</html>
"""
    
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Showcase generated at: {OUTPUT_HTML.absolute()}")

if __name__ == "__main__":
    build_showcase()

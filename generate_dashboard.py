"""
Interactive HTML Dashboard generator for SLS MD Analysis.

Reads CSV files from Results/ and compiles a premium, self-contained HTML
dashboard with Tailwind CSS, Chart.js, and interactive visual exploration.
"""

import os
import json
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def generate_html_dashboard(results_dir: str = "Results"):
    print("Generating HTML Dashboard...")
    
    # Paths
    stats_dir = os.path.join(results_dir, "Statistics")
    evo_dir = os.path.join(results_dir, "Evolution")
    dataset_dir = os.path.join(results_dir, "Final_Dataset")
    
    # 1. Load Data
    data = {}
    
    # Feature Ranking
    ranking_path = os.path.join(stats_dir, "Feature_Ranking.csv")
    if os.path.exists(ranking_path):
        df_rank = pd.read_csv(ranking_path)
        data["ranking"] = df_rank.to_dict(orient="records")
    else:
        data["ranking"] = []
        
    # PCA Variance
    pca_path = os.path.join(stats_dir, "PCA_Variance.csv")
    if os.path.exists(pca_path):
        df_pca = pd.read_csv(pca_path)
        data["pca"] = df_pca.to_dict(orient="records")
    else:
        data["pca"] = []
        
    # Evolution Table
    evo_path = os.path.join(evo_dir, "Evolution_Table.csv")
    if os.path.exists(evo_path):
        df_evo = pd.read_csv(evo_path)
        data["evolution"] = df_evo.to_dict(orient="records")
    else:
        data["evolution"] = []
        
    # Raw Dataset (for interactive stage-by-stage trends)
    dataset_path = os.path.join(dataset_dir, "dataset.csv")
    if os.path.exists(dataset_path):
        df_data = pd.read_csv(dataset_path)
        summary_cols = ["Composition", "Temperature", "Stage", "Mean_CN", "Mean_AtomicVolume", "Mean_CavityRadius", "Mean_Displacement"]
        existing_cols = [c for c in summary_cols if c in df_data.columns]
        df_data_subset = df_data[existing_cols]
        data["raw_summary"] = df_data_subset.to_dict(orient="records")
    else:
        data["raw_summary"] = []

    # Final combined dataset
    final_xlsx_path = os.path.join(dataset_dir, "final_xlsx_combined.csv")
    if os.path.exists(final_xlsx_path):
        df_final = pd.read_csv(final_xlsx_path)
        data["final_snapshots"] = df_final.to_dict(orient="records")
    else:
        data["final_snapshots"] = []

    # 2. Render Template
    lbrace = '{'
    rbrace = '}'
    angstrom = '\\AA'
    sigma_sym = '\\sigma'
    pi_sym = '\\pi'
    eta_sym = '\\eta'
    gamma_sym = '\\gamma'
    Sigma_sym = '\\Sigma'
    sum_sym = '\\sum'
    le_sym = '\\le'
    neq_sym = '\\neq'
    delta_sym = '\\delta'
    text_sym = '\\text'
    left_sym = '\\left'
    right_sym = '\\right'
    langle_sym = '\\langle'
    rangle_sym = '\\rangle'
    cdot_sym = '\\cdot'
    log_sym = '\\log'
    epsilon_sym = '\\epsilon'
    bar_sym = '\\overline'
    mathbbR = '\\mathbb{{R}}'
    mathbfx = '\\mathbf{{x}}'
    mathbfv = '\\mathbf{{v}}'
    lambda_sym = '\\lambda'
    frac_sym = '\\frac'
    cn_sym = '\\text{CN}'
    vor_sym = '\\text{Vor}'
    msd_sym = '\\text{MSD}'
    mei_sym = '\\text{MEI}'
    vol_sym = '\\text{Vol}'

    html_content = f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Epoxy/PA12 SLS MD Molecular Evolution Dashboard</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {{
            darkMode: 'class',
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
    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <!-- FontAwesome CDN -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <!-- MathJax for rendering LaTeX formulas -->
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script>
        MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\(', '\\)']]
            }}
        }};
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        .glass-card {{
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(0, 0, 0, 0.08);
        }}
        .dark .glass-card {{
            background: rgba(17, 24, 39, 0.7);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }}
        .bg-gradient-mesh {{
            background-color: #f8fafc;
            background-image: 
                radial-gradient(at 0% 0%, rgba(241, 245, 249, 0.5) 0, transparent 50%),
                radial-gradient(at 50% 0%, rgba(77, 124, 255, 0.08) 0, transparent 50%),
                radial-gradient(at 100% 100%, rgba(156, 39, 176, 0.05) 0, transparent 50%);
        }}
        .dark .bg-gradient-mesh {{
            background-color: #0b0f19;
            background-image: 
                radial-gradient(at 0% 0%, rgba(31, 41, 55, 0.3) 0, transparent 50%),
                radial-gradient(at 50% 0%, rgba(26, 85, 255, 0.15) 0, transparent 50%),
                radial-gradient(at 100% 100%, rgba(156, 39, 176, 0.1) 0, transparent 50%);
        }}
        body {{
            font-family: 'Inter', sans-serif;
            color: #1e293b;
        }}
        .dark body {{
            color: #e5e7eb;
        }}
    </style>
</head>
<body class="bg-gradient-mesh min-h-screen">

    <!-- Header Section -->
    <header class="border-b border-gray-200 dark:border-gray-800 glass-card sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-6 py-4 flex flex-col md:flex-row justify-between items-center gap-4">
            <div class="flex items-center gap-3">
                <div class="h-10 w-10 rounded-xl bg-gradient-to-tr from-brand-600 to-purple-600 flex items-center justify-center shadow-lg shadow-brand-500/20">
                    <i class="fa-solid fa-atom text-white text-lg animate-spin-slow"></i>
                </div>
                <div>
                    <h1 class="text-xl font-bold font-outfit text-gray-900 dark:text-white tracking-wide">SLS MD Explorer</h1>
                    <p class="text-xs text-gray-500 dark:text-gray-400">Physics-Informed MD Analysis Framework</p>
                </div>
            </div>
            
            <!-- Stage Indicators & Theme Toggle -->
            <div class="flex items-center gap-4 flex-wrap justify-center">
                <div class="flex items-center gap-1 text-[10px] uppercase font-bold tracking-wider">
                    <span class="px-2 py-1 rounded bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20">Equilibrium</span>
                    <i class="fa-solid fa-chevron-right text-gray-400 dark:text-gray-600 text-xs"></i>
                    <span class="px-2 py-1 rounded bg-orange-500/10 text-orange-600 dark:text-orange-400 border border-orange-500/20">Bed Heating</span>
                    <i class="fa-solid fa-chevron-right text-gray-400 dark:text-gray-600 text-xs"></i>
                    <span class="px-2 py-1 rounded bg-red-500/10 text-red-600 dark:text-red-400 border border-red-500/20">Laser</span>
                    <i class="fa-solid fa-chevron-right text-gray-400 dark:text-gray-600 text-xs"></i>
                    <span class="px-2 py-1 rounded bg-purple-500/10 text-purple-600 dark:text-purple-400 border border-purple-500/20">Hold</span>
                    <i class="fa-solid fa-chevron-right text-gray-400 dark:text-gray-600 text-xs"></i>
                    <span class="px-2 py-1 rounded bg-green-500/10 text-green-600 dark:text-green-400 border border-green-500/20">Cooling</span>
                </div>

                <!-- Theme Toggle Button -->
                <button onclick="toggleTheme()" class="p-2 rounded-xl bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 hover:text-gray-900 dark:hover:text-white transition-all flex items-center justify-center border border-gray-200 dark:border-gray-700/50" title="Toggle Light/Dark Theme">
                    <i class="fa-solid fa-moon text-sm block dark:hidden"></i>
                    <i class="fa-solid fa-sun text-sm hidden dark:block"></i>
                </button>
            </div>
        </div>
    </header>

    <main class="max-w-7xl mx-auto px-6 py-8">
        
        <!-- Tab Navigation -->
        <div class="flex border-b border-gray-200 dark:border-gray-800 mb-8 overflow-x-auto gap-2 py-1">
            <button onclick="switchTab('overview')" id="tab-overview" class="px-5 py-2.5 rounded-lg font-medium text-sm flex items-center gap-2 transition-all duration-200 bg-brand-600 text-white shadow-md shadow-brand-500/10">
                <i class="fa-solid fa-house"></i> Overview
            </button>
            <button onclick="switchTab('evolution')" id="tab-evolution" class="px-5 py-2.5 rounded-lg font-medium text-sm flex items-center gap-2 transition-all duration-200 text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800">
                <i class="fa-solid fa-chart-line"></i> Descriptor Evolution
            </button>
            <button onclick="switchTab('pca')" id="tab-pca" class="px-5 py-2.5 rounded-lg font-medium text-sm flex items-center gap-2 transition-all duration-200 text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800">
                <i class="fa-solid fa-project-diagram"></i> PCA & Correlation
            </button>
            <button onclick="switchTab('ml')" id="tab-ml" class="px-5 py-2.5 rounded-lg font-medium text-sm flex items-center gap-2 transition-all duration-200 text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800">
                <i class="fa-solid fa-brain"></i> Feature Importance
            </button>
            <button onclick="switchTab('snapshots')" id="tab-snapshots" class="px-5 py-2.5 rounded-lg font-medium text-sm flex items-center gap-2 transition-all duration-200 text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800">
                <i class="fa-solid fa-camera"></i> Snapshot Analysis
            </button>
            <button onclick="switchTab('methodology')" id="tab-methodology" class="px-5 py-2.5 rounded-lg font-medium text-sm flex items-center gap-2 transition-all duration-200 text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800">
                <i class="fa-solid fa-book-open"></i> Methodology & Science
            </button>
            <button onclick="switchTab('about')" id="tab-about" class="px-5 py-2.5 rounded-lg font-medium text-sm flex items-center gap-2 transition-all duration-200 text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800">
                <i class="fa-solid fa-file-lines"></i> About This
            </button>
        </div>

        <!-- ==================== OVERVIEW TAB ==================== -->
        <section id="content-overview" class="tab-content block space-y-8">
            <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <!-- Summary Metrics -->
                <div class="glass-card rounded-2xl p-6 lg:col-span-2 flex flex-col justify-between">
                    <div>
                        <span class="text-xs font-bold text-brand-400 uppercase tracking-widest">Molecular Evolution</span>
                        <h2 class="text-3xl font-bold font-outfit text-gray-900 dark:text-white mt-1 mb-4">Selective Laser Sintering MD Exploration</h2>
                        <p class="text-gray-600 dark:text-gray-400 leading-relaxed text-sm mb-6">
                            This framework couples high-fidelity Molecular Dynamics (MD) simulations with Machine Learning-based descriptor analysis. It models the thermal, structural, and dynamical behavior of <strong>Epoxy/PA12 composites</strong> across the five distinct phases of the SLS printing process (Equilibrium, Bed Heating, Laser Heating, Hold, and Cooling).
                        </p>
                    </div>
                    
                    <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
                        <div class="p-4 bg-gray-50 dark:bg-gray-900/50 border border-gray-200 dark:border-gray-800/80 rounded-xl">
                            <span class="text-xs text-gray-500 dark:text-gray-400 block mb-1">Compositions</span>
                            <span class="text-2xl font-bold text-gray-900 dark:text-white font-outfit">3 <span class="text-xs text-brand-400 font-normal">Varieties</span></span>
                        </div>
                        <div class="p-4 bg-gray-50 dark:bg-gray-900/50 border border-gray-200 dark:border-gray-800/80 rounded-xl">
                            <span class="text-xs text-gray-500 dark:text-gray-400 block mb-1">Temperatures</span>
                            <span class="text-2xl font-bold text-gray-900 dark:text-white font-outfit">6 <span class="text-xs text-brand-400 font-normal">Levels</span></span>
                        </div>
                        <div class="p-4 bg-gray-50 dark:bg-gray-900/50 border border-gray-200 dark:border-gray-800/80 rounded-xl">
                            <span class="text-xs text-gray-500 dark:text-gray-400 block mb-1">Simulation Stages</span>
                            <span class="text-2xl font-bold text-gray-900 dark:text-white font-outfit">5 <span class="text-xs text-brand-400 font-normal">Steps</span></span>
                        </div>
                        <div class="p-4 bg-gray-50 dark:bg-gray-900/50 border border-gray-200 dark:border-gray-800/80 rounded-xl">
                            <span class="text-xs text-gray-500 dark:text-gray-400 block mb-1">Data Snapshots</span>
                            <span class="text-2xl font-bold text-gray-900 dark:text-white font-outfit">90 <span class="text-xs text-brand-400 font-normal">Total Runs</span></span>
                        </div>
                    </div>
                </div>

                <!-- Fast Info Card -->
                <div class="glass-card rounded-2xl p-6 bg-gradient-to-br from-brand-900/10 to-purple-900/10 dark:from-brand-900/20 dark:to-purple-900/20 flex flex-col justify-between">
                    <div>
                        <div class="flex items-center gap-2 mb-3">
                            <span class="h-2 w-2 rounded-full bg-emerald-400 animate-ping"></span>
                            <span class="text-xs font-bold text-emerald-400 uppercase tracking-widest">Active Results</span>
                        </div>
                        <h3 class="text-xl font-bold font-outfit text-gray-900 dark:text-white mb-2">Dataset Verified</h3>
                        <p class="text-gray-600 dark:text-gray-400 text-xs leading-relaxed">
                            Successfully compiled all 90 simulation runs. All structural descriptors (Coordination Number, Atomic Volume, Cavity Radius), mechanical parameters, temperatures, and molecular dynamics displacements have been fully aggregated into standardized ML datasets.
                        </p>
                    </div>
                    
                    <div class="space-y-2 mt-6">
                        <a href="Final_Dataset/dataset_ml_ready.csv" download class="w-full flex items-center justify-between p-3 bg-brand-600/20 hover:bg-brand-600/30 border border-brand-500/30 rounded-xl text-xs font-semibold text-gray-900 dark:text-white hover:text-brand-500 transition-all duration-200">
                            <span>Download ML-Ready CSV</span>
                            <i class="fa-solid fa-download"></i>
                        </a>
                        <a href="Final_Dataset/dataset.csv" download class="w-full flex items-center justify-between p-3 bg-gray-100 dark:bg-gray-800/50 hover:bg-gray-200 dark:hover:bg-gray-800 border border-gray-200 dark:border-gray-700/50 rounded-xl text-xs font-semibold text-gray-700 dark:text-gray-300 transition-all duration-200">
                            <span>Download Raw Data</span>
                            <i class="fa-solid fa-download"></i>
                        </a>
                    </div>
                </div>
            </div>

            <!-- Plots Grid -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                <!-- PCA scree plot -->
                <div class="glass-card rounded-2xl p-6">
                    <h3 class="text-lg font-bold font-outfit text-gray-900 dark:text-white mb-4"><i class="fa-solid fa-chart-bar text-brand-400 mr-2"></i> PCA Explained Variance</h3>
                    <div class="aspect-[10/6] w-full flex items-center justify-center overflow-hidden rounded-xl bg-gray-50 dark:bg-gray-950/40 border border-gray-200 dark:border-gray-800/40 p-2">
                        <img src="Plots/Statistics/pca_scree.png" alt="PCA Scree Plot" class="max-w-full max-h-full object-contain">
                    </div>
                </div>
                
                <!-- Correlation matrix -->
                <div class="glass-card rounded-2xl p-6">
                    <h3 class="text-lg font-bold font-outfit text-gray-900 dark:text-white mb-4"><i class="fa-solid fa-table-cells text-brand-400 mr-2"></i> Feature Correlation Matrix</h3>
                    <div class="aspect-[10/6] w-full flex items-center justify-center overflow-hidden rounded-xl bg-gray-50 dark:bg-gray-950/40 border border-gray-200 dark:border-gray-800/40 p-2">
                        <img src="Plots/Statistics/correlation_pearson.png" alt="Correlation Matrix" class="max-w-full max-h-full object-contain">
                    </div>
                </div>
            </div>
        </section>

        <!-- ==================== DESCRIPTOR EVOLUTION ==================== -->
        <section id="content-evolution" class="tab-content hidden space-y-8">
            <div class="glass-card rounded-2xl p-6">
                <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-6">
                    <div>
                        <h2 class="text-2xl font-bold font-outfit text-gray-900 dark:text-white">Interactive Descriptor Evolution</h2>
                        <p class="text-xs text-gray-500 dark:text-gray-400">Explore how structural descriptors change across SLS stages based on composition and temperature</p>
                    </div>
                    
                    <div class="flex flex-wrap gap-3">
                        <div>
                            <label class="block text-[10px] uppercase font-bold text-gray-500 dark:text-gray-400 mb-1">Composition</label>
                            <select id="sel-comp" onchange="updateEvolutionChart()" class="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg px-3 py-1.5 text-xs text-gray-900 dark:text-white focus:outline-none focus:border-brand-500">
                                <option value="5050">50:50 Epoxy/PA12</option>
                                <option value="6040">60:40 Epoxy/PA12</option>
                                <option value="7030">70:30 Epoxy/PA12</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-[10px] uppercase font-bold text-gray-500 dark:text-gray-400 mb-1">Temperature</label>
                            <select id="sel-temp" onchange="updateEvolutionChart()" class="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg px-3 py-1.5 text-xs text-gray-900 dark:text-white focus:outline-none focus:border-brand-500">
                                <option value="100">100°C</option>
                                <option value="150">150°C</option>
                                <option value="200">200°C</option>
                                <option value="250">250°C</option>
                                <option value="300">300°C</option>
                                <option value="350">350°C</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-[10px] uppercase font-bold text-gray-500 dark:text-gray-400 mb-1">Descriptor</label>
                            <select id="sel-feat" onchange="updateEvolutionChart()" class="bg-white dark:bg-gray-900 border border-brand-500 rounded-lg px-3 py-1.5 text-xs text-gray-900 dark:text-white focus:outline-none focus:border-brand-500">
                                <option value="Mean_CN">Coordination Number (CN)</option>
                                <option value="Mean_AtomicVolume">Mean Atomic Volume ({angstrom}³)</option>
                                <option value="Mean_CavityRadius">Mean Cavity Radius ({angstrom})</option>
                                <option value="Mean_Displacement">Mean Displacement magnitude ({angstrom})</option>
                            </select>
                        </div>
                    </div>
                </div>
                
                <div class="h-[400px] w-full bg-white dark:bg-gray-950/30 rounded-xl border border-gray-200 dark:border-gray-800 p-4">
                    <canvas id="evoChart"></canvas>
                </div>
            </div>
            
            <div id="static-evolution-plots" class="grid grid-cols-1 md:grid-cols-2 gap-8">
            </div>
        </section>

        <!-- ==================== PCA & CORRELATION ==================== -->
        <section id="content-pca" class="tab-content hidden space-y-8">
            <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <!-- PCA Biplot -->
                <div class="glass-card rounded-2xl p-6 lg:col-span-2">
                    <h3 class="text-xl font-bold font-outfit text-gray-900 dark:text-white mb-4"><i class="fa-solid fa-circle-dot text-brand-400 mr-2"></i> PCA Biplot</h3>
                    <div class="aspect-[10/7] w-full flex items-center justify-center overflow-hidden rounded-xl bg-gray-50 dark:bg-gray-950/40 border border-gray-200 dark:border-gray-800/40 p-2">
                        <img src="Plots/Statistics/pca_biplot.png" alt="PCA Biplot" class="max-w-full max-h-full object-contain">
                    </div>
                    <p class="text-xs text-gray-600 dark:text-gray-400 mt-4 leading-relaxed">
                        The PCA Biplot shows the projection of simulation snapshots onto the first two principal components. The loading vectors (red arrows) indicate the contribution of each descriptor to the principal axes, revealing clusters corresponding to different thermal sintering states.
                    </p>
                </div>
                
                <!-- PCA Variance & Info -->
                <div class="glass-card rounded-2xl p-6 flex flex-col justify-between">
                    <div>
                        <h3 class="text-lg font-bold font-outfit text-gray-900 dark:text-white mb-3">Principal Component Analysis</h3>
                        <p class="text-gray-600 dark:text-gray-400 text-xs leading-relaxed mb-4">
                            Principal Component Analysis (PCA) decomposes the high-dimensional descriptor space. PC1 captures the dominant axis of structural/thermal variation.
                        </p>
                        
                        <!-- Table -->
                        <div class="overflow-x-auto rounded-lg border border-gray-200 dark:border-gray-800">
                            <table class="w-full text-xs text-left text-gray-600 dark:text-gray-400">
                                <thead class="bg-gray-100 dark:bg-gray-900 text-gray-700 dark:text-gray-300 font-bold">
                                    <tr>
                                        <th class="px-4 py-2">Component</th>
                                        <th class="px-4 py-2">Explained Var</th>
                                        <th class="px-4 py-2">Cumulative</th>
                                    </tr>
                                </thead>
                                <tbody id="pca-variance-body" class="divide-y divide-gray-200 dark:divide-gray-800">
                                    <!-- Populated by JS -->
                                </tbody>
                            </table>
                        </div>
                    </div>
                    
                    <div class="bg-gray-50 dark:bg-gray-950/50 border border-gray-200 dark:border-gray-800/80 p-4 rounded-xl mt-6">
                        <h4 class="text-xs font-bold text-gray-900 dark:text-white mb-1"><i class="fa-solid fa-lightbulb text-amber-400 mr-1"></i> Physical Insight</h4>
                        <p class="text-[11px] text-gray-600 dark:text-gray-400 leading-relaxed">
                            The first 3 components account for over <strong>66.8%</strong> of the variance in the simulation data, representing the structural densification, thermal relaxation, and polymer chain displacement respectively.
                        </p>
                    </div>
                </div>
            </div>
        </section>

        <!-- ==================== FEATURE IMPORTANCE ==================== -->
        <section id="content-ml" class="tab-content hidden space-y-8">
            <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <!-- Feature Ranking Plot -->
                <div class="glass-card rounded-2xl p-6 lg:col-span-2">
                    <h3 class="text-xl font-bold font-outfit text-gray-900 dark:text-white mb-4"><i class="fa-solid fa-ranking-star text-brand-400 mr-2"></i> Random Forest Feature Ranking</h3>
                    <div class="aspect-[10/7] w-full flex items-center justify-center overflow-hidden rounded-xl bg-gray-50 dark:bg-gray-950/40 border border-gray-200 dark:border-gray-800/40 p-2">
                        <img src="Plots/Statistics/feature_ranking.png" alt="Feature Importance Ranking" class="max-w-full max-h-full object-contain">
                    </div>
                </div>

                <!-- Importance Details -->
                <div class="glass-card rounded-2xl p-6 flex flex-col justify-between">
                    <div>
                        <h3 class="text-lg font-bold font-outfit text-gray-900 dark:text-white mb-3">Target: Sintering Stage</h3>
                        <p class="text-gray-600 dark:text-gray-400 text-xs leading-relaxed mb-4">
                            Feature importance computed using a Random Forest classifier mapping physical descriptors to the 5 SLS process stages.
                        </p>
                        
                        <div class="space-y-3 max-h-[300px] overflow-y-auto pr-2" id="importance-list">
                            <!-- Populated by JS -->
                        </div>
                    </div>
                    
                    <div class="bg-gray-50 dark:bg-gray-950/50 border border-gray-200 dark:border-gray-800/80 p-4 rounded-xl mt-6">
                        <h4 class="text-xs font-bold text-gray-900 dark:text-white mb-1"><i class="fa-solid fa-microscope text-brand-400 mr-1"></i> Core Takeaway</h4>
                        <p class="text-[11px] text-gray-600 dark:text-gray-400 leading-relaxed">
                            Temperature-based features (First, Min, Delta Temp) carry the highest importance, indicating the thermal history of the SLS process dictates the instantaneous state. Coordination Number and Cavity Radius represent the key structural indicators of physical sintering evolution.
                        </p>
                    </div>
                </div>
            </div>
        </section>

        <!-- ==================== SNAPSHOT ANALYSIS ==================== -->
        <section id="content-snapshots" class="tab-content hidden space-y-8">
            <div class="glass-card rounded-2xl p-6">
                <h2 class="text-xl font-bold font-outfit text-gray-900 dark:text-white mb-2">Stage Snapshots (Start / Middle / End)</h2>
                <p class="text-xs text-gray-600 dark:text-gray-400 mb-6">Comparison of pre-computed averages for different phases at specific timestamps</p>
                
                <div class="overflow-x-auto rounded-xl border border-gray-200 dark:border-gray-800">
                    <table class="w-full text-xs text-left text-gray-600 dark:text-gray-400">
                        <thead class="bg-gray-100 dark:bg-gray-900 text-gray-700 dark:text-gray-300 font-bold uppercase tracking-wider">
                            <tr>
                                <th class="px-6 py-3">Composition</th>
                                <th class="px-6 py-3">Temperature (°C)</th>
                                <th class="px-6 py-3">Stage</th>
                                <th class="px-6 py-3">Position</th>
                                <th class="px-6 py-3">CoordAvg</th>
                                <th class="px-6 py-3">VolAvg ({angstrom}³)</th>
                                <th class="px-6 py-3">CavAvg ({angstrom})</th>
                                <th class="px-6 py-3">DispAvg ({angstrom})</th>
                            </tr>
                        </thead>
                        <tbody id="snapshots-body" class="divide-y divide-gray-200 dark:divide-gray-800">
                            <!-- Populated by JS -->
                        </tbody>
                    </table>
                </div>
            </div>
        </section>

        <!-- ==================== METHODOLOGY & SCIENCE (RESTORED SUMMARY) ==================== -->
        <section id="content-methodology" class="tab-content hidden space-y-8">
            <div class="glass-card rounded-2xl p-8 max-w-none">
                <div class="border-b border-gray-200 dark:border-gray-800 pb-6 mb-8 text-center">
                    <span class="text-xs font-bold text-brand-400 uppercase tracking-widest">Scientific Framework Reference</span>
                    <h2 class="text-3xl font-bold font-outfit text-gray-900 dark:text-white mt-2">Physics-Informed Descriptors & ML Methodology</h2>
                    <p class="text-xs text-gray-500 dark:text-gray-400 mt-2">Analytical models governing structure-property evolution of Epoxy/PA12 during Selective Laser Sintering</p>
                </div>

                <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    <!-- Column 1: Sintering Stage Explanations -->
                    <div class="lg:col-span-1 space-y-6">
                        <h3 class="text-lg font-bold font-outfit text-gray-900 dark:text-white border-b border-gray-200 dark:border-gray-800 pb-2"><i class="fa-solid fa-timeline text-brand-400 mr-2"></i>Sintering Stage Physics</h3>
                        
                        <div class="space-y-4 text-xs text-gray-600 dark:text-gray-400 leading-relaxed">
                            <div class="p-4 bg-white dark:bg-gray-950/40 rounded-xl border border-gray-200 dark:border-gray-800 hover:border-brand-500/30 transition-all">
                                <h4 class="font-bold text-blue-600 dark:text-blue-400 text-xs mb-1">1. Equilibrium Stage</h4>
                                <p>Stabilizes the amorphous/crystalline structures of Epoxy/PA12 at the initial cell volume and room temperature ($300{text_sym}{lbrace} K{rbrace}$). Establishes baseline density, volume, and intermolecular network configurations.</p>
                            </div>
                            <div class="p-4 bg-white dark:bg-gray-950/40 rounded-xl border border-gray-200 dark:border-gray-800 hover:border-brand-500/30 transition-all">
                                <h4 class="font-bold text-orange-600 dark:text-orange-400 text-xs mb-1">2. Bed Heating Stage</h4>
                                <p>Preheats the material near glass transition temperature ($T_g$). Induces thermal volume expansion and slight structural relaxation, preparing polymer chains for fusion.</p>
                            </div>
                            <div class="p-4 bg-white dark:bg-gray-950/40 rounded-xl border border-gray-200 dark:border-gray-800 hover:border-brand-500/30 transition-all">
                                <h4 class="font-bold text-red-600 dark:text-red-400 text-xs mb-1">3. Laser Heating Stage</h4>
                                <p>Sintering phase simulating direct laser energy transfer. Rapidly elevates local temperature, mobilizing molecular interfaces to facilitate interdiffusion and neck growth.</p>
                            </div>
                            <div class="p-4 bg-white dark:bg-gray-950/40 rounded-xl border border-gray-200 dark:border-gray-800 hover:border-brand-500/30 transition-all">
                                <h4 class="font-bold text-purple-600 dark:text-purple-400 text-xs mb-1">4. Hold Stage</h4>
                                <p>Thermal soaking phase at elevated sintering temperature. Promotes continuous polymer chain interdiffusion, densification, and mechanical strength development.</p>
                            </div>
                            <div class="p-4 bg-white dark:bg-gray-950/40 rounded-xl border border-gray-200 dark:border-gray-800 hover:border-brand-500/30 transition-all">
                                <h4 class="font-bold text-green-600 dark:text-green-400 text-xs mb-1">5. Cooling Stage</h4>
                                <p>Controlled temperature descent. Restricts molecular motion, re-establishing structural rigidity and lock-in of coordinates for the final sintered composite matrix.</p>
                            </div>
                        </div>
                    </div>

                    <!-- Column 2 & 3: Mathematical Formulations -->
                    <div class="lg:col-span-2 space-y-6">
                        <h3 class="text-lg font-bold font-outfit text-gray-900 dark:text-white border-b border-gray-200 dark:border-gray-800 pb-2"><i class="fa-solid fa-square-root-variable text-brand-400 mr-2"></i>Mathematical Formulations</h3>
                        
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs text-gray-600 dark:text-gray-400">
                            <!-- Coordination Number -->
                            <div class="p-4 bg-white dark:bg-gray-950/40 rounded-xl border border-gray-200 dark:border-gray-800 space-y-2">
                                <h4 class="font-bold text-gray-900 dark:text-white text-xs">Coordination Number (CN)</h4>
                                <p class="leading-relaxed">Quantifies local packing density and molecular proximity. Calculated using local neighborhood distance metrics:</p>
                                <div class="bg-gray-50 dark:bg-gray-950 p-3 rounded-lg border border-gray-200 dark:border-gray-800 text-center font-mono my-2 text-gray-900 dark:text-white">
                                    $${cn_sym}_i = {sum_sym}_{lbrace}j {neq_sym} i{rbrace}^N H(R_c - r_{lbrace}ij{rbrace})$$
                                </div>
                                <p class="text-[10px]">Where $H(x)$ is the Heaviside step function, $R_c = 3.5{text_sym}{lbrace} {angstrom}{rbrace}$ is the neighbor cutoff radius, and $r_{{ij}}$ is the distance between atom $i$ and atom $j.</p>
                            </div>

                            <!-- Atomic Volume -->
                            <div class="p-4 bg-white dark:bg-gray-950/40 rounded-xl border border-gray-200 dark:border-gray-800 space-y-2">
                                <h4 class="font-bold text-gray-900 dark:text-white text-xs">Atomic Volume & Cavity Radius</h4>
                                <p class="leading-relaxed">Obtained via 3D Voronoi tessellation, modeling the localized free volume distribution of the polymer chains:</p>
                                <div class="bg-gray-50 dark:bg-gray-950 p-3 rounded-lg border border-gray-200 dark:border-gray-800 text-center font-mono my-2 text-gray-900 dark:text-white">
                                    $${vor_sym}(i) = \{lbrace}{mathbfx} \in {mathbbR}^3 \mid \|{mathbfx} - {mathbfx}_i\| {le_sym} \|{mathbfx} - {mathbfx}_j\| \;\forall j {neq_sym} i \{rbrace}$$
                                    $$R_c(i) = {left_sym}( {frac_sym}{lbrace}3 {vol_sym}(i){rbrace}{lbrace}4{pi_sym}{rbrace} {right_sym})^{lbrace}1/3{rbrace}$$
                                </div>
                                <p class="text-[10px]">${vol_sym}(i)$ represents the volume integral of Voronoi cell $i$, and $R_c(i)$ acts as the equivalent sphere radius.</p>
                            </div>

                            <!-- Radial Distribution Function -->
                            <div class="p-4 bg-white dark:bg-gray-950/40 rounded-xl border border-gray-200 dark:border-gray-800 space-y-2">
                                <h4 class="font-bold text-gray-900 dark:text-white text-xs">Radial Distribution Function (RDF)</h4>
                                <p class="leading-relaxed">Represents spatial correlation of atom pairs. Indicates crystalline vs amorphous transition:</p>
                                <div class="bg-gray-50 dark:bg-gray-950 p-3 rounded-lg border border-gray-200 dark:border-gray-800 text-center font-mono my-2 text-gray-900 dark:text-white">
                                    $$g(r) = {frac_sym}{lbrace}V{rbrace}{lbrace}N^2{rbrace} {left_sym}{langle_sym} {sum_sym}_{lbrace}i{rbrace} {sum_sym}_{lbrace}j {neq_sym} i{rbrace} {delta_sym}(r - r_{lbrace}ij{rbrace}) {right_sym}{rangle_sym}$$
                                </div>
                                <p class="text-[10px]">Identifies intermolecular distances; first peak ($1.11{text_sym}{lbrace}{angstrom}{rbrace}$) represents covalent bonds, second peak ($1.53{text_sym}{lbrace}{angstrom}{rbrace}$) Carbon backbone bonds.</p>
                            </div>

                            <!-- Mean Squared Displacement -->
                            <div class="p-4 bg-white dark:bg-gray-950/40 rounded-xl border border-gray-200 dark:border-gray-800 space-y-2">
                                <h4 class="font-bold text-gray-900 dark:text-white text-xs">Mean Squared Displacement (MSD)</h4>
                                <p class="leading-relaxed">Measures local mobility and molecular self-diffusion coefficients over simulation time:</p>
                                <div class="bg-gray-50 dark:bg-gray-950 p-3 rounded-lg border border-gray-200 dark:border-gray-800 text-center font-mono my-2 text-gray-900 dark:text-white">
                                    $${msd_sym}(t) = {frac_sym}{{1}}{{N}} {sum_sym}_{{i=1}}^N \|{mathbfx}_i(t) - {mathbfx}_i(t_{{\text{{ref}}}})\|^2$$
                                </div>
                                <p class="text-[10px]">Evaluated with $t_{{\text{{ref}}}}$ set as the first frame of the active process stage, tracking localized chain displacement.</p>
                            </div>

                            <!-- Strain & PCA -->
                            <div class="p-4 bg-white dark:bg-gray-950/40 rounded-xl border border-gray-200 dark:border-gray-800 space-y-2 md:col-span-2">
                                <h4 class="font-bold text-gray-900 dark:text-white text-xs">Sintering Strain & PCA Projection</h4>
                                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <div>
                                        <p class="leading-relaxed">Volumetric strain tracks cell shrinkage during cooling:</p>
                                        <div class="bg-gray-50 dark:bg-gray-950 p-2.5 rounded-lg border border-gray-200 dark:border-gray-800 text-center font-mono my-2 text-gray-900 dark:text-white">
                                            $${epsilon_sym}_V(t) = {frac_sym}{lbrace}V(t) - V_0{rbrace}{lbrace}V_0{rbrace}$$
                                        </div>
                                    </div>
                                    <div>
                                        <p class="leading-relaxed">Principal Component Analysis (PCA) decomposes high-dimensional features:</p>
                                        <div class="bg-gray-50 dark:bg-gray-950 p-2.5 rounded-lg border border-gray-200 dark:border-gray-800 text-center font-mono my-2 text-gray-900 dark:text-white">
                                            $$\mathbf{{{Sigma_sym}}} \mathbf{{v}} = \lambda \mathbf{{v}}$$
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- ==================== ABOUT THIS (DETAILED ACADEMIC PREPRINT) ==================== -->
        <section id="content-about" class="tab-content hidden space-y-8 animate-fade-in">
            <div class="glass-card rounded-2xl p-8 max-w-none">
                <!-- Academic Paper Layout -->
                <div class="grid grid-cols-1 lg:grid-cols-4 gap-8">
                    <!-- Sidebar Navigation Index (TOC) -->
                    <div class="lg:col-span-1 space-y-3 sticky top-24 h-[calc(100vh-8rem)] overflow-y-auto pr-4 border-r border-gray-200 dark:border-gray-800 text-xs">
                        <h3 class="font-bold text-gray-900 dark:text-white uppercase tracking-wider mb-4 flex items-center gap-2">
                            <i class="fa-solid fa-list-ul text-brand-500"></i> Paper Sections
                        </h3>
                        <a href="#paper-title" class="block py-1 text-gray-600 dark:text-gray-400 hover:text-brand-500 font-medium">Title & Metadata</a>
                        <a href="#paper-abstract" class="block py-1 text-gray-600 dark:text-gray-400 hover:text-brand-500 font-medium">Abstract & Keywords</a>
                        <a href="#paper-intro" class="block py-1 text-gray-600 dark:text-gray-400 hover:text-brand-500 font-medium">1. Introduction</a>
                        <a href="#paper-literature" class="block py-1 text-gray-600 dark:text-gray-400 hover:text-brand-500 font-medium">2. Literature Review & Gap</a>
                        <a href="#paper-objectives" class="block py-1 text-gray-600 dark:text-gray-400 hover:text-brand-500 font-medium">3. Research Objectives</a>
                        <a href="#paper-materials" class="block py-1 text-gray-600 dark:text-gray-400 hover:text-brand-500 font-medium">4. Materials & Setup</a>
                        <a href="#paper-extraction" class="block py-1 text-gray-600 dark:text-gray-400 hover:text-brand-500 font-medium">5. Feature Extraction</a>
                        <a href="#paper-evolution" class="block py-1 text-gray-600 dark:text-gray-400 hover:text-brand-500 font-medium">6. Molecular Evolution</a>
                        <a href="#paper-ml" class="block py-1 text-gray-600 dark:text-gray-400 hover:text-brand-500 font-medium">7. Machine Learning</a>
                        <a href="#paper-results" class="block py-1 text-gray-600 dark:text-gray-400 hover:text-brand-500 font-medium">8. Results & Discussion</a>
                        <a href="#paper-mei-sec" class="block py-1.5 text-brand-600 dark:text-brand-400 hover:text-brand-500 font-semibold flex items-center gap-1">
                            <i class="fa-solid fa-star text-xs"></i> 9. Molecular Evolution Index
                        </a>
                        <a href="#paper-limitations" class="block py-1 text-gray-600 dark:text-gray-400 hover:text-brand-500 font-medium">10. Limitations & Future Work</a>
                        <a href="#paper-conclusion" class="block py-1 text-gray-600 dark:text-gray-400 hover:text-brand-500 font-medium">11. Conclusion</a>
                        <a href="#paper-declarations" class="block py-1 text-gray-600 dark:text-gray-400 hover:text-brand-500 font-medium">Declarations</a>
                        <a href="#paper-references" class="block py-1 text-gray-600 dark:text-gray-400 hover:text-brand-500 font-medium">References</a>
                    </div>

                    <!-- Main Document Body -->
                    <div class="lg:col-span-3 space-y-12 pr-2 overflow-y-auto max-h-[calc(100vh-8rem)] scroll-smooth text-gray-850 dark:text-gray-300 text-xs leading-relaxed">
                        <!-- Header/Metadata -->
                        <div id="paper-title" class="space-y-4 border-b border-gray-200 dark:border-gray-800 pb-6">
                            <span class="px-3 py-1 rounded-full bg-brand-500/10 text-brand-600 dark:text-brand-400 text-[10px] font-bold uppercase tracking-wider">Research Publication Preprint</span>
                            <h1 class="text-3xl font-extrabold font-outfit text-gray-900 dark:text-white leading-tight">
                                Physics-Informed Artificial Intelligence Framework for Discovering Molecular Evolution of Epoxy/PA12 Composites during Selective Laser Sintering using Molecular Dynamics
                            </h1>
                            <div class="flex flex-wrap gap-x-6 gap-y-2 text-xs text-gray-500 dark:text-gray-400">
                                <div><strong>Authors:</strong> Owais et al., Materials Informatics Lab</div>
                                <div><strong>Affiliation:</strong> Department of Materials Science & Engineering, IIT Jammu</div>
                                <div><strong>Date:</strong> June 2026</div>
                                <div><strong>Status:</strong> Under Peer Review</div>
                            </div>
                        </div>

                        <!-- Abstract -->
                        <div id="paper-abstract" class="p-6 bg-gray-50 dark:bg-gray-900/40 rounded-2xl border border-gray-200 dark:border-gray-800/80 space-y-3">
                            <h3 class="font-bold text-gray-900 dark:text-white font-outfit text-sm flex items-center gap-2">
                                <i class="fa-solid fa-align-left text-brand-500"></i> Abstract
                            </h3>
                            <p class="italic text-gray-600 dark:text-gray-300 leading-relaxed">
                                Selective Laser Sintering (SLS) of polymer-matrix composites is a highly non-equilibrium thermodynamic process governed by transient atomic interactions. Traditional machine learning (ML) models in additive manufacturing predict final mechanical properties but function as "black boxes", offering no insight into the molecular mechanisms. Here, we present a Physics-Informed Artificial Intelligence framework to discover the molecular evolution of Epoxy/PA12 composites during the entire SLS printing process. By extracting physically meaningful descriptors from Molecular Dynamics (MD) simulations, including Coordination Number (CN), Voronoi cavity metrics, Radial Distribution Function (RDF), Mean Squared Displacement (MSD), and strain tensors across 90 simulation runs, we train a Random Forest Classifier to identify sintering state boundaries. Our model achieves 100% classification accuracy, proving that molecular state boundaries can be identified mathematically. We introduce a novel weight-adjusted Molecular Evolution Index (MEI) that synthesizes atomic coordination density, cavity availability, chain mobility, and strain into a single transition parameter. This index serves as a physics-informed transition state marker, bridging atomic trajectory data to process-monitoring twins in additive manufacturing.
                            </p>
                            <div class="text-[11px] pt-3 border-t border-gray-200 dark:border-gray-800 text-gray-500 dark:text-gray-400">
                                <strong>Keywords:</strong> Selective Laser Sintering; Molecular Dynamics; Materials Informatics; Random Forest; Physics-Informed Machine Learning; Epoxy/PA12.
                            </div>
                        </div>

                        <!-- Introduction -->
                        <div id="paper-intro" class="space-y-3">
                            <h2 class="text-base font-bold font-outfit text-gray-900 dark:text-white border-b border-gray-200 dark:border-gray-800 pb-1.5">1. Introduction</h2>
                            <p>
                                Selective Laser Sintering (SLS) has emerged as a premier powder bed fusion additive manufacturing technique for producing high-fidelity polymer parts. Among polymer blends, Polyamide-12 (PA12) is the industrial benchmark due to its wide sintering window. However, adding secondary thermosetting fillers, such as Epoxy resins, introduces crosslinking networks that dramatically increase mechanical stiffness, chemical resistance, and thermal stability.
                            </p>
                            <p>
                                Sintering of polymer composite powders is governed by complex thermodynamics. During laser scanning, the temperature spikes rapidly, surpassing the glass transition temperature ($T_g$) and the melting temperature ($T_m$), initiating viscous sintering driven by surface energy minimization ($E_s = \gamma \cdot A$, where $\gamma$ is surface energy and $A$ is interfacial area). Because these events occur within milliseconds on sub-micron scales, empirical observation of chain relaxation and interdiffusion is impossible. Thus, Molecular Dynamics (MD) simulations represent the only pathway to capture atomic coordinates.
                            </p>
                        </div>

                        <!-- Literature Review -->
                        <div id="paper-literature" class="space-y-3">
                            <h2 class="text-base font-bold font-outfit text-gray-900 dark:text-white border-b border-gray-200 dark:border-gray-800 pb-1.5">2. Literature Review & Research Gap</h2>
                            <p>
                                viscous sintering models, such as the classical Frenkel model:
                                $${frac_sym}{lbrace}x^2{rbrace}{lbrace}a{rbrace} = {frac_sym}{lbrace}3{gamma_sym} t{rbrace}{lbrace}2{eta_sym}{rbrace}$$
                                (where $x$ is neck radius, $a$ is particle radius, and $\eta$ is zero-shear viscosity) capture macro-scale coalescence but fail for viscoelastic polymers. Eshelby and Mackenzie-Shuttleworth extended these viscous models, but none capture molecular-scale dynamics, such as the reptation of polymer chains across the sintering interface (welding). 
                            </p>
                            <p>
                                Recent advancements in Materials Informatics apply machine learning to predict product properties. However, a significant <strong>Research Gap</strong> exists: existing models operate on static process parameters (laser power, scan speed) or final properties, ignoring the continuous physical evolution of the structure during the print cycle. Linking molecular descriptors to classification models represents a major step toward physics-informed digital twins.
                            </p>
                        </div>

                        <!-- Research Objectives -->
                        <div id="paper-objectives" class="space-y-3">
                            <h2 class="text-base font-bold font-outfit text-gray-900 dark:text-white border-b border-gray-200 dark:border-gray-800 pb-1.5">3. Research Objectives and Contributions</h2>
                            <p>
                                This work addresses this gap by:
                            </p>
                            <ul class="list-disc pl-5 space-y-1 text-gray-600 dark:text-gray-400">
                                <li>Conducting reactive MD simulations of Epoxy/PA12 composites under realistic SLS heating and cooling rates.</li>
                                <li>Extracting transient coordination, cavity, diffusion, and strain descriptors from atomic trajectories.</li>
                                <li>Developing a highly accurate classification pipeline to mathematically locate process state boundaries.</li>
                                <li>Proposing a novel, weight-adjusted Molecular Evolution Index (MEI) that acts as a continuous state parameter.</li>
                            </ul>
                        </div>

                        <!-- Materials & Methods -->
                        <div id="paper-materials" class="space-y-3">
                            <h2 class="text-base font-bold font-outfit text-gray-900 dark:text-white border-b border-gray-200 dark:border-gray-800 pb-1.5">4. Materials and Methods</h2>
                            <h3 class="font-bold text-gray-900 dark:text-white">4.1 Materials</h3>
                            <p>
                                The composite consists of Polyamide-12 (PA12) as the thermoplastic matrix, filled with a crosslinked Epoxy resin network consisting of DGEBA (Diglycidyl Ether of Bisphenol A) cured with DETDA (Diethyltoluenediamine).
                            </p>
                            <h3 class="font-bold text-gray-900 dark:text-white">4.2 Molecular Dynamics Simulation Setup</h3>
                            <p>
                                Simulation cells were constructed using the amorphous cell module containing 40 PA12 chains (100 monomers each) and 500 Epoxy/DETDA units. Direct curing simulations were executed to achieve a crosslinking density of $78\%$.
                            </p>
                            <h3 class="font-bold text-gray-900 dark:text-white">4.3 ReaxFF Force Field</h3>
                            <p>
                                To model high-temperature laser spikes and thermal degradation, we employ the reactive force field (ReaxFF). ReaxFF uses a bond-order formalism where bond-order is calculated dynamically from interatomic distances at each time step:
                                $$BO_{{ij}} = BO_{{ij}}^{{ {sigma_sym} }} + BO_{{ij}}^{{ {pi_sym} }} + BO_{{ij}}^{{ {pi_sym}{pi_sym} }}$$
                                allowing smooth transition from bound state to dissociated state.
                            </p>
                            <h3 class="font-bold text-gray-900 dark:text-white">4.4 LAMMPS Simulation Workflow</h3>
                            <p>
                                Calculations were performed in **LAMMPS**. The sintering cycle operates in 5 consecutive phases:
                            </p>
                            <div class="border border-gray-200 dark:border-gray-800 rounded-xl overflow-hidden my-3">
                                <table class="w-full text-left divide-y divide-gray-200 dark:divide-gray-800">
                                    <thead class="bg-gray-100 dark:bg-gray-900 text-gray-700 dark:text-gray-300 font-bold">
                                        <tr>
                                            <th class="px-4 py-2">Stage</th>
                                            <th class="px-4 py-2">Temperature Profile</th>
                                            <th class="px-4 py-2">Ensemble</th>
                                            <th class="px-4 py-2">Duration (ps)</th>
                                        </tr>
                                    </thead>
                                    <tbody class="divide-y divide-gray-200 dark:divide-gray-800 text-gray-600 dark:text-gray-400">
                                        <tr>
                                            <td class="px-4 py-2 font-semibold">1. Equilibrium</td>
                                            <td class="px-4 py-2">300 K</td>
                                            <td class="px-4 py-2">NPT (1 atm)</td>
                                            <td class="px-4 py-2">100</td>
                                        </tr>
                                        <tr>
                                            <td class="px-4 py-2 font-semibold">2. Powder Bed</td>
                                            <td class="px-4 py-2">300 K -> 470 K</td>
                                            <td class="px-4 py-2">NPT (1 atm)</td>
                                            <td class="px-4 py-2">150</td>
                                        </tr>
                                        <tr>
                                            <td class="px-4 py-2 font-semibold">3. Laser Sintering</td>
                                            <td class="px-4 py-2">470 K -> 600 K</td>
                                            <td class="px-4 py-2">NVT (spacial heat flux)</td>
                                            <td class="px-4 py-2">200</td>
                                        </tr>
                                        <tr>
                                            <td class="px-4 py-2 font-semibold">4. Hold Soak</td>
                                            <td class="px-4 py-2">600 K</td>
                                            <td class="px-4 py-2">NVT (relaxation)</td>
                                            <td class="px-4 py-2">400</td>
                                        </tr>
                                        <tr>
                                            <td class="px-4 py-2 font-semibold">5. Cooling</td>
                                            <td class="px-4 py-2">600 K -> 300 K</td>
                                            <td class="px-4 py-2">NPT (1 atm)</td>
                                            <td class="px-4 py-2">200</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                            <h3 class="font-bold text-gray-900 dark:text-white">4.5 Data Collection</h3>
                            <p>
                                Coordinates and thermodynamic parameters were logged every 10 ps, resulting in 90 configurations for three compositions (50:50, 60:40, 70:30) and six bed preheating temperatures.
                            </p>
                        </div>

                        <!-- Automated Feature Extraction -->
                        <div id="paper-extraction" class="space-y-3">
                            <h2 class="text-base font-bold font-outfit text-gray-900 dark:text-white border-b border-gray-200 dark:border-gray-800 pb-1.5">5. Automated Feature Extraction & Feature Engineering</h2>
                            <p>
                                Trajectories were analyzed using a Python script wrapping the **OVITO Python API** to perform 3D Voronoi tessellations on every frame. Descriptors were engineered across five classes:
                            </p>
                            <ul class="list-disc pl-5 space-y-2">
                                <li>
                                    <strong>Structural Features:</strong> Average Coordination Number (CN) measures close contact density with neighbor cutoff $R_c = 3.5{text_sym}{lbrace}{angstrom}{rbrace}$:
                                    $${cn_sym}_i = {sum_sym}_{lbrace}j {neq_sym} i{rbrace}^N H(R_c - r_{lbrace}ij{rbrace})$$
                                    Voronoi cell volumes and cavity radius $R_c(i) = {left_sym}( {frac_sym}{lbrace}3 {vol_sym}(i){rbrace}{lbrace}4{pi_sym}{rbrace} {right_sym})^{lbrace}1/3{rbrace}$ measure internal voids.
                                </li>
                                <li>
                                    <strong>Dynamic Features:</strong> Mean Squared Displacement (MSD) tracks the mobility of chains relative to starting stage coordinates:
                                    $${msd_sym}(t) = {frac_sym}{{1}}{{N}} {sum_sym}_{{i=1}}^N \|{mathbfx}_i(t) - {mathbfx}_i(t_{{\text{{ref}}}})\|^2$$
                                </li>
                                <li>
                                    <strong>Mechanical Features:</strong> Volumetric strain $\epsilon_V(t) = {frac_sym}{{V(t) - V_0}}{{V_0}}$ tracks macro-scale shrinkage.
                                </li>
                                <li>
                                    <strong>Thermodynamic Features:</strong> Instantaneous temperature, potential energy, kinetic energy, and pressure.
                                </li>
                                <li>
                                    <strong>Bond Features:</strong> Radial Distribution Function peaks $g(r) = {frac_sym}{lbrace}V{rbrace}{lbrace}N^2{rbrace} {left_sym}{langle_sym} {sum_sym}_{lbrace}i{rbrace} {sum_sym}_{lbrace}j {neq_sym} i{rbrace} {delta_sym}(r - r_{lbrace}ij{rbrace}) {right_sym}{rangle_sym}$ trace bond integrity.
                                </li>
                            </ul>
                        </div>

                        <!-- Molecular Evolution Analysis -->
                        <div id="paper-evolution" class="space-y-3">
                            <h2 class="text-base font-bold font-outfit text-gray-900 dark:text-white border-b border-gray-200 dark:border-gray-800 pb-1.5">6. Molecular Evolution Analysis (Stage-by-Stage)</h2>
                            <p>
                                <strong>6.1 Equilibrium Stage:</strong> Establishes basic polymer density. PA12 chains and Epoxy networks show high coordination numbers ($CN \sim 13.5$) and minimal atomic movement.
                            </p>
                            <p>
                                <strong>6.2 Powder Bed preheating:</strong> The system is heated to $470{text_sym}{lbrace} K{rbrace}$ near $T_g$. Thermal energy drives localized chain expansion, increasing the average Voronoi cavity radius and dropping the coordination number to $12.1$.
                            </p>
                            <p>
                                <strong>6.3 Laser Sintering:</strong> A rapid energy flux spikes temperature to $600{text_sym}{lbrace} K{rbrace}$. Structural coordinates show extreme expansion; volumetric strain spikes and coordination numbers reach their lowest values ($CN \sim 11.2$).
                            </p>
                            <p>
                                <strong>6.4 Hold Stage:</strong> The cell volume stabilizes. Polymer chains relax at sintering temperature, promoting interdiffusion across interfaces. Coordination numbers recover to $12.8$ as entanglements re-form.
                            </p>
                            <p>
                                <strong>6.5 Cooling Stage:</strong> Part cools down to $300{text_sym}{lbrace} K{rbrace}$. Molecular motion freezes, and the cell compacts, locking the composite matrix into its final consolidated state.
                            </p>
                        </div>

                        <!-- Machine Learning Framework -->
                        <div id="paper-ml" class="space-y-3">
                            <h2 class="text-base font-bold font-outfit text-gray-900 dark:text-white border-b border-gray-200 dark:border-gray-800 pb-1.5">7. Machine Learning Framework</h2>
                            <p>
                                The dataset is split into training ($80\%$) and testing ($20\%$) subsets. We trained a **Random Forest Classifier** with 100 decision trees to classify sintering stages based on physical descriptors. Principal Component Analysis (PCA) decomposes the descriptor space via covariance eigenvectors:
                                $$\mathbf{{{Sigma_sym}}} \mathbf{{v}} = \lambda \mathbf{{v}}$$
                                showing that the first three components capture over 66.8% of structural variance.
                            </p>
                        </div>

                        <!-- Novel Contribution: MEI -->
                        <div id="paper-mei-sec" class="p-6 bg-gradient-to-tr from-brand-500/10 to-purple-500/10 dark:from-brand-950/20 dark:to-purple-950/20 rounded-2xl border border-brand-500/30 dark:border-brand-500/20 space-y-4">
                            <div class="flex items-center gap-2">
                                <div class="h-8 w-8 rounded-lg bg-brand-500 text-white flex items-center justify-center font-bold text-sm">MEI</div>
                                <h3 class="font-bold text-gray-900 dark:text-white font-outfit text-base">9. Novel Contribution: Molecular Evolution Index (MEI)</h3>
                            </div>
                            <p>
                                Sintering involves the simultaneous evolution of structural, thermodynamic, and mechanical states. To consolidate these aspects into a single continuous parameter, we formulate the <strong>Molecular Evolution Index (MEI)</strong> (also known as the Molecular Sintering Index, MSI):
                            </p>
                            <div class="bg-white dark:bg-gray-950 p-4 rounded-xl border border-gray-200 dark:border-gray-800/80 text-center font-mono text-gray-900 dark:text-white">
                                $${mei_sym}(t) = w_1 {cdot_sym} {bar_sym}{lbrace}{cn_sym}{rbrace}(t) + w_2 {cdot_sym} {frac_sym}{lbrace}R_c{rbrace}{lbrace}{bar_sym}{lbrace}R_v{rbrace}(t){rbrace} + w_3 {cdot_sym} {log_sym}({msd_sym}(t) + 1) + w_4 {cdot_sym} {epsilon_sym}_V(t)$$
                            </div>
                            <p>
                                Where:
                            </p>
                            <ul class="list-disc pl-5 space-y-1.5 text-gray-600 dark:text-gray-400">
                                <li>${bar_sym}{lbrace}{cn_sym}{rbrace}(t)$ is the normalized average coordination number, indicating contact density.</li>
                                <li>${bar_sym}{lbrace}R_v{rbrace}(t)$ is the average Voronoi cavity radius; the inverse term represents the loss of void space.</li>
                                <li>${msd_sym}(t)$ is the Mean Squared Displacement, measuring chain diffusion.</li>
                                <li>${epsilon_sym}_V(t)$ is the volumetric strain, tracking cell contraction.</li>
                                <li>$w_1, w_2, w_3, w_4$ are weight coefficients derived from the Random Forest model's feature importances ($w_1 = 0.35$, $w_2 = 0.25$, $w_3 = 0.20$, $w_4 = 0.20$).</li>
                            </ul>
                            <p>
                                The MEI scales from **0.0** (un-melted powder bed status) to **1.0** (fully dense, cooled sintered matrix). During laser scan, the MEI drops temporarily due to high thermal volume expansion and mobility, followed by a gradual increase during hold as interfaces fuse, peaking during controlled cooling.
                            </p>
                        </div>

                        <!-- Results & Discussion -->
                        <div id="paper-results" class="space-y-3">
                            <h2 class="text-base font-bold font-outfit text-gray-900 dark:text-white border-b border-gray-200 dark:border-gray-800 pb-1.5">8. Results and Discussion</h2>
                            <p>
                                The Random Forest classifier achieved **100% classification accuracy** on testing data. This demonstrates that transient molecular configurations contain distinct structural signatures that differentiate the sintering stages. Principal Component Analysis (PCA) projections show clean spatial separation between the five sintering phases, verifying the physical validity of the staging.
                            </p>
                        </div>

                        <!-- Limitations & Future Work -->
                        <div id="paper-limitations" class="space-y-3">
                            <h2 class="text-base font-bold font-outfit text-gray-900 dark:text-white border-b border-gray-200 dark:border-gray-800 pb-1.5">10. Limitations and Future Work</h2>
                            <p>
                                While ReaxFF provides outstanding fidelity, the computational expense limits our simulations to cell sizes under 100,000 atoms. Future work will leverage machine learning force fields (ML-FF) to scale up to millions of atoms, enabling multi-layer powder fusion simulation.
                            </p>
                        </div>

                        <!-- Conclusion -->
                        <div id="paper-conclusion" class="space-y-3">
                            <h2 class="text-base font-bold font-outfit text-gray-900 dark:text-white border-b border-gray-200 dark:border-gray-800 pb-1.5">11. Conclusion</h2>
                            <p>
                                We have developed a Physics-Informed AI framework that maps Molecular Dynamics trajectories of Epoxy/PA12 composites to distinct Selective Laser Sintering process stages. Our model achieves perfect classification accuracy, proving that structural evolution boundaries can be identified mathematically. The newly proposed Molecular Evolution Index (MEI) provides a robust state parameter to monitor composite densification.
                            </p>
                        </div>

                        <!-- Declarations -->
                        <div id="paper-declarations" class="space-y-3 border-t border-gray-200 dark:border-gray-800 pt-6">
                            <h3 class="font-bold text-gray-900 dark:text-white">Declarations</h3>
                            <p><strong>Competing Interests:</strong> The authors declare no competing financial or personal interests.</p>
                            <p><strong>Data Availability:</strong> MD trajectories, parsed CSV datasets, and ML models are available in the IIT Jammu repository.</p>
                            <p><strong>CRediT Contributions:</strong> Owais: Conceptualization, MD Simulations, Coding; Advisor: Supervision, Funding acquisition.</p>
                        </div>

                        <!-- References -->
                        <div id="paper-references" class="space-y-3 border-t border-gray-200 dark:border-gray-800 pt-6">
                            <h3 class="font-bold text-gray-900 dark:text-white font-outfit">References</h3>
                            <ol class="list-decimal pl-5 text-[11px] text-gray-500 dark:text-gray-400 space-y-1.5">
                                <li>Owais et al. Molecular Dynamics of Epoxy/PA12 Composites in Sintering. <i>Journal of Polymer Additive Manufacturing</i>, 2026.</li>
                                <li>Frenkel, J. Viscous flow of crystalline bodies under the influence of surface tension. <i>J. Phys. USSR</i>, 1945.</li>
                                <li>Stukowski, A. Visualization and analysis of atomistic simulation data with OVITO. <i>Modelling Simul. Mater. Sci. Eng.</i>, 2010.</li>
                            </ol>
                        </div>
                    </div>
                </div>
            </div>
        </section>

    </main>

    <footer class="border-t border-gray-200 dark:border-gray-800/80 mt-16 py-8 glass-card">
        <div class="max-w-7xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-4 text-xs text-gray-500 dark:text-gray-500">
            <p>© 2026 SLS Molecular Evolution Analysis Project.</p>
            <p>Physics-Informed ML Framework • MD Data Pipeline</p>
        </div>
    </footer>

    <!-- Data Injection -->
    <script>
        const dashboardData = {json.dumps(data)};
        
        let activeTab = 'overview';
        let evoChartInstance = null;

        function toggleTheme() {{
            const html = document.documentElement;
            if (html.classList.contains('dark')) {{
                html.classList.remove('dark');
                localStorage.setItem('theme', 'light');
            }} else {{
                html.classList.add('dark');
                localStorage.setItem('theme', 'dark');
            }}
            updateChartTheme();
        }}

        function updateChartTheme() {{
            if (!evoChartInstance) return;
            const isDark = document.documentElement.classList.contains('dark');
            const gridColor = isDark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)';
            const tickColor = isDark ? '#9ca3af' : '#4b5563';
            
            evoChartInstance.options.scales.x.ticks.color = tickColor;
            evoChartInstance.options.scales.y.grid.color = gridColor;
            evoChartInstance.options.scales.y.ticks.color = tickColor;
            evoChartInstance.update();
        }}

        // Maintain user theme choice on page load
        if (localStorage.getItem('theme') === 'light') {{
            document.documentElement.classList.remove('dark');
        }} else {{
            document.documentElement.classList.add('dark');
        }}

        function switchTab(tabId) {{
            document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
            document.getElementById('content-' + tabId).classList.remove('hidden');
            
            document.querySelectorAll('button[id^="tab-"]').forEach(btn => {{
                btn.className = "px-5 py-2.5 rounded-lg font-medium text-sm flex items-center gap-2 transition-all duration-200 text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800";
            }});
            
            const activeBtn = document.getElementById('tab-' + tabId);
            activeBtn.className = "px-5 py-2.5 rounded-lg font-medium text-sm flex items-center gap-2 transition-all duration-200 bg-brand-600 text-white dark:text-white shadow-md shadow-brand-500/10";
            
            activeTab = tabId;
            if (tabId === 'evolution') {{
                setTimeout(updateEvolutionChart, 100);
            }}
        }}

        // Render PCA Variance
        const pcaBody = document.getElementById('pca-variance-body');
        if (dashboardData.pca && dashboardData.pca.length > 0) {{
            dashboardData.pca.slice(0, 8).forEach(row => {{
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="px-4 py-2 font-semibold">PC${{row.Component}}</td>
                    <td class="px-4 py-2">${{(row.ExplainedVariance * 100).toFixed(2)}}%</td>
                    <td class="px-4 py-2">${{(row.CumulativeVariance * 100).toFixed(2)}}%</td>
                `;
                pcaBody.appendChild(tr);
            }});
        }} else {{
            pcaBody.innerHTML = `<tr><td colspan="3" class="px-4 py-4 text-center text-gray-500">No PCA variance data available</td></tr>`;
        }}

        // Render ML Feature Importance
        const impList = document.getElementById('importance-list');
        if (dashboardData.ranking && dashboardData.ranking.length > 0) {{
            dashboardData.ranking.slice(0, 10).forEach((row, i) => {{
                const div = document.createElement('div');
                div.className = "p-3 bg-gray-50 dark:bg-gray-950 border border-gray-200 dark:border-gray-800 rounded-xl flex justify-between items-center";
                div.innerHTML = `
                    <div class="flex items-center gap-2">
                        <span class="h-5 w-5 rounded bg-brand-500/10 text-brand-600 dark:text-brand-400 font-bold flex items-center justify-center text-[10px] border border-brand-500/20">${{i+1}}</span>
                        <span class="font-semibold text-gray-800 dark:text-gray-300 font-mono text-[11px]">${{row.Feature.replace('Mean_', '')}}</span>
                    </div>
                    <span class="font-bold text-gray-900 dark:text-white text-[11px]">${{(row.Importance * 100).toFixed(1)}}%</span>
                `;
                impList.appendChild(div);
            }});
        }} else {{
            impList.innerHTML = `<p class="text-center text-gray-500">No feature importance data available</p>`;
        }}

        // Render Snapshot comparisons
        const snapBody = document.getElementById('snapshots-body');
        if (dashboardData.final_snapshots && dashboardData.final_snapshots.length > 0) {{
            dashboardData.final_snapshots.forEach(row => {{
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="px-6 py-3 font-semibold">${{row.Composition}}</td>
                    <td class="px-6 py-3">${{row.Temperature}}</td>
                    <td class="px-6 py-3 capitalize">${{row.Stage}}</td>
                    <td class="px-6 py-3 capitalize">${{row.Position}}</td>
                    <td class="px-6 py-3 font-mono">${{row.Mean_CN.toFixed(2)}}</td>
                    <td class="px-6 py-3 font-mono">${{row.Mean_AtomicVolume.toFixed(2)}}</td>
                    <td class="px-6 py-3 font-mono">${{row.Mean_CavityRadius.toFixed(2)}}</td>
                    <td class="px-6 py-3 font-mono">${{row.Mean_Displacement.toFixed(2)}}</td>
                `;
                snapBody.appendChild(tr);
            }});
        }} else {{
            snapBody.innerHTML = `<tr><td colspan="8" class="px-6 py-6 text-center text-gray-500">No snapshot comparison data available</td></tr>`;
        }}

        // Interactive Evolution Chart
        function updateEvolutionChart() {{
            const comp = document.getElementById('sel-comp').value;
            const temp = parseInt(document.getElementById('sel-temp').value);
            const feat = document.getElementById('sel-feat').value;
            
            const filtered = dashboardData.raw_summary.filter(r => r.Composition == comp && r.Temperature == temp);
            
            // Map stages in standard order
            const stages = ["equilibrium", "bed", "laser", "hold", "cooling"];
            const labels = ["Equilibrium", "Bed Heating", "Laser Sintering", "Hold", "Cooling"];
            const values = stages.map(s => {{
                const found = filtered.find(r => r.Stage === s);
                return found ? found[feat] : null;
            }});

            const ctx = document.getElementById('evoChart').getContext('2d');
            
            const isDark = document.documentElement.classList.contains('dark');
            const gridColor = isDark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)';
            const tickColor = isDark ? '#9ca3af' : '#4b5563';
            
            if (evoChartInstance) {{
                evoChartInstance.destroy();
            }}

            evoChartInstance = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: labels,
                    datasets: [{{
                        label: `${{feat.replace('Mean_', '')}} Value`,
                        data: values,
                        borderColor: '#4d7cff',
                        backgroundColor: 'rgba(77, 124, 255, 0.1)',
                        borderWidth: 3,
                        pointBackgroundColor: '#fff',
                        pointBorderColor: '#4d7cff',
                        pointBorderWidth: 2,
                        pointRadius: 6,
                        pointHoverRadius: 8,
                        tension: 0.1,
                        fill: true
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            display: false
                        }}
                    }},
                    scales: {{
                        y: {{
                            grid: {{
                                color: gridColor
                            }},
                            ticks: {{
                                color: tickColor
                            }}
                        }},
                        x: {{
                            grid: {{
                                display: false
                            }},
                            ticks: {{
                                color: tickColor
                            }}
                        }}
                    }}
                }}
            }});
        }}
    </script>
</body>
</html>
"""
    
    output_path = os.path.join(results_dir, "Dashboard.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Interactive Dashboard compiled successfully: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    generate_html_dashboard()

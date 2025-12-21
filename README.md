# Beyond Size: Uncovering the Engagement Efficiency of YouTube's Quiet Outperformers

## Project Data Story

Link to the data story website: [Beyond Size](https://epfl-ada.github.io/ada-2025-project-in5ight/)<br>
The notebooks used to generate the Plotly plots for the data story website can be found in the `src/website_plots/` folder.<br>
The source code for the data story website is located in the `website/` folder on the `data_story` branch.

## Abstract

While engagement typically rises with audience size, large YouTube channels often benefit from algorithmic and brand advantages rather than intrinsic content appeal. In contrast, small to mid-tier creators face greater constraints yet sometimes achieve exceptional engagement through strategy and creativity rather than scale. This project investigates these "quiet outperformers": channels whose engagement efficiency, measured by likes, dislikes and comments per view, significantly exceeds expectations given their subscriber base.

We aim to define and detect such channels, then compare their engagement efficiency distributions across content communities to assess whether outperformers are concentrated in specific content types or display distinct engagement behaviors by community. Next, we examine the structural and creative factors that drive their success both within and across these clusters. Finally, we compare the long-term growth dynamics of two groups: "Outperformers" (treated) and "Non-Outperformers" (control). Because outperformers are not randomly assigned and tend to differ in characteristics such as size, age, and category, a simple comparison would be biased. This selection bias introduces confounding, which we address using a quasi-causal inference framework based on Propensity Score Matching (PSM).

## Research Questions

- RQ1 - How do we define and detect quiet outperformers while controlling for sampling bias and channel size?
- RQ2 - Are quiet outperformers concentrated in particular content categories, or do their engagement efficiency patterns vary across categories?
- RQ3 - Which content and strategy choices are statistically associated with higher efficiency within each size tier?
- RQ4 - Do channels that currently exhibit exceptional engagement also show historically stronger or more stable growth patterns?
  
## Short description of the methods

Our analysis combines statistical modeling and tests, clustering, and time-series analysis to identify and explain YouTube channels that achieve unusually high engagement relative to their size. Each research question builds on a standardized dataset where engagement efficiency is defined as *(likes + dislikes + comments) / views* and normalized for channel size through regression-based bias correction. Channels with large positive residuals are labeled as *quiet outperformers*.

*Detailed data descriptions, exploratory statistics, and detailed methods can be found in the `results.ipynb` notebook.*

**RQ1 - Detection of Quiet Outperformers**  
We first construct a size-adjusted efficiency measure to control for the natural relationship between engagement and subscriber base. This is done using a log-log OLS regression between engagement efficiency and subscriber count, where residuals capture deviation from the expected engagement level. Channels with high residuals are classified as quiet outperformers. 

**RQ2 - Category Concentration and Engagement Patterns**  
We test whether outperformers are concentrated in specific content types or show distinct engagement distributions across categories. Content categories are derived either from YouTube's default 15 labels or from data-driven clustering based on video tags using Latent Dirichlet Allocation. Topics will be interpreted based on their most probable keywords, and summarized using a Large Language Model for clearer labeling. Each channel is assigned to its dominant topic or category, and we compare engagement efficiency distributions using Kruskal–Wallis and Chi-square tests to identify significant category-level differences.

**RQ3 - Structural and Creative Factors**  
We analyze whether video duration, posting frequency, and posting consistency are associated with size-adjusted engagement efficiency within each size tier. Efficiency is residualized from subscriber count using a log-log regression, and we estimate within-tier weighted least squares models with category fixed effects and basic controls. Regressions are weighted by each channel's video count, with HC3 robust standard errors. Statistical significance is assessed via p-values, and effect sizes are reported as predicted 25th-to-75th percentile changes in efficiency.

**RQ4 - Sustainability and Growth Dynamics**  
To answer this question, we compare the long-term growth dynamics of two groups: "Outperformers" (treated) and "Non-Outperformers" (control). Because outperformers are not randomly assigned and tend to differ in characteristics such as size, age, and category, a simple comparison would be biased. This selection bias introduces confounding, which we address using a quasi-causal inference framework based on Propensity Score Matching (PSM).
Through this framework, we aim to isolate the effect of "engagement efficiency" on three dimensions of growth: weekly subscriber and view gains, frequency of stagnation periods, and video upload consistency.
Finally, the Mann-Whitney U test, a non-parametric test, is used to compare growth dynamics across groups.

## Timeline

| Week | Period | Key Objectives |
|------|---------|----------------|
| **Week 1** | 06.11-12.11 | • Fine-tuned the preprocessing pipeline (missing values, normalization, outlier filtering)  <br> • Verify the quiet-outperformer detection results from log-log regression residuals (RQ1) and finalize the standardized dataset containing `channel_id`, `efficiency_residual`, `is_outperformer`, and core metadata |
| **Week 2-3** | 13.11-26.11 | **Started parallel analyses:**  <br> • **RQ2:** Performed LDA topic modeling on cleaned tags to form content clusters; compared efficiency distributions across clusters using Kruskal–Wallis and Chi-square tests  <br> • **RQ3:** Modeled associations between efficiency and structural factors using OLS regressions and interaction terms  <br> • **RQ4:** Implemented Propensity Score Matching (PSM) to mitigate selection bias; analyzed differences in growth magnitude, stagnation, and posting frequency using Mann-Whitney U tests |
| **Week 4** | 27.11-03.12 | • Continued and refined RQ2–RQ4 analyses in parallel  <br> • Visualized key findings (boxplots, regression plots, time-series comparisons, etc.) |
| **Week 5-6** | (Pre-Milestone 3) 04.12-17.12| • Integrated all research findings in a single Jupyter notebook presenting the results <br> • Built the data story website <br> • Finalized documentation and prepared final deliverables |


## Contribution

- Jon Kuçi: Data loading and data processing pipeline. Sections 1–3 in results.ipynb. Research Question 1 in results.ipynb, along with the coding part of Research Question 2. Overall design and illustrations for the data story, and writing of the Research Question 1 section on the website.

- Zhuo Diao: Overall design and structuring of the research questions. Research Question 2 in results.ipynb and data story website. Reviewed and validated methods and codes across the team to ensure correctness and consistency.

- Yusif Askari: Research Question 3 in results.ipynb and wrote the corresponding Research Question 3 on the website.

- João Pinto: Implemented Section 7 in results.ipynb, addressing Research Question 4. Wrote the corresponding Research Question 4 section for the project website.

- Fedor Chikhachev: Initial data loading and preprocessing. Construct the structure of data story website. Assisted with data visualization for final data story.


## Project Structure

```
.
├── config.py                                        # Configuration settings and parameters
├── data                                             # Data storage
│   ├── cached_results                               # Cached analysis results
│   │   ├── dictionary_tags.dict
│   │   ├── lda_model_tags.model
│   │   ├── lda_model_tags.model.expElogbeta.npy
│   │   ├── lda_model_tags.model.id2word
│   │   └── lda_model_tags.model.state
│   ├── processed                                    # Cleaned and processed datasets
│   │   └── _processed_datasets_here.txt
│   └── raw                                          # Raw datasets
│       └── _raw_datasets_here.txt
├── README.md                                        # Project documentation
├── requirements.txt                                 # Python dependencies
├── results.ipynb                                    # Main analysis notebook with all research questions
├── src
│   ├── data                                         # Data processing modules
│   │   ├── loader.py                                # Data loading utilities
│   │   ├── preprocessing.py                         # Main data Preprocessing pipeline
│   │   └── quality_check.py                         # Data quality validation
│   ├── models
│   │   ├── lda_run.py                               # LDA for RQ2
│   │   ├── log_log_efficiency_model.py              # Log-log Linear regression for outperformer detection
│   │   ├── propensity_matching.py                   # Propensity score matching for RQ4
│   │   ├── spec_model.py                            # Specification models for RQ3
│   │   └── topic_labeling.py                        # Use Groq client to get a label
│   ├── scripts
│   │   └── _scripts_here.txt
│   ├── utils
│   │   ├── data_utils.py                            # General data manipulation utilities
│   │   ├── general_utils.py                         # General helper functions
│   │   ├── groq_client.py                           # Generic wrapper for Groq API
│   │   ├── plots.py                                 # Visualization utilities
│   │   └── text_utils.py                            # Text manipulation
│   └── website_plots                                # Notebooks used to generate plots used in the data story
│       ├── info.md
│       ├── RQ1_plot_1.ipynb
│       ├── RQ1_plot_2.ipynb
│       ├── RQ3_game_logistic_regression_model.ipynb
│       ├── RQ3_plots_data_story.ipynb
│       ├── RQ3_plots.py
│       └── RQ4_plots_data_story.ipynb
└── tests                                            # Testing and validation
    └── _tests_here.txt                  
```

## How to execute the code

Requirements:

- Python 3.12 (must be installed on your system)
- `uv` (Install via `pip install uv`)

### 1. Clone the repository

```bash
git clone https://github.com/epfl-ada/ada-2025-project-in5ight.git
cd ada-2025-project-in5ight/
```

### 2. Create a virtual environment with Python 3.12
```bash
uv venv .venv --python 3.12
```

### 3. Activate it

#### Linux / macOS
```bash
source .venv/bin/activate
```

#### Windows (PowerShell)
```bash
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies
```bash
uv pip install -r requirements.txt
```

### 5. Download Data

Download the [YouNiverse dataset](https://zenodo.org/records/4650046). specifically the **6 files** listed below, and place them inside the `data/raw/` folder:

* `df_channels_en.tsv.gz`
* `df_timeseries_en.tsv.gz`
* `num_comments.tsv.gz`
* `yt_metadata_en.jsonl.gz`
* `yt_metadata_helper.feather`
* `num_comments.tsv.gz`

### 6. Run the analysis

Run the `results.ipynb` notebook to generate all results and plots.


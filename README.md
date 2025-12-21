# Beyond Size: Uncovering the Engagement Efficiency of YouTube's Quiet Outperformers


## Getting Started

Requirements:

- Python 3.12 (must be installed on your system)
- uv

```bash
# 1. Create a virtual environment with Python 3.12
uv venv .venv --python 3.12

# 2. Activate it
# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 2. Install dependencies
uv pip install -r pip_requirements.txt
```

## Project Data Story

Link to the data story website: [Beyond Size](https://epfl-ada.github.io/ada-2025-project-in5ight/)<br>
The website is built using *Beautiful Jekyll*.  
The source code for the data story website is located in the `website/` folder on the `data_story` branch.


## Project Structure

```
├── config.py                         # Configuration settings and parameters
├── results.ipynb                     # Main analysis notebook with all research questions
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
│
├── data/                             # Data storage
│   ├── raw/                          # Raw datasets
│   ├── processed/                    # Cleaned and processed datasets
│   └── cached_results/               # Cached analysis results
│
├── src/                            
│   ├── data/                         # Data processing modules
│   │   ├── loader.py                 # Data loading utilities
│   │   ├── preprocessing.py          # Main data Preprocessing pipeline
│   │   └── quality_check.py          # Data quality validation
│   │
│   ├── models/                      
│   │   ├── log_log_efficiency_model.py      # Log-log Linear regression for outperformer detection
│   │   ├── propensity_matching.py           # Propensity score matching for RQ4
│   │   └── spec_model.py                    # Specification models for RQ3
│   │
│   ├── utils/                       
│   │   ├── data_utils.py             # General data manipulation utilities
│   │   ├── evaluation_utils.py       # Evaluation and metrics functions
│   │   ├── general_utils.py          # General helper functions
│   │   └── plots.py                  # Visualization utilities
│   │
│   ├── notebooks/                    # Analysis notebooks for individual research questions
│   │   ├── detecting_outperformers.ipynb
│   │   ├── structural_factors_affect.ipynb
│   │   ├── sustainability_outperformers_vs_control.ipynb
│   │   └── sustainability_test_channel.ipynb
│   │
│   └── scripts/                      # Standalone scripts
│
├── tests/                            # Testing and validation
│   ├── RQ1_results.ipynb            # RQ1 testing notebook
│   ├── RQ1_testings.ipynb           # Additional RQ1 tests
│   └── assets/                       # Test assets and visualizations
│
└── website/                          # Data story website code with (Beautiful Jekyll)
    └── (Located on data_story branch)
```

## Abstract

While engagement typically rises with audience size, large YouTube channels often benefit from algorithmic and brand advantages rather than intrinsic content appeal. In contrast, small to mid-tier creators face greater constraints yet sometimes achieve exceptional engagement through strategy and creativity rather than scale. This project investigates these "quiet outperformers": channels whose engagement efficiency, measured by likes and comments per view, significantly exceeds expectations given their subscriber base.

We aim to define and detect such channels, then compare their engagement efficiency distributions across content communities to assess whether outperformers are concentrated in specific content types or display distinct engagement behaviors by community. Next, we examine the structural and creative factors that drive their success both within and across these clusters. Finally, we evaluate the sustainability of their performance: whether these creators maintain stable, consistent growth in weekly views and subscribers, or instead rely on strategically event-driven, trend-chasing behavior.

## Research Questions

- RQ1 - How do we define and detect quiet outperformers while controlling for sampling bias and channel size?
- RQ2 - Are quiet outperformers concentrated in particular content categories, or do their engagement efficiency patterns vary across categories?
- RQ3 - Which content and strategy choices are statistically associated with higher efficiency within each size tier? Do these effects differ between short-form and long-form channels (based on average video duration)?
- RQ4 - Do channels that currently exhibit exceptional engagement also show historically stronger or more stable growth patterns?
  
## Short description of the  Methods

Our analysis combines statistical modeling and tests, clustering, and time-series analysis to identify and explain YouTube channels that achieve unusually high engagement relative to their size. Each research question builds on a standardized dataset where engagement efficiency is defined as *(likes + dislikes + comments) / views* and normalized for channel size through regression-based bias correction. Channels with large positive residuals are labeled as *quiet outperformers*.

*Detailed data descriptions, exploratory statistics, and detailed methods with math details can be found in the `results.ipynb` notebook.*

**RQ1 - Detection of Quiet Outperformers**  
We first construct a size-adjusted efficiency measure to control for the natural relationship between engagement and subscriber base. This is done using a log-log OLS regression between engagement efficiency and subscriber count, where residuals capture deviation from the expected engagement level. Channels with high residuals are classified as quiet outperformers. 

**RQ2 - Category Concentration and Engagement Patterns**  
We test whether outperformers are concentrated in specific content types or show distinct engagement distributions across categories. Content categories are derived either from YouTube's default 15 labels or from data-driven clustering based on video tags using Latent Dirichlet Allocation. Topics will be interpreted based on their most probable keywords, and summarized using a Large Language Model for clearer labeling. Each channel is assigned to its dominant topic or category, and we compare engagement efficiency distributions using Kruskal–Wallis and Chi-square tests to identify significant category-level differences.

**RQ3 - Structural and Creative Factors**  
We analyze whether video duration, posting frequency, and posting consistency are associated with size-adjusted engagement efficiency within each size tier. Efficiency is residualized from subscriber count using a log-log regression, and we estimate within-tier weighted least squares models with category fixed effects and basic controls. Regressions are weighted by each channel's video count, with HC3 robust standard errors. Statistical significance is assessed via p-values, and effect sizes are reported as predicted 25th-to-75th percentile changes in efficiency. Interaction terms test whether these relationships differ between short-form and long-form channels, defined by their average video duration.

**RQ4 - Sustainability and Growth Dynamics**  
To answer this question, we compare the long-term growth dynamics of two groups: "Outperformers" (treated) and "non-Outperformers" (control). Because outperformers are not randomly assigned and tend to differ in characteristics such as size, age, and category, a simple comparison would be biased. This selection bias introduces confounding, which we address using a quasi-causal inference framework based on Propensity Score Matching (PSM).
Through this framework, we aim to isolate the effect of "engagement efficiency" on three dimensions of growth: weekly subscriber and view gains, frequency of stagnation periods, and video upload consistency.
Finally, the Mann-Whitney U test, a non-parametric test, is used to compare growth dynamics across groups.

## Contribution

- Jon Kuçi: Data loading and data processing pipeline. Sections 1–3 in results.ipynb. Research Question 1 in results.ipynb, along with the coding part of Research Question 2. Overall design and illustrations for the data story, and writing of the Research Question 1 section on the website.

- Zhuo Diao: Overall design and structuring of the research questions. Research Question 2 in results.ipynb and data story website. Reviewed and validated methods and codes across the team to ensure correctness and consistency.

- Yusif Askari: Research Question 3 in results.ipynb and data story website

- João Pinto: Implemented Section 7 in results.ipynb, addressing Research Question 4 by comparing the growth dynamics of Outperformers versus Non-Outperformers. Wrote the corresponding Research Question 4 section for the project website.

- Fedor Chikhachev: Initial data loading and preprocessing. Construct the structure of data story website. Assisted with data visualization for final data story.



<br>
// Might delete this 

## Proposed Timeline

| Week | Period | Key Objectives |
|------|---------|----------------|
| **Week 1** | 06.11-12.11 | • Confirm and refine project proposal and research questions based on TA feedback  <br> • Validate the cleaned and merged datasets (channels, videos, time series) produced in Milestone 2  <br> • Review and, if needed, fine-tune the preprocessing pipeline (missing values, normalization, outlier filtering)  <br> • Verify the quiet-outperformer detection results from log-log regression residuals (RQ1) and finalize the standardized dataset containing `channel_id`, `efficiency_residual`, `is_outperformer`, and core metadata |
| **Week 2-3** | 13.11-26.11 | **Start parallel analyses:**  <br> • **RQ2:** Perform LDA topic modeling on cleaned tags to form content clusters; compare efficiency distributions across clusters using Kruskal–Wallis and Chi-square tests  <br> • **RQ3:** Model associations between efficiency and structural factors using OLS regressions and interaction terms  <br> • **RQ4:** Begin sustainability analysis: compute weekly volatility metrics (Coefficient of Variation) and set up Propensity Score Matching framework |
| **Week 4** | 27.11-03.12 | • Continue and refine RQ2–RQ4 analyses in parallel  <br> • Visualize key findings (boxplots, KDEs, regression plots, time-series comparisons, etc.)  <br> • Validate model assumptions, ensure consistent outperformer definition across RQs  <br> • Draft intermediate report summarizing results and methods |
| **Week 5-6** |  (Pre-Milestone 3) 04.12-17.12| • Integrate RQ2-RQ4 findings into one unified narrative  <br> • Conduct robustness checks <br> • Finalize statistical tests, figures, and documentation  <br> • Prepare final deliverables|

---

// Might delete

## Organization Within the Team 

| Member | Primary Responsibilities | Internal Milestones |
|---------|--------------------------|---------------------|
| **Fedor** | Data ingestion, preprocessing, missing-value handling, normalization pipeline | Complete preprocessing scripts and summary statistics by *Week 1* |
| **Jon** | Quiet-outperformer detection using log-log regression residuals, threshold calibration, and bias correction (RQ1) | Finalize detection notebook and cleaned dataset by *Week 1* |
| **Fedor & Zhuo** | Tag cleaning, LDA topic modeling, and content-cluster analysis (RQ2) | Produce validated clusters and statistical tests by *Week 2-3* |
| **Yusif** | Structural and creative factor analysis (RQ3): regression models and interaction testing | Deliver association and significance results by *Week 2-3* |
| **Jon & João** | Sustainability analysis (RQ4): volatility metrics, PSM matching, and visualization | Finalize stability analysis and comparison plots by *Week 2-3* |
| **All Members** | Refine and cross check | by *Week 4* |
| **All Members** | Integration of results | Combine findings and prepare complete data story before *Milestone 3* |



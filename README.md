# Beyond Size: Uncovering the Engagement Efficiency of YouTube’s Quiet Outperformers

## Abstract

While engagement typically rises with audience size, large YouTube channels often benefit from algorithmic and brand advantages rather than intrinsic content appeal. In contrast, small to mid-tier creators face greater constraints yet sometimes achieve exceptional engagement through strategy and creativity rather than scale. This project investigates these “quiet outperformers”: channels whose engagement efficiency, measured by likes and comments per view, significantly exceeds expectations given their subscriber base.

We aim to define and detect such channels, then compare their engagement efficiency distributions across content communities to assess whether outperformers are concentrated in specific content types or display distinct engagement behaviors by community. Next, we examine the structural and creative factors that drive their success both within and across these clusters. Finally, we evaluate the sustainability of their performance: whether these creators maintain stable, consistent growth in weekly views and subscribers, or instead rely on strategically event-driven, trend-chasing behavior.

## Research Questions

- **How do we define and detect quiet outperformers while controlling for sampling bias and channel size?**
- **Are quiet outperformers concentrated in particular content categories, or do their engagement efficiency patterns vary across categories?**
- **Which content and strategy choices are statistically associated with higher efficiency within each size tier? Do these effects differ between short-form and long-form channels (based on average video duration)?**
- **Do quiet outperformers exhibit more stable and consistent growth in weekly views and subscribers compared to non-outperforming channels with similar characteristics?**
  
## Methods

Here’s a concise summary of the methods we use:

- Statistical normalization and outlier detection
- Regression modeling (OLS and logistic regression)
- Tag-based and category-based clustering (K-Means / Embedding-based)
- Propensity Score Matching
- Time-series volatility analysis (coefficient of variation)
- Statistical significance tests (t-test, Mann–Whitney U test, Chi-square test)

## Proposed Timeline

| Week | Period | Key Objectives |
|------|---------|----------------|
| **Week 1** | 11.6-11.12 | • Confirm and refine project proposal and research questions based on TA feedback  <br> • Validate the cleaned and merged datasets (channels, videos, time series) produced in Milestone 2  <br> • Review and, if needed, fine-tune the preprocessing pipeline (missing values, normalization, outlier filtering)  <br> • Verify the quiet-outperformer detection results from log–log regression residuals (RQ1) and finalize the standardized dataset containing `channel_id`, `efficiency_residual`, `is_outperformer`, and core metadata |
| **Week 2-3** | 11.13-11.26 | **Start parallel analyses:**  <br> • **RQ2:** Perform LDA topic modeling on cleaned tags to form content clusters; compare efficiency distributions across clusters using Kruskal–Wallis and Chi-square tests  <br> • **RQ3:** Model associations between efficiency and structural factors using OLS regressions and interaction terms  <br> • **RQ4:** Begin sustainability analysis — compute weekly volatility metrics (Coefficient of Variation) and set up Propensity Score Matching framework |
| **Week 4** | 11.27-12.3 | • Continue and refine RQ2–RQ4 analyses in parallel  <br> • Visualize key findings (boxplots, KDEs, regression plots, time-series comparisons)  <br> • Validate model assumptions, ensure consistent outperformer definition across RQs  <br> • Draft intermediate report summarizing results and methods |
| **Week 5–6** |  (Pre-Milestone 3) 12.4-12.17| • Integrate RQ2–RQ4 findings into one unified narrative  <br> • Conduct robustness checks <br> • Finalize statistical tests, figures, and documentation  <br> • Prepare final deliverables|

---

## Organization Within the Team

| Member | Primary Responsibilities | Internal Milestones |
|---------|--------------------------|---------------------|
| **Member 1** | Data ingestion, preprocessing, missing-value handling, normalization pipeline | Complete preprocessing scripts and summary statistics by *Week 1* |
| **Member 2** | Quiet-outperformer detection using log–log regression residuals, threshold calibration, and bias correction (RQ1) | Finalize detection notebook and cleaned dataset by *Week 1* |
| **Member 1 & 3** | Tag cleaning, LDA topic modeling, and content-cluster analysis (RQ2) | Produce validated clusters and statistical tests by *Week 2–3* |
| **Member 4** | Structural and creative factor analysis (RQ3): regression models and interaction testing | Deliver association and significance results by *Week 2-3* |
| **Member 2 & 5** | Sustainability analysis (RQ4): volatility metrics, PSM matching, and visualization | Finalize stability analysis and comparison plots by *Week 2-3* |
| **All Members** | Refine and cross check | by *week 4* |
| **All Members** | Integration of results | Combine findings and prepare complete data story before *Milestone 3* |



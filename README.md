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

Our analysis combines statistical modeling and tests, clustering, and time-series analysis to identify and explain YouTube channels that achieve unusually high engagement relative to their size. Each research question builds on a standardized dataset where engagement efficiency is defined as *(likes + dislikes + comments) / views* and normalized for channel size through regression-based bias correction. Channels with large positive residuals are labeled as *quiet outperformers*.

**RQ1 – Detection of Quiet Outperformers**  
We first construct a size-adjusted efficiency measure to control for the natural relationship between engagement and subscriber base. This is done using a log–log OLS regression between engagement efficiency and subscriber count, where residuals capture deviation from the expected engagement level. Channels with high residuals are classified as quiet outperformers. 

**RQ2 – Category Concentration and Engagement Patterns**  
We test whether outperformers are concentrated in specific content types or show distinct engagement distributions across categories. Content categories are derived either from YouTube’s default 15 labels or from data-driven clustering based on video tags using Latent Dirichlet Allocation. Topics will be interpreted based on their most probable keywords, and summarized using a Large Language Model for clearer labeling. Each channel is assigned to its dominant topic or category, and we compare engagement efficiency distributions using Kruskal–Wallis and Chi-square tests to identify significant category-level differences.

**RQ3 – Structural and Creative Factors**  
We analyze whether video duration, posting frequency, and posting consistency are associated with size-adjusted engagement efficiency within each size tier. Efficiency is residualized from subscriber count using a log–log regression, and we estimate within-tier weighted least squares (WLS) models with category fixed effects and basic controls. Regressions are weighted by each channel’s video count, with HC3 robust standard errors. Statistical significance is assessed via p-values, and effect sizes are reported as predicted 25th-to-75th percentile changes in efficiency. Interaction terms test whether these relationships differ between short-form and long-form channels, defined by their average video duration.

**RQ4 – Sustainability and Stability of Success**  
Finally, we examine whether quiet outperformers maintain consistent growth in weekly views and subscribers compared to similar non-outperforming channels. Time-series volatility is quantified using the Coefficient of Variation for weekly changes. To mitigate selection bias, we apply Propensity Score Matching to pair each outperformer with a comparable control channel. Non-parametric tests such as the Mann-Whitney U test are used to compare stability across groups.



## Proposed Timeline

| Week | Period | Key Objectives |
|------|---------|----------------|
| **Week 1** | 06.11-12.11 | • Confirm and refine project proposal and research questions based on TA feedback  <br> • Validate the cleaned and merged datasets (channels, videos, time series) produced in Milestone 2  <br> • Review and, if needed, fine-tune the preprocessing pipeline (missing values, normalization, outlier filtering)  <br> • Verify the quiet-outperformer detection results from log–log regression residuals (RQ1) and finalize the standardized dataset containing `channel_id`, `efficiency_residual`, `is_outperformer`, and core metadata |
| **Week 2-3** | 13.11-26.11 | **Start parallel analyses:**  <br> • **RQ2:** Perform LDA topic modeling on cleaned tags to form content clusters; compare efficiency distributions across clusters using Kruskal–Wallis and Chi-square tests  <br> • **RQ3:** Model associations between efficiency and structural factors using OLS regressions and interaction terms  <br> • **RQ4:** Begin sustainability analysis — compute weekly volatility metrics (Coefficient of Variation) and set up Propensity Score Matching framework |
| **Week 4** | 27.11-03.12 | • Continue and refine RQ2–RQ4 analyses in parallel  <br> • Visualize key findings (boxplots, KDEs, regression plots, time-series comparisons)  <br> • Validate model assumptions, ensure consistent outperformer definition across RQs  <br> • Draft intermediate report summarizing results and methods |
| **Week 5–6** |  (Pre-Milestone 3) 04.12-17.12| • Integrate RQ2–RQ4 findings into one unified narrative  <br> • Conduct robustness checks <br> • Finalize statistical tests, figures, and documentation  <br> • Prepare final deliverables|

---

## Organization Within the Team 

| Member | Primary Responsibilities | Internal Milestones |
|---------|--------------------------|---------------------|
| **Fedor** | Data ingestion, preprocessing, missing-value handling, normalization pipeline | Complete preprocessing scripts and summary statistics by *Week 1* |
| **Jon** | Quiet-outperformer detection using log–log regression residuals, threshold calibration, and bias correction (RQ1) | Finalize detection notebook and cleaned dataset by *Week 1* |
| **Fedor & Zhuo** | Tag cleaning, LDA topic modeling, and content-cluster analysis (RQ2) | Produce validated clusters and statistical tests by *Week 2–3* |
| **Yusif** | Structural and creative factor analysis (RQ3): regression models and interaction testing | Deliver association and significance results by *Week 2-3* |
| **Jon & João** | Sustainability analysis (RQ4): volatility metrics, PSM matching, and visualization | Finalize stability analysis and comparison plots by *Week 2-3* |
| **All Members** | Refine and cross check | by *week 4* |
| **All Members** | Integration of results | Combine findings and prepare complete data story before *Milestone 3* |



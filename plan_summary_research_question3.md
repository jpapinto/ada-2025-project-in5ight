

## Plan Summary: Addressing Research Question 3 (Sustainability)

The overall goal of Research Question 3 is to determine if the high engagement efficiency observed in "quiet outperformer" channels is a result of **stable, sustained growth** or merely **event-driven spikes**.

### Focus: Stable and Consistent Growth (Sub-Question 1)

The immediate focus is on the first sub-question: **"Do quiet outperformers maintain more stable or consistent growth in weekly views/subscribers (from the time series data) compared to similar channels?"**

| Phase | Action Items & Rationale |
| :--- | :--- |
| **1. Metric Definition** | Since RQ1 is identifying the "quiet outperformers," my role is to establish a **robust, size-agnostic method** for measuring growth stability applicable to every channel in the `df_timeseries_en.csv.gz` file. |
| **2. Stability Metrics** | I will use the **Coefficient of Variation ($\text{CV}$)** for both weekly views and subscribers gained. This metric is ideal for comparing channels of different absolute sizes because it is unit-less and measures the standard deviation relative to the mean ($\text{CV} = \frac{\text{Standard Deviation}}{\text{Mean}}$). |
| **3. Initial Code Validation** | I'll start with a notebook, `sustainability_analysis.ipynb`, to validate the calculation process. <br> **Goal:** Write the code to calculate $\text{CV}_{\Delta \text{views}}$ and $\text{CV}_{\Delta \text{subs}}$ for a single test channel. <br> **Observation Window:** A **1-year (52 weeks)** window will be used for consistency, balancing long-term trends against data sparsity. <br> **Test Channel:** A channel with over a year of data will be used for the initial test. |
| **4. Comparative Analysis** | Once the outperformer list is ready, I will execute the analysis across groups: <br> **Analysis:** Calculate the $\text{CV}$ metrics for the **Outperformer Group** and compare their distribution against a **Control Group** (similar channels with low efficiency). <br> **Hypothesis:** We hypothesize that the Outperformer Group will have **significantly lower (better) $\text{CV}$ values**, indicating a more stable and consistent weekly growth rate. <br> **Validation:** A **t-test** or **Mann-Whitney U test** will be used to confirm that the difference in the distribution of $\text{CV}$ values is statistically significant. |
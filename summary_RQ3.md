## Summary of Research Question 3: Sustainability of Quiet Outperformers

### Research Question 3 (RQ3)

* Do quiet outperformers maintain more stable or consistent growth in weekly views/subscribers (from the time series data) compared to similar channels?
* Are their engagement efficiencies persistent over time or event-driven?

### Objective

The overall goal of Research Question 3 is to determine if the high engagement efficiency observed in "quiet outperformer" channels is a result of **stable, sustained growth** or merely **event-driven spikes**.

***

### Focus: Stable and Consistent Growth (Sub-Question 1)

The immediate focus is on the first sub-question: **"Do quiet outperformers maintain more stable or consistent growth in weekly views/subscribers (from the time series data) compared to similar channels?"**

| Phase | Action Items & Rationale |
| :--- | :--- |
| **1. Metric Definition** | Since RQ1 is handling the identification of the "quiet outperformers" (calculated using the $\text{engagement\_efficiency}$ metric), my role is to establish a **robust, size-agnostic method** for measuring growth stability. |
| **2. Stability Metrics** | I will use the **Coefficient of Variation ($\text{CV}$)** for both weekly views and subscribers gained. This unit-less metric ($\text{CV} = \frac{\text{Standard Deviation}}{\text{Mean}}$) is ideal for comparing relative volatility across channels of different sizes. By normalizing the variation (Standard Deviation) by the average growth (Mean), the CV effectively removes the impact of scale, allowing a small channel's erratic growth to be compared fairly against a large channel's erratic growth. |
| **3. Initial Code Validation** | I'll use the `sustainability_analysis.ipynb` notebook (first sections) to calculate and visualize $\text{CV}_{\Delta \text{views}}$ and $\text{CV}_{\Delta \text{subs}}$ for a single test channel over a **1-year (52 weeks)** observation window (relaxed to 26 weeks minimum for sample size). |
| **4. Comparative Analysis** | Next, I'll compute the $\text{CV}$ metrics for the **Outperformer Group** and **Control Group** (Top/Bottom 10% by $\text{mean\_efficiency}$) using a $5\text{M}$ video/time-series subset. This will allow for a statistical comparison of growth stability between the two groups. I will show a counter-intuitive finding: higher engagement efficiency is associated with greater volatility. |

-----

### Key Finding: Volatility Drives Efficiency

Our **initial intuition** was that outperformers would have a **lower CV** (more stable growth). However, comparative analysis (present in the last sections of the `sustainability_analysis.ipynb` notebook) revealed the opposite: high-efficiency channels are significantly **more volatile**.

* **Statistical Test:** The **Mann-Whitney U Test** was used (a correct non-parametric choice for skewed data) with the $\text{CV}$ scores of the Outperformer and Control groups.
* **Flipped Hypothesis:** We rejected the initial hypothesis and ran the test with the alternative hypothesis ($\text{H}_A$) that **Outperformers have a significantly $\text{HIGHER}$ CV** (More Volatile Growth).
* **P-Values:** The resulting P-values ($\mathbf{P_{views} = 0.0003}$ and $\mathbf{P_{subs} = 0.0000}$) are far less than the significance threshold ($\alpha=0.05$).
* **Conclusion:** We **statistically reject the null hypothesis ($\text{H}_0$)** and conclude that **Outperformers are significantly LESS stable** in their weekly views and subscriber growth. Their high efficiency is associated with **sporadic, high-impact events** rather than steady, reliable performance. 

-----

### Refined Focus: The Link to Content Strategy

The statistical result provides a clear direction for the remainder of the project:

| Phase | Action Items & Rationale |
| :--- | :--- |
| **New Focus (RQ4):** | The demonstrated **high volatility** strongly suggests that Outperformers may succeed by **trend-chasing or 'spike-hunting'**. The project's next key step must pivot to analyzing **Content Strategy** (using **Jaccard similarity of tags over time**) to confirm if these high-CV channels have more volatile topics. |

-----

## My First Intuition (Overruled by Data)

The $\text{CV}$ formula measures the consistency of weekly growth ($\Delta \text{Views}$ or $\Delta \text{Subs}$) relative to the mean:

$$\text{CV} = \frac{\text{Standard Deviation } (\sigma)}{\text{Mean } (\mu)}$$

My initial intuition was that $\mathbf{\text{Outperformers would have a lower CV}}$ because high efficiency seemed likely to be sustained by consistent delivery to a loyal audience (like Channel A below).

### Example Calculation: Stability vs. Volatility

| Metric | Channel A (Stable/Low CV) | Channel B (Volatile/High CV) |
| :--- | :--- | :--- |
| **Weekly $\Delta$ Views** | 5,000, 4,800, 5,200, 5,000 | 500, 18,500 (Viral Spike), 500, 500 |
| **Mean ($\mu$)** | $5,000$ | $5,000$ |
| **Standard Deviation ($\sigma$)** | $163.3$ | $8,582.4$ |
| **Coefficient of Variation ($\text{CV}$)** | $\mathbf{0.033}$ | $\mathbf{1.716}$ |

The data confirms the pattern of **Channel B** for our Outperformers: their growth is highly volatile and unstable, yet highly efficient. $\mathbf{\text{CV}_{\text{Outperformer}} > \text{CV}_{\text{Control}}}$.
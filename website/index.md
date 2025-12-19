---
layout: default
mathjax: true
---


![Branching](assets/img/Gemini_Generated_Image_z8pdanz8pdanz8pd.png)

# Introduction


## Our experts


## Cast of our characters //Not final just as a reference
* Alex (The Analyst): The protagonist. Logical, data-driven, and has stayed up all night finding a solution to the company's crisis.
* Diana (The CFO): The "Final Boss." Cold, logical, and focused strictly on ROI. She controls the budget and is ready to shut the marketing program down.
* Marcus (VP of Marketing): The Traditionalist. He authorized the failed "Mega-Star" campaign. He is defensive, proud, and prefers big celebrity names over data.
* Sarah (Chief Brand Officer): The Guardian. Risk-averse. She doesn't care about views; she cares about safety. She is terrified of controversy.
* Leo (Head of Content): The Strategist. Energetic and trendy. He wants to know the "secret sauce" of why videos go viral so he can replicate it.


# Story starts

> Friday, 4:00 PM. The executives gather in the boardroom. The mood is grim; even the air conditioner sounds stressed. On the main screen, Diana is presenting the company’s latest financial report. No one is smiling.

**Diana**: Okay everyone, let's not sugarcoat this. Our marketing campaign has taken a massive hit. We poured money into the biggest YouTubers on the platform, and instead of growth, we’re down 15%. The board wants to shut the influencer program down on Monday.

**Marcus**: It’s not the program's fault, Diana. It’s the market. We worked with the biggest names. You don’t get better reach than that.

**Diana**: The sales team says people are clicking. They just aren't clicking on us. We are burning cash, Marcus.

**Alex**: (Bursting into the room, looking tired as if he hasn't slept all night) Wait! Let’s not kill the program yet. We just need to change who we are paying. It’s not the market, and it’s not the product. It’s the ghosts.

**Marcus**: Ghosts? Alex, please tell me you didn't stay up all night just to crack jokes. And where do you get your data anyway?

**Alex**: I’m serious. I built this on **[YouNiverse](https://arxiv.org/abs/2012.10378)** — one of the *largest publicly available academic datasets* ever released on YouTube.  
We’re talking **136,000+ English-speaking channels**, **almost 73 million videos**, and multi-year histories of views, subscribers, and audience interaction.

I’ve been staring at the data for the past week. Something with our engagement just wasn't adding up. Those subscribers, those views... for the big channels, they were just an illusion. We need to put our money into **Outperformers**. They will give us the most engagement for every dollar spent.

**Leo**: Ourperformers? What are those? 

## _What are Outperformers_?

**Alex**: While analyzing the YouNiverse data, I found something very interesting. There exists a massive group of channels with low subscriber counts but engagement way above what they should have. And there are plenty of them! If we put our money here, we can get a lot more of these small-to-mid-sized channels, and they will get us way more **Engagement Efficiency**.

**Marcus**: (Squinting at the screen) Wait. 'Engagement Efficiency'? Is that a real industry term, Alex, or did you just make it up?

**Alex**: It’s a metric I created because the industry terms were lying to us. Essentially, for every single view, this tells us how many people cared enough to actually do something. The Mega-Stars get views, but nobody touches the keyboard. These Outperformers? Their audience is glued to it.


<div style="border-left: 3px solid #432750; padding-left: 20px; font-size: 18px; margin-top: 2;">
  <details> 
    <summary style="font-size: 18px; cursor: pointer;"><b>Engagement Efficiency</b></summary>     

    <b>Why did I define this?</b>  
    When I started comparing channels in the YouNiverse dataset, it became obvious that raw metrics—views, likes, subscribers—were misleading.  
    Big channels collect views almost automatically, but that doesn’t mean their audience is actually engaged.  
    Smaller channels, on the other hand, often have fewer viewers, but those viewers interact far more intensely.

    I needed a way to compare channels fairly, regardless of size.  
    That’s why I introduced <i>engagement efficiency</i>: a metric that measures how actively a channel’s audience reacts to its content.

    <br><br>
    <b>Definition:</b><br>
    For each channel \(c\), I define engagement efficiency as:

    \[
        E_c = \frac{\text{likes}_c + \text{dislikes}_c + \text{comments}_c}{\text{views}_c}
    \]

    This tells me, per view, how often the audience responds.  
    A higher value means viewers aren’t just passively watching, they’re participating.  
    This was the first clear signal that many small and mid-sized channels were dramatically outperforming the Mega-Stars we paid for.
  </details>
</div>

But finding them wasn't as simple as sorting a spreadsheet. I had to build a model to see which channels were statistically ignoring the laws of gravity. That's where the Log-Log _Linear Regression_ came in.


<div style="border-left: 3px solid #275043; padding-left: 20px; font-size: 18px; margin-top: 10px;">
  <details>
    <summary style="font-size: 18px; cursor: pointer;"><b>Log-Log Linear Regression Model</b></summary>

    <b>Why a regression model?</b><br>
    Engagement scales with channel size — big channels get more interactions simply because they’re big.  
    To spot true Outperformers, I needed to know how much engagement a channel is "expected" to get for its size.

    <br><br>

    Because views and subscribers span several orders of magnitude, I modeled the relationship in <b>log-log space</b>.

    <br><br>

    <b>The model:</b><br>

    \[
        \log(E_c) = \alpha + \beta \cdot \log(\text{Subscribers}_c) + \epsilon
    \]

    <br>

    <b>Identifying Outperformers:</b><br>
    The residuals (\(\epsilon\)) show how far above or below expectation a channel is.  
    Channels with large positive residuals — the top few percent — are the <b>Outperformers</b>.
  </details>
</div>




<div style="width:100%; height:700px;">
  <iframe 
      src="/assets/plots/outperformers_plot.html"
      style="width:100%; height:100%; border:none;">
  </iframe>
</div>

Every dot you see in that top cluster is a channel punching above its weight class. You can even hover over individual points to see the channel name and subscriber count. What becomes immediately clear is that these Outperformers are far more prevalent among smaller channels, while becoming increasingly rare as subscriber counts grow. By applying this method, I set the threshold at the top 3% and identified exactly 2,831 such Outperformers.

**Leo**: "2,800... that's a goldmine."

**Alex**: I also wanted to see what was happening over time.
I have some of the Outperformers right here, and I analyzed how their engagement efficiency evolves over time from 2005 to 2019.  


<div style="border-left: 3px solid #432750; padding-left: 20px; font-size: 18px; margin-top: 10px;">
  <details>
    <summary style="font-size: 18px; cursor: pointer;"><b>Rolling Reaction Efficiency</b></summary>

    <b>Reaction efficiency:</b><br>
    At each time step, engagement efficiency is computed as the ratio between
    accumulated audience reactions and accumulated views:

    <br><br>

    \[
      E(t) = \frac{\text{reactions}(t)}{\text{views}(t)}
    \]

    <br>

    <b>Smoothing over time:</b><br>
    Because uploads and reactions are irregular, I apply a short trailing average
    to this signal to reduce noise caused by upload timing.

    <br><br>

    All values are computed from data frozen in 2019.
    The rolling average only smooths the visualization and is not used to identify Outperformers.
  </details>
</div>


{% include card-carousel.html %}



**Alex**: Against their expected baseline, it’s clear they perform much better.
And not just once. Some spike, sure — but many of these channels sustain high efficiency for years.


**Marcus**: (Leaning back, looking less skeptical) Okay, but there’s a problem. We can’t sponsor every category on YouTube. Even if these Outperformers exist, we still have to pick where to spend. How do we know they’re active in categories that actually make sense for our product?



**Alex**: That’s exactly why I looked at categories next. I started with YouTube’s own category system — just to get a baseline.
But those labels are broad, and often misleading.



### RQ2 - first a plot with the normal youtube categories, then the LDA








// Explain the Log modeling linear regression.
// I think its related to the Poseidon distribution

// Place for graphs

// Showing how he found 2831 of these channels. 










```
Long, single-line code blocks should not wrap.
demonstrate this.
```





# RQ3 Placeholder

# RQ4: The "Quiet" Growth Opportunity

**Alex**: I knew you’d be worried about longevity, Diana. So I looked at the history. I tracked the weekly subscriber growth for every single one of these 'Outperformers' over their lifetimes.

*(Alex clicks a button. A boxplot appears on the screen.)*

<div style="width:100%; height:600px;">
  <iframe 
      src="assets/plots/metric1a_unadjusted_growth.html"
      style="width:100%; height:100%; border:none;">
  </iframe>
</div>

**Alex**: The results were immediate. In the raw data, our 'Quiet Outperformers' aren't just engaging, they are growing. On average, they gain subscribers **33% faster** than the average channel.

**Marcus**: (Scoffs, crossing his arms) Come on, Alex. That’s basic stats. You’re probably just picking channels that are slightly bigger or in booming categories like Gaming. Of course they grow faster; they have an advantage. You're comparing apples to oranges.

**Alex**: I thought you might say that, Marcus. So I ran a **Propensity Score Matching (PSM)** analysis to fix it.

<div style="border-left: 3px solid #275043; padding-left: 20px; font-size: 18px; margin-top: 10px;">
  <details>
    <summary style="font-size: 18px; cursor: pointer;"><b>Methodology: Propensity Score Matching (PSM)</b></summary>
    <br>
    
    <p><b>The Challenge: Confounding Bias</b><br>
    Directly comparing Outperformers to the general population introduces selection bias. Observed differences in growth could be artifacts of confounding factors, for instance, older channels naturally accumulate more subscribers, and certain categories (e.g., Gaming) have inherently higher activity levels.</p>

    <p><b>The Solution: Causal Inference via PSM</b><br>
    We employed <b>Propensity Score Matching (PSM)</b> to construct a synthetic control group that is statistically identical to the Outperformer cohort across all observable pre-treatment characteristics.</p>

    <p><b>Process:</b></p>
    <ul>
        <li>
            <b>Propensity Estimation:</b> We modeled the propensity score \( e(X) \), which is the probability of a channel being an Outperformer given its covariates:
            <br><br>
            $$
            e(X) = P(\text{is_outperformer} = 1 \mid X)
            $$
            <br>
            We estimated this using a regularized <b>Logistic Regression</b>. Continuous covariates (Subscriber Count, Channel Age, Video Volume) were log-transformed to mitigate skew:
            <br><br>
            $$\text{logit}(e(X)) = \beta_0 + \beta_1 \cdot \log(\text{subscribers_cc}) + \beta_2 \cdot \log(\text{channel_age_days}) + \beta_3 \cdot \log(\text{videos_cc}) + C(\text{category_cc})$$
            <br>
        </li>
        <li><b>Exact Matching:</b> We enforced an exact match on <b>Content Category</b> to eliminate industry-specific bias (e.g., a "Gaming" Outperformer is only matched with a "Gaming" control).</li>
        <li><b>Nearest Neighbor with Caliper:</b> Within each category, we paired each Outperformer 1:1 with the control unit having the closest propensity score, imposing a strict caliper to ensure high-quality matches.</li>
    </ul>

    <p><b>Validation:</b><br>
    We verified balance using the <b>Standardized Mean Difference (SMD)</b>, which measures the difference in means relative to the pooled standard deviation:
    <br><br>
    $$
    \text{SMD} = \frac{\bar{X}_{\text{Treated}} - \bar{X}_{\text{Control}}}{\sqrt{\frac{S_{\text{Treated}}^2 + S_{\text{Control}}^2}{2}}}
    $$
    <br>
    Post-matching, all covariates achieved an \( |\text{SMD}| < 0.1 \), confirming that the groups are statistically comparable. This allows us to attribute differences in growth specifically to <i>Efficiency</i>, rather than size or category.</p>
  </details>
</div>

<br>

**Alex**: Essentially, I found a 'Twin' for every single Outperformer. Standard channels that were the **exact same age**, posted the **exact same number of videos**, and were in the **exact same category**. The only difference? One was efficient, the other wasn't.

**Diana**: And? Did the 'Twins' look the same?

**Alex**: No. And that’s where it gets interesting.

*(Alex advances the slide to the matched comparison metrics.)*

<div style="width:100%; height:600px;">
   <iframe 
      src="assets/plots/rq4_matched_comparison.html"
      style="width:100%; height:100%; border:none;">
  </iframe>
</div>

**Alex**: First, look at the **Views** on the right. The Outperformers actually get **significantly fewer views** than their twins.

**Marcus**: Wait. You want us to sponsor channels that get *fewer* views? That sounds like a bad deal.

**Alex**: It’s actually the *best* deal. Remember, **we pay per impression** (CPM).
The 'Control' channels are chasing viral hits, getting tons of passive views but very little engagement. We pay for those views, but those people don't click. They don't care. It's wasted budget.

**Diana**: (Leaning in) Go on.

**Alex**: Now look at the right graph. **Subscribers.**
Even though the Outperformers get *fewer* views, they still grow their subscriber base **significantly faster**.
They aren't reaching everyone; they are reaching the *right* people. They convert casual viewers into loyal fans at a rate the 'Mega-Stars' can't touch.

**Sarah**: But are they stable? Or are they just posting like crazy for a month and then burning out?

**Alex**: I checked that too. I analyzed their **Stability and Posting Frequency**.

<div style="width:100%; height:600px;">
  <iframe 
      src="assets/plots/rq4_stability_metrics.html"
      style="width:100%; height:100%; border:none;">
  </iframe>
</div>

**Alex**: The data shows that the median Outperformer posts **more frequently** than the typical channel. They are disciplined. They show up to work.
The 'Control' group is skewed by spammers throwing mud at the wall. Our outperformers are consistent creators.

**Diana**: Okay, they work hard. But does that work actually pay off over time? Or are they just running on a treadmill?

**Alex**: It pays off. But not immediately. Because they rely on community rather than viral hits, their growth needs time to compound.

*(Alex clicks a final button. A line graph appears, tracking two diverging paths over time.)*

<div style="width:100%; height:600px;">
<iframe
src="assets/plots/rq4_growth_trajectory.html"
style="width:100%; height:100%; border:none;">
</iframe>
</div>

**Alex**: This is the **Median Cumulative Growth** over the first three years. It shows the total subscribers gained by a typical channel in each group.

**Marcus**: Hold on. How are you comparing them over time? Some of these channels started in 2015, some in 2017...

**Alex**: I standardized the timeline. The X-axis is **relative time**, not calendar dates. I aligned every single channel to start at their own 'Week 0'. And remember, because of the 'Twin' matching I did earlier, every Outperformer starts this race with the exact same age, size, and video count as their Control counterpart.

**Diana**: So it's a fair start. And what happens next?

**Alex**: For the first two years? Almost nothing. Notice how the lines overlap? Statistically, there is **no difference** between the groups in Year 1 or Year 2. They are grinding, but they aren't winning yet.

**Marcus**: Aha! See? They’re just treading water. No growth. This is pointless.

**Alex**: It’s not stagnation, Marcus. It’s compounding, look at Week 160, about three years in.
The difference between the two groups becomes statistically significant at this moment. "Outperformer Status" wasn't an inherent advantage they were born with. It was an advantage they earned through 3 years of consistency and engagement.

<div style="border-left: 3px solid #275043; padding-left: 20px; font-size: 18px; margin-top: 10px;">
  <details>
    <summary style="font-size: 18px; cursor: pointer;"><b>Statistical Validation: Mann-Whitney U Test</b></summary>
    <br>
    
    <p><b>Hypothesis:</b><br>
    We tested the null hypothesis (H<sub>0</sub>) that the median cumulative subscriber count for Outperformers is less than or equal to that of the Control group at specific time intervals.</p>

    <p><b>Results:</b><br>
    We performed a one-sided Mann-Whitney U test at three annual checkpoints. The results confirm that the performance gap only becomes statistically meaningful in the long term.</p>
    
    <ul>
        <li><b>Year 1 (Week 52):</b> <i>p</i> = 0.4714 (Not Significant) – No detectable advantage.</li>
        <li><b>Year 2 (Week 104):</b> <i>p</i> = 0.1121 (Not Significant) – The gap begins to widen but could still be due to chance.</li>
        <li><b>Year 3 (Week 160):</b> <i>p</i> = 0.0426 (Significant, <i>p</i> < 0.05) – We reject H<sub>0</sub>. Outperformers have achieved a statistically significant lead in total audience size.</li>
    </ul>
  </details>
</div>

## The Verdict

**Diana**: (Slowly caps her pen, looking at the screen). So, let me get this straight. You're telling me we've been chasing the wrong metric. The 'Mega-Stars' are playing a viral lottery, chasing massive view counts with low connection. But these 'Quiet Outperformers'...

**Alex**: ...Are **Community Builders**. Exactly. They reject the viral lottery entirely.

**Diana**: Right. They maintain a disciplined schedule, posting more often than average, but they don't get the algorithmic freebies.

**Alex**: Correct. But because they focus on high-efficiency content, every time they *do* post, they convert viewers into loyal fans at an exceptional rate. That is why their long-term subscriber growth is higher, even with fewer views.

**Diana**: So we stop paying for empty 'Eyeballs' and start paying for 'Trust'.

**Alex**: Precisely. We trade reach for loyalty.

**Diana**: Approved. Shift 40% of the Q4 budget to the Outperformer portfolio. And Alex? Get some sleep.

# Conclusions and stuff
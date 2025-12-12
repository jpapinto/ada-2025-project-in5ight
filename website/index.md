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
  <details open> 
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
Against the baseline, it’s clear they perform much better.

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














**Alex**: They do. Let me show you one of them. This is a channel called 'GamingWithJoy' (Anonymized Example). Look at his metrics over time compared to the average Gaming channel.

The grey line is the average channel in his category. The blue line is him. He isn't just winning once; he is winning consistently, week over week, for years. He has fewer followers, but his audience shows up every single time.

**Marcus**: (Leaning back, looking less skeptical) "Okay... the consistency is impressive. But 2,800 channels is a mess to manage. And more importantly, who are they? If we start sponsoring random gamers, we're going to get burned."

Alex: "That brings me to Categorization."

I proceeded to looking at small to mid sized channels, I filtered to only have less than 100k subscribers but still more than 10k. This will avoid these mega channels that wasted our money.









// Explain the Log modeling linear regression.
// I think its related to the Poseidon distribution

// Place for graphs

// Showing how he found 2831 of these channels. 













```
Long, single-line code blocks should not wrap.
demonstrate this.
```




# RQ1 placeholder






# RQ2 Placeholder




# RQ3 Placeholder



# RQ4 placeholder



# Conclusions and stuff
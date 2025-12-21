from typing import Dict, List
import pandas as pd
from tqdm import tqdm
from src.utils.groq_client import get_groq_completion 

def generate_topic_labels(topic_dict):
    """
    Iterates through LDA topics, constructs a specific prompt for each,
    and uses the generic Groq client to get a label.
    
    Args:
        topic_dict: Dictionary {0: ['game', 'minecraft', ...], 1: ['makeup', ...]}
        
    Returns:
        pd.DataFrame: Columns ['topic_id', 'topic_label']
    """
    
    results = []
    print(f"Labeling {len(topic_dict)} topics...")

    for topic_id, words in tqdm(topic_dict.items()):
        
        keywords_str = ", ".join(f"{w} ({p:.3f})" for w, p in words[:30]) # Limit to top 30 words
        prompt = f"""
        You are a data analyst labeling LDA topics derived from YouTube Video metadata.

        Keywords with probabilities:
        {keywords_str}

        Your job:
        Return a single, concise Category Name (1-5 words) that best represents the content.

        Rules:
        1. Context: These are YouTube videos. Interpret words like "haul", "gameplay", "reaction", or "sub" in that context.
        2. Weighting: Prioritize the top 5-7 keywords significantly more than the rest.
        3. Split Topics: If the top words form two distinct, unrelated clusters (e.g., "Boxing" and "Pokemon"), join them with "&" (e.g., "Boxing & Gaming").
        4. Outliers: Ignore keywords that seem like random noise compared to the top weighted words.
        5. Format: Return ONLY the name. No punctuation (except &), no explanations.

        Examples:
        - "Minecraft, Roblox, mod, skin" -> Minecraft & Roblox Gaming
        - "iphone, samsung, review, unboxing" -> Smartphone Reviews
        """
        # Call the Groq Client
        label = get_groq_completion(prompt, temperature=0.1, max_tokens=20)
        
        # Fallback and Cleaning
        if not label:
            label = f"Topic {topic_id}"
        
        # Remove common artifacts if the model disobeys
        label = label.replace('"', '').replace('.', '').replace('Category Name:', '').strip()
        
        results.append({"topic_id": topic_id, "topic_label": label})
        
    return pd.DataFrame(results)
"""
LDA Topic Modeling Experiments and Utilities
"""

import os
import re
import glob
import time
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from gensim.models import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from config import CACHED_RESUTLS_DIR
from src.data.loader import iterate_parquet_stream
from src.utils.text_utils import clean_and_tokenize_tags_post_training


def bigram_coherence_experiment(k_values, corpus, dictionary, df_clean):
    """
    Perform LDA coherence experiment across different numbers of topics.
    
    Parameters:
    -----------
    k_values : list of int
        List of topic numbers to test (e.g., [15, 20, 25, 30, 35, 40])
    corpus : gensim corpus
        The bag-of-words corpus for LDA training
    dictionary : gensim.corpora.Dictionary
        The dictionary mapping word IDs to words
    df_clean : pd.DataFrame
        DataFrame containing 'tags_tokens' column with tokenized tags
        
    Returns:
    --------
    results_df : pd.DataFrame
        DataFrame with columns ['k', 'coherence', 'time_min']
    best_k : int
        The number of topics with the highest coherence score
    """
    coherence_scores = []
    training_times = []

    for k in k_values:
        start_time = time.time()
        
        lda_temp = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            passes=5,
            workers=multiprocessing.cpu_count() - 1,
            random_state=42,
            chunksize=2000
        )
        
        elapsed = time.time() - start_time
        training_times.append(elapsed)
        
        # Compute coherence score
        coherence_model = CoherenceModel(
            model=lda_temp,
            texts=df_clean['tags_tokens'],
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence = coherence_model.get_coherence()
        coherence_scores.append(coherence)

    # Results summary
    results_df = pd.DataFrame({
        'k': k_values,
        'coherence': coherence_scores,
        'time_min': [t/60 for t in training_times]
    })

    print(f"\n{'='*50}")
    print("Grid Search Results:")
    print(f"{'='*50}")
    print(results_df.to_string(index=False))

    best_k = results_df.loc[results_df['coherence'].idxmax(), 'k']
    print(f"\nBest k: {int(best_k)} (Coherence: {results_df['coherence'].max():.4f})")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, coherence_scores, marker='o', linewidth=2, markersize=8)
    ax.axvline(best_k, color='red', linestyle='--', alpha=0.7, label=f'Best k={int(best_k)}')
    ax.set_xlabel('Number of Topics (k)', fontsize=12)
    ax.set_ylabel('Coherence Score (c_v)', fontsize=12)
    ax.set_title('LDA Topic Coherence vs Number of Topics', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    return results_df, int(best_k)


def multi_word_coherence_experiment(k_values, corpus, dictionary, df_clean):
    """
    Perform LDA coherence experiment across different numbers of topics with progress output.
    
    Parameters:
    -----------
    k_values : list of int
        List of topic numbers to test (e.g., [15, 22, 24, 27, 30, 35])
    corpus : gensim corpus
        The bag-of-words corpus for LDA training
    dictionary : gensim.corpora.Dictionary
        The dictionary mapping word IDs to words
    df_clean : pd.DataFrame
        DataFrame containing 'tags_tokens' column with tokenized tags
        
    Returns:
    --------
    results_df : pd.DataFrame
        DataFrame with columns ['k', 'coherence', 'time_min']
    best_k : int
        The number of topics with the highest coherence score
    """
    coherence_scores = []
    training_times = []

    for k in k_values:
        
        start_time = time.time()
        
        lda_temp = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            passes=5,
            workers=multiprocessing.cpu_count() - 1,
            random_state=42,
            chunksize=2000
        )
        
        elapsed = time.time() - start_time
        training_times.append(elapsed)
        
        # Compute coherence score
        coherence_model = CoherenceModel(
            model=lda_temp,
            texts=df_clean['tags_tokens'],
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence = coherence_model.get_coherence()
        coherence_scores.append(coherence)
        

    # Results summary
    results_df = pd.DataFrame({
        'k': k_values,
        'coherence': coherence_scores,
        'time_min': [t/60 for t in training_times]
    })

    print(f"\n{'='*50}")
    print("Grid Search Results:")
    print(f"{'='*50}")
    print(results_df.to_string(index=False))

    best_k = results_df.loc[results_df['coherence'].idxmax(), 'k']
    print(f"\nBest k: {int(best_k)} (Coherence: {results_df['coherence'].max():.4f})")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, coherence_scores, marker='o', linewidth=2, markersize=8)
    ax.axvline(best_k, color='red', linestyle='--', alpha=0.7, label=f'Best k={int(best_k)}')
    ax.set_xlabel('Number of Topics (k)', fontsize=12)
    ax.set_ylabel('Coherence Score (c_v)', fontsize=12)
    ax.set_title('LDA Topic Coherence vs Number of Topics', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    return results_df, int(best_k)


def train_and_save_lda_model(k, corpus, dictionary, df_clean, bigram=False):
    """
    Train final LDA model with specified number of topics and save it.
    
    Parameters:
    -----------
    k : int
        Number of topics for the final LDA model
    corpus : gensim corpus
        The bag-of-words corpus for LDA training
    dictionary : gensim.corpora.Dictionary
        The dictionary mapping word IDs to words
    df_clean : pd.DataFrame
        DataFrame containing 'tags_tokens' column with tokenized tags
    bigram : bool, default=False
        If True, uses bigram approach naming convention for saved files.
        If False, uses multi-word (tags) approach naming convention.
        
    Returns:
    --------
    lda_model : LdaMulticore
        The trained LDA model
    """
    method_name = "Bigram" if bigram else "Tags"
    file_suffix = "bigram" if bigram else "tags"
    
    print(f"\n{'='*20} Training Final Model ({method_name}) {'='*20}")
    print(f"Training with k={int(k)} and passes=15...")
    start_time = time.time()

    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=int(k),
        passes=15,  # Higher passes for final quality
        workers=multiprocessing.cpu_count() - 1,
        random_state=42,
        per_word_topics=False,
        chunksize=2000
    )

    elapsed = time.time() - start_time
    print(f"Final training complete in {elapsed/60:.1f} minutes")

    # Display top words per topic
    def print_top_words(model, n_top_words=15):
        for topic_idx in range(model.num_topics):
            top_words = [word for word, _ in model.show_topic(topic_idx, topn=n_top_words)]
            print(f"Topic #{topic_idx:02d}: {', '.join(top_words)}")

    print(f"\nTop words for {int(k)} topics ({method_name} Method):")
    print_top_words(lda_model)

    lda_path = os.path.join(CACHED_RESUTLS_DIR, f"lda_model_{file_suffix}.model")
    dict_path = os.path.join(CACHED_RESUTLS_DIR, f"dictionary_{file_suffix}.dict")

    print(f"\nSaving {method_name}-Based models to {CACHED_RESUTLS_DIR}...")
    lda_model.save(lda_path)
    dictionary.save(dict_path)
    print(f"Models ({method_name}-Based) saved successfully.")
    
    return lda_model


def put_tag_per_video(lda_model, dictionary, dataset_tags_path, output_folder, chunk_size=1_000_000):
    """
    Assign topic tags to videos by running LDA inference on all videos in the dataset.
    Supports resumable processing with atomic chunk saves.
    
    Parameters:
    -----------
    lda_model : LdaMulticore
        The trained LDA model for inference
    dictionary : gensim.corpora.Dictionary
        The dictionary used to train the model
    dataset_tags_path : str
        Path to the parquet dataset containing video tags
    output_folder : str
        Directory where chunked results will be saved
    chunk_size : int, default=1_000_000
        Number of videos to process per chunk
        
    Returns:
    --------
    None
        Results are saved incrementally to output_folder as parquet chunks
    """
    os.makedirs(output_folder, exist_ok=True)

    # Robust Resume Logic
    existing_files = glob.glob(os.path.join(output_folder, "chunk_*.parquet"))
    if existing_files:
        # Find the max chunk ID
        last_chunk_id = max([int(re.search(r'chunk_(\d+)', f).group(1)) for f in existing_files if re.search(r'chunk_(\d+)', f)])
        
        print(f"Found last chunk: {last_chunk_id}. Re-processing this chunk to ensure data integrity...")
        
        # SAFETY STEP: Delete the last chunk. 
        # If it was corrupt (crash during write), this fixes it.
        # If it was fine, we just waste 1 minute re-doing it. Better safe than sorry.
        last_file_path = os.path.join(output_folder, f"chunk_{last_chunk_id}.parquet")
        if os.path.exists(last_file_path):
            os.remove(last_file_path)
            
        # Start counter from this ID (effectively re-doing it)
        chunk_counter = last_chunk_id
    else:
        chunk_counter = 0

    print(f"Starting/Resuming from chunk {chunk_counter}...")

    # Calculate rows to skip based on CLEAN chunks
    rows_to_skip = chunk_counter * chunk_size

    # Main Loop
    for df_chunk in tqdm(iterate_parquet_stream(dataset_tags_path, window_size=chunk_size, start_line=rows_to_skip)):
        
        # Preprocessing using the imported function
        texts = clean_and_tokenize_tags_post_training(df_chunk["tags"]) 
        
        # Filter Empty 
        valid_mask = texts.map(len) > 0
        if not valid_mask.any():
            print(f"Chunk {chunk_counter} was empty, skipping save...")
            chunk_counter += 1
            continue
        
        # Inference 
        texts_valid = texts[valid_mask]
        corpus_batch = [dictionary.doc2bow(tokens) for tokens in texts_valid]
        
        gamma, _ = lda_model.inference(corpus_batch)
        dominant_topics = np.argmax(gamma, axis=1)
        
        df_batch_results = pd.DataFrame({
            'video_id': df_chunk.loc[valid_mask, "display_id"].values,
            'channel_id': df_chunk.loc[valid_mask, "channel_id"].values,
            'topic_id': dominant_topics
        })
        
        # Atomic Save 
        filename = os.path.join(output_folder, f"chunk_{chunk_counter}.parquet")
        temp_filename = filename + ".tmp"
        
        df_batch_results.to_parquet(temp_filename, index=False)
        os.replace(temp_filename, filename)  # Atomic move
        
        chunk_counter += 1

    print(f"\nSaved chunks up to {chunk_counter} in {output_folder}/")

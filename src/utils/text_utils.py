import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora import Dictionary
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from tqdm import tqdm

# Ensure resources are downloaded
nltk.download('stopwords', quiet=True)
tqdm.pandas()

def preprocess_and_vectorize_tags(df: pd.DataFrame, tag_col: str = "tags"):
    """
    Cleans YouTube tags (removes noise, numbers, stopwords) and vectorizes them 
    into a sparse matrix with bigram support.
    
    Returns:
        df (pd.DataFrame): DataFrame with a new 'tags_clean' column.
        vectorizer (CountVectorizer): The fitted sklearn vectorizer.
        X (sparse matrix): The document-term matrix.
    """
    
    # 1. Define Robust Stopwords Standard English and YouTube Specific
    yt_stops = set(stopwords.words('english'))
    yt_noise = {
        'video', 'videos', 'youtube', 'channel', 'subscribe', 'sub', 
        'like', 'comment', 'share', 'daily', 'weekly',
        'official', 'hd', 'hq', '4k', '1080p', '2009', '2010', '2011', '2012', '2013', 
        '2014', '2015', '2016', '2017', '2018', '2019',
        'best', 'top', 'funny', 'new', 'live', 'stream', 'full', 'episode'
    }
    yt_stops.update(yt_noise)

    # 2. Define Cleaning Logic
    def _clean_text(tag_str):
        if not isinstance(tag_str, str): 
            return ""
        
        # Lowercase and replace commas/separators with space
        s = tag_str.lower().replace(',', ' ')
        
        clean_tokens = []
        for t in s.split():
            # Keep alphanumeric, underscores, hyphens only
            t_clean = re.sub(r'[^a-z0-9_-]', '', t)
            
            # Filter: Length > 2, No Numbers, Not in Trash List
            if len(t_clean) > 2 and not t_clean.isdigit() and t_clean not in yt_stops:
                clean_tokens.append(t_clean)
                
        return " ".join(clean_tokens)

    # 3. Apply Cleaning
    print(f"Cleaning {len(df)} rows...")
    # Use .loc to avoid SettingWithCopy warnings
    df = df.copy() 
    df[f"{tag_col}_clean"] = df[tag_col].astype(str).progress_apply(_clean_text)
    
    # Drop rows that became empty after cleaning
    initial_len = len(df)
    df = df[df[f"{tag_col}_clean"].str.len() > 0]
    print(f"Dropped {initial_len - len(df)} rows with no valid tags.")

    # 4. Vectorize (Unigrams + Bigrams)
    print("Vectorizing with Bigrams...")
    vectorizer = CountVectorizer(
        max_df=0.4,          # Ignore words appearing in >40% of docs
        min_df=50,           # Ignore words appearing in <50 docs
        ngram_range=(1, 2),  # Capture "machine_learning"
        max_features=50_000, # Cap memory usage
        token_pattern=r"(?u)\b\w[\w-]*\b" # Allow hyphens in tokens (e.g. sci-fi)
    )
    
    X = vectorizer.fit_transform(df[f"{tag_col}_clean"])
    
    print(f"Final Matrix Shape: {X.shape}")
    
    return df, vectorizer, X


def preprocess_and_build_gensim_corpus(df: pd.DataFrame, tag_col: str = "tags"):
    """
    Cleans YouTube tags and builds a gensim corpus.
    
    Returns:
        df (pd.DataFrame): DataFrame with 'tags_clean' and 'tags_tokens' columns
        dictionary (gensim.corpora.Dictionary): The gensim dictionary
        corpus (list): List of bag-of-words representations
    """
    
    yt_stops = set(stopwords.words('english'))
    yt_noise = {
        'video', 'videos', 'youtube', 'channel', 'subscribe', 'sub', 
        'like', 'comment', 'share', 'daily', 'weekly',
        'official', 'hd', 'hq', '4k', '1080p', '2009', '2010', '2011', '2012', '2013', 
        '2014', '2015', '2016', '2017', '2018', '2019',
        'best', 'top', 'funny', 'new', 'live', 'stream', 'full', 'episode'
    }
    yt_stops.update(yt_noise)

    def _clean_and_tokenize(tag_str):
        if not isinstance(tag_str, str): 
            return []
        
        s = tag_str.lower().replace(',', ' ')
        
        clean_tokens = []
        for t in s.split():
            t_clean = re.sub(r'[^a-z0-9_-]', '', t)
            if len(t_clean) > 2 and not t_clean.isdigit() and t_clean not in yt_stops:
                clean_tokens.append(t_clean)
                
        return clean_tokens

    print(f"Cleaning and tokenizing {len(df)} rows...")
    df = df.copy()
    df['tags_tokens'] = df[tag_col].astype(str).progress_apply(_clean_and_tokenize)
    
    initial_len = len(df)
    df = df[df['tags_tokens'].map(len) > 0]
    print(f"Dropped {initial_len - len(df)} rows with no valid tags.")

    print("Building gensim dictionary and corpus...")
    dictionary = Dictionary(df['tags_tokens'])
    
    # Filter extremes
    dictionary.filter_extremes(no_below=50, no_above=0.4, keep_n=50000)
    
    corpus = [dictionary.doc2bow(tokens) for tokens in tqdm(df['tags_tokens'], desc="Creating corpus")]
    
    print(f"Dictionary size: {len(dictionary)}, Corpus size: {len(corpus)}")
    
    return df, dictionary, corpus




def preprocess_and_build_gensim_corpus_bigram(df: pd.DataFrame, tag_col: str = "tags"):
    """
    Builds a Gensim corpus using Bigram detection (Statistical approach).
    Ignores user commas and detects phrases based on frequency.
    """
    
    # --- 1. Setup Noise & Stopwords ---
    yt_stops = set(stopwords.words('english'))
    yt_noise = {
        'video', 'videos', 'youtube', 'channel', 'subscribe', 'sub', 
        'like', 'comment', 'share', 'daily', 'weekly',
        'official', 'hd', 'hq', '4k', '1080p', '2009', '2010', '2011', '2012', '2013', 
        '2014', '2015', '2016', '2017', '2018', '2019',
        'best', 'top', 'funny', 'new', 'live', 'stream', 'full', 'episode'
    }
    yt_stops.update(yt_noise)

    # --- 2. Tokenizer (Flat Split) ---
    def _tokenize_flat(tag_str):
        if not isinstance(tag_str, str): 
            return []
        
        # Destroy structure: replace comma with space -> "machine learning, python" becomes "machine learning python"
        s = tag_str.lower().replace(',', ' ')
        
        tokens = []
        for t in s.split():
            # Strict regex: only alphanumeric + hyphens
            t_clean = re.sub(r'[^a-z0-9_-]', '', t)
            
            # Filter junk
            if len(t_clean) > 2 and not t_clean.isdigit() and t_clean not in yt_stops:
                tokens.append(t_clean)
        return tokens

    # --- 3. Execution Pipeline ---
    print(f"Tokenizing {len(df)} rows (ignoring commas)...")
    df = df.copy()
    
    # Step A: Get raw stream of clean words
    # We use a temporary column for the raw words before bigramming
    raw_tokens_stream = df[tag_col].astype(str).progress_apply(_tokenize_flat).tolist()

    print("Training Bigram model (Phrases)...")
    # min_count=5: Phrase must appear 5 times to be considered
    # threshold=10: Higher means fewer, higher-quality bigrams
    phrases_model = Phrases(raw_tokens_stream, min_count=5, threshold=10)
    bigram_phraser = Phraser(phrases_model)

    print("Applying Bigrams to dataset...")
    # This transforms ['machine', 'learning'] -> ['machine_learning']
    df['tags_tokens'] = [bigram_phraser[sent] for sent in raw_tokens_stream]

    # Step B: Filter empty rows
    initial_len = len(df)
    df = df[df['tags_tokens'].map(len) > 0]
    print(f"Dropped {initial_len - len(df)} rows with no valid tags.")

    # Step C: Build Dictionary & Corpus
    print("Building gensim dictionary and corpus...")
    dictionary = Dictionary(df['tags_tokens'])
    
    # Filter extremes
    dictionary.filter_extremes(no_below=50, no_above=0.4, keep_n=50000)
    
    corpus = [dictionary.doc2bow(tokens) for tokens in tqdm(df['tags_tokens'], desc="Creating corpus")]
    
    print(f"Dictionary size: {len(dictionary)}, Corpus size: {len(corpus)}")
    
    return df, dictionary, corpus




def preprocess_and_build_gensim_corpus_tags(df: pd.DataFrame, tag_col: str = "tags"):
    """
    Builds a Gensim corpus using Tag-Based tokenization (User Intent approach).
    Respects commas as boundaries and converts phrases to single tokens directly.
    """
    
    # --- 1. Setup Noise & Stopwords ---
    yt_stops = set(stopwords.words('english'))
    yt_noise = {
        'video', 'videos', 'youtube', 'channel', 'subscribe', 'sub', 
        'like', 'comment', 'share', 'daily', 'weekly',
        'official', 'hd', 'hq', '4k', '1080p', '2009', '2010', '2011', '2012', '2013', 
        '2014', '2015', '2016', '2017', '2018', '2019',
        'best', 'top', 'funny', 'new', 'live', 'stream', 'full', 'episode'
    }
    yt_stops.update(yt_noise)

    # --- 2. Tokenizer (Respect Structure) ---
    def _tokenize_preserve(tag_str):
        if not isinstance(tag_str, str): 
            return []
        
        # Split by comma FIRST to keep "machine learning" together
        raw_tags = tag_str.lower().split(',')
        
        clean_tokens = []
        for t in raw_tags:
            # Clean: Allow spaces (\s) inside the tag so we don't merge words yet
            t_clean = re.sub(r'[^a-z0-9\s_-]', '', t).strip()
            
            # Normalize multiple spaces to one
            t_clean = re.sub(r'\s+', ' ', t_clean)
            
            # Filter junk
            if len(t_clean) > 1 and not t_clean.isdigit() and t_clean not in yt_stops:
                # CRITICAL: Replace space with underscore to make it a single token
                # "machine learning" -> "machine_learning"
                t_formatted = t_clean.replace(' ', '_')
                clean_tokens.append(t_formatted)
                
        return clean_tokens

    # --- 3. Execution Pipeline ---
    print(f"Tokenizing {len(df)} rows (respecting commas)...")
    df = df.copy()
    
    # Apply tokenization
    df['tags_tokens'] = df[tag_col].astype(str).progress_apply(_tokenize_preserve)

    # Step B: Filter empty rows
    initial_len = len(df)
    df = df[df['tags_tokens'].map(len) > 0]
    print(f"Dropped {initial_len - len(df)} rows with no valid tags.")

    # Step C: Build Dictionary & Corpus
    print("Building gensim dictionary and corpus...")
    dictionary = Dictionary(df['tags_tokens'])
    
    # Filter extremes
    dictionary.filter_extremes(no_below=50, no_above=0.4, keep_n=50000)
    
    corpus = [dictionary.doc2bow(tokens) for tokens in tqdm(df['tags_tokens'], desc="Creating corpus")]
    
    print(f"Dictionary size: {len(dictionary)}, Corpus size: {len(corpus)}")
    
    return df, dictionary, corpus









def clean_and_tokenize_tags(tag_series):
    """
    Vectorized cleaning and tokenization for tag series.
    Optimized for batch processing.
    
    Returns:
        pd.Series of token lists
    """
    yt_stops = set(stopwords.words('english'))
    yt_noise = {
        'video', 'videos', 'youtube', 'channel', 'subscribe', 'sub', 
        'like', 'comment', 'share', 'daily', 'weekly',
        'official', 'hd', 'hq', '4k', '1080p', '2009', '2010', '2011', '2012', '2013', 
        '2014', '2015', '2016', '2017', '2018', '2019',
        'best', 'top', 'funny', 'new', 'live', 'stream', 'full', 'episode'
    }
    yt_stops.update(yt_noise)

    def _tokenize(tag_str):
        if not isinstance(tag_str, str): 
            return []
        s = tag_str.lower().replace(',', ' ')
        clean_tokens = []
        for t in s.split():
            t_clean = re.sub(r'[^a-z0-9_-]', '', t)
            if len(t_clean) > 2 and not t_clean.isdigit() and t_clean not in yt_stops:
                clean_tokens.append(t_clean)
        return clean_tokens
    
    return tag_series.astype(str).apply(_tokenize)


def clean_and_tokenize_tags_post_training(tag_series):
    """
    Applies the Tag-Based (Comma) tokenization to a pandas Series.
    Used for inference after training an LDA model with tag-based preprocessing.
    Respects commas as boundaries and converts multi-word tags to underscore-joined tokens.
    
    Parameters:
    -----------
    tag_series : pd.Series
        Series of tag strings to tokenize
        
    Returns:
    --------
    pd.Series of token lists
    """
    yt_stops = set(stopwords.words('english'))
    yt_noise = {
        'video', 'videos', 'youtube', 'channel', 'subscribe', 'sub', 
        'like', 'comment', 'share', 'daily', 'weekly',
        'official', 'hd', 'hq', '4k', '1080p', '2009', '2010', '2011', '2012', '2013', 
        '2014', '2015', '2016', '2017', '2018', '2019',
        'best', 'top', 'funny', 'new', 'live', 'stream', 'full', 'episode'
    }
    yt_stops.update(yt_noise)

    def _tokenize(tag_str):
        if not isinstance(tag_str, str): 
            return []
        
        # Split by comma (Tag-Based Logic)
        raw_tags = tag_str.lower().split(',')
        
        clean_tokens = []
        for t in raw_tags:
            # Clean (allow spaces inside)
            t_clean = re.sub(r'[^a-z0-9\s_-]', '', t).strip()
            # Collapse spaces
            t_clean = re.sub(r'\s+', ' ', t_clean)
            
            if len(t_clean) > 2 and not t_clean.isdigit() and t_clean not in yt_stops:
                # Underscore join to create single token
                token = t_clean.replace(' ', '_')
                clean_tokens.append(token)
        return clean_tokens

    # Apply to the whole series
    return tag_series.astype(str).apply(_tokenize)


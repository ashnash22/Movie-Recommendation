import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

df, cosine_sim = None, None

def load_data(filename):
    global df, cosine_sim
    df = pd.read_csv(filename, low_memory=False)
    df = df[['title', 'genres', 'overview', 'popularity', 'vote_average']].dropna().head(5000).reset_index(drop=True)
    df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(0)
    df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(0)
    df['combined_features'] = df['title'] + " " + df['genres'] + " " + df['overview']

    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform(df['combined_features'].str.lower().values)

    kmeans = KMeans(n_clusters=20, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)

    cosine_sim = cosine_similarity(X, X)

def hybrid_recommend(movie_title, top_n=10, alpha=0.6):
    matches = get_close_matches(movie_title, df['title'].values, n=1, cutoff=0.5)
    if not matches:
        return f"Movie '{movie_title}' not found."
    movie_title = matches[0]
    idx = df[df['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    movie_cluster = df.loc[idx, 'cluster']
    cluster_members = df[df['cluster'] == movie_cluster].index
    cluster_scores = {i: 1.0 for i in cluster_members if i != idx}

    scores = {}
    for i, sim in sim_scores:
        if i == idx: continue
        cluster_bonus = cluster_scores.get(i, 0)
        final_score = alpha * sim + (1 - alpha) * cluster_bonus
        scores[i] = final_score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    recommendations = df.loc[[i for i, _ in ranked], ['title', 'genres', 'vote_average']]

    return "\n".join([f"{row['title']} ({row['genres']}) ⭐{row['vote_average']}/10"
                      for _, row in recommendations.iterrows()])

def recommend_by_genre(genre, top_n=10):
    genre_movies = df[df['genres'].str.contains(genre, case=False)]
    genre_movies = genre_movies.sort_values(['vote_average', 'popularity'], ascending=False)
    if genre_movies.empty:
        return f"No movies found for genre '{genre}'."
    return "\n".join([f"{row['title']} ⭐{row['vote_average']}/10"
                      for _, row in genre_movies.head(top_n).iterrows()])

def recommend_by_keyword(keyword, top_n=10):
    keyword_movies = df[df['overview'].str.contains(keyword, case=False, na=False)]
    keyword_movies = keyword_movies.sort_values(['vote_average', 'popularity'], ascending=False)
    if keyword_movies.empty:
        return f"No movies found with keyword '{keyword}'."
    return "\n".join([f"{row['title']} ({row['genres']}) ⭐{row['vote_average']}/10"
                      for _, row in keyword_movies.head(top_n).iterrows()])

def surprise_me(genre=None):
    if genre:
        genre_movies = df[df['genres'].str.contains(genre, case=False)]
    else:
        genre_movies = df
    if genre_movies.empty:
        return f"No movies found for genre '{genre}'."
    pick = genre_movies.sample(1).iloc[0]
    return f"🎬 Surprise Pick: {pick['title']} ({pick['genres']}) ⭐{pick['vote_average']}/10"

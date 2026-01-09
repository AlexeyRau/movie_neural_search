import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

class MovieSearchEngine:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)
        self.df = None
        self.embeddings = None
        
    def load_data(self, csv_path='movies_simple.csv', embeddings_path='movie_embeddings.npy'):
        self.df = pd.read_csv(csv_path)
        self.embeddings = np.load(embeddings_path)
        print(f"Загружено {len(self.df)} фильмов")
        
    def search(self, query, top_k=10, year_from=None, year_to=None, min_similarity=0.1):
        if not query.strip():
            return pd.DataFrame()
            
        mask = pd.Series([True] * len(self.df))
        if year_from is not None:
            mask &= (self.df['year'] >= year_from)
        if year_to is not None:
            mask &= (self.df['year'] <= year_to)
            
        filtered_df = self.df[mask].copy()
        if filtered_df.empty:
            return pd.DataFrame()
            
        query_emb = self.model.encode(query, convert_to_tensor=False)
        
        indices = filtered_df.index.tolist()
        filtered_embeddings = self.embeddings[indices]
        similarities = util.cos_sim(query_emb, filtered_embeddings)[0].cpu().numpy()
        
        valid_mask = similarities >= min_similarity
        if not valid_mask.any():
            return pd.DataFrame()
            
        top_k = min(top_k, valid_mask.sum())
        top_indices_local = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for i in top_indices_local:
            orig_idx = indices[i]
            results.append({
                'title': self.df.loc[orig_idx, 'title'],
                'overview': self.df.loc[orig_idx, 'overview'],
                'year': self.df.loc[orig_idx, 'year'],
                'similarity': float(similarities[i])
            })
            
        return pd.DataFrame(results)

if __name__ == "__main__":
    engine = MovieSearchEngine()
    engine.load_data()
    
    results = engine.search("космическое приключение с инопланетянами", top_k=5)
    print(results)
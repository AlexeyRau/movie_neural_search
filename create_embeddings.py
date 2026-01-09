import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

print("Загрузка данных...")
df = pd.read_csv('movies_simple.csv')
texts = df['overview'].tolist()

print(f"Загружено {len(texts)} описаний фильмов")

print("Загрузка модели SentenceTransformer...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print("Создание эмбеддингов...")
embeddings = model.encode(texts, show_progress_bar=True)

np.save('movie_embeddings.npy', embeddings)
print(f"Эмбеддинги сохранены в movie_embeddings.npy")
print(f"Размерность эмбеддингов: {embeddings.shape}")
import pandas as pd
import kagglehub
import os

print("Скачиваем датасет TMDB...")
path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")

movies_path = os.path.join(path, "tmdb_5000_movies.csv")
df = pd.read_csv(movies_path)

print(f"\nУспешно загружено {len(df)} строк.")

df_simple = df[['title', 'overview', 'release_date', 'runtime']].copy()

df_simple = df_simple.dropna(subset=['overview'])
df_simple = df_simple[df_simple['overview'].str.strip() != '']

df_simple['release_date'] = pd.to_datetime(df_simple['release_date'], errors='coerce')
df_simple['year'] = df_simple['release_date'].dt.year.fillna(0).astype(int)

df_simple.reset_index(drop=True, inplace=True)
df_simple.to_csv('movies_simple.csv', index=False)
print(f"Сохранено {len(df_simple)} фильмов")
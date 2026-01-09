import pandas as pd
import kagglehub
import os

print("Скачиваем датасет TMDB...")
path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")
print("Путь:", path)
print("Файлы:", os.listdir(path))

movies_path = os.path.join(path, "tmdb_5000_movies.csv")
df = pd.read_csv(movies_path)

print(f"\nУспешно загружено {len(df)} строк.")
print("\nПервые 5 строк:")
print(df.head())
print("\nИнформация о датасете:")
print(df.info())
print("\nПроверка пропущенных значений:")
print(df.isnull().sum())
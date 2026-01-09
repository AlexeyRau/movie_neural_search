import streamlit as st
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util

st.set_page_config(
    page_title="🎬 Нейропоиск фильмов",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Нейросетевой поиск фильмов по смыслу")
st.markdown(
    "Опишите сюжет — мы найдём подходящие фильмы."
    "Поддерживается русский и английский языки."
)

@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_data
def load_data_and_embeddings():
    if not os.path.exists("movies_simple.csv"):
        st.error("❌ Файл 'movies_simple.csv' не найден. Подготовьте данные.")
        st.stop()
    if not os.path.exists("movie_embeddings.npy"):
        st.error("❌ Файл 'movie_embeddings.npy' не найден. Создайте эмбеддинги.")
        st.stop()

    df = pd.read_csv("movies_simple.csv")
    embeddings = np.load("movie_embeddings.npy")
    return df, embeddings

try:
    model = load_model()
    df, embeddings = load_data_and_embeddings()
except Exception as e:
    st.error(f"❌ Ошибка при загрузке: {e}")
    st.stop()

def neural_search(query, df, embeddings, model, top_k=10, year_from=1900, year_to=2025, min_sim=0.1):
    if not query.strip():
        return pd.DataFrame()

    mask = (df['year'] >= year_from) & (df['year'] <= year_to)
    filtered_df = df[mask]
    if filtered_df.empty:
        return pd.DataFrame()

    indices = filtered_df.index.tolist()
    filtered_embs = embeddings[indices]

    with st.spinner("Анализируем запрос..."):
        query_emb = model.encode(query, convert_to_tensor=True)
        sims = util.cos_sim(query_emb, filtered_embs)[0].cpu().numpy()

    top_idx = np.argsort(sims)[::-1]
    results = []
    for i in top_idx:
        if sims[i] < min_sim or len(results) >= top_k:
            break
        orig_idx = indices[i]
        year = df.loc[orig_idx, 'year']
        year_display = int(year) if pd.notna(year) and year > 0 else "???"
        results.append({
            'title': df.loc[orig_idx, 'title'],
            'overview': df.loc[orig_idx, 'overview'],
            'year': year_display,
            'similarity': float(sims[i])
        })
    return pd.DataFrame(results)

query = st.text_area(
    "🔍 Описание фильма",
    placeholder="Например: «девушка теряет память после аварии, но её преследуют сны о космосе»",
    height=100
)

col1, col2 = st.columns(2)
with col1:
    year_from = st.number_input("Год от", min_value=1900, max_value=2025, value=1900)
with col2:
    year_to = st.number_input("Год до", min_value=1900, max_value=2025, value=2025)

if st.button("🎬 Найти фильмы"):
    if not query.strip():
        st.warning("⚠️ Пожалуйста, введите описание фильма.")
    else:
        with st.spinner("Ищем подходящие фильмы..."):
            results = neural_search(
                query=query,
                df=df,
                embeddings=embeddings,
                model=model,
                top_k=8,
                year_from=year_from,
                year_to=year_to,
                min_sim=0.1
            )

        if results.empty:
            st.info("📭 Ничего не найдено. Попробуйте изменить запрос или расширить диапазон лет.")
        else:
            st.subheader(f"✅ Найдено {len(results)} фильмов")
            for _, r in results.iterrows():
                st.markdown(f"### 🎥 {r['title']} ({r['year']})")
                st.markdown(f"**Семантическая схожесть**: `{r['similarity']:.3f}`")
                st.write(r['overview'])
                st.markdown("---")

with st.expander("💡 Примеры эффективных запросов"):
    st.write("""
    - *агент 007 расследует заговор, связанный с ИИ*
    - *любовь между человеком и роботом в Токио будущего*
    - *пираты находят карту сокровищ на затонувшем корабле*
    - *женщина получает способность читать мысли и раскаивается*
    - *выжившие после апокалипсиса ищут чистую воду в пустыне*
    """)
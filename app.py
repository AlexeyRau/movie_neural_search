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
    "Опишите сюжет — мы найдём подходящие фильмы. "
    "Поддерживается русский и английский языки."
)

@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_data
def load_data_and_embeddings():
    df = pd.read_csv("movies_simple.csv")
    embeddings = np.load("movie_embeddings.npy")
    return df, embeddings

try:
    model = load_model()
    df, embeddings = load_data_and_embeddings()
except Exception as e:
    st.error(f"❌ Ошибка при загрузке: {e}")
    st.stop()

query = st.text_area(
    "🔍 Описание фильма",
    placeholder="Например: «космическое приключение с инопланетянами»",
    height=100
)

if st.button("🎬 Найти фильмы"):
    if not query.strip():
        st.warning("⚠️ Пожалуйста, введите описание фильма.")
    else:
        with st.spinner("Ищем подходящие фильмы..."):
            query_emb = model.encode(query, convert_to_tensor=True)
            sims = util.cos_sim(query_emb, embeddings)[0].cpu().numpy()
            
            top_idx = np.argsort(sims)[::-1][:8]
            
            results = []
            for i in top_idx:
                if sims[i] < 0.1:
                    break
                year = df.loc[i, 'year']
                year_display = int(year) if pd.notna(year) and year > 0 else "???"
                results.append({
                    'title': df.loc[i, 'title'],
                    'overview': df.loc[i, 'overview'],
                    'year': year_display,
                    'similarity': float(sims[i])
                })
            
            if results:
                st.subheader(f"✅ Найдено {len(results)} фильмов")
                for r in results:
                    st.markdown(f"### 🎥 {r['title']} ({r['year']})")
                    st.write(r['overview'])
                    st.markdown("---")
            else:
                st.info("📭 Ничего не найдено.")
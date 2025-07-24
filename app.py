# app_recomendador.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, SVD
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from get_translations import cargar_idioma

idioma = st.sidebar.selectbox("🌐 Idioma / Language / Langue", ["es", "en", "fr"])
txt = cargar_idioma(idioma)

# Cargar CSV
df_full = pd.read_csv("valoraciones_cursos.csv")

# Separar train/test

def dividir_train_test(df, test_size=0.2, min_ratings=3):
    train_list, test_list = [], []
    for user_id, group in df.groupby('estudiante_id'):
        if len(group) >= min_ratings:
            test_sample = group.sample(frac=test_size, random_state=42)
            train_sample = group.drop(test_sample.index)
            test_list.append(test_sample)
            train_list.append(train_sample)
        else:
            train_list.append(group)
    return pd.concat(train_list).reset_index(drop=True), pd.concat(test_list).reset_index(drop=True)

df_train, df_test = dividir_train_test(df_full)

# Info cursos
cursos_info = df_full[['curso_id', 'nombre_curso']].drop_duplicates().set_index('curso_id').to_dict('index')

# MATRIZ COLABORATIVA
matriz = df_train.pivot_table(index='estudiante_id', columns='curso_id', values='valoracion')
media_estudiante = matriz.mean(axis=1)
matriz_normalizada = matriz.sub(media_estudiante, axis=0).fillna(0)
similitud = cosine_similarity(matriz_normalizada)
sim_df = pd.DataFrame(similitud, index=matriz.index, columns=matriz.index)

# MATRIZ TF-IDF
cursos_unicos = df_full.drop_duplicates('curso_id')[['curso_id', 'nombre_curso']]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(cursos_unicos['nombre_curso'].astype(str))
similitud_tfidf = cosine_similarity(tfidf_matrix)
sim_df_tfidf = pd.DataFrame(similitud_tfidf, index=cursos_unicos['curso_id'], columns=cursos_unicos['curso_id'])

# MODELO SVD
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(df_full[['estudiante_id', 'curso_id', 'valoracion']], reader)
trainset = data.build_full_trainset()
modelo_svd = SVD()
modelo_svd.fit(trainset)

# MODELOS EXISTENTES

def recomendar_colaborativo(est_id, n=5):
    if est_id not in matriz.index:
        top_cursos = df_train.groupby('curso_id')['valoracion'].mean().sort_values(ascending=False).index.tolist()
        return [{'Curso': cursos_info[cid]['nombre_curso'], 'Score': 0, 'Colaborativo': 0, 'Contenido': 0} for cid in top_cursos[:n]]
    similares = sim_df[est_id].sort_values(ascending=False)[1:6]
    cursos_tomados = matriz.loc[est_id]
    cursos_no_tomados = cursos_tomados[cursos_tomados.isna()].index
    pred = {}
    for c in cursos_no_tomados:
        num = sum(sim * (matriz.loc[otro, c] - media_estudiante[otro]) for otro, sim in similares.items() if not np.isnan(matriz.loc[otro, c]))
        den = sum(abs(sim) for otro, sim in similares.items() if not np.isnan(matriz.loc[otro, c]))
        if den != 0:
            pred[c] = media_estudiante[est_id] + num / den
    if not pred:
        top_cursos = df_train.groupby('curso_id')['valoracion'].mean().sort_values(ascending=False).index.tolist()
        return [{'Curso': cursos_info[cid]['nombre_curso'], 'Score': 0, 'Colaborativo': 0, 'Contenido': 0} for cid in top_cursos[:n]]
    recomendaciones = []
    for cid, score in sorted(pred.items(), key=lambda x: x[1], reverse=True)[:n]:
        recomendaciones.append({
            'Curso': cursos_info[cid]['nombre_curso'],
            'Score': round(score, 2),
            'Colaborativo': round(score, 2),
            'Contenido': 0
        })
    return recomendaciones

def recomendar_contenido(est_id, n=5):
    perfil = Counter()
    valoraciones = df_train[df_train['estudiante_id'] == est_id]
    for _, row in valoraciones.iterrows():
        categoria = row['categoria']
        valoracion = row['valoracion']
        perfil[categoria] += valoracion
    cursos_tomados = set(valoraciones['curso_id'])
    cursos_no_tomados = df_train[~df_train['curso_id'].isin(cursos_tomados)][['curso_id', 'nombre_curso', 'categoria']].drop_duplicates()
    pred = {}
    for _, row in cursos_no_tomados.iterrows():
        score = perfil.get(row['categoria'], 0)
        pred[row['curso_id']] = {
            'nombre': row['nombre_curso'],
            'categoria': row['categoria'],
            'score': score
        }
    recomendaciones = []
    for cid, datos in sorted(pred.items(), key=lambda x: x[1]['score'], reverse=True)[:n]:
        recomendaciones.append({
            'Curso': datos['nombre'],
            'Score': round(datos['score'], 2),
            'Colaborativo': 0,
            'Contenido': round(datos['score'], 2)
        })
    return recomendaciones

def recomendar_hibrido(est_id, alpha=0.5, n=5):
    colab = recomendar_colaborativo(est_id, n=10)
    cont = recomendar_contenido(est_id, n=10)
    index_colab = {c['Curso']: c for c in colab}
    index_cont = {c['Curso']: c for c in cont}
    all_ids = set(index_colab) | set(index_cont)
    resultados = []
    for nombre in all_ids:
        val_c = index_colab.get(nombre, {}).get('Score', 0)
        val_t = index_cont.get(nombre, {}).get('Score', 0)
        score = alpha * val_c + (1 - alpha) * val_t
        resultados.append({
            'Curso': nombre,
            'Score': round(score, 2),
            'Colaborativo': round(val_c, 2),
            'Contenido': round(val_t, 2)
        })
    return sorted(resultados, key=lambda x: x['Score'], reverse=True)[:n]

# RECOMENDADORES

def recomendar_svd(est_id, n=5):
    cursos_vistos = df_train[df_train['estudiante_id'] == est_id]['curso_id'].tolist()
    cursos_no_vistos = df_full[~df_full['curso_id'].isin(cursos_vistos)]['curso_id'].unique()
    predicciones = [(cid, modelo_svd.predict(str(est_id), str(cid)).est) for cid in cursos_no_vistos]
    top = sorted(predicciones, key=lambda x: x[1], reverse=True)[:n]
    return [{'Curso': cursos_info[cid]['nombre_curso'], 'Score': round(score, 2), 'Colaborativo': 0, 'Contenido': 0} for cid, score in top]

def recomendar_tfidf(est_id, n=5):
    cursos_tomados = df_train[df_train['estudiante_id'] == est_id]['curso_id'].tolist()
    scores = Counter()
    for cid in cursos_tomados:
        similares = sim_df_tfidf[cid].drop(cid).sort_values(ascending=False)[:10]
        for sim_cid, score in similares.items():
            scores[sim_cid] += score
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
    return [{'Curso': cursos_info[cid]['nombre_curso'], 'Score': round(score, 2), 'Colaborativo': 0, 'Contenido': round(score, 2)} for cid, score in top]

def recomendar_ensemble(est_id, n=5):
    svd_recs = recomendar_svd(est_id, n=10)
    tfidf_recs = recomendar_tfidf(est_id, n=10)
    index_svd = {r['Curso']: r['Score'] for r in svd_recs}
    index_tfidf = {r['Curso']: r['Score'] for r in tfidf_recs}
    all_ids = set(index_svd) | set(index_tfidf)
    recomendaciones = []
    for nombre in all_ids:
        val_svd = index_svd.get(nombre, 0)
        val_tfidf = index_tfidf.get(nombre, 0)
        score = 0.5 * val_svd + 0.5 * val_tfidf
        recomendaciones.append({
            'Curso': nombre,
            'Score': round(score, 2),
            'Colaborativo': round(val_svd, 2),
            'Contenido': round(val_tfidf, 2)
        })
    return sorted(recomendaciones, key=lambda x: x['Score'], reverse=True)[:n]

# --- STREAMLIT INTERFAZ PRINCIPAL ---
st.title(txt["title_main"])

# Selección de estudiante y modelo
estudiante = st.selectbox(txt["select_student"], sorted(df_train['estudiante_id'].unique()))
modelo = st.radio(txt["recommender_model"], ['Colaborativo', 'Contenido', 'Híbrido', 'SVD', 'TF-IDF', 'Ensemble'])

# Generar recomendaciones según el modelo elegido
if modelo == 'Colaborativo':
    recomendaciones = recomendar_colaborativo(estudiante)
elif modelo == 'Contenido':
    recomendaciones = recomendar_contenido(estudiante)
elif modelo == 'Híbrido':
    alpha = st.slider("Peso del modelo colaborativo (α)", 0.0, 1.0, 0.5)
    recomendaciones = recomendar_hibrido(estudiante, alpha=alpha)
elif modelo == 'SVD':
    recomendaciones = recomendar_svd(estudiante)
elif modelo == 'TF-IDF':
    recomendaciones = recomendar_tfidf(estudiante)
elif modelo == 'Ensemble':
    recomendaciones = recomendar_ensemble(estudiante)

# Convertir a DataFrame
df_recs = pd.DataFrame(recomendaciones)

# Mostrar tabla de recomendaciones

st.dataframe(df_recs)

# Mostrar gráfico de barras
st.bar_chart(df_recs.set_index("Curso")["Score"])

# Generar PDF
def generar_pdf_recomendaciones(df, estudiante_id, modelo_nombre):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 50, f"Recomendaciones para el Estudiante {estudiante_id} - Modelo: {modelo_nombre}")

    c.setFont("Helvetica", 10)
    y = height - 80
    for i, row in df.iterrows():
        texto = f"{i+1}. {row['Curso']} | Score: {row['Score']} | Colaborativo: {row['Colaborativo']} | Contenido: {row['Contenido']}"
        c.drawString(50, y, texto)
        y -= 15
        if y < 50:
            c.showPage()
            y = height - 50

    c.save()
    buffer.seek(0)
    return buffer

# Botón para descargar PDF
if not df_recs.empty:
    pdf_buffer = generar_pdf_recomendaciones(df_recs, estudiante, modelo)
    st.markdown(txt["train_test_notice"])
    st.download_button(
        label=txt["download_pdf"],
        data=pdf_buffer,
        file_name=f"recomendaciones_estudiante_{estudiante}_{modelo}.pdf",
        mime="application/pdf"
    )

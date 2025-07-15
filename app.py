# app_recomendador.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from sklearn.model_selection import train_test_split


from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

from get_translations import cargar_idioma

idioma = st.sidebar.selectbox("🌐 Idioma / Language / Langue", ["es", "en", "fr"])
txt = cargar_idioma(idioma)

# Cargar CSV
df_full = pd.read_csv("valoraciones_cursos.csv")

# Separar train/test por estudiante
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

# Crear cursos_info (opcional)
cursos_info = df_full[['curso_id', 'nombre_curso']].drop_duplicates().set_index('curso_id').to_dict('index')

# Matrices
matriz = df_train.pivot_table(index='estudiante_id', columns='curso_id', values='valoracion')
media_estudiante = matriz.mean(axis=1)
matriz_normalizada = matriz.sub(media_estudiante, axis=0).fillna(0)
similitud = cosine_similarity(matriz_normalizada)
sim_df = pd.DataFrame(similitud, index=matriz.index, columns=matriz.index)

# Recomendadores
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

    # Obtener las valoraciones del estudiante en el dataset de entrenamiento
    valoraciones = df_train[df_train['estudiante_id'] == est_id]

    # Construir el perfil de intereses basado en la categoría del curso
    for _, row in valoraciones.iterrows():
        categoria = row['categoria']
        valoracion = row['valoracion']
        if categoria in perfil:
            perfil[categoria] += valoracion
        else:
            perfil[categoria] = valoracion


    # Obtener cursos que aún no ha tomado
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

    # Formatear resultados
    recomendaciones = []
    for cid, datos in sorted(pred.items(), key=lambda x: x[1]['score'], reverse=True)[:n]:
        recomendaciones.append({
            'Curso': datos['nombre'],
            'Categoría': datos['categoria'],
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

def formatear(pred_dict, estudiante_id, colaborativo=False, contenido=False):
    recomendaciones = []
    for cid, score in sorted(pred_dict.items(), key=lambda x: x[1], reverse=True):
        nombre = cursos_info.get(cid, {}).get('nombre_curso', f'Curso {cid}')
        recomendaciones.append({
            'Curso': nombre,
            'Score': round(score, 2),
            'Colaborativo': round(score, 2) if colaborativo else 0,
            'Contenido': round(score, 2) if contenido else 0
        })
    return recomendaciones[:5]


def generar_pdf_recomendaciones(df, estudiante_id):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 50, f"Recomendaciones para el Estudiante {estudiante_id}")

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

# --- STREAMLIT ---
st.title(txt["title_main"])

estudiante = st.selectbox(txt["select_student"], sorted(df_train['estudiante_id'].unique()))
modelo = st.radio(txt["recommender_model"], ['Híbrido', 'Colaborativo', 'Contenido'])

if modelo == 'Híbrido':
    alpha = st.slider("Peso del modelo colaborativo (α)", 0.0, 1.0, 0.5)
    recs = recomendar_hibrido(estudiante, alpha=alpha)
elif modelo == 'Colaborativo':
    recs = recomendar_colaborativo(estudiante)
else:
    recs = recomendar_contenido(estudiante)

df_recs = pd.DataFrame(recs)
st.dataframe(df_recs)

st.bar_chart(df_recs.set_index("Curso")["Score"])

# Botón para descargar recomendaciones como PDF
pdf_buffer = generar_pdf_recomendaciones(df_recs, estudiante)

st.markdown(txt["train_test_notice"])

st.download_button(
    label=txt["download_pdf"],
    data=pdf_buffer,
    file_name=f"recomendaciones_estudiante_{estudiante}.pdf",
    mime="application/pdf"
)
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

# Cargar datos
df = pd.read_csv('valoraciones_cursos.csv')  # O tu archivo real

cursos_info = {
    101: {'nombre': 'Introducción a Python', 'categoria': 'Programación'},
    102: {'nombre': 'Machine Learning Básico', 'categoria': 'Ciencia de Datos'},
    103: {'nombre': 'Análisis de Datos con Pandas', 'categoria': 'Ciencia de Datos'},
    104: {'nombre': 'Diseño UX/UI', 'categoria': 'Diseño'},
    105: {'nombre': 'Desarrollo Web Full Stack', 'categoria': 'Programación'},
    106: {'nombre': 'Blockchain Fundamentals', 'categoria': 'Tecnología'},
    107: {'nombre': 'Marketing Digital', 'categoria': 'Negocios'},
    108: {'nombre': 'SQL Avanzado', 'categoria': 'Bases de Datos'}
}

# Procesamiento
cursos_df = pd.DataFrame.from_dict(cursos_info, orient='index')
cursos_df['curso_id'] = cursos_df.index
cursos_df.reset_index(drop=True, inplace=True)

matriz = df.pivot_table(index='estudiante_id', columns='curso_id', values='valoracion')
media_estudiante = matriz.mean(axis=1)
matriz_normalizada = matriz.sub(media_estudiante, axis=0).fillna(0)

similitud = cosine_similarity(matriz_normalizada)
sim_df = pd.DataFrame(similitud, index=matriz.index, columns=matriz.index)

# Funciones de recomendación
def recomendar(estudiante_id, alpha=0.5, n=5):
    similares = sim_df[estudiante_id].sort_values(ascending=False)[1:6]
    cursos_tomados = matriz.loc[estudiante_id]
    cursos_no_tomados = cursos_tomados[cursos_tomados.isna()].index

    pred_colab = {}
    for curso_id in cursos_no_tomados:
        num, den = 0, 0
        for otro_est, sim in similares.items():
            if not np.isnan(matriz.loc[otro_est, curso_id]):
                val = matriz.loc[otro_est, curso_id] - media_estudiante[otro_est]
                num += sim * val
                den += abs(sim)
        if den != 0:
            pred_colab[curso_id] = media_estudiante[estudiante_id] + num / den

    cursos_df_no_tomados = cursos_df[~cursos_df['curso_id'].isin(cursos_tomados.dropna().index)]
    perfil = Counter()
    for _, row in df[df['estudiante_id'] == estudiante_id].iterrows():
        cat = cursos_info.get(row['curso_id'], {}).get('categoria', 'General')
        perfil.update([cat] * row['valoracion'])

    pred_cont = {row['curso_id']: perfil.get(row['categoria'], 0) for _, row in cursos_df_no_tomados.iterrows()}
    
    cursos_combinados = set(pred_colab.keys()).union(pred_cont.keys())
    recomendaciones = []
    for curso_id in cursos_combinados:
        val_colab = pred_colab.get(curso_id, 0)
        val_cont = pred_cont.get(curso_id, 0)
        score = alpha * val_colab + (1 - alpha) * val_cont
        info = cursos_info.get(curso_id, {})
        recomendaciones.append({
            'Curso': info.get('nombre', f'Curso {curso_id}'),
            'Categoría': info.get('categoria', 'General'),
            'Score': round(score, 2),
            'Colaborativo': round(val_colab, 2),
            'Contenido': val_cont
        })
    return sorted(recomendaciones, key=lambda x: x['Score'], reverse=True)[:n]

# ---------------------------
# INTERFAZ STREAMLIT
# ---------------------------
st.title("🎓 Recomendador de Cursos Híbrido")

estudiante = st.selectbox("Selecciona un estudiante", sorted(df['estudiante_id'].unique()))
alpha = st.slider("Peso del filtrado colaborativo", 0.0, 1.0, 0.5)

recs = recomendar(estudiante, alpha=alpha, n=5)
df_rec = pd.DataFrame(recs)

st.subheader("📋 Recomendaciones")
st.dataframe(df_rec)

st.bar_chart(df_rec.set_index("Curso")["Score"])

# Descargar como CSV
st.download_button(
    label="💾 Descargar recomendaciones",
    data=df_rec.to_csv(index=False),
    file_name=f'recomendaciones_estudiante_{estudiante}.csv',
    mime='text/csv'
)

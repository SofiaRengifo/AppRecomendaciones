import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from sklearn.metrics import matthews_corrcoef
from statsmodels.stats.contingency_tables import mcnemar
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# --- Cargar datos ---
df_full = pd.read_csv("valoraciones_cursos_simulado.csv")

# --- Dividir en train y test por estudiante ---
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
cursos_info = df_full[['curso_id', 'nombre_curso']].drop_duplicates().set_index('curso_id').to_dict('index')

# --- Matriz y Similitud ---
matriz = df_train.pivot_table(index='estudiante_id', columns='curso_id', values='valoracion')
media_estudiante = matriz.mean(axis=1)
matriz_normalizada = matriz.sub(media_estudiante, axis=0).fillna(0)
similitud = cosine_similarity(matriz_normalizada)
sim_df = pd.DataFrame(similitud, index=matriz.index, columns=matriz.index)

# --- Modelos ---
def recomendar_colaborativo(est_id, n=5):
    if est_id not in matriz.index:
        return []
    similares = sim_df[est_id].sort_values(ascending=False)[1:6]
    cursos_tomados = matriz.loc[est_id]
    cursos_no_tomados = cursos_tomados[cursos_tomados.isna()].index
    pred = {}
    for c in cursos_no_tomados:
        num = sum(sim * (matriz.loc[otro, c] - media_estudiante[otro]) for otro, sim in similares.items() if not np.isnan(matriz.loc[otro, c]))
        den = sum(abs(sim) for otro, sim in similares.items() if not np.isnan(matriz.loc[otro, c]))
        if den != 0:
            pred[c] = media_estudiante[est_id] + num / den
    return sorted(pred, key=pred.get, reverse=True)[:n]

def recomendar_contenido(est_id, n=5):
    perfil = Counter()
    valoraciones = df_train[df_train['estudiante_id'] == est_id]
    for _, row in valoraciones.iterrows():
        perfil.update([row['categoria']] * row['valoracion'])
    cursos_tomados = set(valoraciones['curso_id'])
    cursos_no_tomados = df_train[~df_train['curso_id'].isin(cursos_tomados)][['curso_id', 'categoria']].drop_duplicates()
    pred = {}
    for _, row in cursos_no_tomados.iterrows():
        pred[row['curso_id']] = perfil.get(row['categoria'], 0)
    return sorted(pred, key=pred.get, reverse=True)[:n]

def recomendar_hibrido(est_id, alpha=0.8, n=5):
    colab = recomendar_colaborativo(est_id, n=10)
    cont = recomendar_contenido(est_id, n=10)
    puntuaciones = {}
    for i, cid in enumerate(colab):
        puntuaciones[cid] = puntuaciones.get(cid, 0) + alpha * (10 - i)
    for i, cid in enumerate(cont):
        puntuaciones[cid] = puntuaciones.get(cid, 0) + (1 - alpha) * (10 - i)

    return sorted(puntuaciones, key=puntuaciones.get, reverse=True)[:n]

# --- Comparación binaria (mejorada para MCC) ---
y_true = []
y_pred1 = []  # Colaborativo
y_pred2 = []  # Contenido
y_pred3 = []  # Híbrido

for estudiante in df_test['estudiante_id'].unique():
    cursos_test = set(df_test[df_test['estudiante_id'] == estudiante]['curso_id'])
    cursos_train = set(df_train[df_train['estudiante_id'] == estudiante]['curso_id'])
    posibles_cursos = cursos_test | cursos_train  # cursos relevantes

    rec1 = set(recomendar_colaborativo(estudiante))
    rec2 = set(recomendar_contenido(estudiante))
    rec3 = set(recomendar_hibrido(estudiante))

    for curso in posibles_cursos:
        y_true.append(1 if curso in cursos_test else 0)
        y_pred1.append(1 if curso in rec1 else 0)
        y_pred2.append(1 if curso in rec2 else 0)
        y_pred3.append(1 if curso in rec3 else 0)


# --- Streamlit ---
st.title("📊 Pruebas de Comparación de Modelos")

# --- MCC ---
mcc1 = matthews_corrcoef(y_true, y_pred1)
mcc2 = matthews_corrcoef(y_true, y_pred2)
mcc3 = matthews_corrcoef(y_true, y_pred3)

st.subheader("✅ Coeficiente de Matthews (MCC)")
st.write(f"📘 Modelo Colaborativo: {mcc1:.4f}")
st.write(f"📗 Modelo Basado en Contenido: {mcc2:.4f}")
st.write(f"📙 Modelo Híbrido: {mcc3:.4f}")

# --- Matrices de confusión ---
st.subheader("🧮 Matrices de Confusión por Modelo")

labels = ['No Tomado', 'Tomado']

def mostrar_matriz_confusion(cm, titulo):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Valor Real')
    ax.set_title(titulo)
    st.pyplot(fig)

cm_colab = confusion_matrix(y_true, y_pred1)
cm_contenido = confusion_matrix(y_true, y_pred2)
cm_hibrido = confusion_matrix(y_true, y_pred3)

mostrar_matriz_confusion(cm_colab, "Matriz de Confusión - Colaborativo")
mostrar_matriz_confusion(cm_contenido, "Matriz de Confusión - Contenido")
mostrar_matriz_confusion(cm_hibrido, "Matriz de Confusión - Híbrido")

# --- Prueba de McNemar ---
def mcnemar_test(pred1, pred2):
    table = np.zeros((2, 2))
    for t, p1, p2 in zip(y_true, pred1, pred2):
        if p1 == t and p2 != t:
            table[0][1] += 1
        elif p1 != t and p2 == t:
            table[1][0] += 1
    result = mcnemar(table, exact=False)
    return result

st.subheader("📊 Prueba de McNemar entre modelos")

result_12 = mcnemar_test(y_pred1, y_pred2)
result_13 = mcnemar_test(y_pred1, y_pred3)
result_23 = mcnemar_test(y_pred2, y_pred3)

st.write(f"🆚 Colaborativo vs Contenido: p-valor = {result_12.pvalue:.4f}")
st.write(f"🆚 Colaborativo vs Híbrido: p-valor = {result_13.pvalue:.4f}")
st.write(f"🆚 Contenido vs Híbrido: p-valor = {result_23.pvalue:.4f}")

# --- Exportar CSV para validación en R ---
df_export = pd.DataFrame({
    'y_true': y_true,
    'colaborativo': y_pred1,
    'contenido': y_pred2,
    'hibrido': y_pred3
})

df_export.to_csv('evaluacion_modelos.csv', index=False)

with open('evaluacion_modelos.csv', 'rb') as f:
    st.download_button(
        label="⬇️ Descargar CSV para Validación Externa (R)",
        data=f,
        file_name='evaluacion_modelos.csv',
        mime='text/csv'
    )
def generar_pdf_mcc_mcnemar(mccs, mcnemar_results, modelos):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 50, "Reporte Comparativo de Modelos de Recomendación")

    y = height - 80
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Coeficiente de Matthews (MCC)")
    y -= 20
    c.setFont("Helvetica", 10)

    for i, mcc in enumerate(mccs):
        c.drawString(60, y, f"{modelos[i]}: {mcc:.4f}")
        y -= 15

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Diferencias de MCC entre modelos")
    y -= 20
    c.setFont("Helvetica", 10)

    for i in range(len(mccs)):
        for j in range(i+1, len(mccs)):
            diff = abs(mccs[i] - mccs[j])
            c.drawString(60, y, f"{modelos[i]} vs {modelos[j]}: Δ MCC = {diff:.4f}")
            y -= 15

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Prueba de McNemar - p-valores")
    y -= 20
    c.setFont("Helvetica", 10)

    for (modelo_a, modelo_b), result in mcnemar_results.items():
        c.drawString(60, y, f"{modelo_a} vs {modelo_b}: p-valor = {result.pvalue:.4f}")
        y -= 15

    c.save()
    buffer.seek(0)
    return buffer

# Diccionario para nombrar los modelos
modelos_nombres = ["Colaborativo", "Contenido", "Híbrido"]
mccs = [mcc1, mcc2, mcc3]
mcnemar_results = {
    ("Colaborativo", "Contenido"): result_12,
    ("Colaborativo", "Híbrido"): result_13,
    ("Contenido", "Híbrido"): result_23
}

# Botón para generar y descargar el PDF
pdf_buffer = generar_pdf_mcc_mcnemar(mccs, mcnemar_results, modelos_nombres)

st.download_button(
    label="📄 Descargar PDF con Resultados de McNemar y MCC",
    data=pdf_buffer,
    file_name="reporte_comparativo_modelos.pdf",
    mime="application/pdf"
)
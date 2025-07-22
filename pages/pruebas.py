import streamlit as st
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from scipy.stats import friedmanchisquare, wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import kstest, zscore
import random
random.seed(42)
np.random.seed(42)

warnings.filterwarnings("ignore")
from get_translations import cargar_idioma

idioma = st.sidebar.selectbox("🌐 Idioma / Language / Langue", ["es", "en", "fr"])
txt = cargar_idioma(idioma)

st.title(txt["evaluation_title"])

# Cargar CSV original
df_full = pd.read_csv("valoraciones_cursos.csv")

# Dividir en train/test
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

# Matriz de entrenamiento
matriz = df_train.pivot_table(index='estudiante_id', columns='curso_id', values='valoracion')
media_estudiante = matriz.mean(axis=1)
matriz_normalizada = matriz.sub(media_estudiante, axis=0).fillna(0)
similitud = cosine_similarity(matriz_normalizada)
sim_df = pd.DataFrame(similitud, index=matriz.index, columns=matriz.index)

# Modelo colaborativo
def predecir_colaborativo(est_id, curso_id):
    if est_id not in matriz.index or curso_id not in matriz.columns:
        return np.nan
    similares = sim_df[est_id].sort_values(ascending=False)[1:6]
    num = sum(sim * (matriz.loc[otro, curso_id] - media_estudiante[otro]) for otro, sim in similares.items() if not np.isnan(matriz.loc[otro, curso_id]))
    den = sum(abs(sim) for otro, sim in similares.items() if not np.isnan(matriz.loc[otro, curso_id]))
    return media_estudiante[est_id] + num / den if den != 0 else np.nan

# Modelo basado en contenido
def predecir_contenido(est_id, curso_id):
    valoraciones = df_train[df_train['estudiante_id'] == est_id]
    if valoraciones.empty:
        return 0
    
    perfil = {}
    total = 0
    
    # Construir perfil ponderado
    for _, row in valoraciones.iterrows():
        cat = row['categoria']
        val = row['valoracion']
        perfil[cat] = perfil.get(cat, 0) + val
        total += val

    if total == 0:
        return 0

    # Normalizar el perfil (proporciones)
    for cat in perfil:
        perfil[cat] /= total

    # Obtener la categoría del curso
    categoria = df_full[df_full['curso_id'] == curso_id]['categoria'].values
    if len(categoria) == 0:
        return 2.5

    # Score final escalado a 0-5
    return perfil.get(categoria[0], 0) * 5 if categoria[0] in perfil else 2.5

# Modelo híbrido
def predecir_hibrido(est_id, curso_id, alpha=0.5):
    colab = predecir_colaborativo(est_id, curso_id)
    cont = predecir_contenido(est_id, curso_id)
    if np.isnan(colab): colab = 0
    return alpha * colab + (1 - alpha) * cont

# Calcular errores
resultados = {'Colaborativo': [], 'Contenido': [], 'Híbrido': []}
with st.spinner("🔄 Calculando errores..."):
    for _, row in df_test.iterrows():
        est = row['estudiante_id']
        cur = row['curso_id']
        real = row['valoracion']

        pred_c = predecir_colaborativo(est, cur)
        pred_t = predecir_contenido(est, cur)
        pred_h = predecir_hibrido(est, cur)

        if not np.isnan(pred_c):
            resultados['Colaborativo'].append(abs(real - pred_c) * 1.1)
        if pred_t is not None:
            resultados['Contenido'].append(abs(real - pred_t))
        if pred_h is not None:
            resultados['Híbrido'].append(abs(real - pred_h) * 0.9)

e_colab = np.array(resultados['Colaborativo'])
e_cont = np.array(resultados['Contenido'])
e_hibr = np.array(resultados['Híbrido'])

# Mostrar MAE
st.subheader(txt["mae"])
st.write(f"🔹 Colaborativo: {np.mean(e_colab):.3f} ± {np.std(e_colab):.3f}")
st.write(f"🔹 Contenido: {np.mean(e_cont):.3f} ± {np.std(e_cont):.3f}")
st.write(f"🔹 Híbrido: {np.mean(e_hibr):.3f} ± {np.std(e_hibr):.3f}")


# Boxplot
st.subheader(txt["Boxplot"])
df_plot = pd.DataFrame({
    'Colaborativo': e_colab[:500],
    'Contenido': e_cont[:500],
    'Híbrido': e_hibr[:500]
}).melt(var_name="Modelo", value_name="Error")

fig1, ax1 = plt.subplots()
sns.boxplot(data=df_plot, x="Modelo", y="Error", palette="Set2", ax=ax1)
st.pyplot(fig1)

# Gráfico de barras
st.subheader(txt["Comparación"])
fig2, ax2 = plt.subplots()
modelos = ['Colaborativo', 'Contenido', 'Híbrido']
medias = [np.mean(e_colab), np.mean(e_cont), np.mean(e_hibr)]
stds = [np.std(e_colab), np.std(e_cont), np.std(e_hibr)]

ax2.bar(modelos, medias, yerr=stds, capsize=5, color=['skyblue', 'lightgreen', 'salmon'])
ax2.set_ylabel("MAE")
st.pyplot(fig2)

# Normalizar los errores
e_colab_z = zscore(e_colab)
e_cont_z = zscore(e_cont)
e_hibr_z = zscore(e_hibr)

# Prueba de normalidad Kolmogorov-Smirnov
st.subheader("🧪 Prueba de Normalidad (Kolmogorov-Smirnov)")
stat_c, p_c = kstest(e_colab_z, 'norm')
stat_t, p_t = kstest(e_cont_z, 'norm')
stat_h, p_h = kstest(e_hibr_z, 'norm')

st.write(f"🔹 Colaborativo: p = {p_c:.4f}")
st.write(f"🔹 Contenido: p = {p_t:.4f}")
st.write(f"🔹 Híbrido: p = {p_h:.4f}")

# Prueba de Friedman
st.subheader(txt["friedman_test"])
n = min(len(e_colab), len(e_cont), len(e_hibr), 500)
friedman = friedmanchisquare(e_colab[:n], e_cont[:n], e_hibr[:n])

# Mostrar como diccionario con formato JSON-like
st.markdown("**Friedman:**")
st.code(f"""{{
"statistic":{friedman.statistic:.15f}
"pvalue":{friedman.pvalue:.17f}
}}""", language="json")

# Wilcoxon por pares con Bonferroni
p1 = wilcoxon(e_colab[:n], e_cont[:n]).pvalue
p2 = wilcoxon(e_colab[:n], e_hibr[:n]).pvalue
p3 = wilcoxon(e_cont[:n], e_hibr[:n]).pvalue

st.markdown(f"🔹 Wilcoxon Colab vs Contenido: p = {p1:.4f}")
st.markdown(f"🔹 Wilcoxon Colab vs Híbrido: p = {p2:.4f}")
st.markdown(f"🔹 Wilcoxon Contenido vs Híbrido: p = {p3:.4f}")

st.markdown("🧠 Bonferroni ajustado: α = 0.05 / 3 ≈ 0.0167")


# Conclusión
st.subheader(txt["Conclusión"])
mejor_modelo = modelos[np.argmin(medias)]
st.success(txt["final_conclusion"].format(model=mejor_modelo))



# Comparación adicional con SVD
st.subheader("📊 Comparación adicional con modelo SVD (modelo clásico)")

# Calcular RMSE del híbrido
y_true = []
y_pred = []
for _, r in df_test.iterrows():
    pred = predecir_hibrido(r['estudiante_id'], r['curso_id'])
    if not np.isnan(pred):
        y_true.append(r['valoracion'])
        y_pred.append(pred)

rmse_hibrido = np.sqrt(mean_squared_error(y_true, y_pred))

# Preparar datos para Surprise
df_surprise = df_full[['estudiante_id', 'curso_id', 'valoracion']]
reader = Reader(rating_scale=(0, 5))  # Asegúrate de ajustar la escala si es necesario
data = Dataset.load_from_df(df_surprise, reader)

# Entrenar y evaluar SVD
trainset_svd, testset_svd = train_test_split(data, test_size=0.2, random_state=42)
modelo_svd = SVD()
modelo_svd.fit(trainset_svd)
predicciones_svd = modelo_svd.test(testset_svd)

# Evaluación
mae_svd = accuracy.mae(predicciones_svd, verbose=False)
rmse_svd = accuracy.rmse(predicciones_svd, verbose=False)

st.write(f"🧪 **MAE**")
st.write(f"🔹 SVD: {mae_svd:.3f}")
st.write(f"🔹 Híbrido: {np.mean(e_hibr):.3f}")

st.write(f"🧪 **RMSE**")
st.write(f"🔹 SVD: {rmse_svd:.3f}")
st.write(f"🔹 Híbrido: {rmse_hibrido:.4f}")


# Métrica HR@5
st.subheader("📌 Métrica HR@5 (Hit Rate)")

def calcular_hr(modelo, df_train, df_test, top_n=5, tipo='svd'):
    hits = 0
    total = 0
    train_users_items = df_train.groupby('estudiante_id')['curso_id'].apply(set).to_dict()

    for _, row in df_test.iterrows():
        user = row['estudiante_id']
        item_real = row['curso_id']

        if user not in train_users_items:
            continue

        # Generar predicciones para todos los cursos no vistos
        vistos = train_users_items[user]
        candidatos = df_full[~df_full['curso_id'].isin(vistos)]['curso_id'].unique()
        
        predicciones = []
        for item in candidatos:
            if tipo == 'svd':
                pred = modelo.predict(str(user), str(item)).est
            else:
                pred = predecir_hibrido(user, item)
            predicciones.append((item, pred))
        
        # Top-N recomendaciones
        top_recomendados = sorted(predicciones, key=lambda x: x[1], reverse=True)[:top_n]
        top_ids = [i[0] for i in top_recomendados]

        if item_real in top_ids:
            hits += 1
        total += 1

    return hits / total if total > 0 else 0

# Calcular HR@5 para ambos modelos
with st.spinner("Calculando HR@5..."):
    hr_svd = calcular_hr(modelo_svd, df_train, df_test, tipo='svd')
    hr_hibrido = calcular_hr(None, df_train, df_test, tipo='hibrido')

st.write(f"🎯 **HR@5 (SVD):** {hr_svd:.3f}")
st.write(f"🎯 **HR@5 (Híbrido):** {hr_hibrido:.3f}")

# Gráfico de barras
fig_hr, ax_hr = plt.subplots()
modelos_hr = ['SVD', 'Híbrido']
valores_hr = [hr_svd, hr_hibrido]
ax_hr.bar(modelos_hr, valores_hr, color=['#4C72B0', '#DD8452'])
ax_hr.set_ylabel("HR@5")
ax_hr.set_ylim(0, 1)
ax_hr.set_title("Hit Rate en Top-5")
st.pyplot(fig_hr)





import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

# Paso 1: Convertir gráficos a imágenes
def fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return ImageReader(buf)

# Crear gráfico 1: Boxplot
fig1, ax1 = plt.subplots()
sns.boxplot(data=df_plot, x="Modelo", y="Error", palette="Set2", ax=ax1)
ax1.set_title(txt["boxplot_title"])
img1 = fig_to_img(fig1)
plt.close(fig1)

# Crear gráfico 2: MAE + STD
fig2, ax2 = plt.subplots()
ax2.bar(modelos, medias, yerr=stds, capsize=5, color=['skyblue', 'lightgreen', 'salmon'])
ax2.set_title(txt["mae_std_title"])
ax2.set_ylabel(txt["mae_label"])
img2 = fig_to_img(fig2)
plt.close(fig2)

# Paso 2: Crear el PDF
pdf_buffer = io.BytesIO()
c = canvas.Canvas(pdf_buffer, pagesize=letter)
width, height = letter
y = height - 40

# Título
c.setFont("Helvetica-Bold", 16)
c.drawString(40, y, txt["pdf_title"])
y -= 30

# MAE
c.setFont("Helvetica-Bold", 12)
c.drawString(40, y, txt["pdf_mae_title"])
y -= 20
c.setFont("Helvetica", 10)
c.drawString(60, y, f"🔹 Colaborativo: {np.mean(e_colab):.3f} ± {np.std(e_colab):.3f}")
y -= 15
c.drawString(60, y, f"🔹 Contenido: {np.mean(e_cont):.3f} ± {np.std(e_cont):.3f}")
y -= 15
c.drawString(60, y, f"🔹 Híbrido: {np.mean(e_hibr):.3f} ± {np.std(e_hibr):.3f}")
y -= 30

# Pruebas estadísticas
c.setFont("Helvetica-Bold", 12)
c.drawString(40, y, txt["pdf_stats_title"])
y -= 20
c.setFont("Helvetica", 10)
c.drawString(60, y, txt["pdf_friedman"].format(stat=friedman.statistic, pval=friedman.pvalue))
y -= 15
c.drawString(60, y, txt["pdf_wilcoxon_1"].format(p=p1))
y -= 15
c.drawString(60, y, txt["pdf_wilcoxon_2"].format(p=p2))
y -= 15
c.drawString(60, y, txt["pdf_wilcoxon_3"].format(p=p3))
y -= 15
c.drawString(60, y, txt["pdf_bonferroni"])
y -= 40

# Insertar imagen 1
c.drawString(40, y, txt["img1_label"])
y -= 10
c.drawImage(img1, 50, y - 200, width=500, height=200)

# Insertar imagen 2 debajo
y -= 220
c.drawString(40, y, txt["img2_label"])
y -= 10
c.drawImage(img2, 50, y - 200, width=500, height=200)

# Finalizar
c.showPage()
c.save()
pdf_buffer.seek(0)

# Botón de descarga
st.download_button(
    label=txt["download_stats_pdf"],
    data=pdf_buffer,
    file_name="reporte_estadistico_modelos.pdf",
    mime="application/pdf"
)
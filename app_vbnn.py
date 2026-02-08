import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial import distance, Voronoi, voronoi_plot_2d
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="VBNN + Voronoi", layout="wide")

st.title("VBNN (Voronoi-Based kNN) + PCA Visualization")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_preprocessed.csv")
    return df

df = load_data()

st.write("### Preview Dataset")
st.dataframe(df.head())

# Pisahkan fitur dan target
X = df.drop(columns=['Revenue']).values
y = df['Revenue'].values

# ===============================
# PARAMETER DI STREAMLIT
# ===============================
st.sidebar.header("⚙️ Parameter Model")

k = st.sidebar.slider("Jumlah Cluster (k)", 2, 6, 3)
k_nn = st.sidebar.slider("Nilai k pada kNN", 1, 15, 5)

# ===============================
# STANDARISASI
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# K-MEANS
# ===============================
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_scaled)
centroids = kmeans.cluster_centers_

df['cluster'] = labels

st.write("### Centroid K-Means")
# Buat nama fitur (tanpa Revenue, cluster, dll)
feature_names = df.drop(columns=['Revenue','cluster','voronoi_region'], errors='ignore').columns

# Ubah centroid jadi tabel berlabel
centroid_df = pd.DataFrame(
    centroids,
    columns=feature_names,
    index=[f"Centroid {i}" for i in range(len(centroids))]
)

st.dataframe(centroid_df)


# ===============================
# VORONOI REGION
# ===============================
voronoi_region = []

for x in X_scaled:
    d = distance.cdist([x], centroids)
    region = np.argmin(d)
    voronoi_region.append(region)

df['voronoi_region'] = voronoi_region

st.write("### Distribusi Wilayah Voronoi")
st.write(df['voronoi_region'].value_counts())

# ===============================
# FUNGSI VBNN
# ===============================
def vbnn_predict(new_point, X, y, centroids, regions, k_nn=5):

    d_cent = distance.cdist([new_point], centroids)
    region = np.argmin(d_cent)

    mask = np.array(regions) == region
    X_region = X[mask]
    y_region = y[mask]

    if len(X_region) == 0:
        d = distance.cdist([new_point], X)[0]
        idx = np.argsort(d)[:k_nn]
        pred = int(np.round(np.mean(y[idx])))
        return pred, -1

    d = distance.cdist([new_point], X_region)[0]
    idx = np.argsort(d)[:k_nn]

    pred = int(np.round(np.mean(y_region[idx])))

    return pred, region

# ===============================
# EVALUASI AKURASI
# ===============================
predictions = []

for i in range(len(X_scaled)):
    p, _ = vbnn_predict(
        X_scaled[i],
        X_scaled,
        y,
        centroids,
        voronoi_region,
        k_nn=k_nn
    )
    predictions.append(p)

st.write("### Hasil Evaluasi VBNN")
st.write(f"Akurasi: **{accuracy_score(y, predictions):.4f}**")
st.text(classification_report(y, predictions))

# ===============================
# 7) PERBANDINGAN DENGAN KNN Standar
# ===============================
st.write("## Perbandingan: VBNN vs KNN Standar")

# ---- Train KNN Standar (tanpa Voronoi) ----
knn = KNeighborsClassifier(n_neighbors=k_nn)
knn.fit(X_scaled, y)

knn_pred = knn.predict(X_scaled)
acc_knn = accuracy_score(y, knn_pred)
acc_vbnn = accuracy_score(y, predictions)

# Tabel ringkas perbandingan
compare_df = pd.DataFrame({
    "Metode": ["KNN Standar", "VBNN"],
    "Akurasi": [acc_knn, acc_vbnn]
})

st.dataframe(compare_df)

st.write("### Classification Report KNN Standar")
st.text(classification_report(y, knn_pred))


# ===============================
# PCA + VORONOI PLOT
# ===============================
st.write("### Visualisasi PCA + Voronoi")

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)
centroids_2d = pca.transform(centroids)

vor = Voronoi(centroids_2d)

fig, ax = plt.subplots(figsize=(7,6))

# --- Plot data dulu (background) ---
scatter = ax.scatter(
    X_2d[:,0], 
    X_2d[:,1], 
    c=labels, 
    s=6, 
    alpha=0.6,
    cmap="viridis"
)

# --- Gambar garis Voronoi DI ATAS data ---
voronoi_plot_2d(
    vor,
    ax=ax,
    show_vertices=False,
    line_colors="black",
    line_width=1.2,
    point_size=0
)

# --- Plot centroid (biar jelas) ---
ax.scatter(
    centroids_2d[:,0], 
    centroids_2d[:,1], 
    marker='X', 
    s=200,
    c="red",
    edgecolor="black",
    linewidth=2,
    label="Centroid"
)

ax.set_title("Peta Voronoi  - Online Shoppers Intention")
ax.set_xlabel("PCA Component 1 (X)")
ax.set_ylabel("PCA Component 2 (Y)")
ax.grid(True, linestyle="--", alpha=0.4)

# --- Legenda cluster ---
legend1 = ax.legend(*scatter.legend_elements(),
                    title="Cluster",
                    loc="upper right")
ax.add_artist(legend1)

# --- Legenda centroid ---
ax.legend(loc="lower right")

st.pyplot(fig)


# ===============================
# INPUT UJI DATA BARU
# ===============================
st.write("##  Uji Data Baru (VBNN Prediction)")

st.write("Masukkan nilai fitur sesuai kolom dataset:")

feature_names = df.drop(columns=['Revenue','cluster','voronoi_region']).columns

new_input = []

for col in feature_names:
    val = st.number_input(col, value=float(df[col].mean()))
    new_input.append(val)

if st.button("Prediksi"):
    new_point = np.array(new_input).reshape(1, -1)
    new_point_scaled = scaler.transform(new_point)

    pred, region = vbnn_predict(
        new_point_scaled[0],
        X_scaled,
        y,
        centroids,
        voronoi_region,
        k_nn=k_nn
    )

    st.success(f"Prediksi Revenue: **{pred}**")
    st.info(f"Masuk wilayah Voronoi: **{region}**")

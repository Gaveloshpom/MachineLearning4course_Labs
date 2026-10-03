# Автор Скицко Руслан, 642П, 20 варіант

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

# Завантаження даних
X = np.loadtxt('data_clustering.txt', delimiter=',')

silhouette_scores = []
dbi_scores = []
ch_scores = []
K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    
    # Обчислюємо метрики
    silhouette = metrics.silhouette_score(X, labels)
    dbi = metrics.davies_bouldin_score(X, labels)
    chi = metrics.calinski_harabasz_score(X, labels)
    
    silhouette_scores.append(silhouette)
    dbi_scores.append(dbi)
    ch_scores.append(chi)

# Візуалізація
plt.figure(figsize=(14,4)).canvas.manager.set_window_title("Скицко Руслан, 642П, вар. 20")

plt.subplot(1,3,1)
plt.plot(K, silhouette_scores, 'o-', color='blue')
plt.title("Silhouette Score")
plt.xlabel("k (кількість кластерів)")
plt.ylabel("Score")

plt.subplot(1,3,2)
plt.plot(K, dbi_scores, 'o-', color='red')
plt.title("Davies-Bouldin Index")
plt.xlabel("k (кількість кластерів)")
plt.ylabel("Index")

plt.subplot(1,3,3)
plt.plot(K, ch_scores, 'o-', color='green')
plt.title("Calinski-Harabasz Index")
plt.xlabel("k (кількість кластерів)")
plt.ylabel("Score")

plt.tight_layout()
plt.show()

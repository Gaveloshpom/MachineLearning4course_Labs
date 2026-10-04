# Автор Скицко Руслан, 642П, 20 варіант

import json
import numpy as np
import yfinance as yf
from sklearn import cluster

# Параметри
INPUT_JSON = "company_symbol_mapping_clean.json"
START_DATE = "2003-07-03"
END_DATE = "2007-05-04"

# --- Завантажуємо мапу компаній
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    company_symbols_map = json.load(f)

symbols = list(company_symbols_map.keys())
names = [company_symbols_map[s] for s in symbols]

# --- Завантаження котирувань, фільтрація порожніх
quotes = []
valid_symbols = []
valid_names = []

print("Завантаження даних...")
for sym, nm in zip(symbols, names):
    try:
        df = yf.download(sym, start=START_DATE, end=END_DATE,
                         progress=False, auto_adjust=False)
    except Exception as e:
        print(f"❌ Помилка завантаження {sym}: {e}")
        continue
    if df is None or df.empty:
        print(f"⚠️ Пропущено {sym} (немає даних у цей період)")
        continue
    quotes.append(df)
    valid_symbols.append(sym)
    valid_names.append(nm)
print(f"✅ Завантажено для {len(quotes)} символів: {valid_symbols}\n")

if not quotes:
    raise SystemExit("Немає жодного валідного символу "
                     "для заданого періоду — змініть список або період.")

# --- Вирівнюємо по спільному індексу (перетин дат)
common_index = quotes[0].index
for q in quotes[1:]:
    common_index = common_index.intersection(q.index)

if len(common_index) == 0:
    raise SystemExit("Перетин дат між символами порожній — "
                     "змініть період або список компаній.")

quotes = [q.loc[common_index].copy() for q in quotes]
print(f"Спільних дат після вирівнювання: {len(common_index)}")

# --- Переконаємось, що немає NaN у Open/Close: робимо forward/backfill
for i, q in enumerate(quotes):
    quotes[i] = q[["Open", "Close"]].ffill().bfill()

# --- Побудова масивів: використаємо np.vstack/np.stack, щоб отримати 2D
# shape (n_symbols, n_dates)
opens = np.vstack([q["Open"].values.reshape(1, -1) for q in quotes])   
closes = np.vstack([q["Close"].values.reshape(1, -1) for q in quotes])

print("shapes -> opens:", opens.shape, "closes:", closes.shape)

# --- Різниця і перетворення у X (samples x features)
quotes_diff = closes - opens           # shape (n_symbols, n_dates)
# Переносимо: кожен рядок — день, кожен стовпчик — символ
X = quotes_diff.T                      # shape (n_dates, n_symbols)
print("Початковий X.shape:", X.shape, " dtype:", X.dtype, " ndim:", X.ndim)

# --- Перевірка — якщо трапилось щось 3D, примусово перетворимо
X = np.asarray(X)
if X.ndim > 2:
    print("⚠️ X має >2 вимірів — спробую стиснути зайві виміри.")
    X = np.squeeze(X)
    print("Після squeeze X.shape:", X.shape, " ndim:", X.ndim)

if X.ndim != 2:
    # Остання спроба — сформувати через vstack/stack заново з quotes_diff
    try:
        X = np.vstack([np.ravel(row) for row in quotes_diff])
        X = X.T
        print("Після vstack/rebuild X.shape:", X.shape)
    except Exception as e:
        raise RuntimeError(f"Не вдалося привести X до 2D: {e}")

# --- Обробка NaN / нульової дисперсії
# замінимо NaN на 0 (або можна - np.nan_to_num)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# нормалізація по стовпцях (features); захист від ділення на нуль
stds = X.std(axis=0)
# якщо std == 0, замінюємо на 1 (тобто стовпчик не масштабуватиметься)
zero_std_mask = (stds == 0)
if zero_std_mask.any():
    print(f"⚠️ Знайдено {zero_std_mask.sum()} стовпчик(ів) із нульовою "
          "дисперсією — вони будуть виключені перед GraphicalLassoCV.")
# Видалимо постійні стовпці, бо вони не годяться для оцінки коваріації
keep_mask = ~zero_std_mask
X = X[:, keep_mask]
stds = stds[keep_mask]
if X.size == 0 or X.shape[1] < 2:
    raise SystemExit("Після видалення постійних стовпців залишилось "
                     "менше 2-х ознак — змініть список символів або період.")
# Нормалізуємо
X = X / stds

print("Після фільтрації X.shape:", X.shape)

# --- Графове моделювання
from sklearn.covariance import GraphicalLassoCV
edge_model = GraphicalLassoCV()

# Навчання
with np.errstate(invalid='ignore'):
    edge_model.fit(X)

# --- Кластеризація на основі коваріації
_, labels = cluster.affinity_propagation(edge_model.covariance_)
num_labels = labels.max() + 1

# Виведення результатів
print("\nClustering of stocks based on difference in "
      "opening and closing quotes:\n")
# синхронізуємо імена з keep_mask
valid_names_kept = [n for (n, keep) in zip(valid_names, keep_mask) if keep]  
for i in range(num_labels):
    members = [valid_names_kept[j] for j in range(len(valid_names_kept))
               if labels[j] == i]
    print(f"Cluster {i+1} ==> {', '.join(members)}")
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# візуалізація у вигляді графа взаємозв’язків акцій на основі матриці
# коваріацій, яку навчив GraphicalLassoCV
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl

# Отримуємо матрицю прецизій
prec_matrix = edge_model.precision_.copy()

# Створюємо граф
G = nx.Graph()
for i, name in enumerate(valid_names_kept):
    G.add_node(name, cluster=labels[i])

# Додаємо ребра
threshold = 0.02
for i in range(len(valid_names_kept)):
    for j in range(i + 1, len(valid_names_kept)):
        if abs(prec_matrix[i, j]) > threshold:
            G.add_edge(valid_names_kept[i], valid_names_kept[j], weight=prec_matrix[i, j])

# Кольори вузлів за кластерами
unique_clusters = sorted(set(labels))
color_map = mpl.colormaps.get_cmap("tab10")
# обмеження 10 кольорів
node_colors = [color_map(labels[i] % 10) for i in range(len(valid_names_kept))]

# Малюємо граф
plt.figure(figsize=(10, 10)).canvas.manager.set_window_title("Скицко Руслан, 642П, вар. 20")
pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, alpha=0.9)
nx.draw_networkx_labels(G, pos, font_size=7)
nx.draw_networkx_edges(G, pos, alpha=0.4)

plt.title("Граф зв’язків між акціями (з кластерами)", fontsize=12)
plt.axis("off")
plt.tight_layout()  # прибирає великі поля
plt.show()

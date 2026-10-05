# Автор Скицко Руслан, 642П, 20 варіант

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Compute similarity score')
    parser.add_argument('--user1', dest='user1', required=True,
                        help='First user')
    parser.add_argument('--user2', dest='user2', required=True,
                        help='Second user')
    parser.add_argument("--score-type", dest="score_type", required=True,
                        choices=['Euclidean', 'Pearson'],
                        help='Similarity metric to be used')
    return parser

# Euclidean score
def euclidean_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')
    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')

    common_movies = {item for item in dataset[user1] if item in dataset[user2]}
    if len(common_movies) == 0:
        return 0

    squared_diff = [(dataset[user1][item] - dataset[user2][item])**2
                    for item in common_movies]
    return 1 / (1 + np.sqrt(np.sum(squared_diff)))

# Pearson score
def pearson_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')
    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')

    common_movies = {item for item in dataset[user1] if item in dataset[user2]}
    num_ratings = len(common_movies)
    if num_ratings == 0:
        return 0

    user1_ratings = np.array([dataset[user1][item] for item in common_movies])
    user2_ratings = np.array([dataset[user2][item] for item in common_movies])

    user1_sum, user2_sum = user1_ratings.sum(), user2_ratings.sum()
    user1_sq_sum, user2_sq_sum = (user1_ratings**2).sum(),  \
        (user2_ratings**2).sum()
    sum_of_products = (user1_ratings * user2_ratings).sum()

    Sxy = sum_of_products - (user1_sum * user2_sum / num_ratings)
    Sxx = user1_sq_sum - (user1_sum**2) / num_ratings
    Syy = user2_sq_sum - (user2_sum**2) / num_ratings

    if Sxx * Syy == 0:
        return 0
    return Sxy / np.sqrt(Sxx * Syy)

# Візуалізація
def visualize_comparison(dataset, user1, user2):
    common_movies = [m for m in dataset[user1] if m in dataset[user2]]
    if not common_movies:
        print("\n⚠️ У користувачів немає спільних фільмів для порівняння.")
        return

    print(f"\nЗнайдено {len(common_movies)} спільних фільм(ів) "
          "для {user1} і {user2}.")

    df = pd.DataFrame({
        "Film": common_movies,
        user1: [dataset[user1][m] for m in common_movies],
        user2: [dataset[user2][m] for m in common_movies]
    })

    print("\nПорівняння оцінок:\n")
    print(df)

    # Лінійний графік
    plt.figure(figsize=(8,5)).canvas.manager.set_window_title("Скицко Руслан, 642П, 20 варіант")
    x = range(len(common_movies))
    plt.plot(x, df[user1], marker='o', label=user1)
    plt.plot(x, df[user2], marker='s', label=user2)
    plt.xticks(x, common_movies, rotation=45)
    plt.xlabel("Фільми")
    plt.ylabel("Оцінка")
    plt.title(f"Порівняння оцінок: {user1} vs {user2}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Scatter-графік
    plt.figure(figsize=(6,6)).canvas.manager.set_window_title("Скицко Руслан, 642П, 20 варіант")
    plt.scatter(df[user1], df[user2], c='blue', s=80)
    for i, film in enumerate(common_movies):
        plt.text(df[user1][i]+0.02, df[user2][i]+0.02, film, fontsize=9)
    plt.xlabel(user1)
    plt.ylabel(user2)
    plt.title(f"Scatter-порівняння оцінок ({user1} vs {user2})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    args = build_arg_parser().parse_args()
    user1, user2, score_type = args.user1, args.user2, args.score_type

    ratings_file = 'ratings.json'
    with open(ratings_file, 'r', encoding="utf-8") as f:
        data = json.loads(f.read())

    if score_type == 'Euclidean':
        print("\nEuclidean score:")
        print(euclidean_score(data, user1, user2))
    else:
        print("\nPearson score:")
        print(pearson_score(data, user1, user2))

    # Візуалізація спільних оцінок
    visualize_comparison(data, user1, user2)

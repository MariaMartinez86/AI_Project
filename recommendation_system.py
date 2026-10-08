"""Sistema básico de recomendación con KNN y scikit-learn."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class RecommendationSystem:
    """Recomienda películas a un usuario usando los gustos de otros usuarios."""

    def __init__(self, ratings: pd.DataFrame) -> None:
        self.ratings = ratings.copy()
        required_columns = {"user_id", "item_id", "rating"}
        missing_columns = required_columns - set(self.ratings.columns)
        if missing_columns:
            raise ValueError(f"Faltan columnas: {sorted(missing_columns)}")
        if self.ratings.empty:
            raise ValueError("ratings no puede estar vacío")
        if self.ratings["rating"].isna().any():
            raise ValueError("Las puntuaciones no pueden ser nulas")

        self.users = self.ratings["user_id"].unique()
        self.items = self.ratings["item_id"].unique()
        self.matrix = self._build_user_item_matrix()

    def _build_user_item_matrix(self) -> pd.DataFrame:
        """Construye una matriz donde filas son usuarios y columnas películas."""
        pivot = self.ratings.pivot(
            index="user_id", columns="item_id", values="rating"
        )
        return pivot.reindex(index=self.users, columns=self.items).fillna(0.0)

    def recommend(self, user_id: int, n_recommendations: int = 5) -> list[tuple[int, float]]:
        """Devuelve las mejores películas para un usuario, sin repetirlas."""
        if user_id not in self.users:
            raise ValueError(f"Usuario '{user_id}' no existe en los datos")
        if n_recommendations <= 0:
            raise ValueError("n_recommendations debe ser mayor que cero")

        user_ratings = self.matrix.loc[user_id]
        rated_items = self.ratings.loc[
            self.ratings["user_id"] == user_id, "item_id"
        ]

        # Se usa KNN para encontrar usuarios similares al objetivo.
        neighbors = NearestNeighbors(n_neighbors=5, metric="cosine")
        neighbors.fit(self.matrix)
        distances, indices = neighbors.kneighbors(
            self.matrix.loc[[user_id]], n_neighbors=min(5, len(self.users))
        )

        similar_users = self.matrix.iloc[indices[0][1:]]
        recommendations = similar_users.mean(axis=0)
        recommendations = recommendations.loc[~recommendations.index.isin(rated_items)]
        recommendations = recommendations.sort_values(ascending=False)

        return [
            (int(item_id), round(float(score), 2))
            for item_id, score in recommendations.head(n_recommendations).items()
        ]


def create_sample_ratings() -> pd.DataFrame:
    """Crea un conjunto de ejemplo con puntuaciones de usuarios y películas."""
    return pd.DataFrame(
        [
            (1, 1, 5), (1, 2, 4), (1, 3, 3), (1, 4, 2),
            (2, 1, 4), (2, 2, 5), (2, 3, 4), (2, 4, 3),
            (3, 1, 3), (3, 2, 4), (3, 3, 5), (3, 4, 4),
            (4, 1, 2), (4, 2, 3), (4, 3, 4), (4, 4, 5),
        ],
        columns=["user_id", "item_id", "rating"],
    )


def main() -> None:
    ratings = create_sample_ratings()
    system = RecommendationSystem(ratings)

    print("Recomendaciones para el usuario 1:")
    for item_id, score in system.recommend(1, n_recommendations=3):
        print(f"- Película {item_id}: {score}/5.0")


if __name__ == "__main__":
    main()

from typing import Literal, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: Optional[int] = None
    ) -> dict:
        """Reduces document-term matrix dimensions to 2D (PCA, t-SNE, UMAP) and plots scatter plot.

        Performs dimensionality reduction on session.feature_matrix to 2 components,
        saves reduced 2D coordinates into session.artifacts, generates a scatter plot
        colored by category saved to session.pending_figure, and stores the result.

        Args:
            method: Dimensionality reduction technique ("pca", "tsne", "umap").
            perplexity: Perplexity parameter for t-SNE (default: 30.0).
            n_neighbors: Number of neighbors parameter for UMAP (default: 15).
            random_state: Random state seed for reproducibility.
        """
        if session.feature_matrix is None:
            raise ValueError("session.feature_matrix is missing. Call build_dtm_tool first.")

        X = session.feature_matrix.toarray()
        method_lower = method.lower()
        n_components = 2

        explained_variance_ratio = None

        if method_lower == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            coords = reducer.fit_transform(X)
            explained_variance_ratio = reducer.explained_variance_ratio_.tolist()
        elif method_lower == "tsne":
            reducer = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                init="random",
                learning_rate=200.0,
                random_state=random_state
            )
            coords = reducer.fit_transform(X)
        elif method_lower == "umap":
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=random_state
            )
            coords = reducer.fit_transform(X)
        else:
            raise ValueError(f"Unsupported method: {method}. Use 'pca', 'tsne', or 'umap'.")

        coords_list = coords.tolist()

        # 取得類別標籤用於繪圖與呈現
        labels = session.labels
        if labels is None and session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values

        # 繪製 Scatter Plot 並設定至 session.pending_figure
        fig, ax = plt.subplots(figsize=(8, 6))
        if labels is not None:
            sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=labels, ax=ax, s=100)
        else:
            sns.scatterplot(x=coords[:, 0], y=coords[:, 1], ax=ax, s=100)
        
        ax.set_title(f"Dimensionality Reduction ({method_lower.upper()})")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        fig.tight_layout()
        session.pending_figure = fig

        # 儲存降維座標至 session.artifacts
        artifact_key = f"reduced_coords_{method_lower}"
        session.artifacts[artifact_key] = coords_list

        result_id = session.next_result_id("reduce_dimensions")
        summary = {
            "result_id": result_id,
            "method": method_lower,
            "n_components": n_components,
            "n_samples": len(coords_list),
            "coordinates_preview": coords_list[:10],
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        args_dict = {
            "method": method,
            "perplexity": perplexity,
            "n_neighbors": n_neighbors,
            "random_state": random_state
        }

        session.store_result("reduce_dimensions_tool", args_dict, summary)
        return summary

    @tool
    def binarize_labels_tool() -> dict:
        """One-hot encodes document category labels into a binary matrix.

        Reads category labels from session.labels or session.dataframe["category_name"],
        performs one-hot encoding using pandas.get_dummies, stores the category list
        and binary matrix into session.artifacts["binarized_labels"], and returns the result.
        """
        labels = session.labels
        if labels is None and session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values

        if labels is None:
            raise ValueError("No category labels available in session.labels or session.dataframe['category_name'].")

        dummies_df = pd.get_dummies(labels, dtype=int)
        categories = sorted(dummies_df.columns.tolist())
        dummies_df = dummies_df[categories]
        encoded_matrix = dummies_df.values.tolist()

        n_samples = len(encoded_matrix)
        n_categories = len(categories)

        session.artifacts["binarized_labels"] = {
            "categories": categories,
            "matrix": encoded_matrix
        }

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "n_samples": n_samples,
            "n_categories": n_categories,
            "categories": categories,
            "encoded_matrix_preview": encoded_matrix[:10]
        }

        session.store_result("binarize_labels_tool", {}, summary)
        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]

from typing import Dict, Any, Optional
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool


def make_tools(session):

    @tool
    def cosine_similarity_tool(
        doc1_index: Optional[int] = None,
        doc2_index: Optional[int] = None,
        doc1_text: Optional[str] = None,
        doc2_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculates cosine similarity between two documents.

        Can accept either zero-based document indices (referencing session.feature_matrix)
        or raw text strings (vectorized using session.artifacts['count_vectorizer']).

        Args:
            doc1_index: Index of the first document in session.feature_matrix.
            doc2_index: Index of the second document in session.feature_matrix.
            doc1_text: Text string for the first document.
            doc2_text: Text string for the second document.

        Returns:
            Dict containing result_id and cosine_similarity value.
        """
        if doc1_index is not None and doc2_index is not None:
            if session.feature_matrix is None:
                raise ValueError("session.feature_matrix is not initialized. Run build_dtm_tool first.")
            
            n_rows = session.feature_matrix.shape[0]
            if not (0 <= doc1_index < n_rows):
                raise ValueError(f"doc1_index {doc1_index} is out of bounds for feature_matrix with {n_rows} rows.")
            if not (0 <= doc2_index < n_rows):
                raise ValueError(f"doc2_index {doc2_index} is out of bounds for feature_matrix with {n_rows} rows.")

            vec1 = session.feature_matrix[doc1_index]
            vec2 = session.feature_matrix[doc2_index]
        elif doc1_text is not None and doc2_text is not None:
            vectorizer = session.artifacts.get("count_vectorizer")
            if vectorizer is None:
                raise ValueError("count_vectorizer not found in session.artifacts. Run build_dtm_tool first.")
            vec1 = vectorizer.transform([doc1_text])
            vec2 = vectorizer.transform([doc2_text])
        else:
            raise ValueError("Must provide either (doc1_index, doc2_index) or (doc1_text, doc2_text).")

        norm1 = np.linalg.norm(vec1.toarray())
        norm2 = np.linalg.norm(vec2.toarray())

        if norm1 == 0 or norm2 == 0:
            sim_val = 0.0
        else:
            sim_matrix = cosine_similarity(vec1, vec2)
            sim_val = float(sim_matrix[0, 0])

        result_id = session.next_result_id("cosine_similarity")
        summary = {
            "result_id": result_id,
            "cosine_similarity": sim_val,
        }
        session.store_result("cosine_similarity_tool", {"doc1_index": doc1_index, "doc2_index": doc2_index}, summary)
        return summary

    @tool
    def feature_correlation_matrix_tool() -> Dict[str, Any]:
        """Computes feature-vs-feature Pearson correlation matrix for top 20 terms by variance.

        Operates on the global DTM (session.feature_matrix and session.feature_names).
        Generates a heatmap on session.pending_figure and returns term names and correlation matrix.

        Returns:
            Dict containing result_id, terms, and correlation_matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix or session.feature_names is not initialized. Run build_dtm_tool first.")

        X = session.feature_matrix
        feature_names = np.array(session.feature_names)

        # 1. Compute population variance directly on sparse matrix per column without full dense conversion
        # E[X^2] - (E[X])^2 for population variance
        if hasattr(X, "toarray"):
            mean = np.array(X.mean(axis=0)).ravel()
            mean_sq = np.array(X.power(2).mean(axis=0)).ravel()
            variances = mean_sq - (mean ** 2)
        else:
            X_arr = np.array(X)
            variances = np.var(X_arr, axis=0)

        # 2. Sort terms by variance descending and pick top 20
        sorted_indices = np.argsort(-variances)
        selected_indices = sorted_indices[:20]

        selected_terms = [str(term) for term in feature_names[selected_indices]]
        
        # Slice selected columns and convert only those to dense array
        if hasattr(X, "toarray"):
            selected_data = X[:, selected_indices].toarray()
        else:
            selected_data = np.array(X)[:, selected_indices]

        # 3. Compute Pearson correlation matrix across selected term columns
        corr_matrix = np.corrcoef(selected_data, rowvar=False)

        # Handle edge case where corrcoef returns scalar or nan
        if np.isscalar(corr_matrix):
            corr_matrix = np.array([[corr_matrix]])
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # 4. Plot heatmap using seaborn and set session.pending_figure
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_matrix,
            xticklabels=selected_terms,
            yticklabels=selected_terms,
            annot=True,
            fmt=".4f",
            cmap="coolwarm",
            ax=ax
        )
        ax.set_title("Feature Correlation Matrix (Top Variance Terms)")
        plt.tight_layout()
        session.pending_figure = fig

        # 5. Store and return JSON-serializable summary
        result_id = session.next_result_id("feature_correlation")
        summary = {
            "result_id": result_id,
            "terms": selected_terms,
            "correlation_matrix": corr_matrix.tolist(),
        }
        session.store_result("feature_correlation_matrix_tool", {}, summary)
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]

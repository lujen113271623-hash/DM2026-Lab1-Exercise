from typing import List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def build_dtm_tool(
        max_features: Optional[int] = None,
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        ngram_range: List[int] = [1, 1],
    ) -> dict:
        """Build a document-term matrix (DTM) from the current session's dataframe.

        Fits a CountVectorizer on session.dataframe['text'], populates
        session.feature_matrix and session.feature_names, stores the vectorizer at
        session.artifacts['count_vectorizer'], sets session labels via category_name,
        and returns summary statistics including matrix sparsity.

        Args:
            max_features: Maximum number of terms to keep ordered by term frequency.
            min_df: Maximum document frequency for terms to be kept.
            max_df: Maximum document frequency for terms to be kept.
            ngram_range: List of two ints [min_n, max_n] defining the n-gram range.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            raise ValueError("session.dataframe is not loaded or missing 'text' column.")

        # 將 List[int] 轉換為 tuple 以供 CountVectorizer 使用
        ngram_tuple = tuple(ngram_range)

        # 初始化並擬合 CountVectorizer
        vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_tuple,
        )

        # 處理空的或 NaN 的文字欄位（填補為空字串）
        texts = session.dataframe["text"].fillna("")
        feature_matrix = vectorizer.fit_transform(texts)
        feature_names = list(vectorizer.get_feature_names_out())

        # 更新 session 狀態
        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        # 若 session.dataframe 中含有 category_name，設定 session.labels
        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)

        # 計算稀疏度指標 (Sparsity)
        n_rows, n_cols = feature_matrix.shape
        total_elements = n_rows * n_cols
        non_zero = int(feature_matrix.nnz)
        sparsity_pct = 100.0 * (1.0 - non_zero / total_elements) if total_elements > 0 else 0.0

        # 組裝摘要資訊
        result_id = session.next_result_id("dtm")
        summary = {
            "result_id": result_id,
            "shape": [n_rows, n_cols],
            "vocabulary_size": len(feature_names),
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": sparsity_pct,
            "feature_names": feature_names,
        }

        session.store_result("build_dtm_tool", {"max_features": max_features, "min_df": min_df, "max_df": max_df, "ngram_range": ngram_range}, summary)
        return summary

    @tool
    def term_frequency_tool() -> dict:
        """Aggregates and computes total term frequencies across all documents in the DTM."""
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix or session.feature_names is missing. Call build_dtm_tool first.")

        # 計算全域各詞彙總出現次數
        counts = np.asarray(session.feature_matrix.sum(axis=0)).ravel()
        term_frequencies = {term: int(count) for term, count in zip(session.feature_names, counts)}

        # 建立 DataFrame 作為 full_report，方便排序與提供繪圖/進一步分析
        report_df = pd.DataFrame({
            "term": session.feature_names,
            "frequency": counts
        }).sort_values(by="frequency", ascending=False)

        result_id = session.next_result_id("term_freq")
        summary = {
            "result_id": result_id,
            "total_terms": len(term_frequencies),
            "frequencies": term_frequencies,
        }

        session.store_result("term_frequency_tool", {}, summary, full_report=report_df)
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> dict:
        """Generates a heatmap visualization of a positional slice of the raw document-term matrix.

        Args:
            n_terms: Number of terms (columns) to slice from the beginning of session.feature_names.
            n_documents: Number of documents (rows) to slice from the beginning of session.dataframe.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix or session.feature_names is missing. Call build_dtm_tool first.")

        # 切片矩陣與特徵名稱
        sliced_matrix_sparse = session.feature_matrix[:n_documents, :n_terms]
        sliced_matrix = sliced_matrix_sparse.toarray().astype(int)
        sliced_feature_names = session.feature_names[:n_terms]
        document_labels = [f"Doc {i}" for i in range(sliced_matrix.shape[0])]

        # 繪製 Heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            sliced_matrix,
            annot=True,
            fmt="d",
            xticklabels=sliced_feature_names,
            yticklabels=document_labels,
            ax=ax,
            cmap="Blues"
        )
        ax.set_title("Document-Term Matrix Heatmap")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        # 將 matplotlib Figure 設定給 session.pending_figure
        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "n_documents": sliced_matrix.shape[0],
            "n_terms": sliced_matrix.shape[1],
            "feature_names": sliced_feature_names,
            "document_labels": document_labels,
            "matrix": sliced_matrix.tolist(),
        }

        session.store_result("dtm_heatmap_tool", {"n_terms": n_terms, "n_documents": n_documents}, summary)
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]

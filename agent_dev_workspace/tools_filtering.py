from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def variance_filter_tool(threshold: float = 0.0) -> dict:
        """Filters feature terms in the DTM based on variance.

        Calculates the population variance for each term across all documents in
        session.feature_matrix, separates terms into kept and removed sets according
        to threshold, and passes a full_report DataFrame for result visualization.

        Args:
            threshold: Minimum variance threshold required for a term to be kept.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix or session.feature_names is missing. Call build_dtm_tool first.")

        # 將稀疏矩陣轉為 dense 陣列以計算 population variance (ddof=0)
        dense_matrix = session.feature_matrix.toarray()
        variances_arr = np.var(dense_matrix, axis=0, ddof=0)

        kept_features = []
        removed_features = []
        variances_dict = {}

        for term, var_val in zip(session.feature_names, variances_arr):
            var_float = float(var_val)
            variances_dict[term] = var_float
            if var_float >= threshold:
                kept_features.append(term)
            else:
                removed_features.append(term)

        # 建立 full_report DataFrame（第一欄為 term，第二欄為 variance，依 variance 降序排列）
        report_df = pd.DataFrame({
            "term": session.feature_names,
            "variance": variances_arr
        }).sort_values(by="variance", ascending=False)

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "kept_features": kept_features,
            "removed_features": removed_features,
            "variances": variances_dict,
        }

        session.store_result("variance_filter_tool", {"threshold": threshold}, summary, full_report=report_df)
        return summary

    @tool
    def pearson_filter_tool(target_class: str) -> dict:
        """Computes Pearson correlation between DTM terms and a target class (one-vs-rest).

        Args:
            target_class: The target category name to correlate against (one-vs-rest).
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix or session.feature_names is missing. Call build_dtm_tool first.")

        labels = session.labels
        if labels is None and session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values

        if labels is None:
            raise ValueError("session.labels or session.dataframe['category_name'] is missing.")

        target_binary = (np.array(labels) == target_class).astype(float)

        pearson_r_list = []
        pearson_dict = {}

        for i, term in enumerate(session.feature_names):
            term_vec = session.feature_matrix[:, i].toarray().ravel()
            if np.std(term_vec) == 0 or np.std(target_binary) == 0:
                r_val = 0.0
            else:
                r_val, _ = stats.pearsonr(term_vec, target_binary)
                if np.isnan(r_val):
                    r_val = 0.0
            r_float = float(r_val)
            pearson_r_list.append(r_float)
            pearson_dict[term] = r_float

        report_df = pd.DataFrame({
            "term": session.feature_names,
            "pearson_r": pearson_r_list
        }).sort_values(by="pearson_r", ascending=False)

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "correlations": pearson_dict
        }

        session.store_result("pearson_filter_tool", {"target_class": target_class}, summary, full_report=report_df)
        return summary

    @tool
    def spearman_filter_tool(target_class: str) -> dict:
        """Computes Spearman rank correlation between DTM terms and a target class (one-vs-rest).

        Args:
            target_class: The target category name to correlate against (one-vs-rest).
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix or session.feature_names is missing. Call build_dtm_tool first.")

        labels = session.labels
        if labels is None and session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values

        if labels is None:
            raise ValueError("session.labels or session.dataframe['category_name'] is missing.")

        target_binary = (np.array(labels) == target_class).astype(float)

        spearman_r_list = []
        spearman_dict = {}

        for i, term in enumerate(session.feature_names):
            term_vec = session.feature_matrix[:, i].toarray().ravel()
            if np.std(term_vec) == 0 or np.std(target_binary) == 0:
                r_val = 0.0
            else:
                r_val, _ = stats.spearmanr(term_vec, target_binary)
                if np.isnan(r_val):
                    r_val = 0.0
            r_float = float(r_val)
            spearman_r_list.append(r_float)
            spearman_dict[term] = r_float

        report_df = pd.DataFrame({
            "term": session.feature_names,
            "spearman_r": spearman_r_list
        }).sort_values(by="spearman_r", ascending=False)

        result_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "correlations": spearman_dict
        }

        session.store_result("spearman_filter_tool", {"target_class": target_class}, summary, full_report=report_df)
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]

from typing import Optional
import numpy as np
import pandas as pd
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

    return [variance_filter_tool]

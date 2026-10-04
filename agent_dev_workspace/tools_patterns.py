import os
import tempfile
from typing import Literal, Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

from PAMI.extras.convert import DF2DB
from PAMI.frequentPattern.basic import FPGrowth
from PAMI.frequentPattern.topk import FAE
from PAMI.frequentPattern.maximal import MaxFPGrowth


def make_tools(session):

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"] = "variance",
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """Mine frequent patterns for a specific category using PAMI after feature filtering.

        Args:
            category_name: Category label to filter documents for mining.
            filtering_method: Feature filtering strategy ('variance', 'tfidf', or 'term_frequency').
            algorithm: Pattern mining algorithm ('fpgrowth', 'topk', or 'maxfpgrowth').
            min_sup: Minimum support threshold (required for 'fpgrowth' and 'maxfpgrowth').
            k: Top-k patterns count (required for 'topk').
        """
        if session.dataframe is None:
            raise ValueError("No dataset loaded in session.")

        if algorithm in ["fpgrowth", "maxfpgrowth"] and min_sup is None:
            raise ValueError(f"min_sup is required when algorithm is '{algorithm}'.")
        if algorithm == "topk" and k is None:
            raise ValueError("k is required when algorithm is 'topk'.")

        df = session.dataframe
        if "category_name" not in df.columns:
            raise ValueError("DataFrame missing 'category_name' column.")

        category_df = df[df["category_name"] == category_name]
        if category_df.empty:
            raise ValueError(f"No documents found for category: '{category_name}'")

        # 1. Category-local CountVectorizer with empty vocabulary handling
        vectorizer = CountVectorizer(stop_words="english")
        try:
            X = vectorizer.fit_transform(category_df["text"].fillna(""))
            feature_names = np.array(vectorizer.get_feature_names_out())
        except ValueError as exc:
            if "empty vocabulary" not in str(exc).lower():
                raise
            X = None
            feature_names = np.array([])

        if X is None or X.shape[1] == 0:
            kept_indices = []
        else:
            X_dense = X.toarray()
            if filtering_method == "variance":
                scores = np.var(X_dense, axis=0)
                p5, p95 = np.percentile(scores, 5), np.percentile(scores, 95)
                kept_indices = np.where((scores >= p5) & (scores <= p95))[0]
            elif filtering_method == "term_frequency":
                scores = np.sum(X_dense, axis=0)
                p5, p95 = np.percentile(scores, 5), np.percentile(scores, 95)
                kept_indices = np.where((scores >= p5) & (scores <= p95))[0]
            elif filtering_method == "tfidf":
                tfidf = TfidfTransformer().fit_transform(X).toarray()
                scores = np.mean(tfidf, axis=0)
                p20 = np.percentile(scores, 20)
                kept_indices = np.where(scores >= p20)[0]
            else:
                raise ValueError(f"Unsupported filtering_method: {filtering_method}")

        result_id = session.next_result_id("patterns")
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"

        if len(kept_indices) == 0:
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.artifacts[artifact_key] = []
            session.store_result("mine_patterns_tool", {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k
            }, summary)
            return summary

        kept_features = feature_names[kept_indices]
        X_kept = X_dense[:, kept_indices]

        # Convert to DataFrame
        df_filtered = pd.DataFrame(X_kept, columns=kept_features)

        # Mining via PAMI
        patterns_list = []
        with tempfile.TemporaryDirectory() as tmpdir:
            tx_file = os.path.join(tmpdir, "tx_data.txt")
            converter = DF2DB.DF2DB(df_filtered)
            converter.convert2TransactionalDatabase(
                tx_file,
                condition=">=",
                thresholdValue=1
            )

            if algorithm == "fpgrowth":
                miner = FPGrowth.FPGrowth(iFile=tx_file, minSup=min_sup, sep="\t")
            elif algorithm == "topk":
                miner = FAE.FAE(iFile=tx_file, k=k, sep="\t")
            elif algorithm == "maxfpgrowth":
                miner = MaxFPGrowth.MaxFPGrowth(iFile=tx_file, minSup=min_sup, sep="\t")
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

            miner.mine()
            patterns_df = miner.getPatternsAsDataFrame()

            if patterns_df is not None and not patterns_df.empty:
                for _, row in patterns_df.iterrows():
                    pat_str = str(row["Patterns"])
                    pat_items = [item.strip() for item in pat_str.split("\t") if item.strip()]
                    sup_val = int(row["Support"])
                    patterns_list.append({
                        "pattern": pat_items,
                        "support": sup_val
                    })

        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "n_patterns": len(patterns_list),
            "patterns": patterns_list
        }

        session.artifacts[artifact_key] = patterns_list
        session.store_result("mine_patterns_tool", {
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "min_sup": min_sup,
            "k": k
        }, summary)

        return summary

    return [mine_patterns_tool]

import os
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def tokenize_tool() -> dict:
        """Tokenizes each document's text in session.dataframe into lowercased unigrams.

        Uses nltk.word_tokenize on lowercased text. Empty or missing text values
        produce empty token lists []. Stores the token list per document into
        session.dataframe["unigrams"].
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            raise ValueError("session.dataframe or 'text' column is missing. Call load_dataset_tool first.")

        token_lists = []
        for text in session.dataframe["text"]:
            if pd.isna(text) or not str(text).strip():
                token_lists.append([])
            else:
                tokens = word_tokenize(str(text).lower())
                token_lists.append(tokens)

        session.dataframe["unigrams"] = token_lists

        result_id = session.next_result_id("tokenize")
        n_documents = len(token_lists)
        total_tokens = sum(len(t) for t in token_lists)

        summary = {
            "result_id": result_id,
            "n_documents": n_documents,
            "total_tokens": total_tokens,
            "sample_tokens": token_lists[:3]
        }

        session.store_result("tokenize_tool", {}, summary)
        return summary

    @tool
    def list_files_tool(subdirectory: str = "") -> dict:
        """Lists files and directories in the specified workspace subdirectory.

        Args:
            subdirectory: Relative path to the directory to list (e.g. "newdataset").
        """
        target_dir = os.path.join(".", subdirectory) if subdirectory else "."
        
        if not os.path.exists(target_dir):
            raise ValueError(f"Directory '{subdirectory}' does not exist.")

        entries = os.listdir(target_dir)
        directories = [e for e in entries if os.path.isdir(os.path.join(target_dir, e))]
        files = [e for e in entries if os.path.isfile(os.path.join(target_dir, e))]

        directories.sort()
        files.sort()

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files,
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def check_missing_tool() -> dict:
        """Checks for missing or empty text values in session.dataframe.

        Scans the 'text' column of session.dataframe for NaN values or whitespace-only
        strings and returns the missing count and missing row indices.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            raise ValueError("session.dataframe or 'text' column is missing. Call load_dataset_tool first.")

        missing_indices = []
        for idx, text in enumerate(session.dataframe["text"]):
            if pd.isna(text) or not str(text).strip():
                missing_indices.append(idx)

        missing_count = len(missing_indices)
        total_rows = len(session.dataframe)

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "missing_count": missing_count,
            "total_rows": total_rows,
            "missing_indices": missing_indices,
        }

        session.store_result("check_missing_tool", {}, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> dict:
        """Checks for duplicate documents in session.dataframe, with optional removal.

        When drop=True, removes all duplicate copies using keep=False (matching Master
        notebook's drop_duplicates(keep=False, inplace=True) behavior).

        Args:
            drop: Whether to drop duplicate rows in-place using keep=False.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            raise ValueError("session.dataframe or 'text' column is missing. Call load_dataset_tool first.")

        # Identify duplicates (pandas default duplicated() flags subsequent copies)
        duplicated_mask = session.dataframe.duplicated(subset=["text"])
        duplicate_count = int(duplicated_mask.sum())
        initial_rows = len(session.dataframe)

        if drop and duplicate_count > 0:
            # Match Master's drop_duplicates(keep=False, inplace=True) exactly
            session.dataframe.drop_duplicates(subset=["text"], keep=False, inplace=True)
            session.dataframe.reset_index(drop=True, inplace=True)

            if "category_name" in session.dataframe.columns:
                session.set_labels(session.dataframe["category_name"].values)

        remaining_rows = len(session.dataframe)

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "duplicate_count": duplicate_count,
            "drop": drop,
            "initial_rows": initial_rows,
            "remaining_rows": remaining_rows,
        }

        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    @tool
    def sample_data_tool(n: int, random_state: Optional[int] = None) -> dict:
        """Subsamples session.dataframe to n rows using pandas DataFrame.sample.

        Args:
            n: Number of items to sample.
            random_state: Seed for the random number generator.
        """
        if session.dataframe is None:
            raise ValueError("session.dataframe is missing. Call load_dataset_tool first.")

        initial_rows = len(session.dataframe)
        sampled_df = session.dataframe.sample(n=n, random_state=random_state)
        sampled_indices = sampled_df.index.tolist()

        # Update session.dataframe with the subsampled DataFrame and reset index
        session.dataframe = sampled_df.reset_index(drop=True)

        # Sync session.labels and session.categories if category_name column exists
        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "n": n,
            "random_state": random_state,
            "initial_rows": initial_rows,
            "sampled_rows": len(session.dataframe),
            "sampled_indices": sampled_indices,
        }

        session.store_result("sample_data_tool", {"n": n, "random_state": random_state}, summary)
        return summary

    @tool
    def inspect_data_tool(n: int = 5) -> dict:
        """Inspects the top n rows of session.dataframe.

        Args:
            n: Number of rows to inspect from the top of session.dataframe.
        """
        if session.dataframe is None:
            raise ValueError("session.dataframe is missing. Call load_dataset_tool first.")

        if n <= 0:
            raise ValueError("n must be a positive integer.")

        total_rows = len(session.dataframe)
        columns = session.dataframe.columns.tolist()

        # Convert top n rows into records (JSON-serializable)
        preview_df = session.dataframe.head(n)
        preview = preview_df.to_dict(orient="records")

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "n": n,
            "total_rows": total_rows,
            "columns": columns,
            "preview": preview,
        }

        session.store_result("inspect_data_tool", {"n": n}, summary)
        return summary

    @tool
    def describe_data_tool() -> dict:
        """Computes text length statistics and generates a box plot by category.

        Calculates text_length (character count) for session.dataframe, computes overall
        and per-category describe() statistics, and generates a box plot set on
        session.pending_figure.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            raise ValueError("session.dataframe or 'text' column is missing. Call load_dataset_tool first.")

        if "category_name" not in session.dataframe.columns:
            raise ValueError("session.dataframe['category_name'] is missing. Call load_dataset_tool first.")

        # Compute character length matching Master's X['text_length'] = X['text'].apply(len) exactly
        session.dataframe["text_length"] = session.dataframe["text"].apply(len)

        # Overall describe stats
        overall_stats = session.dataframe["text_length"].describe().to_dict()

        # Per-category describe stats
        category_stats = {}
        for cat_name, group in session.dataframe.groupby("category_name"):
            category_stats[cat_name] = group["text_length"].describe().to_dict()

        # Build box plot with seaborn
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=session.dataframe, x="category_name", y="text_length", ax=ax)
        ax.set_title("Text Length Distribution by Category")
        ax.set_xlabel("Category")
        ax.set_ylabel("Text Length (characters)")

        # Store figure in session.pending_figure (no plt.show() or savefig())
        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "overall_stats": overall_stats,
            "category_stats": category_stats,
        }

        session.store_result("describe_data_tool", {}, summary)
        return summary

    return [
        tokenize_tool,
        list_files_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        inspect_data_tool,
        describe_data_tool,
    ]

import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool_known_answer_5():
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # Build DTM using build_dtm_tool
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Execute variance_filter_tool with threshold=0.15
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(
        t for t in filtering_tools if t.name == "variance_filter_tool"
    )
    result = variance_filter_tool.invoke({"threshold": 0.15})

    # Expected variances from Known Answer #5
    expected_variances = {
        "alpha": 1.1875,
        "always": 0.109375,
        "beta": 0.234375,
        "delta": 0.25,
        "gamma": 1.109375,
    }

    # 1. Verify summary fields (Small Summary API)
    assert result["total_features"] == 5
    assert result["n_kept"] == 4
    assert result["n_removed"] == 1
    # top_variance_terms is dict {term: variance}
    assert result["top_variance_terms"] == pytest.approx(
        expected_variances, abs=1e-4
    )

    # 2. Verify complete Known Answer #5 via stored full_report
    stored = session.results_store[result["result_id"]]
    full_report = stored["full_report"]
    assert list(full_report.columns) == ["term", "variance"]

    report_variances = dict(zip(full_report["term"], full_report["variance"]))
    assert report_variances == pytest.approx(expected_variances, abs=1e-4)

    # Derive kept and removed features using threshold=0.15 (>= threshold is kept)
    kept_features = set(
        full_report[full_report["variance"] >= 0.15]["term"]
    )
    removed_features = list(
        full_report[full_report["variance"] < 0.15]["term"]
    )
    assert kept_features == {"alpha", "beta", "delta", "gamma"}
    assert removed_features == ["always"]


def test_pearson_filter_tool_known_answer_6():
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # Build DTM using build_dtm_tool
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Execute pearson_filter_tool with target_class="catB"
    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(
        t for t in filtering_tools if t.name == "pearson_filter_tool"
    )
    result = pearson_filter_tool.invoke({"target_class": "catB"})

    expected_correlations = {
        "alpha": -0.6882,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.8307,
    }

    # 1. Verify summary fields (Small Summary API)
    assert result["total_terms"] == 5
    # Verify values in top_correlations and bottom_correlations
    assert result["top_correlations"] == pytest.approx(
        expected_correlations, abs=1e-3
    )
    assert result["bottom_correlations"] == pytest.approx(
        expected_correlations, abs=1e-3
    )

    # 2. Verify complete Known Answer #6 via stored full_report
    stored = session.results_store[result["result_id"]]
    full_report = stored["full_report"]
    assert list(full_report.columns[:2]) == ["term", "pearson_r"]
    # Verify sorting by pearson_r descending
    assert full_report["pearson_r"].is_monotonic_decreasing

    report_correlations = dict(
        zip(full_report["term"], full_report["pearson_r"])
    )
    assert report_correlations == pytest.approx(
        expected_correlations, abs=1e-3
    )


def test_spearman_filter_tool_known_answer_6():
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # Build DTM using build_dtm_tool
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Execute spearman_filter_tool with target_class="catB"
    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(
        t for t in filtering_tools if t.name == "spearman_filter_tool"
    )
    result = spearman_filter_tool.invoke({"target_class": "catB"})

    expected_correlations = {
        "alpha": -0.7500,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.9363,
    }

    # 1. Verify summary fields (Small Summary API)
    assert result["total_terms"] == 5
    # Verify values in top_correlations and bottom_correlations
    assert result["top_correlations"] == pytest.approx(
        expected_correlations, abs=1e-3
    )
    assert result["bottom_correlations"] == pytest.approx(
        expected_correlations, abs=1e-3
    )

    # 2. Verify complete Known Answer #6 via stored full_report
    stored = session.results_store[result["result_id"]]
    full_report = stored["full_report"]
    assert list(full_report.columns[:2]) == ["term", "spearman_r"]
    # Verify sorting by spearman_r descending
    assert full_report["spearman_r"].is_monotonic_decreasing

    report_correlations = dict(
        zip(full_report["term"], full_report["spearman_r"])
    )
    assert report_correlations == pytest.approx(
        expected_correlations, abs=1e-3
    )

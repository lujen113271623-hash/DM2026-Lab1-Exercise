import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools


def _setup_session():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")
    return session, mine_patterns_tool


def test_mine_patterns_zero_terms():
    """Requirement A: variance & term_frequency zero-term cases."""
    for method in ["variance", "term_frequency"]:
        for cat in ["catA", "catB"]:
            session, mine_patterns_tool = _setup_session()
            res = mine_patterns_tool.invoke({
                "category_name": cat,
                "filtering_method": method,
                "algorithm": "fpgrowth",
                "min_sup": 1
            })
            assert res["patterns"] == []
            assert res["note"] == "no terms survived filtering"
            artifact_key = f"patterns_{cat}_{method}_fpgrowth"
            assert artifact_key in session.artifacts
            assert session.artifacts[artifact_key] == []


def test_mine_patterns_fpgrowth_tfidf():
    """Requirement B: Known Answer #17 (TF-IDF + FPGrowth)."""
    # catA
    session_a, tool_a = _setup_session()
    res_a = tool_a.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_a["patterns"] == [{"pattern": ["alpha"], "support": 3}]
    assert session_a.artifacts["patterns_catA_tfidf_fpgrowth"] == [{"pattern": ["alpha"], "support": 3}]

    # catB
    session_b, tool_b = _setup_session()
    res_b = tool_b.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_b["patterns"] == [{"pattern": ["gamma"], "support": 4}]
    assert session_b.artifacts["patterns_catB_tfidf_fpgrowth"] == [{"pattern": ["gamma"], "support": 4}]


def test_mine_patterns_topk_tfidf():
    """Requirement C: Known Answer #20 (TF-IDF + Top-K)."""
    # catA
    session_a, tool_a = _setup_session()
    res_a = tool_a.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert res_a["patterns"] == [{"pattern": ["alpha"], "support": 3}]
    assert session_a.artifacts["patterns_catA_tfidf_topk"] == [{"pattern": ["alpha"], "support": 3}]

    # catB
    session_b, tool_b = _setup_session()
    res_b = tool_b.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert res_b["patterns"] == [{"pattern": ["gamma"], "support": 4}]
    assert session_b.artifacts["patterns_catB_tfidf_topk"] == [{"pattern": ["gamma"], "support": 4}]


def test_mine_patterns_maxfpgrowth_tfidf():
    """Requirement D: Known Answer #20 (TF-IDF + MaxFPGrowth)."""
    # catA
    session_a, tool_a = _setup_session()
    res_a = tool_a.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_a["patterns"] == []
    assert session_a.artifacts["patterns_catA_tfidf_maxfpgrowth"] == []

    # catB
    session_b, tool_b = _setup_session()
    res_b = tool_b.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_b["patterns"] == []
    assert session_b.artifacts["patterns_catB_tfidf_maxfpgrowth"] == []

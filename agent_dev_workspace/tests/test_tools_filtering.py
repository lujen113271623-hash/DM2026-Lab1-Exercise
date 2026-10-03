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

    # Assertions based strictly on Known Answer #5 and variance_filter_tool spec
    # 1. Check all 5 term variances using pytest.approx
    expected_variances = {
        "alpha": 1.1875,
        "always": 0.109375,
        "beta": 0.234375,
        "delta": 0.25,
        "gamma": 1.109375,
    }
    assert result["variances"] == pytest.approx(
        expected_variances, abs=1e-4
    )

    # 2. Check kept features
    assert set(result["kept_features"]) == {"alpha", "beta", "delta", "gamma"}

    # 3. Check removed features
    assert result["removed_features"] == ["always"]

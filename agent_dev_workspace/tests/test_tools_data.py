import pytest
import matplotlib.figure
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools as make_data_tools


def test_tokenize_tool_known_answer_10():
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

    # Execute tokenize_tool
    data_tools = make_data_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")
    result = tokenize_tool.invoke({})

    # Expected tokens based strictly on Known Answer #10
    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        [],
    ]

    # Verify session.dataframe["unigrams"] matched expected token lists
    assert session.dataframe["unigrams"].tolist() == expected_tokens

    # Verify summary structure
    assert result["n_documents"] == 8
    assert result["total_tokens"] == sum(len(t) for t in expected_tokens)


def test_list_files_tool_known_answer_11():
    session = SessionState()

    data_tools = make_data_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")
    result = list_files_tool.invoke({"subdirectory": "newdataset"})

    # Verify results according strictly to Known Answer #11
    assert result["directories"] == []
    assert result["files"] == ["Reddit-stock-sentiment.csv"]


def test_check_missing_tool_known_answer_2():
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

    # Execute check_missing_tool
    data_tools = make_data_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")
    result = check_missing_tool.invoke({})

    # Verify missing_count according strictly to Known Answer #2
    assert result["missing_count"] == 1
    assert result["missing_indices"] == [7]


def test_check_duplicates_tool_known_answer_3():
    # Case 1: drop=False
    session1 = SessionState()
    premade_tools1, _ = load_workspace_tools("premade_tools", session1)
    load_dataset_tool1 = next(
        t for t in premade_tools1 if t.name == "load_dataset_tool"
    )
    load_dataset_tool1.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    data_tools1 = make_data_tools(session1)
    check_duplicates_tool1 = next(
        t for t in data_tools1 if t.name == "check_duplicates_tool"
    )
    result1 = check_duplicates_tool1.invoke({"drop": False})

    assert result1["duplicate_count"] == 1
    assert result1["remaining_rows"] == 8

    # Case 2: drop=True
    session2 = SessionState()
    premade_tools2, _ = load_workspace_tools("premade_tools", session2)
    load_dataset_tool2 = next(
        t for t in premade_tools2 if t.name == "load_dataset_tool"
    )
    load_dataset_tool2.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    data_tools2 = make_data_tools(session2)
    check_duplicates_tool2 = next(
        t for t in data_tools2 if t.name == "check_duplicates_tool"
    )
    result2 = check_duplicates_tool2.invoke({"drop": True})

    assert result2["duplicate_count"] == 1
    assert result2["remaining_rows"] == 6
    assert "always gamma delta" not in session2.dataframe["text"].values


def test_sample_data_tool_known_answer_13():
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

    # Execute sample_data_tool with n=4, random_state=42
    data_tools = make_data_tools(session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")
    result = sample_data_tool.invoke({"n": 4, "random_state": 42})

    # Assertions according strictly to Known Answer #13 and sync requirement
    assert result["sampled_indices"] == [1, 5, 0, 7]
    assert result["sampled_rows"] == 4
    assert len(session.dataframe) == 4
    assert len(session.labels) == 4


def test_inspect_data_tool():
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

    data_tools = make_data_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    # 1. Test n=3
    result = inspect_data_tool.invoke({"n": 3})
    assert len(result["preview"]) == 3
    assert result["total_rows"] == 8
    assert result["columns"] == session.dataframe.columns.tolist()

    # 2. Test n=0 raises ValueError
    with pytest.raises(ValueError, match="n must be a positive integer"):
        inspect_data_tool.invoke({"n": 0})


def test_describe_data_tool_known_answer_18():
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

    data_tools = make_data_tools(session)
    describe_data_tool = next(t for t in data_tools if t.name == "describe_data_tool")
    result = describe_data_tool.invoke({})

    # 1. Verify text_length array
    assert session.dataframe["text_length"].tolist() == [29, 23, 17, 30, 24, 18, 18, 0]

    # 2. Verify overall stats with Known Answer #18
    expected_overall = {
        "count": 8.0,
        "mean": 19.875,
        "std": 9.4330,
        "min": 0.0,
        "25%": 17.75,
        "50%": 20.5,
        "75%": 25.25,
        "max": 30.0,
    }
    assert result["overall_stats"] == pytest.approx(expected_overall, abs=1e-3)

    # 3. Verify session.pending_figure is populated
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)

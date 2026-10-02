import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool_sample_fixture():
    # 1. 初始化 SessionState 與載入測試資料
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 2. 實例化 tools_dtm 中的工具
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    # 3. 呼叫 build_dtm_tool
    result = build_dtm_tool.invoke({})

    # 4. 根據 TEST_FIXTURE.md 已知答案 4 進行斷言驗證
    assert result["shape"] == [8, 5]
    assert result["feature_names"] == ["alpha", "always", "beta", "delta", "gamma"]
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert result["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)

    # 5. 驗證 CountVectorizer artifact 是否存在
    assert session.artifacts.get("count_vectorizer") is not None


def test_term_frequency_tool_sample_fixture():
    # 1. 初始化獨立的 SessionState 並載入資料與建立 DTM
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    # 先執行 build_dtm_tool
    build_dtm_tool.invoke({})

    # 2. 執行 term_frequency_tool
    result = term_frequency_tool.invoke({})

    # 3. 根據 TEST_FIXTURE.md Known Answer #14 進行斷言驗證
    expected_frequencies = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7,
    }

    assert result["total_terms"] == 5
    assert result["frequencies"] == expected_frequencies


def test_dtm_heatmap_tool_sample_fixture():
    # 1. 初始化獨立的 SessionState 並載入資料與建立 DTM
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    build_dtm_tool.invoke({})

    # 2. 使用預設參數 (n_terms=20, n_documents=20) 測試
    result_default = dtm_heatmap_tool.invoke({})

    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
    ]

    assert result_default["n_documents"] == 8
    assert result_default["n_terms"] == 5
    assert result_default["feature_names"] == ["alpha", "always", "beta", "delta", "gamma"]
    assert result_default["matrix"] == expected_full_matrix
    assert len(result_default["document_labels"]) == 8
    assert session.pending_figure is not None

    # 3. 指定小尺寸切片 (n_terms=2, n_documents=3) 測試
    result_slice = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})

    assert result_slice["n_documents"] == 3
    assert result_slice["n_terms"] == 2
    assert result_slice["feature_names"] == ["alpha", "always"]
    assert result_slice["matrix"] == [[3, 1], [2, 1], [1, 1]]
    assert len(result_slice["document_labels"]) == 3

import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_reduction import make_tools


def test_reduce_dimensions_tool_pca():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    reduce_dimensions_tool = next(t for t in tools if t.name == "reduce_dimensions_tool")

    # 需要先建立 DTM 供降維工具使用
    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    res = reduce_dimensions_tool.invoke({"method": "pca"})

    # Known Answer #7 PCA coordinates
    expected_coords = [
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822]
    ]
    expected_variance = [0.7532, 0.1965]

    # Verify summary fields
    assert res["n_samples"] == 8
    assert np.allclose(res["coordinates_preview"], expected_coords[:10], atol=1e-3)
    assert np.allclose(res["explained_variance_ratio"], expected_variance, atol=1e-3)
    assert session.pending_figure is not None

    # Verify complete coordinates from session.artifacts
    coords_artifact = session.artifacts["reduced_coords_pca"]
    assert np.allclose(coords_artifact, expected_coords, atol=1e-3)


def test_reduce_dimensions_tool_tsne():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    tools = make_tools(session)
    reduce_dimensions_tool = next(t for t in tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({
        "method": "tsne",
        "perplexity": 2.0,
        "random_state": 42
    })

    # Known Answer #8 t-SNE coordinates
    expected_coords = [
        [1330.9991, 32.1039],
        [951.2459, 393.5588],
        [530.8209, 682.9791],
        [-1558.6448, -314.1625],
        [-915.5153, -346.3907],
        [-280.0313, -1261.2115],
        [-280.0313, -1261.2115],
        [-701.3674, 1465.3973]
    ]

    # Verify summary fields
    assert res["n_samples"] == 8
    assert np.allclose(res["coordinates_preview"], expected_coords[:10], atol=1e-3)
    assert session.pending_figure is not None

    # Verify complete coordinates from session.artifacts
    coords_artifact = session.artifacts["reduced_coords_tsne"]
    assert np.allclose(coords_artifact, expected_coords, atol=1e-3)


def test_reduce_dimensions_tool_umap():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    tools = make_tools(session)
    reduce_dimensions_tool = next(t for t in tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({
        "method": "umap",
        "n_neighbors": 3,
        "random_state": 42
    })

    # Known Answer #9 UMAP coordinates
    expected_coords = [
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622]
    ]

    # Verify summary fields
    assert res["n_samples"] == 8
    assert np.allclose(res["coordinates_preview"], expected_coords[:10], atol=1e-3)
    assert session.pending_figure is not None

    # Verify complete coordinates from session.artifacts
    coords_artifact = session.artifacts["reduced_coords_umap"]
    assert np.allclose(coords_artifact, expected_coords, atol=1e-3)


def test_binarize_labels_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    binarize_labels_tool = next(t for t in tools if t.name == "binarize_labels_tool")

    res = binarize_labels_tool.invoke({})

    # 驗證 Summary preview & 元資料
    assert res["categories"] == ["catA", "catB"]
    assert res["n_samples"] == 8
    assert res["n_categories"] == 2
    
    matrix_preview = np.array(res["encoded_matrix_preview"])
    assert matrix_preview.shape == (8, 2)
    assert (matrix_preview[0] == [1, 0]).all()
    assert (matrix_preview[3] == [0, 1]).all()

    # 驗證 session.artifacts 完整矩陣
    assert "binarized_labels" in session.artifacts
    artifact = session.artifacts["binarized_labels"]
    assert artifact["categories"] == ["catA", "catB"]
    assert artifact["matrix"] == res["encoded_matrix_preview"]
    
    full_matrix = np.array(artifact["matrix"])
    assert full_matrix.shape == (8, 2)
    assert (full_matrix[0] == [1, 0]).all()
    assert (full_matrix[3] == [0, 1]).all()

"""Assign each test module to one CI category (see [tool.pytest.ini_options] markers)."""

import pytest

CATEGORIES = {
    "test_authorization_protocol.py": "security", "test_policy.py": "security",
    "test_advanced_risk.py": "security", "test_firewall.py": "security",
    "test_gateway.py": "gateway", "test_trusted_adapter.py": "gateway", "test_api.py": "gateway",
    "test_dataset.py": "dataset", "test_v5_dataset.py": "dataset", "test_v5_benchmark.py": "dataset",
    "test_v5_jev_silver.py": "dataset", "test_v5_jev_diagnostic.py": "dataset",
    "test_external_benchmark_pipeline.py": "dataset", "test_context_windows.py": "dataset",
    "test_v5_annotation_ui.py": "integration", "test_external_real.py": "model_quality",
    "test_inference.py": "model_quality", "test_model.py": "model_quality",
    "test_calibration.py": "model_quality", "test_model_quality.py": "model_quality",
}


def pytest_collection_modifyitems(items):
    for item in items:
        category = CATEGORIES.get(item.path.name)
        if category is None:
            raise pytest.UsageError(f"{item.path.name} has no test category in tests/conftest.py")
        if not any(item.iter_markers(name=category)):
            item.add_marker(getattr(pytest.mark, category))

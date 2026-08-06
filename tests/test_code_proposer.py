import pytest

from skilled_proposer.code_proposer import (
    _primitives_catalog,
    _strip_code_fences,
    _validate_module_source,
)

VALID_SRC = '''class GeneratedModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.step = dspy.Predict("question -> answer")

    def forward(self, **inputs):
        return dspy.Prediction(answer=self.step(**inputs).answer)
'''


def test_strip_code_fences_plain_text_unchanged():
    assert _strip_code_fences(VALID_SRC) == VALID_SRC


def test_strip_code_fences_removes_python_fence():
    fenced = f"```python\n{VALID_SRC}\n```"
    assert _strip_code_fences(fenced).strip() == VALID_SRC.strip()


def test_strip_code_fences_removes_bare_fence():
    fenced = f"```\n{VALID_SRC}\n```\n"
    assert _strip_code_fences(fenced).strip() == VALID_SRC.strip()


def test_validate_accepts_module_source():
    assert _validate_module_source(VALID_SRC) is None


def test_validate_rejects_non_python():
    error = _validate_module_source("Here is my revised module: use two steps.")
    assert error is not None
    assert "parse" in error


def test_validate_rejects_source_without_class():
    error = _validate_module_source("x = 1\n\ndef forward(**inputs):\n    return x")
    assert error is not None
    assert "class" in error


def test_primitives_catalog_available():
    pytest.importorskip("dspy.predict.flex")
    catalog = _primitives_catalog()
    assert isinstance(catalog, str) and catalog.strip()

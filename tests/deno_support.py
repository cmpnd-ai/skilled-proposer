"""Skip marks for tests that need dspy's real Deno/Pyodide sandbox."""

import pytest
from dspy.primitives.code_interpreter import CodeInterpreterError


def _deno_error() -> str | None:
    """Why dspy's Deno sandbox cannot start here, or None when it can."""
    from dspy.primitives.python_interpreter import _find_deno_executable, _validate_deno_version

    try:
        _validate_deno_version(_find_deno_executable())
    except CodeInterpreterError as e:
        return str(e)
    return None


needs_deno = [
    pytest.mark.deno,
    pytest.mark.skipif(_deno_error() is not None, reason="Deno is not installed"),
]

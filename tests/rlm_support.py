"""An in-process CodeInterpreter and scripted LMs so RLM tests run without Deno."""

import contextlib
import io

from dspy.primitives.code_interpreter import CodeExecutionError, FinalOutput
from dspy.utils.dummies import DummyLM


class _Submit(Exception):
    def __init__(self, output):
        self.output = output


class ExecInterpreter:
    """Runs RLM code with exec() in one namespace. SUBMIT ends the run."""

    def __init__(self):
        self._tools = {}
        self.namespace = {}
        self.output_fields = None

    @property
    def tools(self):
        return self._tools

    def start(self):
        pass

    def shutdown(self):
        pass

    def execute(self, code, variables=None):
        names = [f["name"] for f in (self.output_fields or [])]

        def SUBMIT(*args, **kwargs):
            output = dict(zip(names, args))
            output.update(kwargs)
            raise _Submit(output)

        self.namespace.update(self._tools)
        self.namespace["SUBMIT"] = SUBMIT
        self.namespace.update(variables or {})
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                exec(code, self.namespace)
        except _Submit as submit:
            return FinalOutput(submit.output)
        except SyntaxError:
            raise
        except Exception as e:
            raise CodeExecutionError(f"{type(e).__name__}: {e}")
        return buffer.getvalue()


def step(code):
    return {"reasoning": "Next step.", "code": f"```python\n{code}\n```"}


def scripted(*codes):
    """A DummyLM that answers one RLM step per code string, in order."""
    return DummyLM([step(c) for c in codes])

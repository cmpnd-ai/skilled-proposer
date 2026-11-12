"""A DummyLM that chooses each answer from the request it receives.

DSPy 3.4 routes calls through ``BaseLM.forward``. DSPy 3.5 removed that entry
point and routes calls through an engine, so this hooks whichever one exists.
"""

from dspy.utils.dummies import DummyLM

try:
    from dspy.clients.engines.dummy_engine import AsyncDummyEngine, DummyEngine
except ImportError:  # DSPy < 3.5
    AsyncDummyEngine = DummyEngine = None


if DummyEngine is not None:

    class _RespondingEngine(DummyEngine):
        def _complete_messages(self, messages):
            self.owner.answers = iter([self.owner.respond(messages)])
            return super()._complete_messages(messages)


class RespondingLM(DummyLM):
    """Subclasses implement ``respond(messages) -> dict`` of output field values."""

    def __init__(self):
        super().__init__([])
        self._install_engine()

    def _install_engine(self):
        if DummyEngine is not None:
            self._engine_spec = _RespondingEngine(self)
            self._async_engine_spec = AsyncDummyEngine(self._engine_spec)

    def copy(self, **kwargs):
        copied = super().copy(**kwargs)
        copied._install_engine()
        return copied

    def respond(self, messages):
        raise NotImplementedError

    def forward(self, prompt=None, messages=None, **kwargs):  # DSPy < 3.5
        self.answers = iter([self.respond(messages)])
        return super().forward(prompt=prompt, messages=messages, **kwargs)

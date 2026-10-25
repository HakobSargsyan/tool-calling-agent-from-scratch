"""Tests for the loop, with a fake model. No network, no API key.

The interesting part is what these DON'T assert. Three runs with identical
input and temperature=0 chose `limit=10`, `limit=5`, and no `limit` at all, so
asserting on exact tool arguments or exact answer text gives you a test that
fails for no reason. What you can assert is TRAJECTORY — which tools were
called, in what shape — plus whether the answer contains the facts the tools
returned.

Run:  uv run pytest -q
"""

from types import SimpleNamespace

from google.genai import types

import agent


# ---------------------------------------------------------------------------
# The tools: ordinary functions, so ordinary unit tests.
# ---------------------------------------------------------------------------


def test_balance_lookup():
    assert agent.get_account_balance("ACC-1")["balance"] == 4512.30


def test_unknown_account_is_an_error_not_an_exception():
    # Errors are data the model can react to, so they come back as a value.
    assert "error" in agent.get_account_balance("ACC-999")


def test_limit_is_coerced_and_bounded():
    # A model that sends "3" despite the INTEGER schema must not take the
    # process down with `TypeError: slice indices must be integers`.
    assert len(agent.get_recent_transactions("ACC-1", limit="3")["transactions"]) == 3
    assert len(agent.get_recent_transactions("ACC-1", limit=999)["transactions"]) == 5


def test_a_hallucinated_tool_name_becomes_feedback():
    call = types.FunctionCall(name="wire_money_to_me", args={})
    assert "Unknown tool" in agent.execute_tool_call(call)["error"]


# ---------------------------------------------------------------------------
# The loop: a fake client that replays a scripted conversation.
# ---------------------------------------------------------------------------


def _tool_turn(*calls) -> SimpleNamespace:
    parts = [
        types.Part(function_call=types.FunctionCall(name=name, args=args))
        for name, args in calls
    ]
    content = types.Content(role="model", parts=parts)
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)], text=None)


def _text_turn(text: str) -> SimpleNamespace:
    content = types.Content(role="model", parts=[types.Part(text=text)])
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)], text=text)


class FakeClient:
    """Replays scripted responses and records the history it was sent."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.histories = []
        self.models = SimpleNamespace(generate_content=self._generate_content)

    def _generate_content(self, *, model, contents, config):
        self.histories.append(list(contents))
        return self._responses.pop(0)


def test_loop_executes_parallel_calls_and_feeds_results_back():
    client = FakeClient(
        [
            # One turn, two tools — the model does this on its own.
            _tool_turn(
                ("get_account_balance", {"account_id": "ACC-1"}),
                ("get_recent_transactions", {"account_id": "ACC-1", "limit": 5}),
            ),
            _text_turn("Your balance is 4,512.30 AMD because rent and a card fee cleared."),
        ]
    )

    answer = agent.run_agent("Why is my balance on ACC-1 lower than I expected?", client)

    assert "4,512.30 AMD" in answer

    # Trajectory: the second call saw the request AND both results.
    second_history = client.histories[1]
    assert len(second_history) == 3  # user question, model tool request, results
    responses = second_history[2].parts
    assert len(responses) == 2
    assert {p.function_response.name for p in responses} == {
        "get_account_balance",
        "get_recent_transactions",
    }


def test_loop_stops_at_max_iterations():
    # A model that never stops asking must not loop forever.
    client = FakeClient(
        [_tool_turn(("get_account_balance", {"account_id": "ACC-1"}))]
        * agent.MAX_ITERATIONS
    )
    assert "maximum number of iterations" in agent.run_agent("hello", client)

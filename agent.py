"""A tool-calling agent in plain Python. No framework.

An LLM cannot call a tool. It can only emit a structured request — "call
get_account_balance with account_id ACC-1". YOUR code executes the function,
appends the result to the history, and calls the model again. That round trip,
placed in a loop the model controls, is the whole of "an agent".

The example is a bank support agent: two tools, fake account data, one question.

Run:  uv run python -u agent.py
      uv run python -u agent.py "Did I pay any card fees recently on ACC-1?"
"""

import json
import os
import random
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

load_dotenv()

# The newest flash model is also the most congested on the free tier. This one is
# plenty for a two-tool agent and answers in ~2s.
MODEL = "gemini-3.5-flash"
MAX_ITERATIONS = 10

# ---------------------------------------------------------------------------
# 1. The "world" the agent can act on.
#    In reality these would be SQL queries or REST calls to a core banking
#    system. Fake data keeps the lesson about the agent loop, not the plumbing.
# ---------------------------------------------------------------------------

ACCOUNTS = {
    "ACC-1": {"balance": 4512.30, "currency": "AMD", "owner": "H. Sargsyan"},
    "ACC-2": {"balance": 120000.00, "currency": "AMD", "owner": "H. Sargsyan"},
}

TRANSACTIONS = {
    "ACC-1": [
        {"date": "2026-10-21", "desc": "Rent payment", "amount": -250000.00},
        {"date": "2026-10-20", "desc": "Salary", "amount": 450000.00},
        {"date": "2026-10-19", "desc": "ANNUAL CARD SERVICE FEE", "amount": -12000.00},
        {"date": "2026-10-18", "desc": "Grocery store", "amount": -18500.00},
        {"date": "2026-10-15", "desc": "Transfer to ACC-2", "amount": -120000.00},
    ],
    "ACC-2": [],
}


# ---------------------------------------------------------------------------
# 2. The tool implementations — ordinary Python functions. Nothing magic.
# ---------------------------------------------------------------------------


def get_account_balance(account_id: str) -> dict:
    """Look up the current balance of one account."""
    account = ACCOUNTS.get(account_id)
    if account is None:
        # NOTE: we return an error rather than raising. The loop feeds this back
        # to the model, which can then apologise or ask for a valid account id.
        return {"error": f"No such account: {account_id}"}
    return {"account_id": account_id, **account}


def get_recent_transactions(account_id: str, limit: int = 5) -> dict:
    """List the most recent transactions on one account."""
    if account_id not in TRANSACTIONS:
        return {"error": f"No such account: {account_id}"}
    # `limit` arrives from the model, so it is untrusted input like any other
    # request parameter. The schema says INTEGER; coerce anyway and bound it.
    limit = max(1, min(int(limit), 50))
    return {"account_id": account_id, "transactions": TRANSACTIONS[account_id][:limit]}


# A plain name -> function registry. This is how your code finds what to execute
# when the model asks for a tool by name.
TOOL_IMPLEMENTATIONS = {
    "get_account_balance": get_account_balance,
    "get_recent_transactions": get_recent_transactions,
}


# ---------------------------------------------------------------------------
# 3. The tool SCHEMAS — what we send to the model.
#
#    The model never sees your code. It sees only these names, descriptions and
#    parameter types. So the description IS the interface: describe the tool's
#    PURPOSE, not the user phrasing you happen to expect. Treat schema text as
#    part of your API design, because it is.
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="get_account_balance",
            description=(
                "Get the current available balance for a single bank account. "
                "Use this whenever an answer depends on how much money an "
                "account currently holds."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "account_id": types.Schema(
                        type=types.Type.STRING,
                        description="The account identifier, e.g. 'ACC-1'.",
                    ),
                },
                required=["account_id"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_recent_transactions",
            description=(
                "List recent transactions on an account, newest first. Use this "
                "to investigate an unexpected balance, find a specific charge, "
                "or review account activity."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "account_id": types.Schema(
                        type=types.Type.STRING,
                        description="The account identifier, e.g. 'ACC-1'.",
                    ),
                    # INTEGER, not STRING. Declared as a string, the model sends
                    # "5", the function slices a list with it, and you get
                    # `TypeError: slice indices must be integers` — a schema bug
                    # wearing a Python bug's clothes.
                    "limit": types.Schema(
                        type=types.Type.INTEGER,
                        description="How many transactions to return. Defaults to 5, maximum 50.",
                    ),
                },
                required=["account_id"],
            ),
        ),
    ]
)


SYSTEM_INSTRUCTION = """You are a helpful bank support assistant.

Use the provided tools to look up real account data. Never invent numbers, balances or
transactions — if you cannot retrieve something with a tool, say so plainly.
When a user asks why a balance is unexpected, investigate: look at both the balance and
the recent transactions before answering.
Answer concisely, and format money with its currency.
"""


# ---------------------------------------------------------------------------
# 4. Reliability. This is the part that actually broke.
# ---------------------------------------------------------------------------


def build_client() -> genai.Client:
    return genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=types.HttpOptions(
            # Always bound a network call. Without a timeout a congested backend
            # does not fail — it hangs, and your agent looks dead rather than
            # broken. A hang is worse to operate than an error.
            timeout=30_000,  # milliseconds
            # The SDK retries internally too. Left on, its retries nest inside
            # ours and the delays multiply instead of adding. Own the policy at
            # exactly one layer; turn it off everywhere else.
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def generate_with_retry(client, contents, config, attempts: int = 5):
    """Call the model, retrying transient server-side failures.

    Two classes of failure behave completely differently and must not share a
    code path:

      * TRANSIENT (503 overloaded, 429 rate limited, 500) -> retry with backoff.
        The request was fine; the service was busy. Retrying usually works.
      * PERMANENT (400 bad request, 404 unknown model, 401/403 auth) -> fail now.
        Retrying a malformed request only wastes time and money.

    Exponential backoff with JITTER: 1s, 2s, 4s, 8s... plus a random fraction.
    The jitter is not decoration — without it, every client that failed at the
    same moment retries at the same moment and stampedes a service that is
    already recovering.
    """
    for attempt in range(attempts):
        try:
            return client.models.generate_content(
                model=MODEL, contents=contents, config=config
            )
        except genai_errors.ServerError as exc:  # 5xx — the service's problem
            if attempt == attempts - 1:
                raise
            delay = 2**attempt + random.uniform(0, 1)
            print(f"  [transient {exc.code}; retrying in {delay:.1f}s]")
            time.sleep(delay)
        except genai_errors.ClientError as exc:
            # 429 is a 4xx but IS worth retrying; the rest are our own fault.
            if exc.code != 429 or attempt == attempts - 1:
                raise
            delay = 2**attempt + random.uniform(0, 1)
            print(f"  [rate limited; retrying in {delay:.1f}s]")
            time.sleep(delay)
    raise RuntimeError("unreachable")


def execute_tool_call(call) -> dict:
    """Run one tool call and return a dict — never raise.

    A tool failure becomes DATA the model reads on the next iteration and can
    adapt to (retry with a valid account id, apologise, ask a question).
    Crashing throws that recovery away. Note this is the OPPOSITE of how the
    model's own failures are handled above.
    """
    function = TOOL_IMPLEMENTATIONS.get(call.name)
    if function is None:
        # The model asked for a tool that does not exist. Tell it so.
        return {"error": f"Unknown tool: {call.name}"}
    try:
        return function(**dict(call.args))
    except Exception as exc:  # noqa: BLE001 — deliberately broad: errors are data
        return {"error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# 5. The loop.
# ---------------------------------------------------------------------------


def run_agent(user_message: str, client=None) -> str:
    client = client or build_client()

    # `contents` IS the conversation. We append to it and resend the whole thing
    # every turn, because the model is stateless: this list is the only memory
    # that exists.
    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part(text=user_message)])
    ]

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=[TOOL_SCHEMAS],
        # Reliable tool choices, not creativity. But note: temperature=0 is NOT
        # determinism — identical runs still pick different tool arguments.
        temperature=0,
        # The SDK can execute tools for us. Disabled on purpose: running the
        # loop by hand is the entire point.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    for iteration in range(MAX_ITERATIONS):
        print(f"\n--- iteration {iteration + 1} ---")

        response = generate_with_retry(client, contents, config)

        candidate_parts = response.candidates[0].content.parts or []
        function_calls = [p.function_call for p in candidate_parts if p.function_call]

        # No tool requested -> the model is done. This is the loop's exit.
        if not function_calls:
            return response.text

        # Append the model's own turn (its tool request) before the results.
        # Skipping this is the most common bug: the model loses track of what it
        # asked for and asks again.
        contents.append(response.candidates[0].content)

        # A model can request several tools in ONE turn — here it asks for the
        # balance and the transactions together. So this is a list, not a single
        # call; with real I/O-bound tools you would run them concurrently.
        tool_response_parts = []
        for call in function_calls:
            print(f"  model wants: {call.name}({dict(call.args)})")
            result = execute_tool_call(call)
            print(f"  result: {json.dumps(result, default=str)[:160]}")
            # Tag each result with the tool name so the model can match it to
            # the call it made. `response` MUST be a dict.
            tool_response_parts.append(
                types.Part.from_function_response(name=call.name, response=result)
            )

        # role="user" looks wrong — your code is not the user — but it is the API
        # convention: anything that is not the model's own output is "user".
        contents.append(types.Content(role="user", parts=tool_response_parts))

    return "Stopped: hit the maximum number of iterations without a final answer."


if __name__ == "__main__":
    question = (
        " ".join(sys.argv[1:])
        or "Why is my balance on ACC-1 lower than I expected?"
    )
    print(f"USER: {question}")
    print(f"\n=== FINAL ANSWER ===\n{run_agent(question)}")

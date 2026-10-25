# A tool-calling agent, written from scratch

**An LLM cannot call a tool.**

You send it your question plus a list of tool *schemas*. It replies with structured data —
"call `get_account_balance` with `account_id: ACC-1`". That is all it can do. **Your code**
executes the function, appends the result to the history, and calls the model again. The model
never touches your database.

This is that round trip, written by hand on the Gemini API: no LangChain, no ADK, no framework.
Built to be able to explain every step, not to replace the frameworks.

The example is a bank support agent — two tools, fake account data, one question.

## It does something I didn't write

```
USER: Why is my balance on ACC-1 lower than I expected?

--- iteration 1 ---
  model wants: get_account_balance({'account_id': 'ACC-1'})
  result: {"account_id": "ACC-1", "balance": 4512.3, "currency": "AMD", ...}
  model wants: get_recent_transactions({'account_id': 'ACC-1'})
  result: {"account_id": "ACC-1", "transactions": [...]}

--- iteration 2 ---
=== FINAL ANSWER ===
Your current balance on ACC-1 is 4,512.30 AMD. The salary of 450,000 AMD landed on Oct 20,
but these cleared after it:
  Rent payment: -250,000 AMD (Oct 21)
  Annual card service fee: -12,000 AMD (Oct 19)
  ...
```

Both tools in a **single turn**, in parallel. Nothing in the code sequences them, and nothing
tells it that a salary followed by rent explains the gap. That emergent step is the whole
difference between an agent and an app that calls an LLM.

## Run it

```bash
uv sync
cp .env.example .env     # then paste a key: https://aistudio.google.com/apikey
uv run python -u agent.py
```

Ask it something else:

```bash
uv run python -u agent.py "Did I pay any card fees recently on ACC-1?"
```

Run the tests — no API key needed, the model is faked:

```bash
uv run pytest -q
```

- [`agent.py`](agent.py) — the whole agent: tools, schemas, retry policy, loop
- [`test_agent.py`](test_agent.py) — what you *can* assert about an agent, and what you can't

## What the loop is

An LLM is a stateless pure function: `f(history) -> next_message`. No memory, no network, and it
executes nothing. "Memory" is just resending the entire conversation on every call.

```python
while True:
    response = model(history)
    if response.has_tool_calls:
        results = execute(response.tool_calls)   # your code acts
        history += [response, results]           # feed the outcome back
        continue
    return response.text                         # no tool wanted -> done
```

An agent is that call placed in a loop where *the model* decides whether to continue. Three
properties: **autonomy** (it picks the next step), **tool use** (it can act and observe),
**iteration** (it adapts to what it sees).

## The full list of things that broke

Not one of these is an AI problem. They are ordinary distributed-systems problems — which is the
actual lesson: agent engineering is mostly reliability engineering. The prompt is the easy part.

**1. A tool schema is a contract, and it lies silently.** `limit` declared as `STRING` while the
function slices a list: the model dutifully sends `"5"`, and you get
`TypeError: slice indices must be integers`. A schema bug wearing a Python bug's clothes. Fixed in
two places — the schema says `INTEGER`, *and* the function coerces and bounds it, because tool
arguments are untrusted input like any other request parameter.

**2. A tool description is a production interface.** The model only ever sees the name, the
description and the parameter schema. Describing a tool by the user phrasing I expected ("use when
the user asks to list transactions") instead of by its purpose ("use to investigate an unexpected
balance") was enough to stop it being called at all. The question that broke it never used the
word "transactions".

**3. Models call tools in parallel.** Both tools above arrive in one turn, so the executor handles
a *list* of calls, not one. With real I/O-bound tools, run them concurrently.

**4. Tool failures and model failures need opposite handling.** I'd handled tool errors and not
model errors, so a 503 took the whole process down.

| | Tool error (bad account id, timeout in your own backend) | Model/infra error (503, 429) |
|---|---|---|
| Who recovers | the **model** | your **code** |
| What you do | return `{"error": ...}` as data and loop | retry with exponential backoff **+ jitter** |
| Why | it can adapt — ask again, apologise, try another id | the request was fine; the service was busy |

Crashing on a tool error throws the model's chance to recover away. Retrying a *malformed*
request (400, 404, 401) just burns time and money — only 5xx and 429 are worth a retry.

**5. Jitter is not decoration.** Without it, every client that failed at the same moment retries at
the same moment, and you stampede a service that is already recovering.

**6. Always bound a network call.** With no timeout, a congested backend didn't fail — it hung.
That is worse: a hang looks like a dead agent and tells you nothing. A 30s timeout turns it into a
clean, retryable error.

**7. Nested retries multiply instead of adding.** My retry wrapped the SDK's own retry, the delays
compounded, and the agent looked dead. Own the retry policy at exactly one layer and disable it
everywhere else (`HttpRetryOptions(attempts=1)`).

**8. `temperature=0` is not determinism.** Three runs, identical input: `limit=10`, `limit=5`, and
no `limit` at all. So you cannot assert equality. Assert on **trajectory** (were the right tools
called, with results fed back) and on **semantic** similarity of the answer. That is what
[`test_agent.py`](test_agent.py) does, and the reason its fake model is scripted rather than live.

## Why write this by hand at all

Frameworks do exactly these steps for you, and in production you probably want one. But every
failure above lives *underneath* the framework, where you still have to debug it. Generating tool
schemas from type hints and docstrings — as Google's ADK does — doesn't make description quality
matter less. It makes it matter more, because now it is implicit.

import boto3
import json

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")


def get_model_id() -> str:
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    return f"arn:aws:bedrock:us-east-1:{account_id}:inference-profile/us.anthropic.claude-sonnet-4-6"


def call_claude(**kwargs) -> dict:
    body = {"anthropic_version": "bedrock-2023-05-31", **kwargs}
    response = bedrock.invoke_model(modelId=get_model_id(), body=json.dumps(body))
    return json.loads(response["body"].read())


extraction_tools = [
    {
        "name": "extract_action_items",
        "description": "Extract every action item from meeting notes. Only use names and dates that appear in the notes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string"},
                            "owner": {"type": "string", "description": "Person responsible, or 'unassigned' if not specified in the notes"},
                            "due_date": {"type": "string", "description": "Due date, or 'not specified' if not given in the notes"}
                        },
                        "required": ["task", "owner", "due_date"]
                    }
                }
            },
            "required": ["action_items"]
        }
    }
]

MEETING_NOTES = {
    "Client kickoff": """
Kickoff call with Acme Corp's finance team. Maria Lopez will send over the
last three years of general ledger exports by October 15. Our side, Devon is
building the data model and wants a first draft ready for review on
October 22. Maria also asked that we schedule a weekly status call; Devon
will send the invite by end of day tomorrow.
""",
    "Internal sprint review": """
Sprint review for the reporting dashboard. Priya finished the revenue page
and will demo it to the client next Thursday. Kenji is fixing the date filter
bug before the demo. James will write the deployment runbook by Friday,
October 17. Everyone agreed the color palette needs a refresh to match the
client's brand guide.
""",
    "Vague notes (hallucination test)": """
Quick sync on the Q4 audit. Someone needs to follow up with the vendor about
the missing invoices. We should also get the access review done at some
point. Alex mentioned the risk register is out of date and will take a look
when things calm down.
""",
}


def extract_action_items(notes):
    response = call_claude(
        max_tokens=800,
        tools=extraction_tools,
        tool_choice={"type": "tool", "name": "extract_action_items"},
        messages=[{"role": "user", "content": f"Extract action items from these notes:\n\n{notes}"}]
    )
    for block in response["content"]:
        if block["type"] == "tool_use":
            return block["input"]["action_items"]
    return []


def exercise1_extraction():
    print("=== Exercise 1: Structured Extraction ===")
    for title, notes in MEETING_NOTES.items():
        items = extract_action_items(notes)
        print(f"\n--- {title} ({len(items)} action items) ---")
        print(json.dumps(items, indent=2))


def get_stock_price(ticker: str) -> str:
    fake_prices = {"AAPL": "$182.34", "TSLA": "$248.50", "MSFT": "$412.87"}
    return fake_prices.get(ticker.upper(), "Unknown ticker")


def get_company_info(ticker: str) -> str:
    fake_info = {
        "AAPL": "Apple Inc. - consumer electronics and software.",
        "TSLA": "Tesla Inc. - electric vehicles and clean energy.",
        "MSFT": "Microsoft Corp. - software and cloud computing."
    }
    return fake_info.get(ticker.upper(), "Unknown ticker")


stock_tools = [
    {
        "name": "get_stock_price",
        "description": "Get the current stock price for a ticker symbol",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"]
        }
    },
    {
        "name": "get_company_info",
        "description": "Get general company information for a ticker symbol",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"]
        }
    }
]

tool_functions = {
    "get_stock_price": get_stock_price,
    "get_company_info": get_company_info,
}

QUESTIONS = [
    "What's the current price of AAPL?",
    "Tell me about Tesla as a company.",
    "Should I invest in Microsoft?",
]


def text_from(response):
    return "".join(block["text"] for block in response["content"] if block["type"] == "text")


def exercise2_routing():
    print("\n=== Exercise 2: Multi-Tool Selection ===")
    for question in QUESTIONS:
        response = call_claude(
            max_tokens=300,
            tools=stock_tools,
            messages=[{"role": "user", "content": question}]
        )
        print(f"\nQuestion: {question}")
        print(f"Stop reason: {response['stop_reason']}")
        tool_blocks = [b for b in response["content"] if b["type"] == "tool_use"]
        if tool_blocks:
            for block in tool_blocks:
                print(f"Tool chosen: {block['name']}")
                print(f"Arguments: {block['input']}")
        else:
            print(f"No tool called. Response: {text_from(response)}")


def run_tool(name, tool_input):
    function = tool_functions.get(name)
    if function is None:
        return f"Error: unknown tool '{name}'"
    try:
        return function(**tool_input)
    except TypeError as error:
        return f"Error: bad arguments for {name}: {error}"


def handle_query(question, max_rounds=5):
    messages = [{"role": "user", "content": question}]

    for _ in range(max_rounds):
        response = call_claude(max_tokens=600, tools=stock_tools, messages=messages)

        if response["stop_reason"] != "tool_use":
            return text_from(response)

        messages.append({"role": "assistant", "content": response["content"]})
        tool_results = []
        for block in response["content"]:
            if block["type"] == "tool_use":
                result = run_tool(block["name"], block["input"])
                print(f"  -> {block['name']}({block['input']}) = {result}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": result,
                })
        messages.append({"role": "user", "content": tool_results})

    return "Stopped: reached the maximum number of tool rounds."


def exercise3_full_loop():
    print("\n=== Exercise 3: Complete the Loop ===")
    for question in QUESTIONS:
        print(f"\nQ: {question}")
        print(f"A: {handle_query(question)}")


def bonus_parallel_calls():
    print("\n=== Bonus: Parallel Tool Calls ===")
    question = "Compare AAPL and MSFT prices."
    response = call_claude(
        max_tokens=400,
        tools=stock_tools,
        messages=[{"role": "user", "content": question}]
    )
    tool_blocks = [b for b in response["content"] if b["type"] == "tool_use"]
    print(f"Question: {question}")
    print(f"Tool calls in a single response: {len(tool_blocks)}")
    for block in tool_blocks:
        print(f"  {block['name']}({block['input']})")
    print("\nFull loop answer:")
    print(handle_query(question))


REFLECTION = """
Reflection

Exercise 2 (ambiguous routing): "Should I invest in Microsoft?" does not map
cleanly to either tool. Claude tends to call both tools for MSFT (often in a
single response) and then give a hedged answer that it cannot give personal
investment advice, rather than asking a clarifying question. Either way, the
choice was driven entirely by the short tool descriptions. In a real agent,
where a wrong route could mean pulling the wrong client record or triggering
the wrong action, descriptions need to say when to use a tool, when not to,
and what it returns, and ambiguous requests need an explicit fallback path.

Bonus (parallel calls): for "Compare AAPL and MSFT prices", Claude typically
requests get_stock_price twice in the same response, once per ticker, instead
of handling them one after another. That is why handle_query loops over every
tool_use block and returns all tool_result blocks in one user message.

Overall: forcing tool use is far more reliable than asking for JSON in a
prompt. The output always matches the schema, so there is no stripping of
code fences or retrying on parse errors. But the vague meeting notes showed
the limit. The schema guarantees an owner and due_date field exist, not that
their values are true. When the notes said "someone" or "at some point",
Claude used "unassigned" and "not specified" because the schema descriptions
told it to, but it can still be tempted to attach Alex to tasks he was only
near, or to turn "when things calm down" into a date. Tool use guarantees
trustworthy format, not trustworthy content, so extracted values still need
validation against the source text.
"""


if __name__ == "__main__":
    exercise1_extraction()
    exercise2_routing()
    exercise3_full_loop()
    bonus_parallel_calls()
    print(REFLECTION)
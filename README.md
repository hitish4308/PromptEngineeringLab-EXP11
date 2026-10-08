# Prompt Engineering Lab - Experiment 11

## LCEL Chain (LangChain Expression Language)

Builds a simple LCEL chain: prompt | llm | parser
Generates a short story from a hardcoded topic.

## Task

Take a topic (e.g., "a dragon and a knight") and generate a 2-3
sentence story using a three-stage LCEL chain.

## The Chain

    chain = prompt | llm | parser

| Stage | Component | Input | Output |
|-------|-----------|-------|--------|
| 1 | ChatPromptTemplate | dict with {topic} | formatted messages |
| 2 | ChatNVIDIA | messages | AIMessage |
| 3 | StrOutputParser | AIMessage | plain string |

## Model and Parameters

- Model: openai/gpt-oss-20b (via langchain-nvidia-ai-endpoints)
- Temperature: 0.7
- Max tokens: 200

## Why ChatNVIDIA, not ChatOpenAI

The lab manual suggests `langchain-openai.ChatOpenAI` pointed at
OpenAI's endpoint. When redirected at NVIDIA NIM's endpoint
(`https://integrate.api.nvidia.com/v1`), ChatOpenAI returns
`403 Forbidden` because NIM validates headers the OpenAI SDK
does not send by default.

The fix is to use LangChain's official NVIDIA integration:
`langchain-nvidia-ai-endpoints.ChatNVIDIA`.

## Setup

Reuse the environment from Experiment 1:

    Copy-Item ..\EXP_1\.env .
    python -m pip install -r requirements.txt

The requirements file has been extended to include:

    langchain>=0.1.0
    langchain-nvidia-ai-endpoints>=0.1.0
    langchain-core>=0.1.0

## Run

    python lcel_chain.py

## Expected Output

For each of three hardcoded topics, a 2-3 sentence story is printed.
A streaming demo then shows token-by-token output.

## Key LCEL Concepts

- The `|` operator chains Runnables.
- Every Runnable implements: `invoke`, `stream`, `batch`, `ainvoke`.
- `chain.invoke({"topic": "..."})` runs the full chain once.
- `chain.stream({"topic": "..."})` yields tokens as they arrive.
- `chain.batch([...])` runs multiple inputs in parallel.

## Security

- Never commit .env
- Never paste API keys in chat, logs, or screenshots

## License

For educational / lab use only.

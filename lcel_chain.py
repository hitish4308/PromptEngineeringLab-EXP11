"""
Experiment 11 - LCEL Chain (LangChain Expression Language)

Builds a simple LCEL chain: prompt | llm | parser
Generates a short story about a hardcoded topic.

Uses langchain-nvidia-ai-endpoints (ChatNVIDIA) because
langchain-openai (ChatOpenAI) fails with NVIDIA NIM due to
header requirements.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# LCEL imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# ---------- Setup ----------
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.environ.get("NVIDIA_API_KEY")
if not api_key:
    raise SystemExit("NVIDIA_API_KEY not found. Create a .env file.")

# ChatNVIDIA picks up NVIDIA_API_KEY from the environment automatically.
# If it doesn't, we pass it explicitly.
if not os.environ.get("NVIDIA_API_KEY"):
    os.environ["NVIDIA_API_KEY"] = api_key

MODEL = "openai/gpt-oss-20b"


# ---------- Prompt Template ----------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a creative storyteller. Keep stories concise and vivid."),
    ("user", "Write a very short story (2-3 sentences) about {topic}."),
])


# ---------- LLM ----------
llm = ChatNVIDIA(model=MODEL, temperature=0.7, max_tokens=200)


# ---------- Output Parser ----------
parser = StrOutputParser()


# ---------- LCEL Chain ----------
# This is the LCEL pipe syntax:
#   prompt | llm | parser
# Each component's output becomes the next component's input.
chain = prompt | llm | parser


# ---------- Topics ----------
TOPICS = [
    "a dragon and a knight",
    "a lost spaceship",
    "a robot chef",
]


# ---------- Main ----------
if __name__ == "__main__":
    print(f"Model: {MODEL}")
    print(f"Chain: prompt | llm | parser")
    print(f"Topics: {len(TOPICS)}")
    print()

    for i, topic in enumerate(TOPICS, start=1):
        print("=" * 72)
        print(f"  TOPIC {i}: {topic}")
        print("=" * 72)

        # LCEL invoke: pass a dict matching the {topic} placeholder
        try:
            story = chain.invoke({"topic": topic})
            print(f"\nSTORY:\n  {story}\n")
        except Exception as e:
            print(f"\n[ERROR] {type(e).__name__}: {e}\n")

    # ---------- Stream demo (optional) ----------
    print("=" * 72)
    print("  STREAMING DEMO (LCEL .stream())")
    print("=" * 72)
    print("Topic: a lonely lighthouse\n")
    try:
        for chunk in chain.stream({"topic": "a lonely lighthouse"}):
            print(chunk, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")

    # ---------- Summary ----------
    print("=" * 72)
    print("  SUMMARY - LCEL CONCEPTS")
    print("=" * 72)
    print("""
LCEL (LangChain Expression Language) pipe syntax:

    chain = prompt | llm | parser

What each part does:

  1. ChatPromptTemplate
     - Defines the message structure (system + user)
     - {topic} is a placeholder filled at invoke time

  2. ChatNVIDIA (or any chat model)
     - Receives the formatted prompt
     - Returns an AIMessage

  3. StrOutputParser
     - Extracts the .content string from the AIMessage
     - Returns plain text

Why LCEL:

  - Composable: swap any part without touching the others
  - Streaming: chain.stream() yields tokens as they arrive
  - Batching: chain.batch([...]) runs many inputs in parallel
  - Async: chain.ainvoke() for asyncio-based apps
  - Type-safe: each step declares its input/output shape

Key insight: the | operator chains Runnables. Each Runnable
implements invoke / stream / batch / ainvoke, so the whole chain
supports all four methods automatically.
""")

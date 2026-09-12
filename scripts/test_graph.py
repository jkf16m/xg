"""Quick end-to-end test: builds graph, prefills task, streams node progress."""

import sys

from langchain_core.messages import AIMessage

from xg_project.graph import graph

TASK = "Create /tmp/xg_test.txt containing the word hello, then verify it exists."

result = graph.invoke(
    {"task": TASK, "messages": [], "reasoning": []},
    {"recursion_limit": 50},
)

print("\n=== reasoning steps ===")
for r in result["reasoning"]:
    print(r[:200])
    print("---")

print("=== agent messages ===")
for m in result["messages"]:
    if isinstance(m, AIMessage):
        print(type(m).__name__, str(m.content)[:200])
sys.exit(0)

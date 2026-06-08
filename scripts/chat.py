import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag import ask


def print_result(result: dict) -> None:
    print(f"\nBot: {result['answer']}")
    if result["sources"]:
        print(f"Sources: {', '.join(result['sources'])}")
    print()


def main() -> None:
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"You: {question}")
        print_result(ask(question))
        return

    print("QuantumForge Knowledge Base Assistant (type 'exit' to quit)\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("exit", "quit"):
            break
        print_result(ask(question))


if __name__ == "__main__":
    main()

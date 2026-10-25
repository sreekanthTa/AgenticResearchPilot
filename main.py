import sys

from graph import graph


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    question = input("Research question: ")
    result = graph.invoke({"question": question})
    print(result["report"])


if __name__ == "__main__":
    main()

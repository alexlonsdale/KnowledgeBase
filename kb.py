#!/usr/bin/env python3
"""
kb.py — Local LLM knowledge base CLI
Requires: pip install openai
Requires: LM Studio running with local server on localhost:1234

Usage:
  python kb.py -s <information>
  python kb.py -r <query>
  python kb.py list
  python kb.py clear
"""

import sys
import os
from datetime import datetime
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────

KB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge.md")
LM_STUDIO_URL = "http://localhost:1234/v1"
MODEL = "qwen2.5-3b-it"  # match the model name shown in LM Studio

# ── LM Studio client ──────────────────────────────────────────────────────────

client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_notes() -> str:
    if not os.path.exists(KB_FILE):
        return ""
    with open(KB_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()


def store(information: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"- [{timestamp}] {information}\n"
    with open(KB_FILE, "a", encoding="utf-8") as f:
        f.write(entry)
    print(f"✓ Stored: {information}")


def retrieve(query: str):
    notes = load_notes()
    if not notes:
        print("No notes stored yet. Use: python kb.py store <information>")
        return

    system_prompt = (
        "You are a personal knowledge assistant. "
        "The user has stored the following notes:\n\n"
        f"{notes}\n\n"
        "Answer the user's query using only the information in their notes. "
        "Be concise and direct. If the notes don't contain relevant information, say so."
    )

    print()  # breathing room
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        stream=True,
        temperature=0.3,  # low temp = more faithful retrieval
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print()  # final newline


def list_notes():
    notes = load_notes()
    if not notes:
        print("No notes stored yet.")
        return
    print("\n── Knowledge Base ──────────────────────────\n")
    print(notes)
    print("\n────────────────────────────────────────────")
    count = notes.count("\n- ") + 1
    print(f"{count} note(s) total.")


def clear_notes():
    if not os.path.exists(KB_FILE):
        print("Nothing to clear.")
        return
    confirm = input("Clear all notes? This cannot be undone. [y/N] ").strip().lower()
    if confirm == "y":
        with open(KB_FILE, "w", encoding="utf-8") as f:
            f.write("")
        print("✓ Knowledge base cleared.")
    else:
        print("Cancelled.")


def usage():
    print(__doc__)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        usage()
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "-s":
        if len(sys.argv) < 3:
            print("Usage: kb. -s <information>")
            sys.exit(1)
        information = " ".join(sys.argv[2:])
        store(information)

    elif command == "-r":
        if len(sys.argv) < 3:
            print("Usage: kb -r <query>")
            sys.exit(1)
        query = " ".join(sys.argv[2:])
        retrieve(query)

    elif command == "list":
        list_notes()

    elif command == "clear":
        clear_notes()

    else:
        print(f"Unknown command: {command}")
        usage()
        sys.exit(1)


if __name__ == "__main__":
    main()

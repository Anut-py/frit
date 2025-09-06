# This file generates a math fact corpus
# Do not run this anymore, the corpus been saved already in /workspace/corpora/math_corpus.txt

import random
from typing import List

# Basic operations
operations = [
    ("+", lambda x, y: x + y),
    ("-", lambda x, y: x - y),
    ("*", lambda x, y: x * y),
    ("/", lambda x, y: x / y if y != 0 else None),
    ("^", lambda x, y: x ** y),
]

# Different phrasings including symbolic and informal
STYLES = [
    lambda x, op, y, r: f"What is {x} {op} {y}? It's {r}.",
    lambda x, op, y, r: f"{x} {op} {y} equals {r}.",
    lambda x, op, y, r: f"Compute {x} {op} {y}. Answer: {r}",
    lambda x, op, y, r: f"If you take {x} and {op} it by {y}, you get {r}.",
    lambda x, op, y, r: f"The result of {x} {op} {y} is {r}.",
    lambda x, op, y, r: f"{x} {operation_to_words(op)} {y} makes {r}.",
    lambda x, op, y, r: f"{x} {op} {y} = {r}",  # Spaced symbolic style
    lambda x, op, y, r: f"{x} {operation_to_words(op)} {y} is {r}"  # Another English phrasing
]

def operation_to_words(op: str) -> str:
    return {
        "+": "plus",
        "-": "minus",
        "*": "times",
        "/": "divided by"
    }.get(op, op)

def format_result(result: int | float) -> str:
    if result == int(result):
        return str(int(result))
    else:
        formatted = f"{result:.6f}".rstrip('0').rstrip('.')
        return formatted

def generate_fact(x: int, y: int, op: str, func) -> str | None:
    result = func(x, y)
    if result is None:
        return None
    style = random.choice(STYLES)
    return style(format_result(x), op, format_result(y), format_result(result))

def generate_corpus(n: int = 100000, max_val: int = 200, max_exp: int = 5) -> List[str]:
    corpus = []
    while len(corpus) < n:
        op, func = random.choice(operations)
        x = random.randint(-max_val, max_val)
        if op == "/":
            y = random.randint(1, max_val) * (random.randint(0, 1) * 2 - 1)  # Avoid divide by zero
        elif op == "^":
            if x == 0:
                y = random.randint(1, max_exp)  # Avoid 0 to a non-positive power
            else:
                y = random.randint(-max_exp, max_exp)
        else:
            y = random.randint(-max_val, max_val)
        fact = generate_fact(x, y, op, func)
        if fact:
            corpus.append(fact)
    return corpus

def save_corpus(corpus: List[str], filename: str = "/workspace/corpora/math_corpus.txt"):
    with open(filename, "w") as f:
        for line in corpus:
            f.write(line + "\n")

if __name__ == "__main__":
    corpus = generate_corpus()
    save_corpus(corpus)
    print(f"Saved {len(corpus)} math facts to math_corpus.txt")

"""Preserve the documented override path while using the current answer prompt."""
from memondemand.runners.prompts import load_prompt


ANSWER_SYSTEM = load_prompt("answer_v2.txt")

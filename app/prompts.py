from typing import List, Dict
import json

REVERSE_PROMPT_SYSTEM = """You are a prompt optimizer.

Your responsibilities:
- Rewrite the user's query into a clearer and more explicit prompt.
- Do NOT ask clarifying questions.
- Do NOT request more information.
- Do NOT expand beyond user intent.
- Do NOT add extra tasks.
- Keep the meaning identical.
- If the query is extremely short (e.g., "google"), infer the most likely intended meaning minimally.
- Output JSON ONLY. No prose.

Return JSON:
{{
  "final_prompt": "<rewritten_prompt>",
  "context_requirements": []
}}

"""

DYNAMIC_PROMPT_TEMPLATE = """
System: You are an expert assistant. Be concise and cite sources as [[id]].
Instructions: {instructions}
Context:
{context_chunks}
User question: {user_question}
Constraints: {constraints}
"""

def build_reverse_prompt(user_query: str) -> str:
    return REVERSE_PROMPT_SYSTEM + "\nUser query: " + user_query

def assemble_prompt(instructions: str, chunks: List[Dict], user_question: str, constraints: str=""):
    ctx = "\n\n".join([f"[[{c['id']}]]\n{c['text']}" for c in chunks])
    return DYNAMIC_PROMPT_TEMPLATE.format(instructions=instructions, context_chunks=ctx, user_question=user_question, constraints=constraints)

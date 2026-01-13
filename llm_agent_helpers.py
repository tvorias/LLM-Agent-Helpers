from openai import OpenAI
from IPython.display import Markdown, display
import os
from dotenv import load_dotenv


def ask_question(question: str, max_messages: int = 20):
    """
    Ask a question to the LLM, storing conversation state in memory.
    """

    # Load environment variables once
    load_dotenv(override=True)

    # Initialize conversation history on first call
    if not hasattr(ask_question, "conversation_history"):

        system_prompt = (
            """You are a Python and data science assistant specializing in large language models (LLMs)
            and AI agentic workflows.

            You ONLY provide answers related to:
            - Python software development
            - Data science workflows
            - LLM APIs and implementations
            - AI agent design and orchestration

            ALL LLM interactions MUST use the OpenAI Python SDK.
            The word "client" ALWAYS refers to a software API client.
            """
        )

        ask_question.conversation_history = [
            {"role": "system", "content": system_prompt}
        ]

        # Initialize the OpenAI client lazily
        ask_question.client = OpenAI()

    # Trim history if needed (keep system prompt)
    while len(ask_question.conversation_history) > max_messages:
        ask_question.conversation_history.pop(1)

    # Add user question
    ask_question.conversation_history.append(
        {"role": "user", "content": question}
    )

    # Call the model
    response = ask_question.client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=ask_question.conversation_history
    )

    answer = response.choices[0].message.content

    # Store assistant reply
    ask_question.conversation_history.append(
        {"role": "assistant", "content": answer}
    )

    display(Markdown(answer))


def reset_memory():
    """Clear conversation history for ask_question."""
    if hasattr(ask_question, "conversation_history"):
        del ask_question.conversation_history
        del ask_question.client
        print("Conversation history reset.")
    else:
        print("No conversation history exists.")


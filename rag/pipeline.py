"""Production Haystack RAG pipeline.

Retriever (Chroma, top-3) -> ChatPromptBuilder -> Claude on Amazon Bedrock.

The system prompt is NOT hardcoded: it is passed in at run time from a
versioned prompt file (prompts/baseline.txt or prompts/optimized.txt), so the
DSPy/GEPA optimisation loop can improve it without touching pipeline code.
"""

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.amazon_bedrock import (
    AmazonBedrockChatGenerator,
)

import config
from rag.retrieval import retriever

USER_TEMPLATE = """Context from the knowledge base:
{% for doc in documents %}
[{{ loop.index }}] {{ doc.content }}
{% endfor %}

Customer question: {{ question }}"""


def build_pipeline() -> Pipeline:
    prompt_builder = ChatPromptBuilder(
        template=[
            ChatMessage.from_system("{{ system_prompt }}"),
            ChatMessage.from_user(USER_TEMPLATE),
        ],
        required_variables=["system_prompt", "question"],
    )
    # Auth: AmazonBedrockChatGenerator uses boto3, which reads the Bedrock
    # API key from the AWS_BEARER_TOKEN_BEDROCK env var (boto3 >= 1.39).
    llm = AmazonBedrockChatGenerator(
        model=config.PROD_MODEL,
        generation_kwargs={"maxTokens": 1024},
    )

    pipe = Pipeline()
    pipe.add_component("retriever", retriever())
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("llm", llm)
    pipe.connect("retriever.documents", "prompt_builder.documents")
    pipe.connect("prompt_builder.prompt", "llm.messages")
    return pipe


def answer(pipe: Pipeline, question: str, system_prompt: str):
    """Run the pipeline for one question. Returns (reply_text, retrieved_docs)."""
    result = pipe.run(
        {
            "retriever": {"query": question},
            "prompt_builder": {"question": question, "system_prompt": system_prompt},
        },
        include_outputs_from={"retriever"},
    )
    reply = result["llm"]["replies"][0].text
    documents = result["retriever"]["documents"]
    return reply, documents

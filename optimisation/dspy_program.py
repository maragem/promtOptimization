"""DSPy mirror of the generation step of the Haystack pipeline.

DSPy never runs in production. This thin program exists only so GEPA has
something to optimise: it reuses the exact same Chroma retriever as the
Haystack pipeline and wraps the LLM call in a DSPy signature whose
`instructions` field is the prompt under optimisation.
"""

import dspy

from rag.retrieval import format_context, retrieve_documents


class GenerateAnswer(dspy.Signature):
    question: str = dspy.InputField(desc="The customer's question")
    context: str = dspy.InputField(desc="Passages retrieved from the knowledge base")
    answer: str = dspy.OutputField(desc="The reply sent to the customer")


class SupportRag(dspy.Module):
    def __init__(self, instructions: str):
        super().__init__()
        self.generate = dspy.Predict(GenerateAnswer.with_instructions(instructions))

    def forward(self, question: str) -> dspy.Prediction:
        documents = retrieve_documents(question)
        context = format_context(documents)
        prediction = self.generate(question=question, context=context)
        return dspy.Prediction(answer=prediction.answer, context=context)

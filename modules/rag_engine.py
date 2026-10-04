import os
import json
import re
from google import genai
import config

# Initialize Google GenAI client using the Interactions API
client = genai.Client(api_key=config.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY"))
MODEL_NAME = "gemini-3.8-flash"

class RAGEngine:
    def generate_summary(self, context: str) -> str:
        """Generates a high-yield executive summary from context."""
        if not context or not context.strip():
            return "No content available to summarize."

        # Truncate context slice specifically for summaries to speed up PDF processing
        trimmed_context = context[:6000]

        prompt = f"""
        You are an expert academic tutor. Provide a clear, structured, high-yield summary 
        of the following text using bullet points and bold key terms:

        {trimmed_context}
        """
        try:
            interaction = client.interactions.create(
                model=MODEL_NAME,
                input=prompt
            )
            return interaction.output_text
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def generate_flashcards(self, context: str) -> list:
        """Extracts key concepts and formats them into flashcard Q&A objects."""
        if not context or not context.strip():
            return []

        prompt = f"""
        Extract 5 essential concepts from this study text and format them strictly as a JSON 
        list of objects. Each object must have a 'question' and 'answer' key.

        Text:
        {context[:4000]}

        Return ONLY raw JSON list without markdown backticks.
        """
        try:
            interaction = client.interactions.create(
                model=MODEL_NAME,
                input=prompt
            )
            raw_output = interaction.output_text
            
            # Robust extraction to avoid JSON parsing hangs
            json_match = re.search(r"\[.*\]", raw_output, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))

            cleaned_text = raw_output.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned_text)
        except Exception:
            return [{
                "question": "Overview Concept",
                "answer": "Refer to the executive summary tab for detailed notes."
            }]

    def answer_question(self, question: str, context: str) -> str:
        """Answers user queries grounded strictly in the provided retrieved context."""
        if not question or not question.strip():
            return "Please enter a valid question."

        prompt = f"""
        You are a precise study assistant. Answer the user's question using ONLY the provided context. 
        If the answer is not in the context, politely state that it isn't covered in the source.

        Context:
        {context[:6000]}

        Question: {question}
        """
        try:
            interaction = client.interactions.create(
                model=MODEL_NAME,
                input=prompt
            )
            return interaction.output_text
        except Exception as e:
            return f"Error processing query: {str(e)}"
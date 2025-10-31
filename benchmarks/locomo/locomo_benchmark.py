"""
LoComo-specific benchmark implementations.

Provides dataset, answer generator, and evaluator for the LoComo benchmark.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional
import asyncio
import pydantic
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Import common framework
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.benchmark_runner import BenchmarkDataset, LLMAnswerGenerator, LLMAnswerEvaluator

class LoComoDataset(BenchmarkDataset):
    """LoComo dataset implementation."""

    def load(self, path: Path, max_items: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load LoComo dataset from JSON file."""
        with open(path, 'r') as f:
            dataset = json.load(f)

        if max_items:
            dataset = dataset[:max_items]

        return dataset

    def get_item_id(self, item: Dict) -> str:
        """Get sample ID from LoComo item."""
        return item['sample_id']

    def prepare_sessions_for_ingestion(self, item: Dict) -> List[Dict[str, Any]]:
        """
        Prepare LoComo conversation sessions for batch ingestion.

        Returns:
            List of session dicts with 'content', 'context', 'event_date'
        """
        conv = item['conversation']
        speaker_a = conv['speaker_a']
        speaker_b = conv['speaker_b']

        # Get all session keys sorted
        session_keys = sorted([k for k in conv.keys() if k.startswith('session_') and not k.endswith('_date_time')])

        batch_contents = []

        for session_key in session_keys:
            if session_key not in conv or not isinstance(conv[session_key], list):
                continue

            session_data = conv[session_key]

            # Build session content from all turns
            session_parts = []
            for turn in session_data:
                speaker = turn['speaker']
                text = turn['text']
                session_parts.append(f"{speaker}: {text}")

            if not session_parts:
                continue

            # Get session date
            date_key = f"{session_key}_date_time"
            session_date = self._parse_date(conv.get(date_key, "1:00 pm on 1 January, 2023"))

            # Add to batch
            session_content = "\n".join(session_parts)
            batch_contents.append({
                "content": session_content,
                "context": f"Conversation session between {speaker_a} and {speaker_b} (conversation {item['sample_id']} session {session_key})",
                "event_date": session_date
            })

        return batch_contents

    def get_qa_pairs(self, item: Dict) -> List[Dict[str, Any]]:
        """
        Extract QA pairs from LoComo item.

        Returns:
            List of QA dicts with 'question', 'answer', 'category'
        """
        return item['qa']

    def _parse_date(self, date_string: str) -> datetime:
        """Parse LoComo date format to datetime."""
        # Format: "1:56 pm on 8 May, 2023"
        try:
            dt = datetime.strptime(date_string, "%I:%M %p on %d %B, %Y")
            return dt.replace(tzinfo=timezone.utc)
        except:
            return datetime.now(timezone.utc)


class QuestionAnswer(pydantic.BaseModel):
    """Answer format for LoComo questions."""
    answer: str
    reasoning: str


class LoComoAnswerGenerator(LLMAnswerGenerator):
    """LoComo-specific answer generator using Groq."""

    def __init__(self):
        """Initialize with Groq client."""
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")

        base_url = os.getenv('GROQ_BASE_URL', 'https://api.groq.com/openai/v1')
        self.client = AsyncOpenAI(
            api_key=groq_api_key,
            base_url=base_url
        )

    async def generate_answer(
        self,
        question: str,
        memories: List[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """
        Generate answer from retrieved memories using Groq.

        Returns:
            Tuple of (answer, reasoning)
        """
        # Format context
        context_parts = []
        for result in memories:
            context_parts.append({"text": result.get("text"), "context": result.get("context"), "event_date": result.get("event_date")})

        context = json.dumps(context_parts)

        # Use Groq to generate answer
        try:
            response = await self.client.beta.chat.completions.parse(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful expert assistant answering questions from lme_experiment users based on the provided context."
                    },
                    {
                        "role": "user",
                        "content": f"""
# CONTEXT:
You have access to facts and entities from a conversation.

# INSTRUCTIONS:
1. Carefully analyze all provided memories
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. Always convert relative time references to specific dates, months, or years.
6. Be as specific as possible when talking about people, places, and events
7. Timestamps in memories represent the actual time the event occurred, not the time the event was mentioned in a message.

Clarification:
When interpreting memories, use the timestamp to determine when the described event happened, not when someone talked about the event.

Example:

Memory: (2023-03-15T16:33:00Z) I went to the vet yesterday.
Question: What day did I go to the vet?
Correct Answer: March 15, 2023
Explanation:
Even though the phrase says "yesterday," the timestamp shows the event was recorded as happening on March 15th. Therefore, the actual vet visit happened on that date, regardless of the word "yesterday" in the text.


# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question
2. Examine the timestamps and content of these memories carefully
3. Look for explicit mentions of dates, times, locations, or events that answer the question
4. If the answer requires calculation (e.g., converting relative time references), show your work
5. Formulate a precise, concise answer based solely on the evidence in the memories
6. Double-check that your answer directly addresses the question asked
7. Ensure your final answer is specific and avoids vague time references

Context:

{context}

Question: {question}
Answer:

"""
                    }
                ],
                response_format=QuestionAnswer
            )
            answer_obj = response.choices[0].message.parsed
            return answer_obj.answer, answer_obj.reasoning
        except Exception as e:
            return f"Error generating answer: {str(e)}", "Error occurred during answer generation."


class JudgeResponse(pydantic.BaseModel):
    """Judge response format."""
    correct: bool
    reasoning: str


class LoComoAnswerEvaluator(LLMAnswerEvaluator):
    """LoComo-specific answer evaluator using Groq."""

    def __init__(self):
        """Initialize with Groq client."""
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")

        base_url = os.getenv('GROQ_BASE_URL', 'https://api.groq.com/openai/v1')
        self.client = AsyncOpenAI(
            api_key=groq_api_key,
            base_url=base_url
        )

    async def judge_answer(
        self,
        question: str,
        correct_answer: str,
        predicted_answer: str,
        semaphore: asyncio.Semaphore
    ) -> Tuple[bool, str]:
        """
        Evaluate predicted answer using Groq LLM-as-judge.

        Returns:
            Tuple of (is_correct, reasoning)
        """
        async with semaphore:
            try:
                response = await self.client.beta.chat.completions.parse(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert grader that determines if answers to questions match a gold standard answer"
                        },
                        {
                            "role": "user",
                            "content": f"""
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You williolw23 be given the following data:
        (1) a question (posed by one user to another user), 
        (2) a ’gold’ (ground truth) answer, 
        (3) a generated answer
    which you will score as CORRECT/WRONG.

    The point of the question is to ask about something one user should know about the other user based on their prior conversations.
    The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
    Question: Do you remember what I got the last time I went to Hawaii?
    Gold answer: A shell necklace
    The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

    For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

    Now it’s time for the real question:
    Question: {question}
    Gold answer: {correct_answer}
    Generated answer: {predicted_answer}

    First, provide a short (one sentence) explanation of your reasoning. Short reasoning is preferred.
    If it's correct, set correct=true.
"""
                        }
                    ],
                    temperature=0,
                    max_tokens=4096,
                    response_format=JudgeResponse
                )

                judgement = response.choices[0].message.parsed
                return judgement.correct, judgement.reasoning

            except Exception as e:
                print(f"Error judging answer: {e}")
                return False, f"Error: {str(e)}"

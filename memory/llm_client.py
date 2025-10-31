"""
LLM client for fact extraction and other AI-powered operations.

Uses OpenAI-compatible API (works with Groq, OpenAI, etc.)
"""
import os
import json
import re
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Literal
from openai import AsyncOpenAI
from pydantic import BaseModel, Field


class Entity(BaseModel):
    """An entity extracted from text."""
    text: str = Field(
        description="The entity name as it appears in the fact"
    )
    type: Literal["PERSON", "ORG", "PLACE", "PRODUCT", "CONCEPT"] = Field(
        description="Entity type: PERSON, ORG, PLACE, PRODUCT, or CONCEPT"
    )


class ExtractedFact(BaseModel):
    """A single extracted fact from text."""
    fact: str = Field(
        description="Self-contained factual statement with subject + action + context"
    )
    date: str = Field(
        description="Absolute date/time when this fact occurred in ISO format (YYYY-MM-DDTHH:MM:SSZ). If text mentions relative time (yesterday, last week, this morning), calculate absolute date from the provided context date."
    )
    entities: List[Entity] = Field(
        default_factory=list,
        description="List of important entities mentioned in this fact with their types"
    )


class FactExtractionResponse(BaseModel):
    """Response containing all extracted facts."""
    facts: List[ExtractedFact] = Field(
        description="List of extracted factual statements"
    )


def chunk_text(text: str, max_chars: int = 120000) -> List[str]:
    """
    Split text into chunks at sentence boundaries using LangChain's text splitter.

    Uses RecursiveCharacterTextSplitter which intelligently splits at sentence boundaries
    and allows chunks to slightly exceed max_chars to finish sentences naturally.

    Args:
        text: Input text to chunk
        max_chars: Maximum characters per chunk (default 120k ≈ 30k tokens)
                   Note: chunks may slightly exceed this to complete sentences

    Returns:
        List of text chunks, roughly under max_chars
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # If text is small enough, return as-is
    if len(text) <= max_chars:
        return [text]

    # Configure splitter to split at sentence boundaries first
    # Separators in order of preference: paragraphs, newlines, sentences, words
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
        separators=[
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            ". ",    # Sentence endings
            "! ",    # Exclamations
            "? ",    # Questions
            "; ",    # Semicolons
            ", ",    # Commas
            " ",     # Words
            "",      # Characters (last resort)
        ],
    )

    return splitter.split_text(text)


def get_llm_client() -> AsyncOpenAI:
    """
    Get configured async LLM client.

    Supports:
    - Groq (default): Set GROQ_API_KEY and optionally GROQ_BASE_URL
    - OpenAI: Set OPENAI_API_KEY

    Returns:
        Configured AsyncOpenAI client
    """
    # Check for Groq configuration first
    groq_api_key = os.getenv('GROQ_API_KEY')
    if groq_api_key:
        base_url = os.getenv('GROQ_BASE_URL', 'https://api.groq.com/openai/v1')
        return AsyncOpenAI(
            api_key=groq_api_key,
            base_url=base_url
        )

    # Fall back to OpenAI
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if openai_api_key:
        return AsyncOpenAI(api_key=openai_api_key)

    raise ValueError(
        "No LLM API key found. Set GROQ_API_KEY or OPENAI_API_KEY environment variable."
    )


async def _extract_facts_from_chunk(
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    event_date: datetime,
    context: str,
    model: str,
    temperature: float,
    max_tokens: int,
    client: AsyncOpenAI
) -> List[Dict[str, str]]:
    """
    Extract facts from a single chunk (internal helper for parallel processing).
    """
    # Format event_date for the prompt
    event_date_str = event_date.strftime("%Y-%m-%dT%H:%M:%SZ")

    prompt = f"""You are extracting facts from text for an AI memory system. Each fact will be stored and retrieved later.

## CONTEXT INFORMATION
- Current reference date/time: {event_date_str}
- Context: {context if context else 'no context provided'}

## CRITICAL: Facts must be DETAILED and COMPREHENSIVE

Each fact should:
1. Be SELF-CONTAINED - readable without the original context
2. Include ALL relevant details: WHO, WHAT, WHERE, WHEN, WHY, HOW
3. **CRITICAL: ALWAYS include the SUBJECT (who is doing/saying/experiencing)**
4. Preserve specific names, dates, numbers, locations, relationships
5. Resolve pronouns to actual names/entities (I → speaker name, their → possessor name)
6. **CRITICAL: Preserve possessive relationships** (their kids → whose kids, his car → whose car)
7. Include surrounding context that makes the fact meaningful
8. Capture nuances, reasons, causes, and implications

**COMMON MISTAKES TO AVOID:**
- ❌ "The kids were excited" → Missing WHO the kids belong to
- ✅ "Melanie's kids were excited" or "Melanie took her kids who were excited"
- ❌ "Someone went hiking" → Missing WHO
- ✅ "Bob went hiking"
- ❌ "The car broke down" → Missing whose car
- ✅ "Alice's car broke down"

## TEMPORAL INFORMATION (VERY IMPORTANT)
For each fact, extract the ABSOLUTE date/time when it occurred:
- If text mentions ABSOLUTE dates ("on March 15, 2024", "last Tuesday"), use that date
- If text mentions RELATIVE times ("yesterday", "last week", "this morning", "3 days ago"), calculate the absolute date using the reference date above.
- if text mentions a vague relative time without a specific day ("last week", "this morning"), transform the date in relative with absolute context ("last week" + " 2 june 2024" -> "week before June 2 2024") in the text and use the absolute date for the 'date' field 
- If NO specific time is mentioned, use the reference date
- Always output dates in ISO format: YYYY-MM-DDTHH:MM:SSZ

Examples of date extraction:
- Reference: 2024-03-20T10:00:00Z
- "Yesterday I went hiking" → date: 2024-03-19T10:00:00Z
- "Last week I joined Google" → date: 2024-03-13T10:00:00Z (approximately)
- "This morning I had coffee" → date: 2024-03-20T08:00:00Z
- "I work at Google" (no time mentioned) → date: 2024-03-20T10:00:00Z (use reference)

## What to EXTRACT (BE EXHAUSTIVE - DO NOT SKIP ANYTHING):
- **Biographical information (CRITICAL - NEVER MISS)**:
  - Origins: home country, birthplace, where someone is from ("my home country Sweden" = Caroline is from Sweden)
  - Current location: where they live now
  - Jobs, roles, backgrounds, experiences, skills
  - Family background, heritage, cultural identity
  - Education, training, certifications
- **Events (NEVER MISS THESE)**:
  - ANY action that happened (went, did, attended, joined, started, finished, etc.)
  - Photos, images, videos shared or taken ("here's a photo", "took a picture", "captured")
  - Social activities (meetups, gatherings, meals, conversations)
  - Achievements, milestones, accomplishments
  - Travels, visits, locations visited
  - Purchases, acquisitions, creations
- **Identity and personal details**:
  - Origins, nationality, home country, roots
  - Cultural background, heritage
  - Family connections (grandmother from X, parents in Y)
- **Opinions and beliefs**: who believes what and why
- **Recommendations and advice**: specific suggestions with reasoning
- **Descriptions**: detailed explanations of how things work
- **Relationships**: connections between people, organizations, concepts
- **States and conditions**: current status, ongoing situations

## CRITICAL: Extract EVERY event mentioned, even casual ones
- "here's a photo of X" = someone took/shared a photo of X
- "I was with friends last week" = meetup/gathering with friends last week
- "sent you that link" = action of sending a link
- DO NOT skip events just because they seem minor or casual

## What to SKIP (ONLY these):
- Greetings, thank yous, acknowledgments (unless they reveal information)
- Filler words ("um", "uh", "like")
- Pure reactions without content ("wow", "cool", "nice")
- Incomplete thoughts or sentence fragments with no meaning

## ENTITY EXTRACTION (CRITICAL):
For EACH fact, extract ALL important entities mentioned with their types:
- **PERSON**: Names of individuals (Alice, Bob, Dr. Smith)
- **ORG**: Companies, institutions, teams (Google, MIT, AI Team)
- **PLACE**: Cities, countries, locations, venues (Mountain View, Yosemite, The Coffee Shop)
- **PRODUCT**: Specific products, tools, technologies (iPhone, Python, TensorFlow)
- **CONCEPT**: Important topics, projects, subjects (AI, machine learning, Project Phoenix)

Entity extraction rules:
- Use the EXACT form as it appears in the fact (preserve capitalization)
- Assign the correct type to distinguish ambiguous entities (Apple the company = ORG, apple the fruit = PRODUCT/CONCEPT)
- Include both full names and commonly used short forms if both appear
- Extract proper nouns and key identifying terms
- Skip generic terms (the, a, some) and pronouns (he, she, they)
- Each entity must have both text and type

## EXAMPLES of GOOD facts (detailed, comprehensive):

Input: "Alice mentioned she works at Google in Mountain View. She joined the AI team last year."
GOOD fact: "Alice works at Google in Mountain View on the AI team, which she joined last year"
GOOD date: Calculate based on reference date (if reference is 2024-03-20, "last year" = 2023-03-20)
GOOD entities: [
  {{"text": "Alice", "type": "PERSON"}},
  {{"text": "Google", "type": "ORG"}},
  {{"text": "Mountain View", "type": "PLACE"}},
  {{"text": "AI team", "type": "ORG"}}
]

Input: "Yesterday Bob went hiking in Yosemite because it helps him clear his mind."
GOOD fact: "Bob went hiking in Yosemite because it helps him clear his mind"
GOOD date: Reference date minus 1 day
GOOD entities: [
  {{"text": "Bob", "type": "PERSON"}},
  {{"text": "Yosemite", "type": "PLACE"}}
]

Input: "Here's a photo of me with my friends taken last week at the beach."
GOOD fact: "Someone shared/took a photo with their friends at the beach"
GOOD date: Reference date minus 7 days (last week)
GOOD entities: []
NOTE: Extract the event (photo taken/shared with friends at beach), NOT just that a photo exists

Input: "I sent you that article about AI last Tuesday."
GOOD fact: "Someone sent an article about AI"
GOOD date: Calculate last Tuesday from reference date
GOOD entities: [
  {{"text": "AI", "type": "CONCEPT"}}
]

Input: "I bought an Apple laptop and some apples from the store."
GOOD fact: "Someone bought an Apple laptop and some apples from the store"
GOOD entities: [
  {{"text": "Apple", "type": "ORG"}},
  {{"text": "apples", "type": "PRODUCT"}}
]
NOTE: Use type to distinguish "Apple" the company from "apples" the fruit

Input: "Melanie said 'Yesterday I took the kids to the museum - it was so cool seeing their eyes light up!'"
BAD fact: "The kids were excited about the museum"
BAD entities: [{{"text": "museum", "type": "PLACE"}}]
PROBLEM: Missing WHO (Melanie) and whose kids!

GOOD fact: "Melanie took her kids to the museum yesterday and they were excited, with their eyes lighting up"
GOOD date: Reference date minus 1 day
GOOD entities: [
  {{"text": "Melanie", "type": "PERSON"}},
  {{"text": "museum", "type": "PLACE"}}
]
NOTE: Preserved the subject (Melanie) and possessive relationship (her kids)

Input: "Caroline said 'This necklace is from my grandma in my home country, Sweden. She gave it to me when I was young.'"
BAD fact: "Caroline received a necklace as a gift from her grandmother when she was young"
BAD entities: [{{"text": "Caroline", "type": "PERSON"}}, {{"text": "necklace", "type": "PRODUCT"}}]
PROBLEM: Missing the CRITICAL biographical info that Caroline is from Sweden!

GOOD facts (extract MULTIPLE facts):
1. "Caroline is from Sweden, which is her home country"
   entities: [{{"text": "Caroline", "type": "PERSON"}}, {{"text": "Sweden", "type": "PLACE"}}]
2. "Caroline's grandmother is from Sweden"
   entities: [{{"text": "Caroline", "type": "PERSON"}}, {{"text": "Sweden", "type": "PLACE"}}]
3. "Caroline received a necklace as a gift from her grandmother in Sweden when she was young"
   entities: [{{"text": "Caroline", "type": "PERSON"}}, {{"text": "Sweden", "type": "PLACE"}}, {{"text": "necklace", "type": "PRODUCT"}}]
NOTE: Extract SEPARATE facts for biographical details (home country) AND events (gift received)

## TEXT TO EXTRACT FROM:
{chunk}

Remember:
1. BE EXHAUSTIVE - Extract EVERY event, action, and fact mentioned
2. DO NOT skip casual mentions like "here's a photo", "I was with X", "sent you Y"
3. **ALWAYS include the SUBJECT** - never say "the kids" without saying whose kids
4. **Preserve possessive relationships** - "their kids" must become "Person's kids"
5. **Extract biographical details as SEPARATE facts** - "my home country Sweden" should create a fact "Person is from Sweden"
6. Include ALL details, names, numbers, reasons, and context in the fact text
7. Extract the absolute date for EACH fact by calculating relative times from the reference date
8. Extract ALL entities with their types (PERSON, ORG, PLACE, PRODUCT, CONCEPT) for each fact
9. Use types to disambiguate entities (Apple the company = ORG, apple the fruit = PRODUCT)
10. When in doubt, EXTRACT IT - better to have too many facts than miss important events"""

    response = await client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an EXHAUSTIVE fact and entity extractor. CRITICAL RULES: 1) ALWAYS include the SUBJECT (never 'the kids' without whose kids), 2) Extract biographical details as SEPARATE facts (if someone mentions 'my home country Sweden', extract 'Person is from Sweden' as its own fact), 3) Extract EVERY event, action, and fact - never skip anything. For each fact, extract ALL important entities with their types: PERSON, ORG, PLACE, PRODUCT, CONCEPT. Use types to disambiguate (Apple=ORG vs apples=PRODUCT). Preserve possessive relationships (their→whose). Include casual mentions (photos, meetups). Calculate absolute dates from relative times. When in doubt, extract it - better too many facts than missing critical biographical/identity information."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=FactExtractionResponse,
        extra_body={"service_tier": "auto"},
    )

    # Extract the parsed response
    extraction_response = response.choices[0].message.parsed

    # Convert to dict format
    chunk_facts = [fact.model_dump() for fact in extraction_response.facts]

    return chunk_facts


async def extract_facts_from_text(
    text: str,
    event_date: datetime,
    context: str = "",
    model: str = "openai/gpt-oss-120b",
    temperature: float = 0.1,
    max_tokens: int = 65000,
    chunk_size: int = 5000
) -> List[Dict[str, str]]:
    """
    Extract semantic facts from conversational or narrative text using LLM.

    For large texts (>chunk_size chars), automatically chunks at sentence boundaries
    to avoid hitting output token limits. Processes ALL chunks in PARALLEL for speed.

    Args:
        text: Input text (conversation, article, etc.)
        event_date: Reference date for resolving relative times
        context: Context about the conversation/document
        model: LLM model to use
        temperature: Sampling temperature (lower = more focused)
        max_tokens: Maximum tokens in response
        chunk_size: Maximum characters per chunk

    Returns:
        List of fact dictionaries with 'fact' and 'date' keys
    """
    client = get_llm_client()

    # Chunk text if necessary
    chunks = chunk_text(text, max_chars=chunk_size)

    # Process all chunks in parallel using asyncio.gather
    tasks = [
        _extract_facts_from_chunk(
            chunk=chunk,
            chunk_index=i,
            total_chunks=len(chunks),
            event_date=event_date,
            context=context,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            client=client
        )
        for i, chunk in enumerate(chunks)
    ]

    # Wait for all chunks to complete in parallel
    chunk_results = await asyncio.gather(*tasks)

    # Flatten results from all chunks
    all_facts = []
    for chunk_facts in chunk_results:
        all_facts.extend(chunk_facts)

    return all_facts

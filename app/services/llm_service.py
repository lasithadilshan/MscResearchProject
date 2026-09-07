import re
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from app.core.config import settings
from app.services.document_service import query_vector_store
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def initialize_llm():
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        openai_api_key=settings.OPENAI_API_KEY
    )

def parse_json_output(text: str):
    """Extract JSON from model output, stripping optional code fences."""
    clean = text.strip()
    fence = re.search(r"```json\s*(.*?)```", clean, re.DOTALL)
    if fence:
        clean = fence.group(1).strip()
    else:
        fence = re.search(r"```\s*(.*?)```", clean, re.DOTALL)
        if fence:
            clean = fence.group(1).strip()
    try:
        return json.loads(clean), None
    except Exception as e:
        return clean, str(e)

def calculate_confidence_level(prompt: str, answer: str) -> float:
    try:
        prompt_clean = re.sub(r'[^\w\s]', '', prompt.lower())
        answer_clean = re.sub(r'[^\w\s]', '', answer.lower())
        
        prompt_keywords = set(prompt_clean.split())
        answer_words = set(answer_clean.split())
        
        common_keywords = prompt_keywords.intersection(answer_words)
        keyword_overlap = len(common_keywords) / len(prompt_keywords) if prompt_keywords else 0
        
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform([prompt_clean, answer_clean])
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        confidence_score = (keyword_overlap * 0.3 + similarity_score * 0.7) * 100
        return min(confidence_score, 100)
    except Exception:
        return 0

def calculate_match_percentage(answer: str, source_text: str) -> float:
    try:
        answer_clean = re.sub(r'[^\w\s]', '', answer.lower())
        source_clean = re.sub(r'[^\w\s]', '', source_text.lower())
        
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([source_clean, answer_clean])
        match_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        match_percentage = match_score * 100
        return min(match_percentage, 100)
    except Exception:
        return 0

def get_confidence_category(percentage: float):
    if percentage >= 70: return "High", "🟢"
    elif percentage >= 30: return "Medium", "🟡"
    else: return "Low", "🔴"

# Prompt constants
USER_STORY_PROMPT = """
You are an Expert Business Analyst. Your responsibility is to read the entire Business Requirement Document (BRD) below and convert it into detailed User Stories.

Think step-by-step and ensure you extract EVERY POSSIBLE user story derived from the BRD.
DO NOT summarize or skip any functionality. Provide fully complete User Stories only.

## DOCUMENT:
{context}

Return ONLY valid JSON (no markdown, no explanations):
{{
  "user_stories": [
    {{
      "id": "US_001",
      "title": "[Title]",
      "story": "As a [role], I want [feature] so that [value]",
      "acceptance_criteria": ["Given [context], when [action], then [outcome]"],
      "priority": "High",
      "story_points": 5,
      "category": "Core",
      "notes": []
    }}
  ]
}}
"""

TEST_CASE_PROMPT = """
You are a Senior QA Engineer. Design a comprehensive test suite for the following user story:

## USER STORY:
{input_text}

Return ONLY valid JSON (no markdown):
{{
  "test_cases": [
    {{
      "id": "TC_001",
      "title": "Title",
      "preconditions": ["Precondition"],
      "test_data": ["data: value"],
      "test_steps": ["1. Step"],
      "expected_results": ["Result"],
      "priority": "High",
      "attachments": []
    }}
  ]
}}
"""

CUCUMBER_PROMPT = """
You are an expert test automation engineer. Transform these test cases into a complete Cucumber test suite.

## TEST CASES:
{input_text}

**FEATURE FILE (FeatureName.feature):**
```gherkin
[content]
```

**STEP DEFINITIONS (FeatureSteps.java):**
```java
[content]
```
"""

SELENIUM_PROMPT = """
You are an expert test automation engineer. Transform these test cases into a complete Selenium (Python) test script.

## TEST CASES:
{input_text}

Return the complete python script.
```python
[content]
```
"""

def generate_artifact(task_type: str, document_id: str, input_text: str = None) -> dict:
    llm = initialize_llm()
    start_time = time.time()
    
    if task_type == "user_stories":
        context = query_vector_store(document_id, "Find all functional and non-functional requirements, business rules, and user workflows.", n_results=10)
        prompt = USER_STORY_PROMPT.format(context=context)
    elif task_type == "test_cases":
        prompt = TEST_CASE_PROMPT.format(input_text=input_text)
        context = input_text
    elif task_type == "cucumber":
        prompt = CUCUMBER_PROMPT.format(input_text=input_text)
        context = input_text
    elif task_type == "selenium":
        prompt = SELENIUM_PROMPT.format(input_text=input_text)
        context = input_text
    else:
        raise ValueError("Invalid task type")

    response = llm.invoke(prompt)
    result_text = response.content
    processing_time = time.time() - start_time
    
    parsed, parse_error = parse_json_output(result_text) if task_type in ["user_stories", "test_cases"] else (result_text, None)
    
    confidence_score = calculate_confidence_level(prompt, result_text)
    match_score = calculate_match_percentage(result_text, context)
    overall_score = (confidence_score + match_score) / 2
    
    conf_level, _ = get_confidence_category(confidence_score)
    match_level, _ = get_confidence_category(match_score)
    overall_level, _ = get_confidence_category(overall_score)
    
    return {
        "result": parsed,
        "parse_error": parse_error,
        "quality_assessment": {
            "confidence_score": round(confidence_score, 2),
            "match_score": round(match_score, 2),
            "overall_score": round(overall_score, 2),
            "confidence_level": conf_level,
            "match_level": match_level,
            "overall_level": overall_level
        },
        "processing_time_seconds": round(processing_time, 2)
    }

import re
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from app.core.config import settings
from app.services.document_service import query_vector_store
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel, Field
from typing import List, Optional

def initialize_llm():
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        openai_api_key=settings.OPENAI_API_KEY
    )

class UserStory(BaseModel):
    id: str = Field(description="Unique sequential ID (US_001, US_002, ...)")
    title: str = Field(description="Specific, searchable title from BRD content")
    story: str = Field(description="As a [specific role], I want [feature] so that [value]")
    acceptance_criteria: List[str] = Field(description="Detailed acceptance criteria covering multiple scenarios")
    priority: str = Field(description="Critical, High, Medium, or Low")
    story_points: int = Field(description="Fibonacci sequence: 1, 2, 3, 5, 8, 13")
    category: str = Field(description="Category name")
    notes: List[str] = Field(description="Technical and business notes")

class UserStoriesResponse(BaseModel):
    user_stories: List[UserStory] = Field(description="List of extracted user stories")

class TestCase(BaseModel):
    id: str = Field(description="TC_001")
    title: str = Field(description="Descriptive title of the test case")
    preconditions: List[str] = Field(description="List of preconditions")
    test_data: List[str] = Field(description="List of test data requirements")
    test_steps: List[str] = Field(description="List of test steps")
    expected_results: List[str] = Field(description="List of expected results")
    priority: str = Field(description="High, Medium, or Low")
    attachments: List[str] = Field(description="List of any attachments or references needed")

class TestCasesResponse(BaseModel):
    test_cases: List[TestCase] = Field(description="List of designed test cases")

PUNCTUATION_REGEX = r'[^\w\s]'

def calculate_confidence_level(prompt: str, answer: str) -> float:
    try:
        prompt_clean = re.sub(PUNCTUATION_REGEX, '', prompt.lower())
        answer_clean = re.sub(PUNCTUATION_REGEX, '', answer.lower())
        
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
        # Cap source text length to prevent massive CPU spikes on large BRDs
        capped_source = source_text[:30000]
        answer_clean = re.sub(PUNCTUATION_REGEX, '', answer.lower())
        source_clean = re.sub(PUNCTUATION_REGEX, '', capped_source.lower())
        
        # Use simple unigrams for faster processing
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
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
You are an Expert Business Analyst with 20+ years of experience in requirements engineering and Agile transformation.

CRITICAL INSTRUCTION: Extract EVERY POSSIBLE user story from the BRD below. No requirement should be missed.

## DOCUMENT TO ANALYZE:
{context}

## EXTRACTION METHODOLOGY:

### PHASE 1: Comprehensive Requirement Mining
1. **Functional Requirements**: Extract ALL features, capabilities, and system behaviors mentioned
2. **User Interactions**: Identify EVERY user action, input, output, and workflow step
3. **Business Rules**: Capture ALL validation rules, constraints, and business logic
4. **Data Requirements**: Extract ALL data fields, entities, relationships, and transformations
5. **Integration Points**: Identify ALL system interfaces, APIs, and external dependencies
6. **Non-Functional Requirements**: Include performance, security, usability, accessibility needs
7. **Reporting & Analytics**: Extract ALL reporting, monitoring, and analytical capabilities
8. **Administrative Functions**: Capture ALL configuration, setup, and maintenance features
9. **Error Scenarios**: Include ALL error handling, validation, and recovery scenarios
10. **Compliance & Audit**: Extract ALL regulatory, compliance, and audit trail requirements

### PHASE 2: User Story Generation Rules

**MANDATORY FORMAT**: "As a [specific role], I want [specific feature/action] so that [measurable business value]"

**Story Categorization** (Generate stories for EACH category where applicable):
- **Core Features**: Primary business functions
- **CRUD Operations**: Create, Read, Update, Delete for each entity
- **Search & Filter**: All search, filter, sort capabilities
- **Validation & Rules**: Input validation, business rule enforcement
- **Workflow & Process**: Multi-step processes, approvals, state transitions
- **Notifications & Alerts**: Email, SMS, in-app notifications
- **Reports & Exports**: All reporting and data export features
- **Security & Access**: Authentication, authorization, role management
- **Integration**: External system interactions, API calls
- **Configuration**: System settings, preferences, customization
- **Audit & Compliance**: Logging, tracking, compliance features
- **Error Handling**: Error recovery, rollback, exception scenarios
- **Performance**: Load handling, response time, scalability
- **Mobile/Responsive**: Device-specific features
- **Accessibility**: Support for users with disabilities

**Acceptance Criteria Requirements**:
- Minimum 3-5 criteria per story
- Use strict Gherkin format: Given [context], When [action], Then [outcome]
- Include: Happy path, Error scenarios, Boundary conditions, Business rules
- Reference specific data fields, values, and thresholds from the BRD

**Priority Assignment Logic**:
- "Critical": Core business functions, regulatory requirements, security
- "High": Primary user workflows, key features
- "Medium": Secondary features, enhancements
- "Low": Nice-to-have, future considerations

**Story Sizing Guidance & Point Calculation (STRICT)**:
- MUST use the Fibonacci sequence: 1, 2, 3, 5, 8, 13
- 1 Point: Simple UI change, static text, or minor configuration (0-1 acceptance criteria).
- 2 Points: Simple feature, minimal logic, no backend changes (1-2 acceptance criteria).
- 3 Points: Standard CRUD operation, basic validation, single table database changes (3-4 acceptance criteria).
- 5 Points: Complex business logic, cross-module impacts, external API integrations, advanced error handling (5-6 acceptance criteria).
- 8 Points: Major architectural changes, highly complex algorithms, third-party system integrations requiring orchestration (>6 acceptance criteria).
- 13 Points: Epic-level feature (MUST be broken down into smaller stories if possible).
- Break complex features into multiple smaller stories using vertical slicing.

### PHASE 3: Quality Checks

**Ensure EVERY story has**:
1. Unique sequential ID (US_001, US_002, ...)
2. Clear, specific, searchable title
3. Complete user story statement with role, feature, and value
4. 3-5 detailed acceptance criteria covering multiple scenarios
5. Realistic priority based on business impact
6. Relevant technical and business notes

**Extraction Completeness Verification**:
- Every paragraph in the BRD should generate at least one user story
- Every user role mentioned should appear in multiple stories
- Every data field should have CRUD stories
- Every business rule should have validation stories
- Every integration point should have connection stories

### OUTPUT REQUIREMENTS:

Your output must exactly match the schema provided. You must be exhaustive and find EVERY requirement.

IMPORTANT RULES:
1. Generate AT LEAST 25-30 stories for a typical BRD
2. Use EXACT terminology, field names, and values from the BRD
3. NO generic placeholders - use specific BRD content
4. NO trailing commas in JSON
5. EVERY requirement in the BRD must be covered
6. Include negative scenarios and edge cases
7. Ensure technical accuracy and business relevance

BEGIN EXTRACTION NOW - BE EXHAUSTIVE!
"""

TEST_CASE_PROMPT = """
You are a highly experienced Senior QA Engineer with over 15 years of expertise in software testing and quality assurance.

Your responsibility is to design a comprehensive test suite for the following user story:


{input_text}


Provide professional, detailed, and well-structured test cases based on the following functional and non-functional requirements:

### Scope of Test Cases:
- Include **positive**, **negative**, **edge**, **database related where applicable**, and **alternative** scenarios.
- Address **input validation**, **error handling**, **security**, **usability**, **performance**, **exploratory**, **exceptional**, and **compatibility** (where applicable).
- Ensure all test cases are **independent**, **clear**, and **suitable for automation**.
- Use **realistic and meaningful** test data.

### Output Format:
Your output must exactly match the schema provided. Generate as many test cases as needed to cover all scenarios.
"""

CUCUMBER_PROMPT = """
You are an expert test automation engineer specializing in BDD and Cucumber.
Follow Cucumber best practices and Gherkin syntax as described in the official documentation (features, scenarios, backgrounds, tags, step definitions, hooks, data tables, doc strings).

Your task is to transform the following test cases into a complete, production-ready Cucumber test suite.

## INPUT TEST CASES
Convert these test cases into Cucumber artifacts:

{input_text}


## STRICT GHERKIN AND CUCUMBER RULES

1. General Gherkin rules
- Use only these step keywords: Feature, Background, Scenario, Scenario Outline, Examples, Given, When, Then, And, But.
- Steps must be written in business-readable language (no implementation details).
- Each Scenario must be independent and executable in isolation.
- Keep steps short, clear, and describing behavior, not UI mechanics.
- Avoid duplication by reusing generic steps across scenarios.

2. Feature file structure
Generate EXACTLY ONE complete feature file.

It MUST include:
- A concise, meaningful Feature name.
- A short description (business value and context).
- A Background section ONLY if there are common preconditions shared by most scenarios.
- Multiple Scenarios that cover ALL provided test cases.
- Use Scenario Outline + Examples where the same workflow is repeated with different data.
- Use tags to organize scenarios:
  - @smoke for core happy paths
  - @regression for wider coverage
  - @critical for high‑risk or business‑critical flows
- Use Given for preconditions, When for actions, Then for verifications.
- Use And / But only to extend the previous Given/When/Then step when it improves readability.
- Use Data Tables for structured multi-field inputs or outputs.
- Use Doc Strings (\"\"\" ... \"\"\") for larger text payloads if appropriate.

3. Scenario quality
- Include both positive and negative scenarios where test cases imply them.
- Make each scenario self-explanatory from a business perspective.
- Prefer reusing generic, parameterized steps (e.g. "I enter \"<username>\" in the username field").
- Avoid referencing UI technology (like “click the blue React button”), keep it domain-focused.

## STEP DEFINITIONS (JAVA, CUCUMBER-JVM)

Create Java step definitions aligned with the feature file:

1. Structure and imports
- Use a realistic package name, e.g. `package steps;`
- Include typical imports (do not reference any specific project framework beyond Selenium + Cucumber + JUnit/TestNG style assertions), for example:
  - Cucumber: io.cucumber.java.{{en.Given, en.When, en.Then, en.And, Before, After}}
  - Selenium: org.openqa.selenium.*
  - Selenium support: org.openqa.selenium.support.ui.WebDriverWait, ExpectedConditions
  - Assertions: org.junit.jupiter.api.Assertions or org.testng.Assert
  - Logging: java.util.logging.Logger or similar

2. Implementation rules
- Each Gherkin step must have a matching annotated Java method:
  - @Given("...")
  - @When("...")
  - @Then("...")
  - @And("...")
- Use parameterized step definitions with capture groups and/or Cucumber expression parameters, for example:
  - @When("I enter {{string}} in the username field")
- Use Page Object Model (POM) style:
  - Assume there are page classes like LoginPage, DashboardPage, etc.
  - Interact with the UI only via page objects (no raw locators in the step class where possible).
- Use explicit waits (WebDriverWait) instead of Thread.sleep.
- Add clear comments for any non-trivial logic.
- Include meaningful assertions that verify outcomes described in the Then steps.
- Include basic error handling where appropriate and log key events.

3. Hooks and test lifecycle
- Add @Before hook to initialize WebDriver, open the application, and any common test setup needed.
- Add @After hook to close/quit the browser and clean up state.
- Keep hooks generic and reusable across scenarios.

4. Data handling
- Support Cucumber DataTable in step definitions when scenarios use tables:
  - Convert DataTable to Map/List or custom objects as appropriate.
- Handle Doc Strings when present as method parameters (String body).

## OUTPUT FORMAT (STRICT)

Generate output EXACTLY in the following structure, with no extra sections, text, or explanations:

**FEATURE FILE (FeatureName.feature):**
```gherkin
[Complete feature file content here]
```

**STEP DEFINITIONS (FeatureSteps.java):**
```java
[Complete Java step definitions here]
```

**TEST DATA NOTES:**
[Concise recommendations for test data management, e.g. using external files, environment-specific data, anonymized production-like data]

**EXECUTION NOTES:**
[Short notes on how to run these tests with Cucumber + Java + Selenium, including any dependencies or runner configuration assumptions]

Constraints:
- Do NOT output any markdown outside the specified code fences and sections.
- Ensure the Gherkin is syntactically valid and would be accepted by Cucumber.
- Ensure every test case from the input is covered by at least one scenario or scenario outline.
"""

SELENIUM_PROMPT = """You are a Senior Test Automation Engineer specializing in Selenium and Python. Convert the following test case into a robust, production-ready Selenium WebDriver script in Python.

INSTRUCTIONS:
- Use best practices for maintainability, reliability, and readability.
- Include all necessary imports, setup, and teardown logic.
- Use explicit waits (WebDriverWait) for element interactions, not time.sleep.
- Add comments for each major step.
- Validate all expected outcomes with assert statements.
- Handle exceptions gracefully and log errors.
- Use Page Object Model if the scenario is complex.
- Ensure the script is ready to run as a standalone test.
- Use realistic locators (id, name, xpath, css selector) based on the test case.
- If data is required, use sample values from the test case.
- If login or setup is needed, include those steps.

Test Case:

{input_text}


Return ONLY the complete Python code, no explanations, no markdown."""

def generate_artifact(task_type: str, document_id: str, input_text: str = None) -> dict:
    llm = initialize_llm()
    start_time = time.time()
    
    schema = None
    if task_type == "user_stories":
        # Retrieve a large number of chunks (e.g. 100) to ensure the LLM sees the ENTIRE document 
        # (up to ~120k chars) rather than just a small semantic sample. This guarantees exhaustive extraction.
        context = query_vector_store(document_id, "Find all functional and non-functional requirements, business rules, and user workflows.", n_results=100)
        prompt = USER_STORY_PROMPT.format(context=context)
        schema = UserStoriesResponse
    elif task_type == "test_cases":
        prompt = TEST_CASE_PROMPT.format(input_text=input_text)
        context = input_text
        schema = TestCasesResponse
    elif task_type == "cucumber":
        prompt = CUCUMBER_PROMPT.format(input_text=input_text)
        context = input_text
    elif task_type == "selenium":
        prompt = SELENIUM_PROMPT.format(input_text=input_text)
        context = input_text
    else:
        raise ValueError("Invalid task type")

    if schema:
        structured_llm = llm.with_structured_output(schema)
        response_obj = structured_llm.invoke(prompt)
        parsed = response_obj.model_dump()
        result_text = json.dumps(parsed)
        parse_error = None
    else:
        response = llm.invoke(prompt)
        result_text = response.content
        parsed = result_text
        parse_error = None
        
    processing_time = time.time() - start_time
    
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

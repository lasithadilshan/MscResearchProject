import os
import re
import time

import numpy as np
import pandas as pd
import pptx
import streamlit as st
from docx import Document
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="SDLC Automate APP",
    page_icon="images/favicon.png"
)

# Get the API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Function to calculate confidence level based on prompt-answer accuracy
def calculate_confidence_level(prompt, answer):
    """Calculate confidence level based on how well the answer addresses the prompt."""
    try:
        # Preprocess texts
        prompt_clean = re.sub(r'[^\w\s]', '', prompt.lower())
        answer_clean = re.sub(r'[^\w\s]', '', answer.lower())
        
        # Extract key terms from prompt
        prompt_keywords = set(prompt_clean.split())
        answer_words = set(answer_clean.split())
        
        # Calculate keyword overlap
        common_keywords = prompt_keywords.intersection(answer_words)
        keyword_overlap = len(common_keywords) / len(prompt_keywords) if prompt_keywords else 0
        
        # Use TF-IDF for semantic similarity
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform([prompt_clean, answer_clean])
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Combined confidence score (weighted average)
        confidence_score = (keyword_overlap * 0.3 + similarity_score * 0.7) * 100
        
        return min(confidence_score, 100)  # Cap at 100%
    except Exception as e:
        st.error(f"Error calculating confidence: {str(e)}")
        return 0

# Function to calculate match percentage with source document
def calculate_match_percentage(answer, source_text):
    """Calculate how well the answer matches the source document content."""
    try:
        # Preprocess texts
        answer_clean = re.sub(r'[^\w\s]', '', answer.lower())
        source_clean = re.sub(r'[^\w\s]', '', source_text.lower())
        
        # Use TF-IDF for document similarity
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([source_clean, answer_clean])
        match_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Convert to percentage
        match_percentage = match_score * 100
        
        return min(match_percentage, 100)  # Cap at 100%
    except Exception as e:
        st.error(f"Error calculating match percentage: {str(e)}")
        return 0

# Function to get confidence level category
def get_confidence_category(percentage):
    """Convert percentage to confidence level category."""
    if percentage >= 70:
        return "High", "🟢"
    elif percentage >= 30:
        return "Medium", "🟡"
    else:
        return "Low", "🔴"

# Function to display confidence metrics
def display_confidence_metrics(confidence_score, match_score):
    """Display confidence and match metrics in a formatted way."""
    conf_level, conf_icon = get_confidence_category(confidence_score)
    match_level, match_icon = get_confidence_category(match_score)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Confidence Level",
            value=f"{conf_level} ({confidence_score:.1f}%)",
            help="How well the answer addresses the prompt"
        )
        st.write(f"{conf_icon} **Confidence Score:** {confidence_score:.1f}%")
    
    with col2:
        st.metric(
            label="Match Percentage", 
            value=f"{match_level} ({match_score:.1f}%)",
            help="How well the answer matches the source document"
        )
        st.write(f"{match_icon} **Match Score:** {match_score:.1f}%")
    
    # Overall assessment
    overall_score = (confidence_score + match_score) / 2
    overall_level, overall_icon = get_confidence_category(overall_score)
    
    st.info(f"{overall_icon} **Overall Assessment:** {overall_level} ({overall_score:.1f}%)")

# Streamlit sidebar setup
with st.sidebar:
    st.title("Your BRD Documents")
    model_selection = st.selectbox(
        "Select AI Model",
        options=["Open AI GPT 4.1", "Google Gemini 2.0 Flash"]
    )
    st.write(f"Selected Model: {model_selection}")
    uploaded_file = st.file_uploader("Upload a file to generate user stories", type=["pdf", "docx", "txt", "xlsx", "pptx"])
    
    # Add confidence settings
    st.subheader("Quality Assessment Settings")
    show_confidence = st.checkbox("Show Confidence & Match Analysis", value=True)
    detailed_metrics = st.checkbox("Show Detailed Metrics", value=False)

# Function to extract text from various file types
@st.cache_resource
def extract_text_from_file(file):
    """Extracts text based on file type, with caching for faster retrieval."""
    text = ""
    file_ext = os.path.splitext(file.name)[1].lower()

    # Handle PDF files
    if file_ext == ".pdf":
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()

    # Handle Word (.docx) files
    elif file_ext == ".docx":
        doc = Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"

    # Handle text (.txt) files
    elif file_ext == ".txt":
        text = file.read().decode("utf-8")

    # Handle Excel files (.xlsx, .xls)
    elif file_ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file)
        text = df.to_string()

    # Handle PowerPoint files (.pptx, .ppt)
    elif file_ext in [".pptx", ".ppt"]:
        ppt = pptx.Presentation(file)
        for slide in ppt.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"

    return text

# Process the uploaded file and extract text for the vector store
@st.cache_resource
def process_uploaded_file(uploaded_file):
    return extract_text_from_file(uploaded_file) if uploaded_file else ""

# Function to create vector store from extracted text
@st.cache_resource
def create_vector_store(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=800,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_texts(chunks, embeddings)

# Streamlit app setup
st.header("BRD to User Story, Test Case, Cucumber Script, and Selenium Script")

# Set up tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["User Story Generation", "User Story to Test Case", "Test Case to Cucumber Script", "Test Case to Selenium Script"])

# Initialize variables
text = ""
vector_store = None
qa_chain = None

# Process uploaded file if available
if uploaded_file:
    text = process_uploaded_file(uploaded_file)
    if text:
        vector_store = create_vector_store(text)
        
        if model_selection == "Open AI GPT 4.1":
            llm = ChatOpenAI(
                model="gpt-4.1",
                temperature=0.5,
            )
        elif model_selection == "Google Gemini 2.0 Flash":
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.7,
            )
        else:
            llm = None
            st.error("Please select a valid AI model.")
        if llm:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever()
            )

# User Story Generation Tab
with tab1:
    start_time = time.time()
    if uploaded_file and qa_chain:
        prompt_message = (
            """ 
            
You are an Expert Business Analyst with 20+ years of experience in requirements engineering and Agile transformation.

CRITICAL INSTRUCTION: Extract EVERY POSSIBLE user story from the BRD below. No requirement should be missed.

## DOCUMENT TO ANALYZE:
{document_text}

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

**Story Sizing Guidance**:
- Break complex features into multiple smaller stories
- Each story should be completable in 1-3 days
- Use vertical slicing (end-to-end functionality)

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

Return ONLY valid JSON (no markdown, no explanations):

{
  "user_stories": [
    {
      "id": "US_001",
      "title": "[Specific, searchable title from BRD content]",
      "story": "As a [specific role from BRD], I want [specific feature from BRD] so that [specific value from BRD]",
      "acceptance_criteria": [
        "Given [specific context from BRD], when [specific action], then [specific outcome with data/thresholds]",
        "Given [error scenario], when [invalid action], then [error handling from BRD]",
        "Given [edge case], when [boundary condition], then [expected behavior]",
        "Given [business rule from BRD], when [rule trigger], then [rule enforcement]",
        "Given [performance requirement], when [load condition], then [performance metric]"
      ],
      "priority": "[Critical/High/Medium/Low]",
      "story_points": [1-13],
      "category": "[category_name]",
      "notes": [
        "Affected users: [specific roles from BRD]",
        "Related module: [specific module/component from BRD]",
        "Dependencies: [specific systems/features from BRD]",
        "Data entities: [specific entities from BRD]",
        "Business rules: [specific rules from BRD]"
      ]
    }
  ]
}

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
        )
        
        start_query_time = time.time()
        matches = vector_store.similarity_search(prompt_message, k=3)
        response = qa_chain.invoke({"query": prompt_message})
        
        st.subheader("Generated User Stories")
        st.write(response['result'])
        
        # Calculate and display confidence metrics
        if show_confidence:
            st.subheader("Quality Assessment")
            
            confidence_score = calculate_confidence_level(prompt_message, response['result'])
            match_score = calculate_match_percentage(response['result'], text)
            
            display_confidence_metrics(confidence_score, match_score)
            
            if detailed_metrics:
                with st.expander("Detailed Analysis"):
                    st.write("**Prompt Analysis:**")
                    st.write(f"- Prompt length: {len(prompt_message)} characters")
                    st.write(f"- Response length: {len(response['result'])} characters")
                    st.write(f"- Source document length: {len(text)} characters")
                    
                    st.write("**Retrieved Context:**")
                    for i, match in enumerate(matches):
                        st.write(f"Match {i+1}: {match.page_content[:200]}...")

        # Display timing info
        st.write(f"Document loading time: {time.time() - start_time:.2f} seconds")
        st.write(f"Query processing time: {time.time() - start_query_time:.2f} seconds")
    else:
        st.write("Please upload a BRD document in the sidebar to generate user stories.")

# User Story to Test Case Tab
with tab2:
    st.subheader("Convert User Story to Test Case")
    user_story_text = st.text_area("Enter the user story text here to generate test cases:")

    if st.button("Generate Test Cases"):
        if user_story_text and qa_chain:
            test_case_prompt = (
                """
              You are a highly experienced Senior QA Engineer with over 15 years of expertise in software testing and quality assurance.

        Your responsibility is to design a comprehensive test suite for the following user story:\n\n"""
        + user_story_text +
        """\n\nProvide professional, detailed, and well-structured test cases based on the following functional and non-functional requirements:

        ### Scope of Test Cases:
        - Include **positive**, **negative**, **edge**, **database related where applicable**, and **alternative** scenarios.
        - Address **input validation**, **error handling**, **security**, **usability**, **performance**, **exploratory**, **exceptional**, and **compatibility** (where applicable).
        - Ensure all test cases are **independent**, **clear**, and **suitable for automation**.
        - Use **realistic and meaningful** test data.

        ### Output Format:
        Respond in **valid JSON only** using the following structure.
        IMPORTANT: Do NOT include trailing commas before closing brackets or braces.

        {{
          "test_cases": [
            {{
              "id": "TC_001",
              "title": "Generate a descriptive title as per the given sample: Verify that an author can be designated as the reprint contact for a publication so that accurate contact information is maintained",
              "preconditions": [
                "1. Browser: Chrome v122",
                "2. OS: Windows 11",
                "3. User is logged in as an administrator",
                "4. A publication record exists",
                "5. An author is linked to an affiliation within the publication",
                "6. Database related preconditions where applicable"
              ],
              "test_data": [
                "publication_id: PUB-2025-1001",
                "author_id: AUTH-3002",
                "contact_type: reprint"
              ],
              "test_steps": [
                "1. Step-by-step instruction with numbering",
                "2. Clear action expected from the tester or system",
                "3. Include test data related steps with proper condition and hierarchy"
              ],
              "expected_results": [
                "The contacts list updates and shows author AUTH-3002 with a “Reprint” badge.",
                "A success message “Reprint contact updated” is displayed.",
                "No other author’s contact type is changed."
              ],
              "priority": "{priority}",
              "attachments": []
            }}
          ]
        }}

        """ 
            )
            start_test_case_time = time.time()
            response = qa_chain.invoke({"query": test_case_prompt})
            
            st.subheader("Generated Test Cases")
            st.write(response['result'])
            
            # Calculate and display confidence metrics
            if show_confidence:
                st.subheader("Quality Assessment")
                
                confidence_score = calculate_confidence_level(test_case_prompt, response['result'])
                match_score = calculate_match_percentage(response['result'], user_story_text)
                
                display_confidence_metrics(confidence_score, match_score)
            
            st.write(f"Test case generation time: {time.time() - start_test_case_time:.2f} seconds")
        elif not qa_chain:
            st.write("Please upload a BRD document first to initialize the AI model.")
        else:
            st.write("Please enter a user story to generate test cases.")

# Test Case to Cucumber Script Tab
with tab3:
    st.subheader("Convert Test Case to Cucumber Script")
    
    # Add configuration options in an expander
    with st.expander("⚙️ Cucumber Generation Settings (Optional)", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            framework = st.selectbox(
                "Testing Framework",
                options=['selenium', 'appium', 'restassured', 'playwright'],
                index=0,
                help="Select the automation framework for your tests"
            )
            
            language = st.selectbox(
                "Programming Language",
                options=['java', 'javascript', 'python', 'ruby'],
                index=0,
                help="Select the language for step definitions"
            )
            
            pattern = st.selectbox(
                "Design Pattern",
                options=['page_object', 'screenplay', 'traditional'],
                index=0,
                help="Select the design pattern for your test code"
            )
        
        with col2:
            tags_input = st.text_input(
                "Tags (comma-separated)",
                value="@automated, @regression",
                help="Enter tags for test categorization"
            )
            tags = [tag.strip() for tag in tags_input.split(',') if tag.strip()]
            
            include_hooks = st.checkbox("Include Setup/Teardown Hooks", value=True)
            include_examples = st.checkbox("Include Example Tables", value=True)
            include_negative = st.checkbox("Include Negative Scenarios", value=True)
        
        use_advanced = st.checkbox(
            "Use Advanced Configuration", 
            value=False,
            help="Enable advanced configuration options above"
        )
    
    # Main text area for test case input
    test_case_text = st.text_area(
        "Enter the test case text here to generate Cucumber script:",
        height=200,
        placeholder="""Example:
Test Case: User Login
- Given the user is on the login page
- When the user enters valid credentials
- And clicks the login button
- Then the user should be redirected to the dashboard"""
    )

    # Generate button with different modes
    col1, col2 = st.columns([3, 1])
    with col1:
        generate_button = st.button("🥒 Generate Cucumber Script", type="primary", use_container_width=True)
    with col2:
        if st.button("📋 Clear", use_container_width=True):
            st.rerun()

    if generate_button:
        if test_case_text and qa_chain:
            # Import the generator module (make sure the module is in your project)
            from cucumber_generator import (
                CucumberConfig, generate_cucumber_script_advanced_streamlit,
                generate_cucumber_script_streamlit)
            
            if use_advanced:
                # Use advanced configuration
                custom_config = CucumberConfig(
                    framework=framework,
                    language=language,
                    pattern=pattern,
                    tags=tags,
                    include_hooks=include_hooks,
                    include_examples=include_examples,
                    include_negative=include_negative,
                    show_confidence=show_confidence
                )
                
                cucumber_script = generate_cucumber_script_advanced_streamlit(
                    qa_chain=qa_chain,
                    test_case_text=test_case_text,
                    config=custom_config,
                    calculate_confidence_fn=calculate_confidence_level,
                    calculate_match_fn=calculate_match_percentage,
                    display_metrics_fn=display_confidence_metrics
                )
            else:
                # Use basic configuration
                cucumber_script = generate_cucumber_script_streamlit(
                    qa_chain=qa_chain,
                    test_case_text=test_case_text,
                    show_confidence=show_confidence,
                    calculate_confidence_fn=calculate_confidence_level,
                    calculate_match_fn=calculate_match_percentage,
                    display_metrics_fn=display_confidence_metrics
                )
            
            # Add download button for the generated script
            if cucumber_script:
                st.download_button(
                    label="📥 Download Cucumber Script",
                    data=cucumber_script,
                    file_name="cucumber_test_suite.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                
        elif not qa_chain:
            st.error("❌ Please upload a BRD document first to initialize the AI model.")
        else:
            st.warning("⚠️ Please enter a test case to generate a Cucumber script.")

# Test Case to Selenium Script Tab
with tab4:
    st.subheader("Convert Test Case to Selenium Script")
    selenium_test_case_text = st.text_area("Enter the test case text here to generate Selenium script:")

    if st.button("Generate Selenium Script"):
        if selenium_test_case_text and qa_chain:
            selenium_prompt = (
                "You are a Senior Test Automation Engineer specializing in Selenium and Python."
                " Convert the following test case into a robust, production-ready Selenium WebDriver script in Python."
                "\n\nINSTRUCTIONS:\n"
                "- Use best practices for maintainability, reliability, and readability.\n"
                "- Include all necessary imports, setup, and teardown logic.\n"
                "- Use explicit waits (WebDriverWait) for element interactions, not time.sleep.\n"
                "- Add comments for each major step.\n"
                "- Validate all expected outcomes with assert statements.\n"
                "- Handle exceptions gracefully and log errors.\n"
                "- Use Page Object Model if the scenario is complex.\n"
                "- Ensure the script is ready to run as a standalone test.\n"
                "- Use realistic locators (id, name, xpath, css selector) based on the test case.\n"
                "- If data is required, use sample values from the test case.\n"
                "- If login or setup is needed, include those steps.\n"
                "\nTest Case:\n" + selenium_test_case_text + "\n\n"
                "Return ONLY the complete Python code, no explanations, no markdown."
            )
            start_selenium_time = time.time()
            response = qa_chain.invoke({"query": selenium_prompt})
            
            st.subheader("Generated Selenium Script")
            st.write(response['result'])
            
            # Calculate and display confidence metrics
            if show_confidence:
                st.subheader("Quality Assessment")
                
                confidence_score = calculate_confidence_level(selenium_prompt, response['result'])
                match_score = calculate_match_percentage(response['result'], selenium_test_case_text)
                
                display_confidence_metrics(confidence_score, match_score)
            
            st.write(f"Selenium script generation time: {time.time() - start_selenium_time:.2f} seconds")
        elif not qa_chain:
            st.write("Please upload a BRD document first to initialize the AI model.")
        else:
            st.write("Please enter a test case to generate a Selenium script.")

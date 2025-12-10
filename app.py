import os

import requests
import streamlit as st

st.set_page_config(
    page_title="SDLC Automate APP",
    page_icon="images/favicon.png"
)

# Get the API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000"

# Streamlit sidebar setup
with st.sidebar:
    st.title("Your BRD Documents")
    uploaded_file = st.file_uploader("Upload a file to generate user stories", type=["pdf", "docx", "txt", "xlsx", "pptx"])

# Streamlit app setup
st.header("BRD to User Story, Test Case, Cucumber Script, and Selenium Script")

# Set up tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["User Story Generation", "User Story to Test Case", "Test Case to Cucumber Script", "Test Case to Selenium Script"])


def render_user_story_cards(stories):
    """Render user stories as cards for readability."""
    if isinstance(stories, dict) and "user_stories" in stories:
        stories = stories.get("user_stories", [])

    if not isinstance(stories, list):
        st.write(stories)
        return

    for story in stories:
        with st.container(border=True):
            st.markdown(f"**{story.get('id', 'Story')} – {story.get('title', '')}**")
            st.write(story.get("story", ""))

            ac = story.get("acceptance_criteria", [])
            if ac:
                st.caption("Acceptance Criteria")
                for item in ac:
                    st.write(f"- {item}")

            meta_cols = st.columns(3)
            meta_cols[0].metric("Priority", story.get("priority", ""))
            meta_cols[1].metric("Points", story.get("story_points", ""))
            meta_cols[2].write(f"Category: {story.get('category', '')}")

            notes = story.get("notes", [])
            if notes:
                st.caption("Notes")
                for note in notes:
                    st.write(f"• {note}")

# Helper function to upload file to backend
def upload_file_to_backend(uploaded_file):
    if uploaded_file:
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        response = requests.post(f"{API_URL}/upload-document", files=files)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error uploading document: {response.text}")
    return None

# Helper function to get document list from backend
def get_documents():
    response = requests.get(f"{API_URL}/documents")
    if response.status_code == 200:
        return response.json().get("documents", [])
    return []

# Store uploaded document info in session to avoid re-uploading on every rerun
if "document_info" not in st.session_state:
    st.session_state["document_info"] = None

document_info = st.session_state.get("document_info")
if uploaded_file:
    # Only upload when the file changes; reuse existing document_id otherwise
    current_filename = uploaded_file.name
    cached_info = st.session_state.get("document_info")
    cached_filename = cached_info.get("filename") if cached_info else None
    if current_filename != cached_filename:
        document_info = upload_file_to_backend(uploaded_file)
        st.session_state["document_info"] = document_info
    else:
        document_info = cached_info

# User Story Generation Tab
with tab1:
    st.subheader("Generate User Stories from BRD")
    if "user_stories_result" not in st.session_state:
        st.session_state["user_stories_result"] = None
    if "user_stories_quality" not in st.session_state:
        st.session_state["user_stories_quality"] = None
    if "user_stories_time" not in st.session_state:
        st.session_state["user_stories_time"] = None
    if document_info:
        document_id = document_info["document_id"]
        if st.button("Generate User Stories"):
            response = requests.post(f"{API_URL}/generate-user-stories?document_id={document_id}")
            if response.status_code == 200:
                result = response.json()
                st.session_state["user_stories_result"] = result.get("user_stories")
                st.session_state["user_stories_quality"] = result.get("quality_assessment")
                st.session_state["user_stories_time"] = result.get("processing_time_seconds")
                st.session_state["user_stories_parse_error"] = result.get("parse_error")
            else:
                st.error(f"Error: {response.text}")

        if st.session_state["user_stories_result"]:
            st.subheader("Generated User Stories")
            if st.session_state.get("user_stories_parse_error"):
                st.warning("The model returned non-JSON output; showing raw text. Parse error: " + str(st.session_state["user_stories_parse_error"]))
                st.code(str(st.session_state["user_stories_result"]), language="json")
            else:
                render_user_story_cards(st.session_state["user_stories_result"])
            st.subheader("Quality Assessment")
            st.json(st.session_state["user_stories_quality"])
            st.write(f"Processing time: {st.session_state['user_stories_time']} seconds")
    else:
        st.info("Please upload a BRD document in the sidebar to generate user stories.")

# User Story to Test Case Tab
with tab2:
    st.subheader("Convert User Story to Test Case")
    user_story_text = st.text_area("Enter the user story text here to generate test cases:")
    if document_info and st.button("Generate Test Cases"):
        payload = {"user_story_text": user_story_text}
        document_id = document_info["document_id"]
        response = requests.post(f"{API_URL}/convert-to-test-cases?document_id={document_id}", json=payload)
        if response.status_code == 200:
            result = response.json()
            st.subheader("Generated Test Cases")
            if result.get("parse_error"):
                st.warning("The model returned non-JSON output; showing raw text. Parse error: " + str(result.get("parse_error")))
                st.code(str(result.get("test_cases")), language="json")
            else:
                st.json(result["test_cases"])
            st.subheader("Quality Assessment")
            st.json(result["quality_assessment"])
            st.write(f"Processing time: {result['processing_time_seconds']} seconds")
        else:
            st.error(f"Error: {response.text}")
    elif not document_info:
        st.info("Please upload a BRD document first to initialize the AI model.")

# Test Case to Cucumber Script Tab
with tab3:
    st.subheader("Convert Test Case to Cucumber Script")
    test_case_text = st.text_area("Enter the test case text here to generate Cucumber script:")
    if document_info and st.button("Generate Cucumber Script"):
        payload = {"test_case_text": test_case_text}
        document_id = document_info["document_id"]
        response = requests.post(f"{API_URL}/convert-to-cucumber?document_id={document_id}", json=payload)
        if response.status_code == 200:
            result = response.json()
            st.subheader("Generated Cucumber Script")
            st.code(result["cucumber_script"], language="gherkin")
            st.subheader("Quality Assessment")
            st.json(result["quality_assessment"])
            st.write(f"Processing time: {result['processing_time_seconds']} seconds")
        else:
            st.error(f"Error: {response.text}")
    elif not document_info:
        st.info("Please upload a BRD document first to initialize the AI model.")

# Test Case to Selenium Script Tab
with tab4:
    st.subheader("Convert Test Case to Selenium Script")
    selenium_test_case_text = st.text_area("Enter the test case text here to generate Selenium script:")
    if document_info and st.button("Generate Selenium Script"):
        payload = {"test_case_text": selenium_test_case_text}
        document_id = document_info["document_id"]
        response = requests.post(f"{API_URL}/convert-to-selenium?document_id={document_id}", json=payload)
        if response.status_code == 200:
            result = response.json()
            st.subheader("Generated Selenium Script")
            st.code(result["selenium_script"], language="python")
            st.subheader("Quality Assessment")
            st.json(result["quality_assessment"])
            st.write(f"Processing time: {result['processing_time_seconds']} seconds")
        else:
            st.error(f"Error: {response.text}")
    elif not document_info:
        st.info("Please upload a BRD document first to initialize the AI model.")
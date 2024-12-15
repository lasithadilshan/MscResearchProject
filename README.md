# BRD to User Story, Test Case, Cucumber Script, and Selenium Script Generator

## Overview

This application is a Streamlit-based tool designed to assist users in generating user stories, test cases, Cucumber scripts, and Selenium scripts from uploaded Business Requirement Document (BRD) files. It uses the LangChain framework and OpenAI's GPT-4 model to automate the extraction and generation of actionable development and testing artifacts.

---

## Features

### 1. **File Upload and Text Extraction**

- Supports file formats: `.pdf`, `.docx`, `.txt`, `.xlsx`, and `.pptx`.
- Extracts text from uploaded files using specialized parsers.

### 2. **User Story Generation**

- Converts the extracted BRD content into user stories using GPT-4.
- Employs a vector store for efficient retrieval and relevance-based processing.

### 3. **User Story to Test Case Conversion**

- Generates comprehensive test cases from user stories, covering functional and edge scenarios.

### 4. **Test Case to Cucumber Script Conversion**

- Converts test cases into Gherkin syntax scripts for automated testing.

### 5. **Test Case to Selenium Script Conversion**

- Generates Python Selenium scripts for automated web testing based on test cases.

---

## Installation

### Prerequisites

- Python 3.9 or higher
- Virtual environment (optional but recommended)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/lasithadilshan/MscResearchProject.git
   cd MscResearchProject
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` or `app_secret.py` file with your OpenAI API key:
   ```python
   OPENAI_API_KEY = "your_openai_api_key"
   ```
4. Run the application:
   ```bash
   streamlit run <application_file_name>.py
   ```

---

## Usage

### Uploading a File

1. Upload a file via the sidebar.
2. Supported formats: `.pdf`, `.docx`, `.txt`, `.xlsx`, `.pptx`.

### Generating User Stories

1. Navigate to the **User Story Generation** tab.
2. View the generated user stories based on the uploaded BRD.

### Converting User Stories to Test Cases

1. Navigate to the **User Story to Test Case** tab.
2. Enter the user story text and click **Generate Test Cases**.

### Converting Test Cases to Cucumber Scripts

1. Navigate to the **Test Case to Cucumber Script** tab.
2. Enter the test case text and click **Generate Cucumber Script**.

### Converting Test Cases to Selenium Scripts

1. Navigate to the **Test Case to Selenium Script** tab.
2. Enter the test case text and click **Generate Selenium Script**.

---

## Dependencies

| Package             | Version |
| ------------------- | ------- |
| streamlit           | 1.38.0  |
| PyPDF2              | 3.0.1   |
| python-docx         | 0.8.11  |
| python-pptx         | 0.6.21  |
| pandas              | 2.1.1   |
| openai              | latest  |
| langchain           | latest  |
| langchain-openai    | latest  |
| faiss-cpu           | 1.8.0   |
| langchain-community | 0.3.0   |
| langchain-core      | 0.3.5   |
| scikit-learn        | latest  |
| openpyxl            | latest  |

---

## Project Structure

```
.
├── app_secret.py          # Contains OpenAI API Key
├── requirements.txt       # Dependencies
├── main.py                # Application entry point
└── README.md              # Project documentation
```

---

## Caching

To optimize performance:

- **Resource-heavy operations** such as file parsing, vector store creation, and text extraction are cached using Streamlit's `@st.cache_resource`.

---

## Notes

- Ensure the OpenAI API key is valid and has sufficient quota.
- Execution times for large files may vary depending on file size and complexity.
- Adjust the `chunk_size` and `chunk_overlap` parameters in the text splitter for fine-tuning.

---

## Troubleshooting

### Common Issues

1. **Missing or invalid OpenAI API Key**:
   - Ensure the API key is set correctly in `app_secret.py` or `.venv` file.
2. **Dependency errors**:
   - Run `pip install -r requirements.txt` again to ensure all dependencies are installed.
3. **File parsing errors**:
   - Verify the file format is supported and the file is not corrupted.

---

## Future Enhancements

- Add support for additional file formats.
- Improve text extraction for more complex document structures.
- Enable multi-language support for user story generation.
- Integrate other LLM models for expanded capabilities.

---

## License

This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).

---

## Contributors

- Lasitha Thilakarathna - Developer

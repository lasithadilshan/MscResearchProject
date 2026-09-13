import pytest
from app.services.document_service import (
    extract_text_from_file,
    create_vector_store,
    query_vector_store
)

def test_extract_text_utf8():
    sample_text = "This is a Business Requirements Document for User Authentication."
    extracted = extract_text_from_file(sample_text.encode("utf-8"), "test.txt")
    assert extracted == sample_text

def test_extract_text_latin1_fallback():
    # Character 'é' in latin-1 is byte b'\xe9' which is invalid in UTF-8
    latin1_bytes = b"Requirement: Support caf\xe9 and resume"
    extracted = extract_text_from_file(latin1_bytes, "requirements.txt")
    assert "caf" in extracted
    assert "resume" in extracted

def test_extract_unsupported_format():
    extracted = extract_text_from_file(b"some raw content", "archive.zip")
    assert extracted == ""

def test_vector_store_create_and_query():
    doc_id = "test-doc-qa-12345"
    sample_brd = (
        "Project: E-Commerce Checkout\n"
        "Requirement 1: The user should be able to pay with Credit Card or PayPal.\n"
        "Requirement 2: If payment succeeds, display an order confirmation page with order number.\n"
        "Requirement 3: Send an email receipt to the customer upon payment completion.\n"
    )
    
    count = create_vector_store(doc_id, sample_brd)
    assert count > 0

    results = query_vector_store(doc_id, "How can the user pay?")
    assert "Credit Card" in results or "PayPal" in results

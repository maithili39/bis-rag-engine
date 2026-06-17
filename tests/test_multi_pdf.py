import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from unittest.mock import MagicMock, patch
from data_processor import BISDataProcessor
from rag_pipeline import BISRAGPipeline


def test_fallback_parser_without_summary_of():
    processor = BISDataProcessor()
    
    # Text resembling a standard without "SUMMARY OF" sections
    non_summary_text = (
        "IS 269 : 1989\n"
        "ORDINARY PORTLAND CEMENT\n"
        "This standard covers the requirements of OPC.\n"
        "IS 383 : 1970\n"
        "COARSE AND FINE AGGREGATES\n"
        "This standard covers aggregates details."
    )
    
    # Run parsing
    standards = processor.extract_standards_from_text(non_summary_text)
    
    # Fallback should kick in because re.split(SUMMARY OF) will return 1 part.
    # It should split by the two IS code headers.
    assert len(standards) >= 2
    
    # Verify first parsed standard
    assert standards[0]["code"] == "IS 269: 1989"
    assert "opc" in standards[0]["content"].lower()
    
    # Verify second parsed standard
    assert standards[1]["code"] == "IS 383: 1970"
    assert "aggregates" in standards[1]["content"].lower()


def test_dynamic_source_metadata():
    processor = BISDataProcessor()
    standards = [
        {"code": "IS 269: 1989", "title": "Cement", "content": "OPC requirements.", "full_text": "..."}
    ]
    
    # Verify custom source name propagates to Document metadata
    docs = processor.create_documents_from_standards(standards, pdf_source_name="custom_handbook.pdf")
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "custom_handbook.pdf"


@patch("data_processor.PdfReader")
def test_initialize_from_pdfs_merges_content(mock_pdf_reader, tmp_path):
    # Mock extract_text for two different PDFs
    mock_pdf1 = MagicMock()
    mock_pdf1.pages = [MagicMock()]
    # Return a long string (> 1000 chars) to pass the extraction length check
    mock_pdf1.pages[0].extract_text.return_value = "IS 269 : 1989\nORDINARY PORTLAND CEMENT\nOPC cement specifications.\n" + ("x" * 1000)
    
    mock_pdf2 = MagicMock()
    mock_pdf2.pages = [MagicMock()]
    mock_pdf2.pages[0].extract_text.return_value = "IS 383 : 1970\nAGGREGATES\nAggregates specifications.\n" + ("y" * 1000)
    
    def side_effect(path):
        if "pdf1.pdf" in path:
            return mock_pdf1
        return mock_pdf2
        
    mock_pdf_reader.side_effect = side_effect
    
    # Build pipeline in temporary directory to avoid touching real vectorstore
    pipeline = BISRAGPipeline(persist_dir=str(tmp_path / "chromadb"))
    
    with patch.object(pipeline.retriever, "build_vectorstore") as mock_build:
        pipeline.initialize_from_pdfs(["/path/to/pdf1.pdf", "/path/to/pdf2.pdf"])
        
        # Verify codes from both PDFs were merged into known_codes
        assert "IS 269: 1989" in pipeline.known_codes
        assert "IS 383: 1970" in pipeline.known_codes
        
        # Verify build_vectorstore was called with both parsed document sets
        mock_build.assert_called_once()
        docs = mock_build.call_args[0][0]
        
        # Find document sources to verify metadata is file-specific
        sources = {doc.metadata.get("source") for doc in docs if "source" in doc.metadata}
        assert "pdf1.pdf" in sources
        assert "pdf2.pdf" in sources

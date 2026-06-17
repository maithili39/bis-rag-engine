"""Data processing and chunking for BIS SP 21 standards documents."""

import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import CHUNK_SIZE, CHUNK_OVERLAP

warnings.filterwarnings("ignore")


class BISDataProcessor:
    """Process and chunk BIS SP 21 standards PDF document."""

    # Regex to match standard headers like "IS 269 : 1989" or "IS 2185 (Part 2) : 1983"
    STANDARD_PATTERN = re.compile(
        r'IS\s+(\d+)\s*(?:\(([^)]+)\))?\s*:\s*(\d{4})',
        re.IGNORECASE
    )

    # Pattern to match "SUMMARY OF" section headers
    SUMMARY_HEADER = re.compile(
        r'SUMMARY\s+OF\s*\n\s*(IS\s+\d+[^A-Z\n]*?)(?:\s{2,}|\n)([A-Z][A-Z\s,/()—\-]+)',
        re.IGNORECASE | re.MULTILINE
    )

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.all_standard_codes = set()  # Track all known codes for hallucination filtering

    def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract full text from the BIS SP 21 PDF."""
        print(f"  Reading PDF: {pdf_path}")
        try:
            reader = PdfReader(pdf_path)
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF at {pdf_path}: {e}") from e

        total_pages = len(reader.pages)
        print(f"  Total pages: {total_pages}")
        if total_pages == 0:
            raise RuntimeError(f"PDF at {pdf_path} has no pages — file may be corrupt or encrypted.")

        full_text = ""
        failed_pages = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text:
                    full_text += text + "\n\n"
            except Exception as e:
                failed_pages.append(i + 1)
                print(f"  Warning: Could not extract page {i+1}: {e}")

        if failed_pages:
            print(f"  Warning: {len(failed_pages)}/{total_pages} pages failed extraction: {failed_pages[:10]}")

        if len(full_text) < 1000:
            raise RuntimeError(
                f"PDF text extraction yielded only {len(full_text)} characters — "
                "the PDF may be scanned/image-only. Use an OCR tool (e.g. pdfminer with OCR) first."
            )

        print(f"  Extracted {len(full_text)} characters of text")
        return full_text

    def extract_standards_from_text(self, full_text: str) -> List[Dict[str, Any]]:
        """
        Parse the SP 21 PDF text into individual standard entries.
        Each entry has: code, title, content.
        """
        standards = []

        # Strategy: Split on "SUMMARY OF" markers to isolate each standard
        # The PDF has sections like:
        #   SUMMARY OF
        #   IS 269 : 1989  ORDINARY PORTLAND CEMENT — 33 GRADE
        #   (Fourth Revision)
        #   1. Scope — ...
        #   2. Requirements — ...

        # Split text at "SUMMARY OF" boundaries
        parts = re.split(r'(?=SUMMARY\s+OF\s*\n)', full_text, flags=re.IGNORECASE)

        # Fallback split strategy for other PDFs (Phase 6)
        if len(parts) <= 2:
            print("  Fallback: Splitting by IS standard code boundaries")
            parts = re.split(r'(?=IS\s+\d+\s*(?:\([^)]+\))?\s*:\s*\d{4})', full_text, flags=re.IGNORECASE)

        print(f"  Found {len(parts)} potential standard sections")

        for part in parts:
            if len(part.strip()) < 50:
                continue

            # Try to extract standard code and title from this section
            std_info = self._parse_standard_section(part)
            if std_info:
                standards.append(std_info)
                self.all_standard_codes.add(std_info["code"])

        if len(standards) < 10:
            print(
                f"  WARNING: Only {len(standards)} standards parsed from 'SUMMARY OF' markers. "
                "The PDF layout may differ from BIS SP 21 : 2005. "
                "Page-level fallback chunks will be used for coverage."
            )

        # Also do a global scan for any IS codes we might have missed
        all_codes = set(self.STANDARD_PATTERN.findall(full_text))
        for num, part_info, year in all_codes:
            code = self._format_code(num, part_info, year)
            # Normalize the code format
            code = self._normalize_code(code)
            self.all_standard_codes.add(code)

        print(f"  Successfully parsed {len(standards)} standard entries")
        print(f"  Total unique standard codes found: {len(self.all_standard_codes)}")

        return standards

    def _parse_standard_section(self, section_text: str) -> Optional[Dict[str, Any]]:
        """Parse a single standard section into structured data."""

        lines = section_text.strip().split('\n')
        if len(lines) < 3:
            return None

        # Look for the IS code in the first few lines
        code = None
        title = None
        code_line_idx = -1

        for i, line in enumerate(lines[:10]):
            # Try to find IS code
            match = self.STANDARD_PATTERN.search(line)
            if match:
                num, part_info, year = match.groups()
                code = self._format_code(num, part_info, year)

                # Title often follows the code on the same line or next line
                # Extract everything after the code pattern on this line
                after_code = line[match.end():].strip()
                if after_code and len(after_code) > 3:
                    title = after_code
                elif i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    # Skip revision info lines
                    if next_line and not next_line.startswith('(') and len(next_line) > 3:
                        title = next_line

                code_line_idx = i
                break

        if not code:
            return None

        # Clean up title
        if title:
            # Remove revision annotations
            title = re.sub(r'\([^)]*[Rr]evision[^)]*\)', '', title).strip()
            title = re.sub(r'\s+', ' ', title).strip()
            title = title.rstrip(' —-')
            # Capitalize properly
            if title.isupper():
                title = title.title()
        else:
            title = "Unknown Title"

        # Content is everything after the code/title lines
        content_start = max(code_line_idx + 1, 0)
        content = '\n'.join(lines[content_start:]).strip()

        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        content = re.sub(r'  +', ' ', content)

        if len(content) < 20:
            return None

        return {
            "code": code,
            "title": title,
            "content": content,
            "full_text": section_text.strip()
        }

    def _format_code(self, num: str, part_info: str, year: str) -> str:
        """Build a canonical IS code from regex groups.

        The 'part_info' capture group sometimes already contains the word
        'Part' (e.g. 'Part 2'), so we must not prepend it again — doing so
        produced malformed codes like 'IS 2185 (Part Part 2): 1983'. A
        plain `startswith` check (rather than a `\\bpart\\b` regex) is used
        because `\\b` does not match between "Part" and a directly-adjacent
        digit/letter (e.g. "PART11", "Part1"), which let those cases slip
        through and get double-prefixed.
        """
        if part_info and part_info.strip():
            pi = part_info.strip()
            if not pi.lower().startswith('part'):
                pi = f"Part {pi}"
            code = f"IS {num} ({pi}): {year}"
        else:
            code = f"IS {num}: {year}"
        return self._normalize_code(code)

    def _normalize_code(self, code: str) -> str:
        """Normalize a standard code to consistent format like 'IS 269: 1989' or 'IS 2185 (Part 2): 1983'."""
        # Remove extra spaces
        code = re.sub(r'\s+', ' ', code).strip()
        # Normalize colon format
        code = re.sub(r'\s*:\s*', ': ', code)
        # Normalize "Part" capitalization
        code = re.sub(r'part\s+', 'Part ', code, flags=re.IGNORECASE)
        return code

    def create_documents_from_standards(self, standards: List[Dict[str, Any]], pdf_source_name: str = "BIS SP 21 : 2005") -> List[Document]:
        """Create LangChain Document objects from parsed standards."""
        documents = []

        for std in standards:
            # Create a rich text representation for embedding
            doc_text = f"""BIS Standard: {std['code']}
Title: {std['title']}

{std['content']}
"""
            doc = Document(
                page_content=doc_text,
                metadata={
                    "standard_code": std["code"],
                    "title": std["title"],
                    "source": pdf_source_name
                }
            )
            documents.append(doc)

        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks, preserving metadata."""
        chunked = []

        for doc in documents:
            if len(doc.page_content) <= self.chunk_size:
                # Small enough to keep as single chunk
                chunked.append(doc)
            else:
                # Split into chunks but preserve metadata
                chunks = self.text_splitter.split_documents([doc])
                for chunk in chunks:
                    # Prepend the standard code to each chunk for better retrieval
                    chunk.page_content = f"[{doc.metadata['standard_code']}] {doc.metadata['title']}\n{chunk.page_content}"
                    chunk.metadata = doc.metadata.copy()
                # Fix #2: only keep chunks, not the original doc (avoids duplicate bias)
                chunked.extend(chunks)

        return chunked

    def process_pdf(self, pdf_path: str, pdf_source_name: Optional[str] = None) -> Tuple[List[Document], set]:
        """
        Full pipeline: PDF -> parsed standards -> chunked documents.
        Returns (documents, all_known_codes).
        """
        if pdf_source_name is None:
            pdf_source_name = os.path.basename(pdf_path)

        print(f"Step 1: Extracting text from PDF {pdf_path}...")
        full_text = self.extract_pdf_text(pdf_path)

        print("Step 2: Parsing individual standards...")
        standards = self.extract_standards_from_text(full_text)

        print("Step 3: Creating document objects...")
        documents = self.create_documents_from_standards(standards, pdf_source_name)

        print("Step 4: Chunking documents...")
        chunked_docs = self.chunk_documents(documents)

        print(f"  Final: {len(chunked_docs)} document chunks from {len(standards)} standards")

        # Also create page-level chunks as fallback for standards we couldn't parse
        print("Step 5: Creating page-level fallback chunks...")
        page_docs = self._create_page_level_chunks(full_text, pdf_source_name)

        all_docs = chunked_docs + page_docs
        print(f"  Total documents (standards + page chunks): {len(all_docs)}")

        return all_docs, self.all_standard_codes

    def _create_page_level_chunks(self, full_text: str, pdf_source_name: str = "BIS SP 21 : 2005") -> List[Document]:
        """Create page-level chunks as fallback for better coverage."""
        # Split text into ~1000 char segments with overlap
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " "]
        )

        doc = Document(
            page_content=full_text,
            metadata={"source": pdf_source_name, "type": "page_chunk"}
        )

        chunks = splitter.split_documents([doc])

        # Enrich metadata by extracting any IS codes in each chunk
        enriched = []
        for chunk in chunks:
            codes = self.STANDARD_PATTERN.findall(chunk.page_content)
            if codes:
                # Add the first found code as metadata
                num, part_info, year = codes[0]
                code = self._format_code(num, part_info, year)
                chunk.metadata["standard_code"] = code
                self.all_standard_codes.add(code)
            enriched.append(chunk)

        return enriched

    def compute_parsing_coverage(self, known_toc: List[str]) -> float:
        """
        Compute coverage of parsed standard codes against a known Table of Contents (TOC).
        Returns a float between 0.0 and 1.0 representing the coverage.
        """
        if not known_toc:
            return 0.0
            
        parsed = set(self._normalize_code(code) for code in self.all_standard_codes)
        expected = set(self._normalize_code(code) for code in known_toc)
        
        found = parsed.intersection(expected)
        missing = expected - parsed
        
        coverage = len(found) / len(expected)
        print(f"  Parsing Coverage: {coverage*100:.1f}% ({len(found)}/{len(expected)})")
        if missing:
            print(f"  Missing expected codes: {missing}")
            
        return coverage

    def get_all_known_codes(self) -> set:
        """Return all known BIS standard codes found during processing."""
        return self.all_standard_codes

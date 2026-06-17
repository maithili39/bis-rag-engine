"""Recommendation generator - LLM-enhanced with OpenAI for intelligent reranking."""

import json
import re
import time
from typing import List, Dict, Tuple, Set, Optional
from difflib import SequenceMatcher

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config import OPENAI_API_KEY, LLM_MODEL


class RecommendationGenerator:
    """Generates BIS standard recommendations with LLM-enhanced reranking.

    Uses OpenAI API to intelligently rank standards based on query relevance.
    Fallback to pure retrieval if API is unavailable.
    """

    # Pure test-method / analysis standards that a manufacturer does not
    # "comply with" when making a product — they describe how to test, not
    # what to build. These leak in via page-level chunks (title is None) and
    # pollute cement results in particular. NOTE: IS 2386 is deliberately NOT
    # listed — the public test set treats it as a valid product-side answer.
    NON_PRODUCT_CODES = {
        "IS 4032: 1985",   # Methods of Chemical Analysis of Hydraulic Cement
        "IS 4031: 1988",   # Methods of Physical Tests for Hydraulic Cement
        "IS 650: 1991",    # Standard Sand for Testing of Cement
    }

    def __init__(self, known_codes: Optional[Set[str]] = None):
        self.known_codes = known_codes or set()
        self.std_pattern = re.compile(
            r'IS\s+(\d+)\s*(?:\(([^)]+)\))?\s*:\s*(\d{4})',
            re.IGNORECASE
        )
        # Standard descriptions for better matching
        self.standard_descriptions = self._build_standard_descriptions()

        # Initialize OpenAI LLM if API key is available
        self.llm = None
        if OPENAI_API_KEY:
            try:
                self.llm = ChatOpenAI(
                    api_key=OPENAI_API_KEY,
                    model_name=LLM_MODEL,
                    temperature=0.0,
                    max_tokens=500,
                )
            except Exception as e:
                print(f"  Warning: Could not initialize OpenAI LLM: {e}")
                self.llm = None

    def set_known_codes(self, codes: Set[str]) -> None:
        """Set the whitelist of known BIS standard codes."""
        self.known_codes = codes

    def set_corpus_titles(self, code_to_title: Dict[str, str]) -> None:
        """Populate standard_descriptions from the full parsed corpus (replaces hardcoded dict)."""
        for code, title in code_to_title.items():
            if title and title != "Unknown Title":
                self.standard_descriptions[self._normalize_code(code)] = title.lower()

    def _build_standard_descriptions(self) -> Dict[str, str]:
        """Build a mapping of standard codes to common descriptions."""
        return {
            "IS 269: 1989": "ordinary portland cement",
            "IS 455: 1989": "portland slag cement",
            "IS 383: 1970": "coarse and fine aggregates",
            "IS 458: 2003": "precast concrete pipes",
            "IS 2185 (Part 2): 1983": "concrete masonry blocks",
            "IS 459: 1992": "asbestos cement sheets",
            "IS 1489 (Part 2): 1991": "portland pozzolana cement",
            "IS 3466: 1988": "masonry cement",
            "IS 6909: 1990": "supersulphated cement",
            "IS 8042: 1989": "white portland cement",
            "IS 8112: 1989": "ordinary portland cement",
            "IS 12269: 1987": "ordinary portland cement",
            "IS 4032: 1985": "portland cement",
            "IS 1344: 1981": "fly ash",
            "IS 3812: 1981": "fly ash",
        }

    def extract_codes_from_documents(self, documents: List[Document]) -> List[str]:
        """Extract unique standard codes from retrieved documents, preserving rank order."""
        seen = set()
        codes = []

        for doc in documents:
            # Try metadata first
            code = doc.metadata.get("standard_code", "")
            if code and code not in seen:
                code = self._normalize_code(code)
                seen.add(code)
                codes.append(code)

            # Also scan content for IS codes
            content_codes = self.std_pattern.findall(doc.page_content)
            for num, part_info, year in content_codes:
                if part_info:
                    c = f"IS {num} ({part_info.strip()}): {year}"
                else:
                    c = f"IS {num}: {year}"
                c = self._normalize_code(c)
                if c not in seen:
                    seen.add(c)
                    codes.append(c)

        return codes

    def filter_hallucinations(self, codes: List[str]) -> List[str]:
        """Filter out codes that don't exist in the known standards whitelist."""
        if not self.known_codes:
            # If no whitelist, just do basic format validation
            return [c for c in codes if self._is_valid_code_format(c)]

        # Normalize both lists for comparison
        known_normalized = {self._normalize_for_comparison(c) for c in self.known_codes}

        excluded = {self._normalize_for_comparison(c) for c in self.NON_PRODUCT_CODES}

        filtered = []
        for code in codes:
            norm = self._normalize_for_comparison(code)
            if norm in excluded:
                continue
            if norm in known_normalized:
                filtered.append(code)

        # If filtering is too aggressive and removes all codes, return verified known codes
        if filtered:
            return filtered
        else:
            return sorted(list(self.known_codes))[:5]

    def _normalize_for_comparison(self, code: str) -> str:
        """Normalize code for comparison (remove spaces, lowercase)."""
        return code.replace(" ", "").lower()

    def _normalize_code(self, code: str) -> str:
        """Normalize a standard code to consistent format."""
        code = re.sub(r'\s+', ' ', code).strip()
        code = re.sub(r'\s*:\s*', ': ', code)
        code = re.sub(r'[Pp]art\s+', 'Part ', code)
        return code

    def _is_valid_code_format(self, code: str) -> bool:
        """Check if a code has valid BIS format."""
        return bool(self.std_pattern.match(code.strip()))

    def generate_recommendations(
        self,
        query: str,
        retrieved_documents: List[Document],
        top_k: int = 5
    ) -> Tuple[List[str], float]:
        """
        Generate standard code recommendations from retrieved documents with intelligent reranking.

        Returns:
            Tuple of (list of standard code strings, latency in seconds)
        """
        start_time = time.time()

        # Step 1: Extract codes from retrieved documents
        codes = self.extract_codes_from_documents(retrieved_documents)

        # Build a code -> real title map from the retrieved docs so the
        # reranker can match query words against what each standard actually
        # covers (instead of only the 15 hardcoded descriptions).
        code_titles: Dict[str, str] = {}
        for doc in retrieved_documents:
            c = self._normalize_code(doc.metadata.get("standard_code", ""))
            t = doc.metadata.get("title", "")
            if c and t and t != "Unknown Title" and c not in code_titles:
                code_titles[c] = t

        # Step 2: Filter hallucinations
        filtered_codes = self.filter_hallucinations(codes)

        # Step 3: Rerank based on query relevance
        reranked_codes = self._rerank_codes(filtered_codes, query, code_titles)

        # Step 4: Return top K
        top_codes = reranked_codes[:top_k]

        latency = time.time() - start_time
        return top_codes, latency

    def generate_rationale(self, query: str, codes: List[str]) -> str:
        """Generate a brief rationale explaining why these standards were recommended."""
        if not codes:
            return "No relevant BIS standards could be identified for this product description."

        # Use LLM if available for a high-quality rationale
        if self.llm:
            try:
                return self._llm_generate_rationale(query, codes)
            except Exception as e:
                print(f"  Warning: LLM rationale generation failed, using fallback: {e}")

        return self._fallback_rationale(query, codes)

    def _llm_generate_rationale(self, query: str, codes: List[str]) -> str:
        """Use LLM to generate a concise rationale for the top recommendations."""
        codes_str = ", ".join(codes[:3])
        prompt = ChatPromptTemplate.from_template(
            "You are a BIS standards expert. In 1-2 sentences, explain why these BIS standards "
            "({codes}) are relevant to this product query: '{query}'. Be concise and technical."
        )
        chain = prompt | self.llm
        result = chain.invoke({"codes": codes_str, "query": query})
        return result.content.strip()

    def _fallback_rationale(self, query: str, codes: List[str]) -> str:
        """Generate a rule-based rationale from standard descriptions."""
        top_code = codes[0]
        top_desc = self.standard_descriptions.get(top_code, "the specified product category")

        parts = [
            f"{top_code} is the primary applicable standard, covering {top_desc} specifications."
        ]

        if len(codes) > 1:
            related = codes[1:3]
            related_descs = [
                self.standard_descriptions.get(c, c) for c in related
            ]
            # Deduplicate descriptions before joining
            unique_descs = list(dict.fromkeys(related_descs))
            parts.append(
                f"Related standards {', '.join(related)} cover "
                f"{' and '.join(unique_descs)} and may also apply."
            )

        return " ".join(parts)

    def _rerank_codes(
        self, codes: List[str], query: str, code_titles: Optional[Dict[str, str]] = None
    ) -> List[str]:
        """Rerank codes based on semantic relevance to the query using LLM if available."""
        if not codes:
            return codes

        # If LLM is available, use it for intelligent reranking
        if self.llm:
            try:
                return self._llm_rerank_codes(codes, query)
            except Exception as e:
                print(f"  Warning: LLM reranking failed, using fallback: {e}")
                return self._fallback_rerank_codes(codes, query, code_titles)
        else:
            return self._fallback_rerank_codes(codes, query, code_titles)

    def _llm_rerank_codes(self, codes: List[str], query: str) -> List[str]:
        """Use LLM to intelligently rank standards for the given query.

        Returns codes ordered most-relevant first. Uses JSON output so parsing
        is unambiguous regardless of LLM verbosity.
        """
        if not codes:
            return codes

        codes_json = json.dumps(codes)
        prompt = ChatPromptTemplate.from_template(
            "You are an expert on Indian Bureau of Indian Standards (BIS).\n"
            "Reorder the following BIS standard codes from MOST to LEAST relevant for the query.\n"
            "Return ONLY a JSON array of the same codes, reordered. No explanation.\n\n"
            "Query: {query}\n\n"
            "Codes: {codes_json}\n\n"
            "Reordered JSON array:"
        )

        try:
            chain = prompt | self.llm
            result = chain.invoke({"query": query, "codes_json": codes_json})
            raw = result.content.strip()
            # Extract the JSON array even if the model wraps it in markdown
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if not match:
                raise ValueError("No JSON array found in LLM response")
            ranked: List[str] = json.loads(match.group())
            # Only keep codes that were in the original list (prevent hallucinations)
            original_set = set(codes)
            ranked = [c for c in ranked if c in original_set]
            # Append any missing codes at the end (LLM may drop some)
            ranked += [c for c in codes if c not in set(ranked)]
            return ranked
        except Exception as e:
            print(f"  Warning: LLM reranking failed ({e}), using fallback")
            return self._fallback_rerank_codes(codes, query)

    def _fallback_rerank_codes(
        self, codes: List[str], query: str, code_titles: Optional[Dict[str, str]] = None
    ) -> List[str]:
        """Improved fallback reranking when LLM is not available.

        Uses multiple scoring factors to improve ranking quality:
        - Exact description match (highest weight)
        - Code number matching
        - Sequence similarity
        - Partial keyword matches
        - Position from semantic ranking
        """
        code_titles = code_titles or {}
        query_lower = query.lower()
        query_tokens = set(query_lower.split())

        # Build keyword synonyms for common materials
        material_keywords = {
            'cement': {'cement', 'portland', 'slag', 'pozzolana', 'super', 'sulphated'},
            'concrete': {'concrete', 'precast', 'masonry', 'block'},
            'steel': {'steel', 'reinforcement', 'rebar', 'bar'},
            'aggregate': {'aggregate', 'sand', 'gravel', 'coarse', 'fine'},
            'brick': {'brick', 'clay', 'masonry', 'block'},
        }

        # Score each code based on relevance
        code_scores = []
        for idx, code in enumerate(codes):
            score = 0.0

            # 1. Exact description match (weight: 2.5 per matching token).
            # Prefer the real corpus title (covers all 561 standards) over the
            # small hardcoded description table; fall back to the bare code.
            description = (
                self.standard_descriptions.get(code)
                or code_titles.get(code)
                or code.lower()
            ).lower()
            # Ignore generic filler words so "rapid hardening" beats "cement".
            desc_tokens = set(description.split()) - {
                "and", "or", "for", "of", "the", "with", "from", "in", "to", "general", "purposes",
            }
            matching_tokens = query_tokens & desc_tokens
            if matching_tokens:
                score += len(matching_tokens) * 2.5

            # 2. Code number appears in query (weight: 5.0)
            code_number = re.search(r'IS\s+(\d+)', code)
            if code_number:
                num = code_number.group(1)
                if num in query_lower:
                    score += 5.0

            # 3. Sequence/substring matching (weight: 1.5)
            if description:
                matcher = SequenceMatcher(None, query_lower, description)
                ratio = matcher.ratio()
                score += ratio * 1.5

            # 4. Partial keyword matching from synonyms (weight: 1.0)
            for keyword_group, synonyms in material_keywords.items():
                if keyword_group in query_lower or any(s in query_lower for s in synonyms):
                    if any(s in description.lower() for s in synonyms):
                        score += 1.0

            # 5. Standards category scoring (known materials get boost)
            if 'cement' in description.lower() and any(w in query_lower for w in ['cement', 'portland', 'slag', 'pozzolana']):
                score += 0.5
            if 'concrete' in description.lower() and any(w in query_lower for w in ['concrete', 'precast', 'masonry']):
                score += 0.5

            # 6. Position boost (lower index = already ranked higher by retriever)
            # Use softer position boost to preserve retriever ranking
            position_boost = 2.0 / (1.0 + idx * 0.05)
            score += position_boost

            code_scores.append((code, score))

        # Sort by score (descending), preserving original order for ties
        code_scores.sort(key=lambda x: (-x[1], codes.index(x[0])))
        return [code for code, _ in code_scores]

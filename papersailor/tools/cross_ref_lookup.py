"""
CrossRef Lookup Tool for smolagents
Looks up paper citations and retrieves metadata from arXiv API
"""

import json
import re
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, Any, Optional, List
from xml.etree import ElementTree as ET
from json_repair import repair_json

from smolagents import Tool
from loguru import logger

from .llm_chat import chat_with_llm


class CrossRefLookup(Tool):
    """
    Tool for looking up paper citations and retrieving metadata.

    This tool takes a citation key (e.g., "Thudi et al., 2024"), extracts the
    references section from a markdown document, uses LLM to match the citation
    to a specific paper title, and then searches arXiv API for metadata.
    """

    name = "cross_ref_lookup"
    description = (
        "Look up paper citations and retrieve metadata from arXiv. Takes a PDF name "
        "and citation key as input, extracts references from the document, matches "
        "the citation using LLM, and searches arXiv API for detailed information."
    )
    inputs = {
        "pdf_name": {
            "type": "string",
            "description": "The name of the PDF file (without .pdf extension)",
        },
        "citation_key": {
            "type": "string",
            "description": "Citation key to look up (e.g., 'Thudi et al., 2024')",
        },
    }
    output_type = "string"

    def __init__(self, output_base_dir: Path | str = None):
        """Initialize the CrossRef lookup tool"""
        super().__init__()
        # Fixed paths for PDF processing
        if output_base_dir is None:
            self.output_base_dir = Path(
                "/data/wdy/AgenticPaperQA/paper-prepare/sampled_jsons"
            )
        else:
            self.output_base_dir = Path(output_base_dir)

    def setup(self):
        """Setup method called before first use"""
        logger.info("🔧 Setting up CrossRef lookup tool...")
        self.is_initialized = True

    def _load_markdown_content(self, pdf_name: str) -> tuple[str, Optional[str]]:
        """
        Load the markdown content from the parsed PDF

        Args:
            pdf_name: Name of the PDF file (without extension)

        Returns:
            Tuple of (markdown content, error message or None)
        """
        md_path = self.output_base_dir / pdf_name / "auto" / f"{pdf_name}.md"

        if not md_path.exists():
            return None, f"Markdown file not found: {md_path}"

        try:
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()

            logger.info(f"📄 Loaded markdown content ({len(content)} characters)")
            return content, None

        except Exception as e:
            error_msg = f"Failed to read markdown file: {e}"
            logger.error(f"❌ {error_msg}")
            return None, error_msg

    def _extract_references_section(self, content: str) -> Optional[str]:
        """
        Extract the references section from markdown content

        Args:
            content: Full markdown content

        Returns:
            References section content or None if not found
        """
        # Find all occurrences of "References" or "# References"
        patterns = [
            r"(?i)^#+\s*References\s*$",  # # References (with any number of #)
            r"(?i)^References\s*$",  # References (standalone line)
        ]

        references_matches = []
        for pattern in patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                references_matches.append(match.start())

        if not references_matches:
            logger.warning("⚠️ No References section found")
            return None

        # Take the last occurrence (as requested)
        last_ref_start = max(references_matches)

        # Find the start of the references content (after the header)
        lines = content[last_ref_start:].split("\n")
        ref_start_line = 0
        for i, line in enumerate(lines):
            if re.match(r"(?i)^#+\s*References\s*$|^References\s*$", line.strip()):
                ref_start_line = i + 1
                break

        ref_content_start = last_ref_start + len("\n".join(lines[:ref_start_line]))
        if ref_start_line > 0:
            ref_content_start += 1  # Add the newline character

        # Find the end of references section (next # heading or end of document)
        remaining_content = content[ref_content_start:]
        next_heading_match = re.search(r"^#+\s+", remaining_content, re.MULTILINE)

        if next_heading_match:
            ref_end = ref_content_start + next_heading_match.start()
            references_content = content[ref_content_start:ref_end].strip()
        else:
            references_content = remaining_content.strip()

        logger.info(
            f"📚 Extracted references section ({len(references_content)} characters)"
        )
        return references_content

    def _match_citation_to_title(
        self, references: str, citation_key: str
    ) -> Optional[str]:
        """
        Use LLM to match citation key to specific paper title

        Args:
            references: References section content
            citation_key: Citation key to match

        Returns:
            Paper title or None if not found
        """
        prompt = f"""You are given a references section from an academic paper and a citation key. Your task is to find the specific reference that matches the citation key and extract its title.

References section:
{references}

Citation key to match: "{citation_key}"

Please analyze the references and find the one that matches the given citation key. Look for matches based on:
1. Author names (including variations like "et al.")
2. Publication year
3. Similar author patterns

Output your response as a JSON object with the following format:
{{
    "found": true/false,
    "title": "exact paper title if found, null if not found",
    "reasoning": "brief explanation of your matching process"
}}

Be precise with the title - extract it exactly as it appears in the reference, without any formatting or extra punctuation."""

        try:
            response = chat_with_llm(
                text_prompt=prompt,
                system_prompt="You are an expert at analyzing academic references and matching citations. Always respond with valid JSON.",
            )

            # Parse JSON response
            result = json.loads(repair_json(response.strip()))

            if result.get("found", False) and result.get("title"):
                title = result["title"].strip()
                logger.info(f"🎯 Matched citation to title: {title}")
                return title
            else:
                logger.info(
                    f"❌ No match found. Reasoning: {result.get('reasoning', 'No reasoning provided')}"
                )
                return None

        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse LLM JSON response: {e}")
            logger.debug(f"Raw response: {response}")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to match citation: {e}")
            return None

    def _search_arxiv(
        self, title: str, max_results: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Search arXiv API for paper information

        Args:
            title: Paper title to search for
            max_results: Maximum number of results to retrieve

        Returns:
            Paper metadata dictionary or None if not found
        """
        try:
            # Prepare search query - search in title and abstract
            query = f"{title}"

            # URL encode the query
            encoded_query = urllib.parse.quote(query)

            # Construct API URL
            api_url = f"http://export.arxiv.org/api/query?search_query={encoded_query}&start=0&max_results={max_results}"

            logger.info(f"🔍 Searching arXiv API: {api_url}")

            # Make API request
            with urllib.request.urlopen(api_url) as response:
                xml_content = response.read().decode("utf-8")

            # Parse XML response
            root = ET.fromstring(xml_content)

            # Define namespaces
            namespaces = {
                "atom": "http://www.w3.org/2005/Atom",
                "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
                "arxiv": "http://arxiv.org/schemas/atom",
            }

            # Get total results
            total_results_elem = root.find("opensearch:totalResults", namespaces)
            total_results = (
                int(total_results_elem.text) if total_results_elem is not None else 0
            )

            if total_results == 0:
                logger.info("❌ No results found on arXiv")
                return None

            # Get the first entry (most relevant)
            entry = root.find("atom:entry", namespaces)
            if entry is None:
                logger.info("❌ No entry found in arXiv response")
                return None

            # Extract metadata
            metadata = {}

            # Title
            title_elem = entry.find("atom:title", namespaces)
            if title_elem is not None:
                metadata["title"] = title_elem.text.strip().replace("\n", " ")

            # Authors
            authors = []
            for author_elem in entry.findall("atom:author", namespaces):
                name_elem = author_elem.find("atom:name", namespaces)
                if name_elem is not None:
                    authors.append(name_elem.text.strip())
            metadata["authors"] = authors

            # Abstract/Summary
            summary_elem = entry.find("atom:summary", namespaces)
            if summary_elem is not None:
                metadata["abstract"] = summary_elem.text.strip().replace("\n", " ")

            # Published date
            published_elem = entry.find("atom:published", namespaces)
            if published_elem is not None:
                metadata["published"] = published_elem.text.strip()

            # arXiv ID
            id_elem = entry.find("atom:id", namespaces)
            if id_elem is not None:
                arxiv_url = id_elem.text.strip()
                # Extract arXiv ID from URL
                arxiv_id = arxiv_url.split("/")[-1]
                metadata["arxiv_id"] = arxiv_id
                metadata["arxiv_url"] = arxiv_url

            # PDF link
            for link_elem in entry.findall("atom:link", namespaces):
                if link_elem.get("title") == "pdf":
                    metadata["pdf_url"] = link_elem.get("href")
                    break

            # Journal reference (if available)
            journal_ref_elem = entry.find("arxiv:journal_ref", namespaces)
            if journal_ref_elem is not None:
                metadata["journal_ref"] = journal_ref_elem.text.strip()

            # DOI (if available)
            doi_elem = entry.find("arxiv:doi", namespaces)
            if doi_elem is not None:
                metadata["doi"] = doi_elem.text.strip()

            # Categories
            categories = []
            for category_elem in entry.findall("atom:category", namespaces):
                term = category_elem.get("term")
                if term:
                    categories.append(term)
            metadata["categories"] = categories

            logger.info(
                f"✅ Found arXiv paper: {metadata.get('title', 'Unknown title')}"
            )
            return metadata

        except Exception as e:
            logger.error(f"❌ arXiv API search failed: {e}")
            return None

    def _format_result(
        self, citation_key: str, title: str, arxiv_metadata: Optional[Dict[str, Any]]
    ) -> str:
        """
        Format the lookup result for output

        Args:
            citation_key: Original citation key
            title: Matched paper title
            arxiv_metadata: arXiv metadata or None

        Returns:
            Formatted result string
        """
        result = f"Citation Lookup Results for: {citation_key}\n"
        result += "=" * 50 + "\n\n"

        result += f"Matched Paper Title: {title}\n\n"

        if arxiv_metadata:
            result += "arXiv Information:\n"
            result += "-" * 20 + "\n"

            if "title" in arxiv_metadata:
                result += f"Title: {arxiv_metadata['title']}\n"

            if "authors" in arxiv_metadata and arxiv_metadata["authors"]:
                authors_str = ", ".join(arxiv_metadata["authors"])
                result += f"Authors: {authors_str}\n"

            if "published" in arxiv_metadata:
                result += f"Published: {arxiv_metadata['published']}\n"

            if "arxiv_id" in arxiv_metadata:
                result += f"arXiv ID: {arxiv_metadata['arxiv_id']}\n"

            if "arxiv_url" in arxiv_metadata:
                result += f"arXiv URL: {arxiv_metadata['arxiv_url']}\n"

            if "pdf_url" in arxiv_metadata:
                result += f"PDF URL: {arxiv_metadata['pdf_url']}\n"

            if "journal_ref" in arxiv_metadata:
                result += f"Journal Reference: {arxiv_metadata['journal_ref']}\n"

            if "doi" in arxiv_metadata:
                result += f"DOI: {arxiv_metadata['doi']}\n"

            if "categories" in arxiv_metadata and arxiv_metadata["categories"]:
                categories_str = ", ".join(arxiv_metadata["categories"])
                result += f"Categories: {categories_str}\n"

            if "abstract" in arxiv_metadata:
                result += f"\nAbstract:\n{arxiv_metadata['abstract']}\n"
        else:
            result += "arXiv Information: Not found on arXiv\n"
            result += "Note: This paper may not be available on arXiv or may be published elsewhere.\n"

        return result

    def forward(self, pdf_name: str, citation_key: str) -> str:
        """
        Look up citation and retrieve metadata

        Args:
            pdf_name: Name of the PDF file (without extension)
            citation_key: Citation key to look up

        Returns:
            Formatted lookup results
        """
        try:
            logger.info(f"🔍 Looking up citation '{citation_key}' in {pdf_name}")

            # Load markdown content
            content, load_error = self._load_markdown_content(pdf_name)
            if load_error:
                return f"Error: {load_error}"

            # Extract references section
            references = self._extract_references_section(content)
            if references is None:
                return f"Error: No references section found in {pdf_name}"
            
            # print(references)

            # Match citation to title using LLM
            title = self._match_citation_to_title(references, citation_key)
            if title is None:
                return f"Error: Could not match citation key '{citation_key}' to any reference"

            # Search arXiv for metadata
            arxiv_metadata = self._search_arxiv(title)

            # Format and return results
            result = self._format_result(citation_key, title, arxiv_metadata)

            logger.info("✅ Citation lookup completed successfully")
            return result

        except Exception as e:
            error_msg = f"❌ Citation lookup failed: {str(e)}"
            logger.error(error_msg)
            return f"Error: {str(e)}"


if __name__ == "__main__":
    # Test the CrossRef lookup tool
    lookup = CrossRefLookup()
    lookup.setup()

    from dotenv import load_dotenv

    load_dotenv('/data/wdy/AgenticPaperQA/smolagents/paper_qa/.env', verbose=True)

    # Example usage
    test_pdf_name = "0A4Y9qRnu9_Leveraging Per-Instance Privacy for Machine Unlearning"  # Replace with actual PDF name
    test_citation = "Thudi et al., 2024"  # Replace with actual citation

    try:
        result = lookup.forward(
            pdf_name=test_pdf_name,
            citation_key=test_citation,
        )
        print("🔍 Citation Lookup Result:")
        print("=" * 50)
        print(result)
    except Exception as e:
        print(f"❌ Citation lookup test failed: {e}")

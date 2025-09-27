"""
Data Loader for Paper QA System
Loads and manages QA data from paper JSON files
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter

from loguru import logger


@dataclass
class QAItem:
    """Single QA item data structure"""

    qa_id: int
    question: str
    answer: str
    qa_type: str
    difficulty: str
    tool_chain: List[str]
    reasoning: str

    # Paper metadata
    pdf_name: str
    paper_title: str
    conference: str
    primary_area: str
    ai_primary_category: str
    ai_secondary_category: Optional[str]
    ai_research_type: str


@dataclass
class PaperInfo:
    """Paper metadata structure"""

    pdf_name: str
    title: str
    status: str
    track: str
    primary_area: str
    paper_id: str
    abstract: str
    pdf_url: str
    conference: str
    ai_primary_category: str
    ai_secondary_category: Optional[str]
    ai_research_type: str
    ai_eval_method: str
    ai_confidence: int
    ai_reasoning: str
    classification_status: str
    qa_count: int


class DataLoader:
    """
    Data loader for Paper QA system

    Loads and manages QA data from paper JSON files, providing
    various filtering and querying capabilities.
    """

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the data loader

        Args:
            data_dir: Directory containing paper JSON files.
                     Defaults to /data/wdy/AgenticPaperQA/paper-prepare/qa
        """
        if data_dir is None:
            data_dir = "/data/wdy/AgenticPaperQA/paper-prepare/qa"

        self.data_dir = Path(data_dir)
        self.papers: Dict[str, PaperInfo] = {}
        self.qa_items: List[QAItem] = []
        self.qa_by_paper: Dict[str, List[QAItem]] = defaultdict(list)

        # Statistics
        self.stats = {}

        logger.info(f"🔧 Initialized DataLoader with directory: {self.data_dir}")

    def load_all_data(self) -> Tuple[bool, Optional[str]]:
        """
        Load all paper data from JSON files

        Returns:
            Tuple of (success, error_message)
        """
        try:
            if not self.data_dir.exists():
                return False, f"Data directory not found: {self.data_dir}"

            # Find all JSON files
            json_files = list(self.data_dir.glob("*.json"))
            if not json_files:
                return False, f"No JSON files found in: {self.data_dir}"

            logger.info(f"📂 Found {len(json_files)} JSON files")

            # Load each file
            loaded_count = 0
            total_qa_count = 0

            for json_file in json_files:
                success, error_msg, qa_count = self._load_single_paper(json_file)
                if success:
                    loaded_count += 1
                    total_qa_count += qa_count
                else:
                    logger.warning(f"⚠️ Failed to load {json_file.name}: {error_msg}")

            if loaded_count == 0:
                return False, "No papers were successfully loaded"

            # Generate statistics
            self._generate_statistics()

            logger.info(
                f"✅ Successfully loaded {loaded_count}/{len(json_files)} papers"
            )
            logger.info(f"📊 Total QA items: {total_qa_count}")

            return True, None

        except Exception as e:
            error_msg = f"Failed to load data: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

    def _load_single_paper(self, json_file: Path) -> Tuple[bool, Optional[str], int]:
        """
        Load a single paper JSON file

        Args:
            json_file: Path to JSON file

        Returns:
            Tuple of (success, error_message, qa_count)
        """
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                paper_data = json.load(f)

            # Extract PDF name from filename
            pdf_name = json_file.stem

            # Validate required fields
            required_fields = ["title", "qa"]
            for field in required_fields:
                if field not in paper_data:
                    return False, f"Missing required field: {field}", 0

            # Create PaperInfo
            paper_info = PaperInfo(
                pdf_name=pdf_name,
                title=paper_data.get("title", ""),
                status=paper_data.get("status", ""),
                track=paper_data.get("track", ""),
                primary_area=paper_data.get("primary_area", ""),
                paper_id=paper_data.get("id", ""),
                abstract=paper_data.get("abstract", ""),
                pdf_url=paper_data.get("pdf", ""),
                conference=paper_data.get("conference", ""),
                ai_primary_category=paper_data.get("ai_primary_category", ""),
                ai_secondary_category=paper_data.get("ai_secondary_category"),
                ai_research_type=paper_data.get("ai_research_type", ""),
                ai_eval_method=paper_data.get("ai_eval_method", ""),
                ai_confidence=paper_data.get("ai_confidence", 0),
                ai_reasoning=paper_data.get("ai_reasoning", ""),
                classification_status=paper_data.get("classification_status", ""),
                qa_count=len(paper_data.get("qa", [])),
            )

            self.papers[pdf_name] = paper_info

            # Load QA items
            qa_items = paper_data.get("qa", [])
            qa_count = 0

            for qa_data in qa_items:
                try:
                    qa_item = QAItem(
                        qa_id=qa_data.get("qa_id", 0),
                        question=qa_data.get("question", ""),
                        answer=qa_data.get("answer", ""),
                        qa_type=qa_data.get("qa_type", ""),
                        difficulty=qa_data.get("difficulty", ""),
                        tool_chain=qa_data.get("tool_chain", []),
                        reasoning=qa_data.get("reasoning", ""),
                        # Paper metadata
                        pdf_name=pdf_name,
                        paper_title=paper_info.title,
                        conference=paper_info.conference,
                        primary_area=paper_info.primary_area,
                        ai_primary_category=paper_info.ai_primary_category,
                        ai_secondary_category=paper_info.ai_secondary_category,
                        ai_research_type=paper_info.ai_research_type,
                    )

                    self.qa_items.append(qa_item)
                    self.qa_by_paper[pdf_name].append(qa_item)
                    qa_count += 1

                except Exception as e:
                    logger.warning(f"⚠️ Failed to load QA item in {pdf_name}: {e}")

            return True, None, qa_count

        except json.JSONDecodeError as e:
            return False, f"JSON parsing error: {e}", 0
        except Exception as e:
            return False, f"Error loading file: {e}", 0

    def _generate_statistics(self):
        """Generate comprehensive statistics about the loaded data"""
        self.stats = {
            "total_papers": len(self.papers),
            "total_qa_items": len(self.qa_items),
            "papers_by_conference": Counter(
                paper.conference for paper in self.papers.values()
            ),
            "papers_by_status": Counter(paper.status for paper in self.papers.values()),
            "papers_by_primary_area": Counter(
                paper.primary_area for paper in self.papers.values()
            ),
            "papers_by_ai_category": Counter(
                paper.ai_primary_category for paper in self.papers.values()
            ),
            "papers_by_research_type": Counter(
                paper.ai_research_type for paper in self.papers.values()
            ),
            "qa_by_type": Counter(qa.qa_type for qa in self.qa_items),
            "qa_by_difficulty": Counter(qa.difficulty for qa in self.qa_items),
            "qa_by_conference": Counter(qa.conference for qa in self.qa_items),
            "tool_chain_usage": Counter(),
            "avg_qa_per_paper": (
                len(self.qa_items) / len(self.papers) if self.papers else 0
            ),
            "avg_confidence": (
                sum(paper.ai_confidence for paper in self.papers.values())
                / len(self.papers)
                if self.papers
                else 0
            ),
        }

        # Count tool chain usage
        for qa in self.qa_items:
            for tool in qa.tool_chain:
                self.stats["tool_chain_usage"][tool] += 1

    def get_papers(
        self,
        conference: Optional[str] = None,
        status: Optional[str] = None,
        primary_area: Optional[str] = None,
        ai_category: Optional[str] = None,
    ) -> List[PaperInfo]:
        """
        Get papers with optional filtering

        Args:
            conference: Filter by conference
            status: Filter by paper status
            primary_area: Filter by primary area
            ai_category: Filter by AI category

        Returns:
            List of filtered PaperInfo objects
        """
        papers = list(self.papers.values())

        if conference:
            papers = [p for p in papers if p.conference.lower() == conference.lower()]
        if status:
            papers = [p for p in papers if p.status.lower() == status.lower()]
        if primary_area:
            papers = [
                p for p in papers if primary_area.lower() in p.primary_area.lower()
            ]
        if ai_category:
            papers = [
                p
                for p in papers
                if p.ai_primary_category.lower() == ai_category.lower()
            ]

        return papers

    def get_qa_items(
        self,
        pdf_name: Optional[str] = None,
        qa_type: Optional[str] = None,
        difficulty: Optional[str] = None,
        tool_required: Optional[str] = None,
        conference: Optional[str] = None,
    ) -> List[QAItem]:
        """
        Get QA items with optional filtering

        Args:
            pdf_name: Filter by specific paper
            qa_type: Filter by QA type
            difficulty: Filter by difficulty level
            tool_required: Filter by required tool in tool chain
            conference: Filter by conference

        Returns:
            List of filtered QAItem objects
        """
        if pdf_name:
            qa_items = self.qa_by_paper.get(pdf_name, [])
        else:
            qa_items = self.qa_items.copy()

        if qa_type:
            qa_items = [qa for qa in qa_items if qa.qa_type.lower() == qa_type.lower()]
        if difficulty:
            qa_items = [
                qa for qa in qa_items if qa.difficulty.lower() == difficulty.lower()
            ]
        if tool_required:
            qa_items = [qa for qa in qa_items if tool_required in qa.tool_chain]
        if conference:
            qa_items = [
                qa for qa in qa_items if qa.conference.lower() == conference.lower()
            ]

        return qa_items

    def get_paper_info(self, pdf_name: str) -> Optional[PaperInfo]:
        """
        Get paper information by PDF name

        Args:
            pdf_name: Name of the PDF (without extension)

        Returns:
            PaperInfo object or None if not found
        """
        return self.papers.get(pdf_name)

    def get_qa_by_id(self, pdf_name: str, qa_id: int) -> Optional[QAItem]:
        """
        Get specific QA item by paper name and QA ID

        Args:
            pdf_name: Name of the PDF
            qa_id: QA item ID

        Returns:
            QAItem object or None if not found
        """
        paper_qa_items = self.qa_by_paper.get(pdf_name, [])
        for qa in paper_qa_items:
            if qa.qa_id == qa_id:
                return qa
        return None

    def search_papers(self, query: str, field: str = "title") -> List[PaperInfo]:
        """
        Search papers by text query

        Args:
            query: Search query
            field: Field to search in ("title", "abstract", "primary_area")

        Returns:
            List of matching PaperInfo objects
        """
        query_lower = query.lower()
        results = []

        for paper in self.papers.values():
            if field == "title" and query_lower in paper.title.lower():
                results.append(paper)
            elif field == "abstract" and query_lower in paper.abstract.lower():
                results.append(paper)
            elif field == "primary_area" and query_lower in paper.primary_area.lower():
                results.append(paper)

        return results

    def search_qa_items(self, query: str, field: str = "question") -> List[QAItem]:
        """
        Search QA items by text query

        Args:
            query: Search query
            field: Field to search in ("question", "answer", "reasoning")

        Returns:
            List of matching QAItem objects
        """
        query_lower = query.lower()
        results = []

        for qa in self.qa_items:
            if field == "question" and query_lower in qa.question.lower():
                results.append(qa)
            elif field == "answer" and query_lower in qa.answer.lower():
                results.append(qa)
            elif field == "reasoning" and query_lower in qa.reasoning.lower():
                results.append(qa)

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the loaded data

        Returns:
            Dictionary containing various statistics
        """
        return self.stats.copy()

    def print_statistics(self):
        """Print formatted statistics"""
        if not self.stats:
            print("📊 No statistics available. Please load data first.")
            return

        print("📊 Data Loader Statistics")
        print("=" * 50)

        print(f"📚 Total Papers: {self.stats['total_papers']}")
        print(f"❓ Total QA Items: {self.stats['total_qa_items']}")
        print(f"📈 Average QA per Paper: {self.stats['avg_qa_per_paper']:.2f}")
        print(f"🎯 Average AI Confidence: {self.stats['avg_confidence']:.2f}")
        print()

        print("🏛️ Papers by Conference:")
        for conf, count in self.stats["papers_by_conference"].most_common():
            print(f"  - {conf}: {count}")
        print()

        print("📄 Papers by Status:")
        for status, count in self.stats["papers_by_status"].most_common():
            print(f"  - {status}: {count}")
        print()

        print("🔬 Papers by Primary Area (Top 5):")
        for area, count in self.stats["papers_by_primary_area"].most_common(5):
            print(f"  - {area}: {count}")
        print()

        print("❓ QA Items by Type:")
        for qa_type, count in self.stats["qa_by_type"].most_common():
            print(f"  - {qa_type}: {count}")
        print()

        print("⭐ QA Items by Difficulty:")
        for difficulty, count in self.stats["qa_by_difficulty"].most_common():
            print(f"  - {difficulty}: {count}")
        print()

        print("🛠️ Tool Chain Usage (Top 5):")
        for tool, count in self.stats["tool_chain_usage"].most_common(5):
            print(f"  - {tool}: {count}")

    def export_summary(self, output_file: Optional[str] = None) -> str:
        """
        Export data summary to JSON file

        Args:
            output_file: Output file path. If None, returns JSON string

        Returns:
            JSON string of the summary
        """
        summary = {
            "statistics": self.stats,
            "papers_summary": [
                {
                    "pdf_name": paper.pdf_name,
                    "title": paper.title,
                    "conference": paper.conference,
                    "status": paper.status,
                    "primary_area": paper.primary_area,
                    "qa_count": paper.qa_count,
                    "ai_confidence": paper.ai_confidence,
                }
                for paper in self.papers.values()
            ],
        }

        json_str = json.dumps(summary, indent=2, ensure_ascii=False)

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(json_str)
            logger.info(f"📁 Summary exported to: {output_file}")

        return json_str


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()

    # Load all data
    success, error = loader.load_all_data()
    if not success:
        print(f"❌ Failed to load data: {error}")
        exit(1)

    # Print statistics
    loader.print_statistics()

    # Test filtering
    print("\n🔍 Sample Queries:")
    print("-" * 30)

    # Get papers by conference
    icml_papers = loader.get_papers(conference="icml2025")
    print(f"ICML 2025 papers: {len(icml_papers)}")

    # Get hard QA items
    hard_qa = loader.get_qa_items(difficulty="Hard")
    print(f"Hard QA items: {len(hard_qa)}")

    # Search for machine learning papers
    ml_papers = loader.search_papers("machine learning", field="title")
    print(f"Papers with 'machine learning' in title: {len(ml_papers)}")

    # Get QA items requiring specific tools
    pdf_parser_qa = loader.get_qa_items(tool_required="PDF Parser")
    print(f"QA items requiring PDF Parser: {len(pdf_parser_qa)}")

    # Sample paper info
    if loader.papers:
        sample_paper = list(loader.papers.values())[0]
        print(f"\nSample paper: {sample_paper.title}")
        print(f"QA count: {sample_paper.qa_count}")

        # Get QA items for this paper
        paper_qa = loader.get_qa_items(pdf_name=sample_paper.pdf_name)
        if paper_qa:
            sample_qa = paper_qa[0]
            print(f"Sample QA: {sample_qa.question[:100]}...")
            print(f"Difficulty: {sample_qa.difficulty}")
            print(f"Tools: {sample_qa.tool_chain}")

import json
import jsonlines
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle

from sklearn.manifold import TSNE
from sklearn_extra.cluster import KMedoids
from scipy.spatial.distance import hamming
import warnings

warnings.filterwarnings("ignore")


class PaperSampler:
    def __init__(self):
        """Initialize the paper sampling system"""
        self.papers = []
        self.vectors = None
        self.tsne_coords = None
        self.medoids_indices = None
        self.sampled_papers = []

        # Define encoding mappings
        self.influence_mapping = {
            "icml2025": {
                "Oral": "Top-Tier",
                "Spotlight": "Mid-Tier",
                "Poster": "Standard-Tier",
            },
            "nips2024": {
                "Oral": "Top-Tier",
                "Spotlight": "Mid-Tier",
                "Poster": "Standard-Tier",
            },
            "iclr2025": {
                "Oral": "Top-Tier",
                "Spotlight": "Mid-Tier",
                "Poster": "Standard-Tier",
            },
            "cvpr2025": {
                "Award Candidate": "Top-Tier",
                "Highlight": "Mid-Tier",
                "Poster": "Standard-Tier",
            },
            "www2025": {
                "Oral": "Top-Tier",
                "Poster": "Standard-Tier",
            },
        }

        # Define category orders for one-hot encoding
        self.influence_categories = ["Top-Tier", "Mid-Tier", "Standard-Tier"]  # 3 dims
        self.primary_categories = [
            "ML",
            "NLP",
            "CV",
            "RL",
            "Gen AI",
            "IR",
            "AI4Science",
            "MLSys",
        ]  # 8 dims
        self.research_types = [
            "Theoretical",
            "Empirical",
            "Architectural",
            "Resource",
        ]  # 4 dims
        self.eval_methods = [
            "Benchmark",
            "Empirical",
            "Ablation",
            "Qualitative",
            "System",
        ]  # 5 dims

        self.total_dims = 3 + 8 + 4 + 5  # 20 dimensions

        # Create output directory for plots
        self.output_dir = Path("sampling_outputs_tmp")
        self.output_dir.mkdir(exist_ok=True)

    def load_classified_papers(self, classified_dir: Path) -> List[Dict]:
        """Load all classified papers from JSONL files"""
        print("📚 Loading classified papers...")

        all_papers = []
        conference_files = [
            "icml2025_classified.jsonl",
            "nips2024_classified.jsonl",
            "iclr2025_classified.jsonl",
            "cvpr2025_classified.jsonl",
            "www2025_classified.jsonl",
        ]

        for filename in conference_files:
            file_path = classified_dir / filename
            if file_path.exists():
                try:
                    with jsonlines.open(file_path, "r") as reader:
                        papers = list(reader)
                        # Only include successfully classified papers
                        successful_papers = [
                            p
                            for p in papers
                            if p.get("classification_status") == "success"
                        ]
                        all_papers.extend(successful_papers)
                        print(
                            f"✅ Loaded {len(successful_papers)} papers from {filename}"
                        )
                except Exception as e:
                    print(f"❌ Error loading {filename}: {str(e)}")
            else:
                print(f"⚠️  File not found: {file_path}")

        print(f"📊 Total papers loaded: {len(all_papers)}")
        return all_papers

    def map_influence_level(self, paper: Dict) -> str:
        """Map paper status to influence level based on conference"""
        conference = paper.get("conference", "")
        status = paper.get("status", "")

        if conference in self.influence_mapping:
            return self.influence_mapping[conference].get(status, "Standard-Tier")

        return "Standard-Tier"  # Default

    def create_feature_vector(self, paper: Dict) -> np.ndarray:
        """Convert paper attributes to 19-dimensional binary vector"""
        vector = np.zeros(self.total_dims, dtype=int)

        # 1. Influence Level (3 dims: positions 0-2)
        influence = self.map_influence_level(paper)
        if influence in self.influence_categories:
            idx = self.influence_categories.index(influence)
            vector[idx] = 1

        # 2. Primary Category (8 dims: positions 3-10)
        primary_cat = paper.get("ai_primary_category", "")
        if primary_cat in self.primary_categories:
            idx = 3 + self.primary_categories.index(primary_cat)
            vector[idx] = 1

        # 3. Research Type (4 dims: positions 11-14)
        research_type = paper.get("ai_research_type", "")
        if research_type in self.research_types:
            idx = 11 + self.research_types.index(research_type)
            vector[idx] = 1

        # 4. Evaluation Method (4 dims: positions 15-18)
        eval_method = paper.get("ai_eval_method", "")
        if eval_method in self.eval_methods:
            idx = 15 + self.eval_methods.index(eval_method)
            vector[idx] = 1

        return vector

    def vectorize_papers(self, papers: List[Dict]):
        """Convert all papers to feature vectors"""
        print("🔢 Converting papers to feature vectors...")

        vectors = []
        valid_papers = []

        for paper in papers:
            vector = self.create_feature_vector(paper)
            # Only include papers with at least some valid features
            if np.sum(vector) > 0:
                vectors.append(vector)
                valid_papers.append(paper)

        self.papers = valid_papers
        self.vectors = np.array(vectors)

        print(
            f"✅ Created {len(vectors)} feature vectors of {self.total_dims} dimensions"
        )
        print(
            f"📊 Vector density: {np.mean(np.sum(self.vectors, axis=1)):.2f} features per paper"
        )

        # Save vectorization results
        self.save_vectorization_results()

    def save_vectorization_results(self):
        """Save vectorization results for later visualization"""
        print("💾 Saving vectorization results...")

        # Save all papers data and vectors
        data = {
            "papers": self.papers,
            "vectors": self.vectors,
            "influence_categories": self.influence_categories,
            "primary_categories": self.primary_categories,
            "research_types": self.research_types,
            "eval_methods": self.eval_methods,
            "total_dims": self.total_dims,
        }

        output_path = self.output_dir / "all_papers_data.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(data, f)

        print(f"✅ All papers data saved to {output_path}")

    def create_research_map(self, perplexity: int = 30, random_state: int = 42):
        """Create t-SNE coordinates for research map"""
        if self.vectors is None:
            print("❌ No vectors available. Run vectorize_papers first.")
            return

        print("🗺️  Creating t-SNE coordinates for AI research map...")

        # Apply t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(self.vectors) - 1),
            random_state=random_state,
            metric="hamming",  # Use Hamming distance for binary vectors
            init="random",
        )

        self.tsne_coords = tsne.fit_transform(self.vectors.astype(float))

        # Save t-SNE results
        self.save_tsne_results()

        print(f"✅ t-SNE coordinates computed and saved")
        return self.tsne_coords

    def save_tsne_results(self):
        """Save t-SNE results for later visualization"""
        print("💾 Saving t-SNE results...")

        data = {
            "tsne_coords": self.tsne_coords,
            "vectors": self.vectors,
            "primary_categories": self.primary_categories,
        }

        output_path = self.output_dir / "tsne_results.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(data, f)

        print(f"✅ t-SNE results saved to {output_path}")

    def hamming_distance_matrix(self, vectors: np.ndarray) -> np.ndarray:
        """Compute pairwise Hamming distances"""
        n = len(vectors)
        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dist = hamming(vectors[i], vectors[j]) * len(vectors[i])
                distances[i, j] = distances[j, i] = dist

        return distances

    def select_prototype_papers(self, k: int = 50) -> List[int]:
        """Select k prototype papers using K-Medoids clustering"""
        if self.vectors is None:
            print("❌ No vectors available. Run vectorize_papers first.")
            return []

        print(f"🎯 Selecting {k} prototype papers using K-Medoids clustering...")

        # Use sklearn-extra K-Medoids with Hamming distance
        kmedoids = KMedoids(
            n_clusters=min(k, len(self.vectors)),
            metric="hamming",
            random_state=42,
            init="k-medoids++",
            max_iter=300,
        )
        cluster_labels = kmedoids.fit_predict(self.vectors)
        medoid_indices = kmedoids.medoid_indices_

        self.medoids_indices = medoid_indices

        print(f"✅ Selected {len(self.medoids_indices)} prototype papers")

        # Analyze cluster distribution
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        print(
            f"📊 Cluster sizes: min={np.min(counts)}, max={np.max(counts)}, avg={np.mean(counts):.1f}"
        )

        return self.medoids_indices

    def farthest_point_sampling(
        self, initial_indices: List[int], n_additional: int
    ) -> List[int]:
        """Select additional papers using Farthest Point Sampling"""
        if self.vectors is None:
            print("❌ No vectors available. Run vectorize_papers first.")
            return []

        print(
            f"🎯 Selecting {n_additional} additional papers using Farthest Point Sampling..."
        )

        selected_indices = set(initial_indices)
        remaining_indices = set(range(len(self.vectors))) - selected_indices

        additional_indices = []

        for iteration in range(n_additional):
            if not remaining_indices:
                break

            max_min_distance = -1
            farthest_idx = None

            # For each remaining paper, find its minimum distance to selected papers
            for candidate_idx in remaining_indices:
                min_distance_to_selected = float("inf")

                for selected_idx in selected_indices:
                    distance = (
                        hamming(self.vectors[candidate_idx], self.vectors[selected_idx])
                        * self.total_dims
                    )
                    min_distance_to_selected = min(min_distance_to_selected, distance)

                # Select the paper with maximum minimum distance (most novel)
                if min_distance_to_selected > max_min_distance:
                    max_min_distance = min_distance_to_selected
                    farthest_idx = candidate_idx

            if farthest_idx is not None:
                additional_indices.append(farthest_idx)
                selected_indices.add(farthest_idx)
                remaining_indices.remove(farthest_idx)

            if (iteration + 1) % 10 == 0:
                print(f"  Progress: {iteration + 1}/{n_additional} papers selected")

        print(f"✅ Selected {len(additional_indices)} additional papers")
        return additional_indices

    def generate_sampling_report(
        self,
        sampled_indices: List[int],
        prototype_indices: List[int],
        additional_indices: List[int],
        output_file: str = "sampled_papers.jsonl",
    ):
        """Generate detailed sampling report and save sampled papers"""
        if not sampled_indices:
            print("❌ No sampled papers available.")
            return

        sampled_papers = [self.papers[i] for i in sampled_indices]
        sampled_vectors = self.vectors[sampled_indices]

        print(f"\n{'='*60}")
        print("📊 PAPER SAMPLING REPORT")
        print(f"{'='*60}")

        print(f"📈 Total Papers Available: {len(self.papers):,}")
        print(f"🎯 Papers Sampled: {len(sampled_papers):,}")
        print(f"📊 Sampling Rate: {len(sampled_papers)/len(self.papers)*100:.2f}%")

        # Analyze sampling distribution
        print(f"\n🏛️  Conference Distribution:")
        conf_dist = {}
        for paper in sampled_papers:
            conf = paper.get("conference", "Unknown")
            conf_dist[conf] = conf_dist.get(conf, 0) + 1

        for conf, count in sorted(conf_dist.items()):
            print(f"  {conf.upper():<12}: {count:>4} papers")

        print(f"\n🎯 Primary Category Distribution:")
        cat_dist = {}
        for paper in sampled_papers:
            cat = paper.get("ai_primary_category", "Unknown")
            cat_dist[cat] = cat_dist.get(cat, 0) + 1

        for cat, count in sorted(cat_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat:<12}: {count:>4} papers")

        print(f"\n🏆 Influence Level Distribution:")
        inf_dist = {}
        for paper in sampled_papers:
            inf = self.map_influence_level(paper)
            inf_dist[inf] = inf_dist.get(inf, 0) + 1

        for inf, count in sorted(inf_dist.items()):
            print(f"  {inf:<15}: {count:>4} papers")

        # Save sampled papers
        output_path = self.output_dir / output_file
        print(f"\n💾 Saving sampled papers to {output_path}...")
        with jsonlines.open(output_path, "w") as writer:
            for paper in sampled_papers:
                writer.write(paper)

        # Save sampling results for visualization
        self.save_sampling_results(
            sampled_indices, prototype_indices, additional_indices, sampled_vectors
        )

        print(f"✅ Sampled papers saved successfully!")
        print(f"{'='*60}")

        return sampled_papers

    def save_sampling_results(
        self,
        sampled_indices: List[int],
        prototype_indices: List[int],
        additional_indices: List[int],
        sampled_vectors: np.ndarray,
    ):
        """Save sampling results for later visualization"""
        print("💾 Saving sampling results...")

        data = {
            "sampled_indices": sampled_indices,
            "prototype_indices": prototype_indices,
            "additional_indices": additional_indices,
            "sampled_papers": [self.papers[i] for i in sampled_indices],
            "sampled_vectors": sampled_vectors,
            "influence_categories": self.influence_categories,
            "primary_categories": self.primary_categories,
            "research_types": self.research_types,
            "eval_methods": self.eval_methods,
            "total_dims": self.total_dims,
        }

        output_path = self.output_dir / "sampling_results.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(data, f)

        print(f"✅ Sampling results saved to {output_path}")

    def run_complete_sampling(
        self, classified_dir: Path, k_prototypes: int = 50, n_additional: int = 200
    ):
        """Run the complete sampling pipeline"""
        print("🚀 Starting Complete Paper Sampling Pipeline")
        print(
            f"🎯 Target: {k_prototypes} prototypes + {n_additional} boundary papers = {k_prototypes + n_additional} total"
        )
        print("=" * 70)

        # Step 1: Load and vectorize papers
        papers = self.load_classified_papers(classified_dir)
        if not papers:
            print("❌ No papers loaded. Exiting.")
            return

        self.vectorize_papers(papers)

        # Step 2: Create t-SNE coordinates
        self.create_research_map()

        # Step 3: Select prototype papers
        prototype_indices = self.select_prototype_papers(k_prototypes)

        # Step 4: Select additional papers using FPS
        additional_indices = self.farthest_point_sampling(
            list(prototype_indices), n_additional
        )

        # Step 5: Combine results
        all_sampled_indices = list(prototype_indices) + list(additional_indices)

        # Step 6: Generate report and save results
        sampled_papers = self.generate_sampling_report(
            all_sampled_indices, prototype_indices, additional_indices
        )

        print(f"\n🎉 Sampling pipeline completed successfully!")
        print(
            f"📊 {len(sampled_papers)} papers selected from {len(papers)} total papers"
        )

        return sampled_papers


def main():
    """Main function to run paper sampling"""
    print("🎯 AI Paper Sampling System")
    print("=" * 50)

    # Initialize sampler
    sampler = PaperSampler()

    # Define paths
    base_dir = Path(__file__).parent
    classified_dir = base_dir / "classified_results"

    # Check if classified results exist
    if not classified_dir.exists():
        print(f"❌ Classified results directory not found: {classified_dir}")
        print("Please run paper_classification.py first to generate classified papers.")
        return

    # Run complete sampling pipeline
    sampled_papers = sampler.run_complete_sampling(
        classified_dir=classified_dir,
        k_prototypes=50,  # Number of prototype papers (K-Medoids)
        n_additional=50,  # Number of additional papers (FPS)
    )

    if sampled_papers:
        print(f"\n✨ Success! {len(sampled_papers)} papers sampled and saved.")
        print("📁 Check the following files:")
        print("  - sampling_outputs/sampled_papers.jsonl: Sampled papers data")
        print("  - sampling_outputs/all_papers_data.pkl: All papers data and vectors")
        print("  - sampling_outputs/tsne_results.pkl: t-SNE coordinates")
        print("  - sampling_outputs/sampling_results.pkl: Sampling results")
        print("\n🎨 To create visualizations, run: python vis_sampling.py")


if __name__ == "__main__":
    main()

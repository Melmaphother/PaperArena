import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# Set style for better plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class SamplingVisualizer:
    def __init__(self, output_dir: str = "sampling_outputs_tmp"):
        """Initialize the sampling visualization system"""
        self.output_dir = Path(output_dir)

        # Data containers
        self.all_papers_data = None
        self.tsne_results = None
        self.sampling_results = None

        # Load data
        self.load_data()

    def load_data(self):
        """Load all saved data files"""
        print("📂 Loading saved data files...")

        # Load all papers data
        all_papers_path = self.output_dir / "all_papers_data.pkl"
        if all_papers_path.exists():
            with open(all_papers_path, "rb") as f:
                self.all_papers_data = pickle.load(f)
            print(
                f"✅ Loaded all papers data: {len(self.all_papers_data['papers'])} papers"
            )
        else:
            print(f"❌ All papers data not found: {all_papers_path}")

        # Load t-SNE results
        tsne_path = self.output_dir / "tsne_results.pkl"
        if tsne_path.exists():
            with open(tsne_path, "rb") as f:
                self.tsne_results = pickle.load(f)
            print(f"✅ Loaded t-SNE results: {self.tsne_results['tsne_coords'].shape}")
        else:
            print(f"❌ t-SNE results not found: {tsne_path}")

        # Load sampling results
        sampling_path = self.output_dir / "sampling_results.pkl"
        if sampling_path.exists():
            with open(sampling_path, "rb") as f:
                self.sampling_results = pickle.load(f)
            print(
                f"✅ Loaded sampling results: {len(self.sampling_results['sampled_papers'])} sampled papers"
            )
        else:
            print(f"❌ Sampling results not found: {sampling_path}")

    def visualize_all_papers_features(self, save_path: Optional[str] = None):
        """Visualize feature distribution for all papers"""
        if self.all_papers_data is None:
            print("❌ All papers data not available.")
            return

        print("📊 Creating all papers feature distribution visualization...")

        vectors = self.all_papers_data["vectors"]
        influence_categories = self.all_papers_data["influence_categories"]
        primary_categories = self.all_papers_data["primary_categories"]
        research_types = self.all_papers_data["research_types"]
        eval_methods = self.all_papers_data["eval_methods"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "All Papers - Feature Distribution Analysis", fontsize=16, fontweight="bold"
        )

        # 1. Influence Level Distribution
        influence_counts = np.sum(vectors[:, :3], axis=0)
        axes[0, 0].bar(influence_categories, influence_counts, color="skyblue")
        axes[0, 0].set_title("Influence Level Distribution")
        axes[0, 0].set_ylabel("Number of Papers")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Add count labels on bars
        for i, count in enumerate(influence_counts):
            axes[0, 0].text(
                i,
                count + max(influence_counts) * 0.01,
                str(int(count)),
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 2. Primary Category Distribution
        primary_counts = np.sum(vectors[:, 3:11], axis=0)
        axes[0, 1].bar(primary_categories, primary_counts, color="lightgreen")
        axes[0, 1].set_title("Primary Category Distribution")
        axes[0, 1].set_ylabel("Number of Papers")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Add count labels on bars
        for i, count in enumerate(primary_counts):
            axes[0, 1].text(
                i,
                count + max(primary_counts) * 0.01,
                str(int(count)),
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 3. Research Type Distribution
        research_counts = np.sum(vectors[:, 11:15], axis=0)
        axes[1, 0].bar(research_types, research_counts, color="lightcoral")
        axes[1, 0].set_title("Research Type Distribution")
        axes[1, 0].set_ylabel("Number of Papers")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Add count labels on bars
        for i, count in enumerate(research_counts):
            axes[1, 0].text(
                i,
                count + max(research_counts) * 0.01,
                str(int(count)),
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 4. Evaluation Method Distribution
        eval_counts = np.sum(vectors[:, 15:20], axis=0)
        axes[1, 1].bar(eval_methods, eval_counts, color="gold")
        axes[1, 1].set_title("Evaluation Method Distribution")
        axes[1, 1].set_ylabel("Number of Papers")
        axes[1, 1].tick_params(axis="x", rotation=45)

        # Add count labels on bars
        for i, count in enumerate(eval_counts):
            axes[1, 1].text(
                i,
                count + max(eval_counts) * 0.01,
                str(int(count)),
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "all_papers_feature_distribution.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"📊 All papers feature distribution saved as '{save_path}'")

    def visualize_sampled_papers_features(self, save_path: Optional[str] = None):
        """Visualize feature distribution for sampled papers"""
        if self.sampling_results is None:
            print("❌ Sampling results not available.")
            return

        print("📊 Creating sampled papers feature distribution visualization...")

        vectors = self.sampling_results["sampled_vectors"]
        influence_categories = self.sampling_results["influence_categories"]
        primary_categories = self.sampling_results["primary_categories"]
        research_types = self.sampling_results["research_types"]
        eval_methods = self.sampling_results["eval_methods"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "Sampled Papers - Feature Distribution Analysis",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Influence Level Distribution
        influence_counts = np.sum(vectors[:, :3], axis=0)
        axes[0, 0].bar(influence_categories, influence_counts, color="skyblue")
        axes[0, 0].set_title("Influence Level Distribution")
        axes[0, 0].set_ylabel("Number of Papers")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Add count labels on bars
        for i, count in enumerate(influence_counts):
            axes[0, 0].text(
                i,
                (
                    count + max(influence_counts) * 0.01
                    if max(influence_counts) > 0
                    else 0.1
                ),
                str(int(count)),
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 2. Primary Category Distribution
        primary_counts = np.sum(vectors[:, 3:11], axis=0)
        axes[0, 1].bar(primary_categories, primary_counts, color="lightgreen")
        axes[0, 1].set_title("Primary Category Distribution")
        axes[0, 1].set_ylabel("Number of Papers")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Add count labels on bars
        for i, count in enumerate(primary_counts):
            axes[0, 1].text(
                i,
                count + max(primary_counts) * 0.01 if max(primary_counts) > 0 else 0.1,
                str(int(count)),
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 3. Research Type Distribution
        research_counts = np.sum(vectors[:, 11:15], axis=0)
        axes[1, 0].bar(research_types, research_counts, color="lightcoral")
        axes[1, 0].set_title("Research Type Distribution")
        axes[1, 0].set_ylabel("Number of Papers")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Add count labels on bars
        for i, count in enumerate(research_counts):
            axes[1, 0].text(
                i,
                (
                    count + max(research_counts) * 0.01
                    if max(research_counts) > 0
                    else 0.1
                ),
                str(int(count)),
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 4. Evaluation Method Distribution
        eval_counts = np.sum(vectors[:, 15:20], axis=0)
        axes[1, 1].bar(eval_methods, eval_counts, color="gold")
        axes[1, 1].set_title("Evaluation Method Distribution")
        axes[1, 1].set_ylabel("Number of Papers")
        axes[1, 1].tick_params(axis="x", rotation=45)

        # Add count labels on bars
        for i, count in enumerate(eval_counts):
            axes[1, 1].text(
                i,
                count + max(eval_counts) * 0.01 if max(eval_counts) > 0 else 0.1,
                str(int(count)),
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "sampled_papers_feature_distribution.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"📊 Sampled papers feature distribution saved as '{save_path}'")

    def visualize_research_map(self, save_path: Optional[str] = None):
        """Visualize all papers on t-SNE research map"""
        if self.tsne_results is None:
            print("❌ t-SNE results not available.")
            return

        print("🗺️  Creating AI research map visualization...")

        tsne_coords = self.tsne_results["tsne_coords"]
        vectors = self.tsne_results["vectors"]
        primary_categories = self.tsne_results["primary_categories"]

        # Create figure with 3:2 aspect ratio
        fig, ax = plt.subplots(figsize=(12, 8))

        # Use brighter, more vibrant colors
        bright_colors = [
            "#FF4444",  # Bright red
            "#4444FF",  # Bright blue
            "#FF69B4",  # Hot pink
            "#00FFFF",  # Cyan
            "#8A2BE2",  # Blue violet
            "#8B4513",  # Saddle brown
            "#808080",  # Gray
            "#FFA500",  # Orange
        ]

        for i, category in enumerate(primary_categories):
            # Find papers in this category
            category_mask = vectors[:, 3 + i] == 1
            if np.any(category_mask):
                color = bright_colors[i % len(bright_colors)]
                ax.scatter(
                    tsne_coords[category_mask, 0],
                    tsne_coords[category_mask, 1],
                    c=color,
                    label=f"{category} ({np.sum(category_mask)})",
                    alpha=0.8,  # Higher alpha for better visibility
                    s=200,  # Increase point size to 200
                    edgecolors="none",  # Remove white edges
                    linewidth=0,
                )

        # Remove titles but keep axis labels blank
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Remove grid lines
        ax.grid(False)

        # Set white background and thicker black border
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(3)

        # Adjust plot limits to create a square box tightly around the data
        x_coords = tsne_coords[:, 0]
        y_coords = tsne_coords[:, 1]

        # Find the data's bounding box
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)

        # Determine the center of the data cloud
        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2

        # Determine the largest range (either width or height)
        max_range = max(x_max - x_min, y_max - y_min)

        # Define the new square limits with a 5% padding
        padding = max_range * 0.05
        new_lim_min = (
            (x_center if x_max - x_min > y_max - y_min else y_center)
            - (max_range / 2)
            - padding
        )
        new_lim_max = (
            (x_center if x_max - x_min > y_max - y_min else y_center)
            + (max_range / 2)
            + padding
        )

        ax.set_xlim(
            x_center - max_range / 2 - padding, x_center + max_range / 2 + padding
        )
        ax.set_ylim(
            y_center - max_range / 2 - padding, y_center + max_range / 2 + padding
        )

        # Let Matplotlib automatically determine the best ticks within the new limits
        ax.locator_params(axis="both", nbins=5)  # Suggests around 5 ticks
        ax.tick_params(axis="both", which="major", labelsize=10)

        # Place legend in top-left corner inside the plot with larger size
        ax.legend(
            loc="upper left",
            frameon=True,
            fancybox=False,
            edgecolor="black",
            facecolor="white",
            fontsize=16,  # Increase font size
            markerscale=1,  # Make legend markers larger
            framealpha=0.9,  # Slightly transparent background
        )

        # Keep data aspect ratio but allow figure to be 3:2
        ax.set_aspect("equal", adjustable="datalim")

        # Remove extra padding around the plot
        plt.tight_layout(pad=0.1)

        if save_path is None:
            save_path = self.output_dir / "research_map.svg"  # Save as SVG

        plt.savefig(
            save_path,
            format="svg",
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white",
        )
        plt.show()

        print(f"🗺️  Research map saved as '{save_path}'")

    def visualize_sampling_results(self, save_path: Optional[str] = None):
        """Visualize sampling results on the research map"""
        print("📊 Creating sampling results visualization...")

        tsne_coords = self.tsne_results["tsne_coords"]
        prototype_indices = self.sampling_results["prototype_indices"]
        additional_indices = self.sampling_results["additional_indices"]

        # Create figure with 1:1 aspect ratio
        fig, ax = plt.subplots(figsize=(10, 10))

        # --- PALETTE DEFINITION ---
        # Here you can easily switch between different color palettes
        # Palette 1: Professional Cool Tones
        all_papers_color = "#E0E0E0"
        prototype_color = "#C51B7D"  # Indigo
        boundary_color = "#2AA198"  # Teal

        # Plot all papers (background)
        ax.scatter(
            tsne_coords[:, 0],
            tsne_coords[:, 1],
            c=all_papers_color,
            alpha=0.7,
            s=200,  # Increase point size to 200
            label=f"All Papers ({len(tsne_coords)})",
            edgecolors="none",  # Remove white edges
            linewidth=0,
        )

        # Highlight prototype papers
        if len(prototype_indices) > 0:
            ax.scatter(
                tsne_coords[prototype_indices, 0],
                tsne_coords[prototype_indices, 1],
                c=prototype_color,
                s=200,  # Set to 200 as requested
                marker="o",
                label=f"Prototype Papers (K-Medoids, n={len(prototype_indices)})",
                edgecolors="none",  # Remove white edges
                linewidth=0,
                alpha=1.0,
                zorder=3,
            )

        # Highlight additional sampled papers
        if len(additional_indices) > 0:
            ax.scatter(
                tsne_coords[additional_indices, 0],
                tsne_coords[additional_indices, 1],
                c=boundary_color,
                s=200,  # Set to 200 as requested
                marker="o",
                label=f"Boundary Papers (FPS, n={len(additional_indices)})",
                edgecolors="none",  # Remove white edges
                linewidth=0,
                alpha=0.9,
                zorder=2,
            )

        # --- AXIS AND LAYOUT CORRECTION (APPLYING THE FIX) ---

        # Remove titles but keep axis labels blank
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Remove grid lines
        ax.grid(False)

        # Set white background and thicker black border
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(2)

        # Adjust plot limits to create a square box tightly around the data
        x_coords = tsne_coords[:, 0]
        y_coords = tsne_coords[:, 1]

        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)

        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2

        max_range = max(x_max - x_min, y_max - y_min)

        padding = max_range * 0.05
        ax.set_xlim(
            x_center - max_range / 2 - padding, x_center + max_range / 2 + padding
        )
        ax.set_ylim(
            y_center - max_range / 2 - padding, y_center + max_range / 2 + padding
        )

        # Let Matplotlib automatically determine the best ticks
        ax.locator_params(axis="both", nbins=5)
        ax.tick_params(axis="both", which="major", labelsize=10)

        # --- END OF FIX ---

        # Place legend with larger size
        ax.legend(
            loc="upper right",
            frameon=True,
            fancybox=False,
            edgecolor="black",
            facecolor="white",
            fontsize=16,  # Larger font size
            markerscale=2.0,  # Larger legend markers
            framealpha=0.9,  # Slightly transparent background
        )

        # Keep data aspect ratio but allow figure to be 3:2
        ax.set_aspect("equal", adjustable="datalim")

        plt.tight_layout(pad=0.1)

        if save_path is None:
            # In a class, this would be: self.output_dir / "..."
            save_path = self.output_dir / "sampling_results.svg"

        plt.savefig(
            save_path,
            format="svg",
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white",
        )
        plt.show()

        print(f"📊 Sampling results visualization saved as '{save_path}'")

    def create_all_visualizations(self):
        """Create all four visualizations"""
        print("🎨 Creating all visualizations...")
        print("=" * 60)

        # 1. All papers feature distribution
        self.visualize_all_papers_features()

        # 2. Sampled papers feature distribution
        self.visualize_sampled_papers_features()

        # 3. Research map (all papers)
        self.visualize_research_map()

        # 4. Sampling results on research map
        self.visualize_sampling_results()

        print("\n🎉 All visualizations completed!")
        print("📁 Check the following files:")
        print(f"  - {self.output_dir}/all_papers_feature_distribution.png")
        print(f"  - {self.output_dir}/sampled_papers_feature_distribution.png")
        print(f"  - {self.output_dir}/research_map.svg")
        print(f"  - {self.output_dir}/sampling_results.svg")


def main():
    """Main function for visualization"""
    parser = argparse.ArgumentParser(description="Visualize paper sampling results")
    parser.add_argument(
        "--output-dir",
        default="sampling_outputs",
        help="Directory containing saved data files (default: sampling_outputs)",
    )
    parser.add_argument(
        "--vis-type",
        choices=["all", "features", "sampled-features", "map", "sampling"],
        default="all",
        help="Type of visualization to create (default: all)",
    )

    args = parser.parse_args()

    print("🎨 Paper Sampling Visualizer")
    print("=" * 50)

    # Initialize visualizer
    visualizer = SamplingVisualizer(args.output_dir)

    # Check if data is available
    if visualizer.all_papers_data is None:
        print("❌ No data available. Please run paper_sampling.py first.")
        return

    # Create visualizations based on type
    if args.vis_type == "all":
        visualizer.create_all_visualizations()
    elif args.vis_type == "features":
        visualizer.visualize_all_papers_features()
    elif args.vis_type == "sampled-features":
        visualizer.visualize_sampled_papers_features()
    elif args.vis_type == "map":
        visualizer.visualize_research_map()
    elif args.vis_type == "sampling":
        visualizer.visualize_sampling_results()


if __name__ == "__main__":
    main()

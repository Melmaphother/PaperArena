#!/usr/bin/env python3
"""
Configuration loader for Paper QA evaluation system
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from loguru import logger


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs"""

    # Model configuration
    model_name: str = "gpt-4.1"
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 4096

    # Search and retrieval parameters
    max_search_results: int = 10
    retrieval_top_k: int = 5

    # Agent configuration
    max_steps: int = 15
    verbosity_level: int = 1

    # Threading
    num_threads: int = 1

    # Directories
    results_dir: str = "results"

    # Filtering (optional)
    filter_conference: Optional[str] = None
    filter_difficulty: Optional[str] = None
    filter_qa_type: Optional[str] = None


class ConfigLoader:
    """
    Loads and manages evaluation configurations from YAML files
    """

    def __init__(self, config_dir: str = "configs"):
        """
        Initialize config loader

        Args:
            config_dir: Directory containing config files
        """
        self.config_dir = Path(config_dir)
        logger.info(f"🔧 ConfigLoader initialized with directory: {self.config_dir}")

    def load_config(self, config_name: str) -> EvaluationConfig:
        """
        Load configuration from YAML file

        Args:
            config_name: Name of config file (with or without .yaml extension)

        Returns:
            EvaluationConfig: Loaded configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        # Add .yaml extension if not present
        if not config_name.endswith(".yaml"):
            config_name += ".yaml"

        config_path = self.config_dir / config_name

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            if not config_data:
                config_data = {}

            # Create config object with loaded data
            config = EvaluationConfig()

            # Update fields from YAML
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    logger.warning(f"⚠️ Unknown config key: {key}")

            logger.info(f"✅ Loaded config: {config_name} (model: {config.model_name})")
            return config

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file {config_path}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load config {config_path}: {e}")

    def list_available_configs(self) -> List[str]:
        """
        List all available configuration files

        Returns:
            List of config file names (without .yaml extension)
        """
        if not self.config_dir.exists():
            return []

        configs = []
        for config_file in self.config_dir.glob("*.yaml"):
            configs.append(config_file.stem)

        return sorted(configs)

    def create_config_template(self, output_path: str) -> None:
        """
        Create a template configuration file

        Args:
            output_path: Path for the template file
        """
        template_config = EvaluationConfig()
        config_dict = {
            "model_name": template_config.model_name,
            "temperature": template_config.temperature,
            "top_p": template_config.top_p,
            "max_tokens": template_config.max_tokens,
            "max_search_results": template_config.max_search_results,
            "retrieval_top_k": template_config.retrieval_top_k,
            "max_steps": template_config.max_steps,
            "verbosity_level": template_config.verbosity_level,
            "num_threads": template_config.num_threads,
            "results_dir": template_config.results_dir,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Template configuration for Paper QA evaluation\n")
            f.write("# Copy and modify this file to create custom configurations\n\n")
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            f.write("\n# Optional filtering (uncomment to use)\n")
            f.write('# filter_conference: "icml2025"\n')
            f.write('# filter_difficulty: "Hard"\n')
            f.write('# filter_qa_type: "Open Answer"\n')

        logger.info(f"✅ Created config template: {output_path}")

    def validate_config(self, config: EvaluationConfig) -> tuple[bool, List[str]]:
        """
        Validate configuration parameters

        Args:
            config: Configuration to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Validate model name
        available_models = [
            "gemini-2.5-pro",
            "o4-mini",
            "claude-sonnet-4",
            "gpt-4.1",
            "claude-3-5",
            "qwen3-235b-thinking",
            "glm-4.5",
            "qwen3-235b-instruct",
            "kimi-k2",
        ]

        if config.model_name not in available_models:
            errors.append(
                f"Unknown model: {config.model_name}. Available: {available_models}"
            )

        # Validate numeric parameters
        if not 0 <= config.temperature <= 2:
            errors.append(
                f"Temperature must be between 0 and 2, got: {config.temperature}"
            )

        if not 0 <= config.top_p <= 1:
            errors.append(f"top_p must be between 0 and 1, got: {config.top_p}")

        if config.max_tokens <= 0:
            errors.append(f"max_tokens must be positive, got: {config.max_tokens}")

        if config.max_steps <= 0:
            errors.append(f"max_steps must be positive, got: {config.max_steps}")

        if config.num_threads <= 0:
            errors.append(f"num_threads must be positive, got: {config.num_threads}")

        # Validate optional filters
        if config.filter_difficulty and config.filter_difficulty not in [
            "Easy",
            "Medium",
            "Hard",
        ]:
            errors.append(f"Invalid difficulty filter: {config.filter_difficulty}")

        if config.filter_qa_type and config.filter_qa_type not in [
            "Multi-Choice Answer",
            "Concise Answer",
            "Open Answer",
        ]:
            errors.append(f"Invalid qa_type filter: {config.filter_qa_type}")

        return len(errors) == 0, errors


def main():
    """Test the config loader"""
    loader = ConfigLoader()

    # List available configs
    configs = loader.list_available_configs()
    print(f"📋 Available configs: {configs}")

    # Test loading a config
    if configs:
        config = loader.load_config(configs[0])
        print(f"✅ Loaded config: {config.model_name}")

        # Validate config
        is_valid, errors = loader.validate_config(config)
        if is_valid:
            print("✅ Config is valid")
        else:
            print(f"❌ Config errors: {errors}")

    # Create template
    loader.create_config_template("template_config.yaml")


if __name__ == "__main__":
    main()

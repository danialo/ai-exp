"""
Extractor version computation for HTN Self-Belief Decomposer.

Computes a stable hash that changes when:
- Code version changes
- Any prompt template changes
- Epistemics cue table changes
- Model IDs change

Does NOT change when:
- Thresholds change (those don't invalidate extracted data)
"""

import hashlib
import json
from pathlib import Path
from typing import Optional

from src.utils.belief_config import BeliefSystemConfig, load_belief_config


# Increment this when making breaking changes to extraction logic
# 1.0.0 - Initial version
# 1.1.0 - Fixed atomizer: stronger prompts, balanced bracket parsing, garbage validation
# 1.2.0 - Fixed segmenter noun-phrase splits, added non-belief pattern filters
# 1.3.0 - Fixed atomizer first-person check: startswith -> contains (matches segmenter)
EXTRACTOR_CODE_VERSION = "1.3.0"


def _compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file's contents."""
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")

    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def _compute_cues_hash(config: BeliefSystemConfig) -> str:
    """Compute hash of the epistemics cue configuration."""
    cues = config.epistemics.cues

    # Build deterministic representation
    cue_data = {
        'negation': sorted(cues.negation),
        'modality': {
            'possible': sorted(cues.modality.possible),
            'likely': sorted(cues.modality.likely),
            'unsure': sorted(cues.modality.unsure),
        },
        'past': sorted(cues.past),
        'transitional': sorted(cues.transitional),
        'habitual_strong': sorted(cues.habitual_strong),
        'habitual_soft': sorted(cues.habitual_soft),
        'ongoing': sorted(cues.ongoing),
        'state': sorted(cues.state),
    }

    # Also include degree cues
    cue_data['degree'] = {
        'strong': sorted(config.epistemics.degree.strong),
        'moderate': sorted(config.epistemics.degree.moderate),
        'weak': sorted(config.epistemics.degree.weak),
    }

    json_str = json.dumps(cue_data, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()


def _compute_models_hash(config: BeliefSystemConfig) -> str:
    """Compute hash of model identifiers."""
    model_data = {
        'atomizer_model': config.extractor.atomizer_model,
        'epistemics_model': config.extractor.epistemics_model,
        'verifier_model': config.extractor.verifier_model,
        'embedding_model': config.embeddings.model,
    }

    json_str = json.dumps(model_data, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()


def get_extractor_version(config: Optional[BeliefSystemConfig] = None) -> str:
    """
    Compute a stable hash that identifies the extractor version.

    The version hash incorporates:
    - EXTRACTOR_CODE_VERSION constant
    - All prompt file contents (atomizer_system, atomizer_user, repair_json,
      epistemics_fallback, verifier)
    - Epistemics cue table (determines rule-based extraction)
    - Model identifiers (LLM outputs may differ between models)

    Changing thresholds does NOT change the version (those affect downstream
    processing but not the extracted data itself).

    Args:
        config: Optional config object. If not provided, loads from default path.

    Returns:
        16-character hexadecimal version string
    """
    if config is None:
        config = load_belief_config()

    # Find project root for prompt files
    current = Path(__file__).resolve()
    project_root = current.parent.parent.parent  # src/utils -> src -> project

    # Compute prompt hashes
    prompt_files = [
        project_root / config.prompts.atomizer_system,
        project_root / config.prompts.atomizer_user,
        project_root / config.prompts.repair_json,
        project_root / config.prompts.epistemics_fallback,
        project_root / config.prompts.verifier,
    ]

    prompt_hashes = []
    for pf in prompt_files:
        prompt_hashes.append(_compute_file_hash(pf))

    # Compute component hashes
    cues_hash = _compute_cues_hash(config)
    models_hash = _compute_models_hash(config)

    # Combine all components
    combined = {
        'code_version': EXTRACTOR_CODE_VERSION,
        'prompt_hashes': prompt_hashes,
        'cues_hash': cues_hash,
        'models_hash': models_hash,
    }

    json_str = json.dumps(combined, sort_keys=True)
    full_hash = hashlib.sha256(json_str.encode()).hexdigest()

    # Return truncated hash (16 chars = 64 bits, plenty for version ID)
    return full_hash[:16]


def get_version_components(config: Optional[BeliefSystemConfig] = None) -> dict:
    """
    Get detailed breakdown of version components for debugging.

    Returns dict with:
    - code_version: The EXTRACTOR_CODE_VERSION constant
    - prompt_files: Dict mapping prompt name to file hash
    - cues_hash: Hash of epistemics cues
    - models_hash: Hash of model identifiers
    - full_version: The complete 16-char version string
    """
    if config is None:
        config = load_belief_config()

    current = Path(__file__).resolve()
    project_root = current.parent.parent.parent

    prompt_details = {}
    for name, path_attr in [
        ('atomizer_system', config.prompts.atomizer_system),
        ('atomizer_user', config.prompts.atomizer_user),
        ('repair_json', config.prompts.repair_json),
        ('epistemics_fallback', config.prompts.epistemics_fallback),
        ('verifier', config.prompts.verifier),
    ]:
        full_path = project_root / path_attr
        prompt_details[name] = {
            'path': str(path_attr),
            'hash': _compute_file_hash(full_path)[:16],
        }

    return {
        'code_version': EXTRACTOR_CODE_VERSION,
        'prompt_files': prompt_details,
        'cues_hash': _compute_cues_hash(config)[:16],
        'models_hash': _compute_models_hash(config)[:16],
        'full_version': get_extractor_version(config),
    }


if __name__ == '__main__':
    # Quick test
    version = get_extractor_version()
    print(f"Extractor version: {version}")

    components = get_version_components()
    print("\nVersion components:")
    print(f"  Code version: {components['code_version']}")
    print(f"  Cues hash: {components['cues_hash']}")
    print(f"  Models hash: {components['models_hash']}")
    print("  Prompt files:")
    for name, details in components['prompt_files'].items():
        print(f"    {name}: {details['hash']} ({details['path']})")

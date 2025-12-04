"""
Configuration loader for HTN Self-Belief Decomposer.

Loads system_config.yaml and provides typed access to all hyperparameters.
"""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing required keys."""
    pass


@dataclass
class ExtractorConfig:
    atomizer_model: str
    epistemics_model: str
    verifier_model: str
    temperature: float
    max_json_repair_attempts: int


@dataclass
class VADConfig:
    enabled: bool
    arousal_field: str
    arousal_weight: float


@dataclass
class HeuristicFallbackConfig:
    enabled: bool
    profanity_penalty: float
    caps_ratio_threshold: float
    caps_penalty: float
    exclaim_density_threshold: float
    exclaim_penalty: float


@dataclass
class SourceContextConfig:
    mode_field: Optional[str]
    mode_weights: Dict[str, float]
    vad: VADConfig
    heuristic_fallback: HeuristicFallbackConfig


@dataclass
class ContextConfig:
    strategy: str
    fallback: str


@dataclass
class PromptsConfig:
    atomizer_system: str
    atomizer_user: str
    repair_json: str
    epistemics_fallback: str
    verifier: str


@dataclass
class AtomizerConfig:
    json_schema_version: str


@dataclass
class ModalityCues:
    possible: List[str]
    likely: List[str]
    unsure: List[str]


@dataclass
class CuesConfig:
    negation: List[str]
    modality: ModalityCues
    past: List[str]
    transitional: List[str]
    habitual_strong: List[str]
    habitual_soft: List[str]
    ongoing: List[str]
    state: List[str]


@dataclass
class DegreeConfig:
    strong: List[str]
    moderate: List[str]
    weak: List[str]


@dataclass
class EpistemicsConfig:
    llm_fallback_threshold: float
    cue_conflict_resolution: str
    default_temporal_scope: str
    default_modality: str
    modality_confidence_caps: Dict[str, float]
    temporal_specificity: Dict[str, int]
    cues: CuesConfig
    degree: DegreeConfig
    degree_values: Dict[str, float]


@dataclass
class StreamMapping:
    primary: str
    secondary: Optional[str]
    confidence: float


@dataclass
class StreamsConfig:
    types: List[str]
    mapping: Dict[str, Dict[str, StreamMapping]]


@dataclass
class EmbeddingsConfig:
    enabled: bool
    model: str
    dimension: int
    batch_size: int
    linear_scan_max_nodes: int
    fallback_to_text_similarity: bool
    text_similarity_method: str


@dataclass
class TentativeLinkConfidenceParams:
    a: float
    b: float
    c: float


@dataclass
class TentativeLinkConfig:
    auto_accept_threshold: float
    auto_reject_threshold: float
    confidence_params: TentativeLinkConfidenceParams
    age_definition: str


@dataclass
class VerifierConfig:
    enabled: bool
    trigger_band: List[float]


@dataclass
class TensionConfig:
    enabled: bool
    embedding_threshold: float
    top_k_conflict_check: int


@dataclass
class ResolutionConfig:
    top_k: int
    match_threshold: float
    no_match_threshold: float
    verifier: VerifierConfig
    tentative_link: TentativeLinkConfig
    tension: TensionConfig


@dataclass
class ConcurrencyConfig:
    strategy: str
    max_retries: int
    retry_delay_ms: int


@dataclass
class ConflictPenaltyConfig:
    enabled: bool
    recent_window_days: int
    weight: float


@dataclass
class StatusThresholdsConfig:
    developing: float
    core: float


@dataclass
class ScoringConfig:
    half_life_days: Dict[str, int]
    support: Dict[str, float]
    spread: Dict[str, float]
    diversity: Dict[str, float]
    conflict_penalty: ConflictPenaltyConfig
    status_thresholds: StatusThresholdsConfig


@dataclass
class PromoteStateToIdentityConfig:
    enabled: bool
    strategy: str
    min_spread: float
    min_diversity: float
    min_activation: float


@dataclass
class RatchetConfig:
    enabled: bool
    allow_demotion: bool
    demotion_triggers: List[str]


@dataclass
class MigrationConfig:
    promote_state_to_identity: PromoteStateToIdentityConfig
    ratchet: RatchetConfig


@dataclass
class BackfillConfig:
    batch_size: int
    checkpoint_every: int
    checkpoint_file: str
    resume_strategy: str
    continue_on_error: bool


@dataclass
class EvalEventsConfig:
    enabled: bool
    path: str
    format: str


@dataclass
class LoggingConfig:
    eval_events: EvalEventsConfig
    level: str


@dataclass
class BeliefSystemConfig:
    """Root configuration object for the HTN Self-Belief Decomposer."""
    extractor: ExtractorConfig
    source_context: SourceContextConfig
    context: ContextConfig
    prompts: PromptsConfig
    atomizer: AtomizerConfig
    epistemics: EpistemicsConfig
    streams: StreamsConfig
    embeddings: EmbeddingsConfig
    resolution: ResolutionConfig
    concurrency: ConcurrencyConfig
    scoring: ScoringConfig
    migration: MigrationConfig
    backfill: BackfillConfig
    logging: LoggingConfig

    # Store raw config for hash computation
    _raw_config: Dict[str, Any] = field(default_factory=dict, repr=False)


def _get_nested(data: Dict[str, Any], key: str, required: bool = True) -> Any:
    """Get nested key from dict, raising ConfigurationError if required and missing."""
    keys = key.split('.')
    value = data
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        elif required:
            raise ConfigurationError(f"Missing required configuration key: {key}")
        else:
            return None
    return value


def _parse_cues_config(data: Dict[str, Any]) -> CuesConfig:
    """Parse the cues configuration section."""
    return CuesConfig(
        negation=data.get('negation', []),
        modality=ModalityCues(
            possible=data.get('modality', {}).get('possible', []),
            likely=data.get('modality', {}).get('likely', []),
            unsure=data.get('modality', {}).get('unsure', []),
        ),
        past=data.get('past', []),
        transitional=data.get('transitional', []),
        habitual_strong=data.get('habitual_strong', []),
        habitual_soft=data.get('habitual_soft', []),
        ongoing=data.get('ongoing', []),
        state=data.get('state', []),
    )


def _parse_stream_mapping(data: Dict[str, Any]) -> Dict[str, Dict[str, StreamMapping]]:
    """Parse the streams mapping configuration."""
    result = {}
    for belief_type, temporal_map in data.items():
        result[belief_type] = {}
        for temporal_scope, mapping in temporal_map.items():
            result[belief_type][temporal_scope] = StreamMapping(
                primary=mapping.get('primary', 'identity'),
                secondary=mapping.get('secondary'),
                confidence=mapping.get('confidence', 0.5),
            )
    return result


def load_belief_config(config_path: Optional[str] = None) -> BeliefSystemConfig:
    """
    Load and validate the belief system configuration.

    Args:
        config_path: Path to config file. Defaults to config/system_config.yaml

    Returns:
        BeliefSystemConfig object with all configuration

    Raises:
        ConfigurationError: If config file is missing or invalid
    """
    if config_path is None:
        # Find config relative to project root
        current = Path(__file__).resolve()
        project_root = current.parent.parent.parent  # src/utils -> src -> project
        config_path = project_root / "config" / "system_config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in configuration file: {e}")

    if not isinstance(raw, dict):
        raise ConfigurationError("Configuration file must contain a YAML mapping")

    # Validate required top-level sections
    required_sections = [
        'extractor', 'source_context', 'context', 'prompts', 'atomizer',
        'epistemics', 'streams', 'embeddings', 'resolution', 'concurrency',
        'scoring', 'migration', 'backfill', 'logging'
    ]

    for section in required_sections:
        if section not in raw:
            raise ConfigurationError(f"Missing required configuration section: {section}")

    # Parse each section
    extractor = ExtractorConfig(
        atomizer_model=raw['extractor']['atomizer_model'],
        epistemics_model=raw['extractor']['epistemics_model'],
        verifier_model=raw['extractor']['verifier_model'],
        temperature=raw['extractor']['temperature'],
        max_json_repair_attempts=raw['extractor']['max_json_repair_attempts'],
    )

    source_context = SourceContextConfig(
        mode_field=raw['source_context'].get('mode_field'),
        mode_weights=raw['source_context']['mode_weights'],
        vad=VADConfig(
            enabled=raw['source_context']['vad']['enabled'],
            arousal_field=raw['source_context']['vad']['arousal_field'],
            arousal_weight=raw['source_context']['vad']['arousal_weight'],
        ),
        heuristic_fallback=HeuristicFallbackConfig(
            enabled=raw['source_context']['heuristic_fallback']['enabled'],
            profanity_penalty=raw['source_context']['heuristic_fallback']['profanity_penalty'],
            caps_ratio_threshold=raw['source_context']['heuristic_fallback']['caps_ratio_threshold'],
            caps_penalty=raw['source_context']['heuristic_fallback']['caps_penalty'],
            exclaim_density_threshold=raw['source_context']['heuristic_fallback']['exclaim_density_threshold'],
            exclaim_penalty=raw['source_context']['heuristic_fallback']['exclaim_penalty'],
        ),
    )

    context = ContextConfig(
        strategy=raw['context']['strategy'],
        fallback=raw['context']['fallback'],
    )

    prompts = PromptsConfig(
        atomizer_system=raw['prompts']['atomizer_system'],
        atomizer_user=raw['prompts']['atomizer_user'],
        repair_json=raw['prompts']['repair_json'],
        epistemics_fallback=raw['prompts']['epistemics_fallback'],
        verifier=raw['prompts']['verifier'],
    )

    atomizer = AtomizerConfig(
        json_schema_version=raw['atomizer']['json_schema_version'],
    )

    epistemics = EpistemicsConfig(
        llm_fallback_threshold=raw['epistemics']['llm_fallback_threshold'],
        cue_conflict_resolution=raw['epistemics']['cue_conflict_resolution'],
        default_temporal_scope=raw['epistemics']['default_temporal_scope'],
        default_modality=raw['epistemics']['default_modality'],
        modality_confidence_caps=raw['epistemics']['modality_confidence_caps'],
        temporal_specificity=raw['epistemics']['temporal_specificity'],
        cues=_parse_cues_config(raw['epistemics']['cues']),
        degree=DegreeConfig(
            strong=raw['epistemics']['degree']['strong'],
            moderate=raw['epistemics']['degree']['moderate'],
            weak=raw['epistemics']['degree']['weak'],
        ),
        degree_values=raw['epistemics']['degree_values'],
    )

    streams = StreamsConfig(
        types=raw['streams']['types'],
        mapping=_parse_stream_mapping(raw['streams']['mapping']),
    )

    embeddings = EmbeddingsConfig(
        enabled=raw['embeddings']['enabled'],
        model=raw['embeddings']['model'],
        dimension=raw['embeddings']['dimension'],
        batch_size=raw['embeddings']['batch_size'],
        linear_scan_max_nodes=raw['embeddings']['linear_scan_max_nodes'],
        fallback_to_text_similarity=raw['embeddings']['fallback_to_text_similarity'],
        text_similarity_method=raw['embeddings']['text_similarity_method'],
    )

    resolution = ResolutionConfig(
        top_k=raw['resolution']['top_k'],
        match_threshold=raw['resolution']['match_threshold'],
        no_match_threshold=raw['resolution']['no_match_threshold'],
        verifier=VerifierConfig(
            enabled=raw['resolution']['verifier']['enabled'],
            trigger_band=raw['resolution']['verifier']['trigger_band'],
        ),
        tentative_link=TentativeLinkConfig(
            auto_accept_threshold=raw['resolution']['tentative_link']['auto_accept_threshold'],
            auto_reject_threshold=raw['resolution']['tentative_link']['auto_reject_threshold'],
            confidence_params=TentativeLinkConfidenceParams(
                a=raw['resolution']['tentative_link']['confidence_params']['a'],
                b=raw['resolution']['tentative_link']['confidence_params']['b'],
                c=raw['resolution']['tentative_link']['confidence_params']['c'],
            ),
            age_definition=raw['resolution']['tentative_link']['age_definition'],
        ),
        tension=TensionConfig(
            enabled=raw['resolution']['tension']['enabled'],
            embedding_threshold=raw['resolution']['tension']['embedding_threshold'],
            top_k_conflict_check=raw['resolution']['tension']['top_k_conflict_check'],
        ),
    )

    concurrency = ConcurrencyConfig(
        strategy=raw['concurrency']['strategy'],
        max_retries=raw['concurrency']['max_retries'],
        retry_delay_ms=raw['concurrency']['retry_delay_ms'],
    )

    scoring = ScoringConfig(
        half_life_days=raw['scoring']['half_life_days'],
        support=raw['scoring']['support'],
        spread=raw['scoring']['spread'],
        diversity=raw['scoring']['diversity'],
        conflict_penalty=ConflictPenaltyConfig(
            enabled=raw['scoring']['conflict_penalty']['enabled'],
            recent_window_days=raw['scoring']['conflict_penalty']['recent_window_days'],
            weight=raw['scoring']['conflict_penalty']['weight'],
        ),
        status_thresholds=StatusThresholdsConfig(
            developing=raw['scoring']['status_thresholds']['developing'],
            core=raw['scoring']['status_thresholds']['core'],
        ),
    )

    migration = MigrationConfig(
        promote_state_to_identity=PromoteStateToIdentityConfig(
            enabled=raw['migration']['promote_state_to_identity']['enabled'],
            strategy=raw['migration']['promote_state_to_identity']['strategy'],
            min_spread=raw['migration']['promote_state_to_identity']['min_spread'],
            min_diversity=raw['migration']['promote_state_to_identity']['min_diversity'],
            min_activation=raw['migration']['promote_state_to_identity']['min_activation'],
        ),
        ratchet=RatchetConfig(
            enabled=raw['migration']['ratchet']['enabled'],
            allow_demotion=raw['migration']['ratchet']['allow_demotion'],
            demotion_triggers=raw['migration']['ratchet']['demotion_triggers'],
        ),
    )

    backfill = BackfillConfig(
        batch_size=raw['backfill']['batch_size'],
        checkpoint_every=raw['backfill']['checkpoint_every'],
        checkpoint_file=raw['backfill']['checkpoint_file'],
        resume_strategy=raw['backfill']['resume_strategy'],
        continue_on_error=raw['backfill']['continue_on_error'],
    )

    logging_config = LoggingConfig(
        eval_events=EvalEventsConfig(
            enabled=raw['logging']['eval_events']['enabled'],
            path=raw['logging']['eval_events']['path'],
            format=raw['logging']['eval_events']['format'],
        ),
        level=raw['logging']['level'],
    )

    config = BeliefSystemConfig(
        extractor=extractor,
        source_context=source_context,
        context=context,
        prompts=prompts,
        atomizer=atomizer,
        epistemics=epistemics,
        streams=streams,
        embeddings=embeddings,
        resolution=resolution,
        concurrency=concurrency,
        scoring=scoring,
        migration=migration,
        backfill=backfill,
        logging=logging_config,
    )

    # Store raw config for version hashing
    config._raw_config = raw

    return config


# Singleton instance for convenience
_config_instance: Optional[BeliefSystemConfig] = None


def get_belief_config() -> BeliefSystemConfig:
    """Get the singleton configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = load_belief_config()
    return _config_instance


def reload_belief_config(config_path: Optional[str] = None) -> BeliefSystemConfig:
    """Reload configuration from disk."""
    global _config_instance
    _config_instance = load_belief_config(config_path)
    return _config_instance

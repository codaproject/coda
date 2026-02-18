"""
Kaggle Ambient Scribe Benchmark for CODA ICD-10 Annotation

Compares ICD-10 annotations between reference transcripts and
Whisper-transcribed audio (from benchmark_asr.py output).

Reads whisper transcripts from benchmark_asr.py's CSV (hyp_text column),
annotates both reference and whisper text with RAGGrounder, and
computes strict and hierarchical code-level metrics.

Usage:
  python benchmark.py --model tiny
  python benchmark.py --model base --asr_csv path/to/results.csv
"""

import csv
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

import kagglehub
import pystow

# Whisper model size: "tiny", "base", "small", "medium", "large"
WHISPER_MODEL_SIZE = "tiny"

# Data paths - kagglehub handles downloading and caching the dataset
DATA_BASE = Path(kagglehub.dataset_download("imeritinc/multilingual-ambient-scribe-dataset")) / \
    "iMerit_Multilingual_Ambient_Scribe_Dataset/UK English"
AUDIO_DIR = DATA_BASE / "audio"
TRANSCRIPTS_DIR = DATA_BASE / "transcripts"

# Pystow cache for benchmark results
RESULTS_BASE = pystow.module("coda", "benchmarks", "kaggle_ambient_scribe")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -------------------------------
# Data access helpers
# -------------------------------

def get_encounter_ids() -> List[int]:
    """Get list of encounter IDs from audio files."""
    pattern = re.compile(r"recording_uk_encounter_(\d+)\.mp3")
    ids = []
    for f in AUDIO_DIR.iterdir():
        match = pattern.match(f.name)
        if match:
            ids.append(int(match.group(1)))
    return sorted(ids)


def get_reference_transcript_path(encounter_id: int) -> Path:
    """Get path to reference transcript for an encounter."""
    return TRANSCRIPTS_DIR / f"Encounter {encounter_id}_UK.txt"


def load_reference_transcript(encounter_id: int) -> str:
    """Load reference transcript text."""
    path = get_reference_transcript_path(encounter_id)
    return path.read_text()


def get_default_asr_csv(model_size: str) -> Path:
    """Get the default ASR results CSV path from pystow cache."""
    return RESULTS_BASE.join("results", name=f"asr_benchmark_results.whisper-{model_size}.csv")


def load_whisper_transcripts_from_csv(csv_path: Path) -> Dict[int, str]:
    """Load whisper transcripts from benchmark_asr.py output CSV.

    Returns a dict mapping encounter_id -> hyp_text.
    """
    transcripts = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            enc_id = row.get("encounter_id", "")
            if not enc_id or enc_id == "" or row.get("audio_file") == "__AGGREGATE__":
                continue
            transcripts[int(enc_id)] = row.get("hyp_text", "")
    return transcripts


# -------------------------------
# Annotation caching
# -------------------------------

def get_annotation_cache_path(encounter_id: int, source: str, model_size: str = None) -> Path:
    """Get cache path for ICD-10 annotations."""
    if source == "reference":
        return RESULTS_BASE.join("annotations-reference", name=f"encounter_{encounter_id}_annotations.json")
    else:
        return RESULTS_BASE.join(f"annotations-whisper-{model_size}", name=f"encounter_{encounter_id}_annotations.json")


def load_cached_annotations(encounter_id: int, source: str, model_size: str = None) -> Optional[List[Dict]]:
    """Load cached annotations if they exist."""
    path = get_annotation_cache_path(encounter_id, source, model_size)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def save_annotations(encounter_id: int, source: str, annotations: List[Dict], model_size: str = None):
    """Save annotations to cache."""
    path = get_annotation_cache_path(encounter_id, source, model_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(annotations, f, indent=2)


def annotations_to_serializable(annotations) -> List[Dict]:
    """Convert gilda Annotation objects to serializable dicts."""
    result = []
    for ann in annotations:
        ann_dict = {
            "text": ann.text,
            "start": ann.start,
            "end": ann.end,
            "matches": []
        }
        for match in ann.matches:
            match_dict = {
                "code": match.term.id,
                "name": match.term.entry_name,
                "score": match.score,
                "db": match.term.db
            }
            ann_dict["matches"].append(match_dict)
        result.append(ann_dict)
    return result


def annotate_text(text: str, encounter_id: int, source: str, model_size: str = None, grounder=None) -> List[Dict]:
    """Annotate text with ICD-10 codes using RAGGrounder, with caching.

    Parameters
    ----------
    text : str
        Text to annotate
    encounter_id : int
        Encounter number for caching
    source : str
        "reference" or "whisper"
    model_size : str, optional
        Whisper model size (required if source is "whisper")
    grounder : RAGGrounder, optional
        Pre-initialized grounder instance. If None, creates a new one.
    """
    # Check cache first
    cached = load_cached_annotations(encounter_id, source, model_size)
    if cached is not None:
        logger.info(f"Encounter {encounter_id}: Using cached {source} annotations")
        return cached

    logger.info(f"Encounter {encounter_id}: Annotating {source} text with ICD-10 codes")

    # Use provided grounder or create new one
    if grounder is None:
        from coda.grounding.icd10_rag_grounder import RAGGrounder
        grounder = RAGGrounder()

    annotations = grounder.annotate(text)

    # Convert to serializable format
    annotations_data = annotations_to_serializable(annotations)

    # Cache the result
    save_annotations(encounter_id, source, annotations_data, model_size)

    return annotations_data


def extract_top_codes(annotations: List[Dict]) -> Set[str]:
    """Extract set of top-ranked ICD-10 codes from annotations.

    Each annotation span has a list of matches. We take the top (first) code
    from each annotation's matches list.
    """
    codes = set()
    for ann in annotations:
        matches = ann.get("matches", [])
        if matches:
            # Top-ranked code is the first one
            top_code = matches[0].get("code", "")
            if top_code:
                codes.add(top_code)
    return codes


def normalize_code_to_parent(code: str) -> str:
    """Normalize ICD-10 code to parent level by removing decimal part.

    Examples:
        G47.0 -> G47
        E11.9 -> E11
        I20.1 -> I20
        C50 -> C50 (unchanged)
    """
    if '.' in code:
        return code.split('.')[0]
    return code


def normalize_codes_to_parent(codes: Set[str]) -> Set[str]:
    """Normalize a set of codes to their parent level."""
    return {normalize_code_to_parent(c) for c in codes}


def compute_metrics(reference_codes: Set[str], predicted_codes: Set[str]) -> Dict[str, float]:
    """Compute precision, recall, F1, and Jaccard similarity.

    Parameters
    ----------
    reference_codes : set
        Set of ICD-10 codes from reference transcript
    predicted_codes : set
        Set of ICD-10 codes from whisper transcript

    Returns
    -------
    dict
        Dictionary with precision, recall, f1, jaccard metrics
    """
    if not reference_codes and not predicted_codes:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "jaccard": 1.0}

    if not predicted_codes:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "jaccard": 0.0}

    if not reference_codes:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "jaccard": 0.0}

    intersection = reference_codes & predicted_codes
    union = reference_codes | predicted_codes

    precision = len(intersection) / len(predicted_codes)
    recall = len(intersection) / len(reference_codes)

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    jaccard = len(intersection) / len(union) if union else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard
    }


def compute_all_metrics(reference_codes: Set[str], predicted_codes: Set[str]) -> Dict[str, Dict[str, float]]:
    """Compute both strict and hierarchical metrics.

    Strict: Exact code matching (e.g., G47.0 != G47)
    Hierarchical: Parent-level matching (e.g., G47.0 == G47)

    Parameters
    ----------
    reference_codes : set
        Set of ICD-10 codes from reference transcript
    predicted_codes : set
        Set of ICD-10 codes from whisper transcript

    Returns
    -------
    dict
        Dictionary with 'strict' and 'hierarchical' sub-dicts,
        each containing precision, recall, f1, jaccard metrics
    """
    # Strict metrics (exact match)
    strict = compute_metrics(reference_codes, predicted_codes)

    # Hierarchical metrics (parent-level match)
    ref_normalized = normalize_codes_to_parent(reference_codes)
    pred_normalized = normalize_codes_to_parent(predicted_codes)
    hierarchical = compute_metrics(ref_normalized, pred_normalized)

    return {
        "strict": strict,
        "hierarchical": hierarchical
    }


def run_benchmark(model_size: str = WHISPER_MODEL_SIZE, asr_csv: str = None) -> Dict:
    """Run the ICD-10 annotation benchmark.

    Parameters
    ----------
    model_size : str
        Whisper model size (used for cache paths and to find ASR CSV)
    asr_csv : str, optional
        Path to benchmark_asr.py output CSV. If None, looks in pystow cache.

    Returns
    -------
    dict
        Benchmark results including per-encounter and aggregate metrics
    """
    # Load whisper transcripts from ASR benchmark CSV
    if asr_csv:
        csv_path = Path(asr_csv)
    else:
        csv_path = get_default_asr_csv(model_size)

    if not csv_path.exists():
        logger.error(f"ASR results CSV not found: {csv_path}")
        logger.error("Run benchmark_asr.py first to generate whisper transcripts.")
        return {}

    whisper_transcripts = load_whisper_transcripts_from_csv(csv_path)
    logger.info(f"Loaded {len(whisper_transcripts)} whisper transcripts from {csv_path}")

    encounter_ids = get_encounter_ids()
    logger.info(f"Found {len(encounter_ids)} encounters in dataset")

    results = {
        "config": {
            "whisper_model": f"whisper-{model_size}",
            "asr_csv": str(csv_path),
            "num_encounters": len(encounter_ids)
        },
        "encounters": {},
        "aggregate": {}
    }

    all_metrics = []

    # Initialize grounder once for all annotations
    grounder = None

    for enc_id in encounter_ids:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing encounter {enc_id}")
        logger.info(f"{'='*60}")

        if enc_id not in whisper_transcripts:
            logger.warning(f"Encounter {enc_id}: No whisper transcript in CSV, skipping")
            continue

        try:
            # Load reference transcript
            reference_text = load_reference_transcript(enc_id)

            # Get whisper transcript from ASR CSV
            whisper_text = whisper_transcripts[enc_id]

            # Lazy-initialize grounder only if needed (not all cached)
            ref_cached = load_cached_annotations(enc_id, "reference")
            wh_cached = load_cached_annotations(enc_id, "whisper", model_size)
            if grounder is None and (ref_cached is None or wh_cached is None):
                logger.info("Initializing RAGGrounder...")
                from coda.grounding.icd10_rag_grounder import RAGGrounder
                grounder = RAGGrounder()

            # Annotate both texts
            reference_annotations = annotate_text(reference_text, enc_id, "reference", grounder=grounder)
            whisper_annotations = annotate_text(whisper_text, enc_id, "whisper", model_size, grounder=grounder)

            # Extract top codes
            reference_codes = extract_top_codes(reference_annotations)
            whisper_codes = extract_top_codes(whisper_annotations)

            # Compute both strict and hierarchical metrics
            metrics = compute_all_metrics(reference_codes, whisper_codes)
            all_metrics.append(metrics)

            # Store results
            results["encounters"][enc_id] = {
                "reference_codes": sorted(list(reference_codes)),
                "whisper_codes": sorted(list(whisper_codes)),
                "metrics": metrics,
                "num_reference_codes": len(reference_codes),
                "num_whisper_codes": len(whisper_codes),
                "num_matching_codes": len(reference_codes & whisper_codes)
            }

            logger.info(f"Reference codes: {sorted(reference_codes)}")
            logger.info(f"Whisper codes: {sorted(whisper_codes)}")
            strict = metrics['strict']
            hier = metrics['hierarchical']
            logger.info(f"Strict:       P={strict['precision']:.3f}, R={strict['recall']:.3f}, F1={strict['f1']:.3f}, J={strict['jaccard']:.3f}")
            logger.info(f"Hierarchical: P={hier['precision']:.3f}, R={hier['recall']:.3f}, F1={hier['f1']:.3f}, J={hier['jaccard']:.3f}")

        except Exception as e:
            logger.error(f"Error processing encounter {enc_id}: {e}")
            import traceback
            traceback.print_exc()
            results["encounters"][enc_id] = {"error": str(e)}

    # Compute aggregate metrics for both strict and hierarchical
    if all_metrics:
        def aggregate_metric_type(metrics_list, metric_type):
            """Aggregate metrics for a given type (strict or hierarchical)."""
            return {
                "precision": {
                    "mean": sum(m[metric_type]["precision"] for m in metrics_list) / len(metrics_list),
                    "min": min(m[metric_type]["precision"] for m in metrics_list),
                    "max": max(m[metric_type]["precision"] for m in metrics_list)
                },
                "recall": {
                    "mean": sum(m[metric_type]["recall"] for m in metrics_list) / len(metrics_list),
                    "min": min(m[metric_type]["recall"] for m in metrics_list),
                    "max": max(m[metric_type]["recall"] for m in metrics_list)
                },
                "f1": {
                    "mean": sum(m[metric_type]["f1"] for m in metrics_list) / len(metrics_list),
                    "min": min(m[metric_type]["f1"] for m in metrics_list),
                    "max": max(m[metric_type]["f1"] for m in metrics_list)
                },
                "jaccard": {
                    "mean": sum(m[metric_type]["jaccard"] for m in metrics_list) / len(metrics_list),
                    "min": min(m[metric_type]["jaccard"] for m in metrics_list),
                    "max": max(m[metric_type]["jaccard"] for m in metrics_list)
                }
            }

        results["aggregate"] = {
            "strict": aggregate_metric_type(all_metrics, "strict"),
            "hierarchical": aggregate_metric_type(all_metrics, "hierarchical"),
            "num_encounters_processed": len(all_metrics)
        }

    return results


def save_results(results: Dict, model_size: str):
    """Save benchmark results to file."""
    output_path = RESULTS_BASE.join("results", name=f"benchmark_whisper-{model_size}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to: {output_path}")
    return output_path


def print_summary(results: Dict):
    """Print benchmark summary to console."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    config = results.get("config", {})
    print(f"\nConfiguration:")
    print(f"  Whisper Model: {config.get('whisper_model', 'unknown')}")
    print(f"  ASR CSV: {config.get('asr_csv', 'unknown')}")
    print(f"  Encounters: {config.get('num_encounters', 0)}")

    agg = results.get("aggregate", {})
    if agg:
        n = agg.get('num_encounters_processed', 0)
        print(f"\nAggregate Metrics (n={n}):")
        print("-" * 50)
        print(f"{'Metric':<12} {'Strict':>12} {'Hierarchical':>14}")
        print("-" * 50)

        strict_agg = agg.get("strict", {})
        hier_agg = agg.get("hierarchical", {})

        for metric in ["precision", "recall", "f1", "jaccard"]:
            s = strict_agg.get(metric, {})
            h = hier_agg.get(metric, {})
            s_mean = s.get('mean', 0)
            h_mean = h.get('mean', 0)
            print(f"{metric.capitalize():<12} {s_mean:>12.3f} {h_mean:>14.3f}")

        print("-" * 50)
        print("\nNote: Hierarchical treats G47.0 as equivalent to G47")

    print("\n" + "=" * 80)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Kaggle Ambient Scribe ICD-10 Annotation Benchmark")
    parser.add_argument(
        "--model", "-m",
        default=WHISPER_MODEL_SIZE,
        choices=["tiny", "base", "small", "medium", "large"],
        help=f"Whisper model size (default: {WHISPER_MODEL_SIZE})"
    )
    parser.add_argument(
        "--asr_csv",
        default=None,
        help="Path to benchmark_asr.py output CSV (default: pystow cache)"
    )
    args = parser.parse_args()

    logger.info(f"Starting annotation benchmark with whisper-{args.model}")

    results = run_benchmark(model_size=args.model, asr_csv=args.asr_csv)
    if not results:
        return

    output_path = save_results(results, args.model)
    print_summary(results)

    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()

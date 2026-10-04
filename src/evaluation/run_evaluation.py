import argparse
import json
import os
from src.correction.inference import CorrectionEngine
from src.evaluation.correction_metrics import evaluate_metrics
from src.evaluation.error_analysis import evaluate_edits, evaluate_terminology

def main():
    parser = argparse.ArgumentParser(description="Run Evaluation")
    parser.add_argument('--dataset', type=str, required=True, help="Test dataset path")
    parser.add_argument('--config', type=str, default='config/training_config.yaml')
    parser.add_argument('--output-dir', type=str, default='evaluation/reports')
    parser.add_argument('--mock', action='store_true', help="Use mock model for testing evaluation pipeline")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Initializing engine...")
    # Minimal config loading for evaluation
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    engine = CorrectionEngine(config, mock_mode=args.mock)
    engine.load_model()
    
    # Optional terminology glossary for this domain (example)
    terminology = {"neural", "network", "deep", "learning", "state", "of", "the", "art"}

    results = []
    with open(args.dataset, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            
            # Predict
            req = {
                "text": item['incorrect_text'],
                "language": item['language'],
                "domain": item.get('domain', 'general')
            }
            pred = engine.correct(req)
            
            hypothesis = pred['corrected_text']
            reference = item['correct_text']
            original = item['incorrect_text']
            
            # Evaluate
            metrics = evaluate_metrics(reference, hypothesis)
            edit_metrics = evaluate_edits(original, reference, hypothesis)
            term_metrics = evaluate_terminology(reference, hypothesis, terminology)
            
            results.append({
                "id": item['id'],
                "original": original,
                "reference": reference,
                "hypothesis": hypothesis,
                "metrics": metrics,
                "edit_metrics": edit_metrics,
                "terminology": term_metrics
            })

    # Summary logic
    summary = {
        "total_samples": len(results),
        "mean_cer": sum(r['metrics']['cer'] for r in results) / max(1, len(results)),
        "mean_wer": sum(r['metrics']['wer'] for r in results) / max(1, len(results)),
        "exact_match_rate": sum(r['metrics']['exact_match'] for r in results) / max(1, len(results)),
        "mean_precision": sum(r['edit_metrics']['precision'] for r in results) / max(1, len(results)),
        "mean_recall": sum(r['edit_metrics']['recall'] for r in results) / max(1, len(results)),
    }
    
    with open(f"{args.output_dir}/results.json", 'w') as f:
        json.dump({"summary": summary, "details": results}, f, indent=2)
        
    print(f"Evaluation complete. Results saved to {args.output_dir}/results.json")
    print(f"Summary: CER={summary['mean_cer']:.3f}, WER={summary['mean_wer']:.3f}, EM={summary['exact_match_rate']:.3f}")
    print(f"Edit Precision={summary['mean_precision']:.3f}, Edit Recall={summary['mean_recall']:.3f}")

if __name__ == "__main__":
    main()

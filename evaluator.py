"""
Evaluation framework for SHL Recommendation System
Calculates Mean Recall@K and other metrics
"""

import pandas as pd
import numpy as np
import time
from typing import List, Dict
from recommendation_engine import RecommendationEngine
from config import settings


class Evaluator:
    def __init__(self):
        """Initialize evaluator"""
        self.engine = RecommendationEngine()

    # ------------------ METRIC HELPERS ------------------

    def recall_at_k(self, predicted: List[str], actual: List[str], k: int) -> float:
        """Calculate Recall@K for a single query"""
        if not actual:
            return 0.0
        predicted_k = predicted[:k]
        hits = sum(1 for url in predicted_k if url in actual)
        return hits / len(actual)

    def mean_recall_at_k(self, predictions: Dict[str, List[str]],
                         ground_truth: Dict[str, List[str]], k: int = 10) -> float:
        """Mean Recall@K across all queries"""
        recalls = [
            self.recall_at_k(predictions[q], ground_truth[q], k)
            for q in ground_truth if q in predictions
        ]
        return np.mean(recalls) if recalls else 0.0

    def precision_at_k(self, predicted: List[str], actual: List[str], k: int) -> float:
        """Calculate Precision@K"""
        predicted_k = predicted[:k]
        hits = sum(1 for url in predicted_k if url in actual)
        return hits / k if k > 0 else 0.0

    def ndcg_at_k(self, predicted: List[str], actual: List[str], k: int) -> float:
        """Normalized Discounted Cumulative Gain@K"""
        predicted_k = predicted[:k]
        dcg = 0.0
        for i, url in enumerate(predicted_k):
            if url in actual:
                dcg += 1.0 / np.log2(i + 2)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual), k)))
        return dcg / idcg if idcg > 0 else 0.0

    # ------------------ EVALUATION ------------------

    def evaluate_on_train_set(self, train_file: str = None) -> Dict:
        """Evaluate on labeled training data"""
        train_file = train_file or settings.TRAIN_DATA
        print("\nLoading training data...")
        df = pd.read_csv(train_file)

        grouped = df.groupby("Query")["Assessment_url"].apply(list).to_dict()
        predictions, ground_truth = {}, {}

        print(f"Evaluating on {len(grouped)} queries...\n")

        for query, actual_urls in grouped.items():
            recs = self.engine.recommend(query, top_k=settings.MAX_RECOMMENDATIONS)
            predicted_urls = [r["assessment_url"] for r in recs]
            predictions[query] = predicted_urls
            ground_truth[query] = actual_urls

            print(f"Query: {query[:60]}...")
            print(f"  Actual: {len(actual_urls)} | Predicted: {len(predicted_urls)}")
            print(f"  Recall@10: {self.recall_at_k(predicted_urls, actual_urls, 10):.3f}\n")

        metrics = {
            "mean_recall@5": self.mean_recall_at_k(predictions, ground_truth, k=5),
            "mean_recall@10": self.mean_recall_at_k(predictions, ground_truth, k=10),
        }

        # Per-query metrics
        per_query = []
        for q in ground_truth:
            per_query.append({
                "query": q,
                "recall@5": self.recall_at_k(predictions[q], ground_truth[q], 5),
                "recall@10": self.recall_at_k(predictions[q], ground_truth[q], 10),
                "precision@5": self.precision_at_k(predictions[q], ground_truth[q], 5),
                "precision@10": self.precision_at_k(predictions[q], ground_truth[q], 10),
                "ndcg@10": self.ndcg_at_k(predictions[q], ground_truth[q], 10),
            })

        return {
            "overall_metrics": metrics,
            "per_query_metrics": per_query,
            "predictions": predictions,
        }

    # ------------------ TEST PREDICTION ------------------

    def generate_test_predictions(self, test_file: str = None,
                                  output_file: str = None):
        """Generate predictions for the unlabeled test set"""
        test_file = test_file or settings.TEST_DATA
        output_file = output_file or "data/predictions.csv"

        print("\nLoading test data...")
        df = pd.read_csv(test_file)
        print(f"Generating predictions for {len(df)} queries...\n")

        results = []
        for i, row in df.iterrows():
            query = row["Query"]
            print(f"[{i+1}/{len(df)}] Processing: {query[:60]}...")
            recs = self.engine.recommend(query, top_k=settings.MAX_RECOMMENDATIONS)

            for rec in recs:
                results.append({
                    "Query": query,
                    "Assessment_url": rec["assessment_url"]
                })
            time.sleep(0.3)  # optional, if using LLM to prevent rate-limit

        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"\n✅ Predictions saved to: {output_file}")
        return results_df

    # ------------------ BALANCE ANALYSIS ------------------

    def analyze_balance(self, predictions: Dict[str, List[Dict]]):
        """Check distribution of test types"""
        report = []
        for q, recs in predictions.items():
            type_counts = {}
            for r in recs:
                t = r.get("test_type", "Unknown")
                type_counts[t] = type_counts.get(t, 0) + 1
            report.append({
                "query": q,
                "type_distribution": type_counts,
                "is_balanced": len(type_counts) > 1
            })
        return report


# ------------------ MAIN ------------------

def main():
    evaluator = Evaluator()

    print("=" * 80)
    print("SHL Assessment Recommendation System - Evaluation")
    print("=" * 80)

    print("\n1. Evaluating on training set...")
    train_results = evaluator.evaluate_on_train_set()

    print("\n" + "=" * 80)
    print("OVERALL METRICS")
    print("=" * 80)
    for k, v in train_results["overall_metrics"].items():
        print(f"{k}: {v:.4f}")

    print("\n" + "=" * 80)
    print("PER-QUERY METRICS (Top 5)")
    print("=" * 80)
    for i, q in enumerate(train_results["per_query_metrics"][:5]):
        print(f"{i+1}. {q['query'][:50]}...")
        print(f"   Recall@10={q['recall@10']:.3f}, Precision@10={q['precision@10']:.3f}, NDCG@10={q['ndcg@10']:.3f}")

    print("\n" + "=" * 80)
    print("2. Generating predictions for test set...")
    print("=" * 80)
    evaluator.generate_test_predictions()

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

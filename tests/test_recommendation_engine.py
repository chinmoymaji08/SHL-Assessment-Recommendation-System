"""
Unit tests for SHL Recommendation Engine
Run with: pytest tests/
"""

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from recommendation_engine import RecommendationEngine


@pytest.fixture(scope="session")
def engine():
    """Create a single engine instance for all tests (faster)"""
    return RecommendationEngine()


class TestRecommendationEngine:
    """Core functionality tests"""

    def test_engine_initialization(self, engine):
        assert engine is not None
        assert engine.is_loaded() is True
        assert isinstance(engine.assessments, list)
        assert len(engine.assessments) > 0

    def test_recommend_returns_list(self, engine):
        results = engine.recommend("Java developer", top_k=5)
        assert isinstance(results, list)

    def test_recommend_returns_correct_count(self, engine):
        query = "Python developer"
        top_k = 5
        results = engine.recommend(query, top_k=top_k)
        assert len(results) <= top_k
        assert all("assessment_name" in r for r in results)

    def test_recommend_returns_valid_structure(self, engine):
        query = "SQL database administrator"
        results = engine.recommend(query, top_k=3)
        for rec in results:
            assert "assessment_name" in rec
            assert "assessment_url" in rec
            assert "relevance_score" in rec
            assert isinstance(rec["assessment_name"], str)
            assert isinstance(rec["assessment_url"], str)
            assert isinstance(rec["relevance_score"], (int, float))

    def test_recommend_scores_are_valid(self, engine):
        results = engine.recommend("JavaScript developer", top_k=5)
        for rec in results:
            assert 0 <= rec["relevance_score"] <= 1

    def test_different_queries_return_different_results(self, engine):
        q1 = "Java developer"
        q2 = "Project manager"
        res1 = [r["assessment_url"] for r in engine.recommend(q1, 5)]
        res2 = [r["assessment_url"] for r in engine.recommend(q2, 5)]
        assert res1 != res2

    def test_semantic_search(self, engine):
        results = engine._semantic_search("Python programming", top_k=10)
        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_extract_requirements(self, engine):
        reqs = engine._extract_requirements("Need Java developer with teamwork skills")
        assert isinstance(reqs, dict)

    def test_balance_by_test_type(self, engine):
        candidates = [
            {"name": "Test1", "test_type": "K", "score": 0.9},
            {"name": "Test2", "test_type": "K", "score": 0.8},
            {"name": "Test3", "test_type": "P", "score": 0.7},
            {"name": "Test4", "test_type": "P", "score": 0.6},
        ]
        reqs = {"technical_skills": ["Java"], "soft_skills": ["Teamwork"]}
        balanced = engine._balance_by_test_type(candidates, reqs, top_k=4)
        types = [c["test_type"] for c in balanced]
        assert "K" in types
        assert "P" in types

    def test_empty_query_handling(self, engine):
        with pytest.raises(Exception):
            engine.recommend("", top_k=5)

    def test_special_characters_query(self, engine):
        res = engine.recommend("C++ & Python developer", top_k=5)
        assert isinstance(res, list)
        assert len(res) > 0

    def test_long_query(self, engine):
        q = "Java " * 500
        res = engine.recommend(q, top_k=5)
        assert isinstance(res, list)
        assert len(res) > 0

    def test_non_english_query(self, engine):
        res = engine.recommend("Développeur Python", top_k=5)
        assert isinstance(res, list)


class TestDataValidation:
    """Test dataset and embeddings"""

    def test_assessments_have_required_fields(self, engine):
        for a in engine.assessments:
            assert "name" in a
            assert "url" in a

    def test_valid_url_format(self, engine):
        for a in engine.assessments:
            url = a.get("url", "")
            assert url.startswith("http")

    def test_embeddings_shape(self, engine):
        if engine.embeddings is not None and engine.embeddings.size > 0:
            assert len(engine.embeddings.shape) == 2
            assert engine.embeddings.shape[0] == len(engine.assessments)
            assert engine.embeddings.shape[1] == 384  # MiniLM


class TestEdgeCases:
    """Edge conditions and exceptions"""

    def test_top_k_larger_than_available(self, engine):
        large_k = len(engine.assessments) + 100
        res = engine.recommend("Developer", top_k=large_k)
        assert len(res) <= len(engine.assessments)

    def test_top_k_zero(self, engine):
        res = engine.recommend("Developer", top_k=0)
        assert isinstance(res, list)

    def test_numeric_query(self, engine):
        res = engine.recommend("12345", top_k=5)
        assert isinstance(res, list)
        assert len(res) > 0


class TestPerformance:
    """Performance constraints"""

    def test_single_query_speed(self, engine):
        import time
        q = "Java developer with SQL skills"
        start = time.time()
        res = engine.recommend(q, top_k=10)
        elapsed = time.time() - start
        assert elapsed < 5.0
        assert len(res) > 0

    def test_multiple_queries_speed(self, engine):
        import time
        queries = ["Java", "Python", "SQL", "Manager", "Designer"]
        start = time.time()
        for q in queries:
            engine.recommend(q, 5)
        elapsed = time.time() - start
        assert elapsed < 15.0


def test_imports():
    """Test import of main modules"""
    import main
    import recommendation_engine
    import evaluator
    import config

    assert main
    assert recommendation_engine
    assert evaluator
    assert config


if __name__ == "__main__":
    pytest.main(["-v"])

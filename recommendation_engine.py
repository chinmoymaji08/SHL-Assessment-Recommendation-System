# recommendation_engine.py
"""
SHL Assessment Recommendation Engine
Hybrid semantic + keyword + optional LLM re-ranking
"""

import json
import numpy as np
import re
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Optional LLM (Gemini)
try:
    import google.generativeai as genai
except Exception:
    genai = None

from config import settings
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationEngine:
    def __init__(self):
        """Initialize recommendation engine with embeddings and data"""
        self.assessments: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.embedding_model = None
        self.loaded = False

        # Load models and data
        self._load_embedding_model()
        self._load_assessments()
        self._load_or_create_embeddings()

        # Optional Gemini setup
        self.llm = None
        if genai and settings.GEMINI_API_KEY:
            try:
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self.llm = genai.GenerativeModel(settings.GEMINI_MODEL)
                logger.info("Gemini LLM configured")
            except Exception as e:
                logger.warning("Failed to configure Gemini: %s", e)
                self.llm = None

        self.loaded = True

    def _load_embedding_model(self):
        """Load sentence transformer model for embeddings"""
        logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("Embedding model loaded")

    def _load_assessments(self):
        """Load assessment data from JSON file (data path from settings)"""
        path = settings.ASSESSMENTS_JSON
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.assessments = json.load(f)
            logger.info("Loaded %d assessments from %s", len(self.assessments), path)
        except Exception as e:
            logger.error("Error loading assessments (%s): %s", path, e)
            self.assessments = []

    def _load_or_create_embeddings(self):
        """Load or create embeddings for all assessments"""
        try:
            self.embeddings = np.load(settings.EMBEDDINGS_FILE)
            logger.info("Loaded precomputed embeddings: %s", settings.EMBEDDINGS_FILE)
        except Exception:
            logger.info("Creating embeddings for %d assessments...", len(self.assessments))
            self._create_embeddings()

    def _create_embeddings(self):
        """Create embeddings for all assessments and save them"""
        texts = []
        for a in self.assessments:
            parts = [
                a.get('name', ''),
                a.get('description', ''),
                a.get('category', ''),
                a.get('test_type', ''),
                ' '.join(a.get('skills_measured', []) or []),
                ' '.join(a.get('features', []) or [])
            ]
            texts.append(' '.join([p for p in parts if p]).strip() or a.get('name', ''))

        if texts:
            embs = self.embedding_model.encode(texts, show_progress_bar=False)
            self.embeddings = np.array(embs)
            np.save(settings.EMBEDDINGS_FILE, self.embeddings)
            logger.info("Embeddings created and saved to %s", settings.EMBEDDINGS_FILE)
        else:
            logger.warning("No texts found to embed; embeddings not created")
            self.embeddings = np.empty((0, self.embedding_model.get_sentence_embedding_dimension()))

    def is_loaded(self) -> bool:
        return self.loaded

    def _extract_requirements(self, query: str) -> Dict:
        """Extract structured requirements from query using LLM (if available)"""
        if not self.llm:
            return {}

        prompt = f"""
Analyze this job query and extract key requirements in JSON format:

Query: {query}

Return ONLY valid JSON with fields:
technical_skills, soft_skills, personality_traits, cognitive_abilities, role_level, key_domains
"""
        try:
            response = self.llm.generate_content(prompt)
            text = getattr(response, "text", "") or ""
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning("LLM requirements extraction failed: %s", e)
        return {}

    def _semantic_search(self, query: str, top_k: int = 20):
        """Return list of (index, score) using cosine similarity"""
        if self.embeddings is None or self.embeddings.size == 0:
            return []
        q_emb = self.embedding_model.encode([query])
        sims = cosine_similarity(q_emb, self.embeddings)[0]
        top_k = min(top_k, len(sims))
        idxs = np.argsort(sims)[-top_k:][::-1]
        return [(int(i), float(sims[i])) for i in idxs]

    def _keyword_search(self, query: str, requirements: Dict) -> List[int]:
        """Simple keyword matching returning indices ordered by match count"""
        keywords = []
        for key in ['technical_skills', 'soft_skills', 'personality_traits', 'cognitive_abilities']:
            items = requirements.get(key, []) if requirements else []
            keywords.extend([k.lower() for k in items if isinstance(k, str)])

        keywords.extend(re.findall(r'\w+', query.lower()))
        matched = []
        for idx, a in enumerate(self.assessments):
            text = ' '.join([
                str(a.get('name', '')),
                str(a.get('description', '')),
                str(a.get('category', '')),
                ' '.join(a.get('skills_measured', []) or []),
                ' '.join(a.get('features', []) or [])
            ]).lower()
            count = sum(1 for kw in set(keywords) if kw and kw in text)
            if count > 0:
                matched.append((idx, count))
        matched.sort(key=lambda x: x[1], reverse=True)
        return [int(i) for i, _ in matched]

    def _balance_by_test_type(self, candidates: List[Dict], requirements: Dict, top_k: int) -> List[Dict]:
        """Balance across K/P/C where possible"""
        needs_technical = bool(requirements.get('technical_skills')) if requirements else False
        needs_personality = bool(requirements.get('soft_skills') or requirements.get('personality_traits')) if requirements else False
        needs_cognitive = bool(requirements.get('cognitive_abilities')) if requirements else False

        by_type = {'K': [], 'P': [], 'C': [], 'other': []}
        for c in candidates:
            t = c.get('test_type') or c.get('type') or 'other'
            t = t if t in by_type else 'other'
            by_type[t].append(c)

        balanced = []
        if needs_technical and needs_personality:
            balanced.extend(by_type['K'][:top_k//2])
            balanced.extend(by_type['P'][:top_k//2])
        elif needs_technical:
            balanced.extend(by_type['K'][:int(top_k*0.7)])
            balanced.extend(by_type['C'][:int(top_k*0.2)])
            balanced.extend(by_type['P'][:int(top_k*0.1)])
        elif needs_personality:
            balanced.extend(by_type['P'][:int(top_k*0.7)])
            balanced.extend(by_type['C'][:int(top_k*0.2)])
            balanced.extend(by_type['K'][:int(top_k*0.1)])
        else:
            per = max(1, top_k // 3)
            balanced.extend(by_type['K'][:per])
            balanced.extend(by_type['P'][:per])
            balanced.extend(by_type['C'][:per])

        # fill remaining with top candidates
        if len(balanced) < top_k:
            remaining = [c for c in candidates if c not in balanced]
            balanced.extend(remaining[:top_k - len(balanced)])
        return balanced[:top_k]

    def _rerank_with_llm(self, query: str, candidates: List[Dict], top_k:int) -> List[Dict]:
        """Rerank using LLM if available, otherwise return candidates[:top_k]"""
        if not self.llm or not candidates:
            return candidates[:top_k]
        # Prepare prompt
        items = []
        for i, c in enumerate(candidates[:20]):
            items.append({'id': i, 'name': c.get('name', ''), 'desc': (c.get('description') or '')[:200], 'type': c.get('test_type', 'N/A')})
        prompt = f"""
Given the requirement: {query}

Rank these assessments by relevance, return ONLY a JSON array of item ids ordered by relevance (e.g. [2,0,1]):
{json.dumps(items, indent=2)}
"""
        try:
            resp = self.llm.generate_content(prompt)
            text = getattr(resp, "text", "") or ""
            arr_match = re.search(r'\[[\d,\s]+\]', text)
            if arr_match:
                ids = json.loads(arr_match.group())
                ranked = [candidates[i] for i in ids if 0 <= i < len(candidates)]
                return ranked[:top_k]
        except Exception as e:
            logger.warning("LLM rerank failed: %s", e)
        return candidates[:top_k]

    def recommend(self, query: str, top_k: int = 10) -> List[Dict]:
        """Main recommendation method"""
        if not query or not str(query).strip():
            raise ValueError("Query empty")
        top_k = int(max(min(top_k, settings.MAX_RECOMMENDATIONS), settings.MIN_RECOMMENDATIONS))

        requirements = self._extract_requirements(query) if self.llm else {}
        pool = settings.CANDIDATE_POOL_SIZE if hasattr(settings, 'CANDIDATE_POOL_SIZE') else 30

        sem = self._semantic_search(query, top_k=pool)
        key_idxs = self._keyword_search(query, requirements)

        candidate_scores = {}
        # Add semantic contributions
        for idx, score in sem:
            candidate_scores[idx] = candidate_scores.get(idx, 0.0) + score * settings.SEMANTIC_WEIGHT

        # Add keyword contributions
        for i, idx in enumerate(key_idxs[:pool]):
            kw_score = (pool - i) / float(pool)
            candidate_scores[idx] = candidate_scores.get(idx, 0.0) + kw_score * settings.KEYWORD_WEIGHT

        # Sort and normalize scores
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:pool]
        if not sorted_candidates:
            # Fallback: return first N assessments
            fallback = []
            for a in self.assessments[:top_k]:
                fallback.append({'name': a.get('name'), 'url': a.get('url'), 'description': a.get('description'), 'test_type': a.get('test_type'), 'score': 0.0})
            return fallback

        scores = np.array([s for _, s in sorted_candidates], dtype=float)
        min_s, max_s = float(scores.min()), float(scores.max())
        candidates = []
        for idx, s in sorted_candidates:
            a = dict(self.assessments[idx])  # copy
            norm = (s - min_s) / (max_s - min_s + 1e-12) if max_s > min_s else 1.0
            a['score'] = float(norm)
            # Ensure keys used by API/frontend exist:
            a['name'] = a.get('name') or a.get('assessment_name') or 'Unknown'
            a['url'] = a.get('url') or a.get('assessment_url') or ''
            a['description'] = a.get('description', '')
            a['test_type'] = a.get('test_type', '') or a.get('type', '')
            candidates.append(a)

        # Balance by test type
        balanced = self._balance_by_test_type(candidates, requirements, top_k * 2)

        # LLM rerank with limited candidate set
        final = self._rerank_with_llm(query, balanced, top_k)

        # Guarantee at least MIN_RECOMMENDATIONS by falling back to candidates list
        if len(final) < settings.MIN_RECOMMENDATIONS:
            final = candidates[:max(settings.MIN_RECOMMENDATIONS, top_k)]

        # Output formatting: map to API fields
        results = []
        for c in final[:top_k]:
            results.append({
                "assessment_name": c.get('name'),
                "assessment_url": c.get('url'),
                "relevance_score": round(float(c.get('score', 0.0)), 4),
                "description": c.get('description', '')[:500],
                "test_type": c.get('test_type', '')
            })
        return results

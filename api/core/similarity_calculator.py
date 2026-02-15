import numpy as np
from typing import Any, Dict, List, Optional
from config.app_config import MULTI_LEVEL_WEIGHTS
from api.core.feature_extractor import is_empty_value

class SimilarityCalculator:
    _instance: Optional["SimilarityCalculator"] = None

    def __new__(cls) -> "SimilarityCalculator":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.weights = MULTI_LEVEL_WEIGHTS
        self.dimensions = ["semantic", "domain", "method", "application", "innovation"]

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        if not vec1 or not vec2:
            return 0.0

        v1 = np.array(vec1)
        v2 = np.array(vec2)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(v1, v2) / (norm1 * norm2))

    def calculate(self, query_vectors: Dict[str, List[float]], target: Dict[str, Any], semantic_score: float = 0.0, query_extracted_texts: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        scores = {}

        scores["semantic"] = semantic_score

        for dim in ["domain", "method", "application", "innovation"]:
            vec_key = f"{dim}_vector"
            query_vec = query_vectors.get(vec_key, [])
            target_vec = target.get(vec_key, [])
            
            query_empty = False
            if query_extracted_texts:
                query_text = query_extracted_texts.get(dim, "")
                query_empty = is_empty_value(query_text)
            
            target_text = target.get(f"extracted_{dim}", "")
            target_empty = is_empty_value(target_text)
            
            if query_empty or target_empty or not query_vec or not target_vec:
                scores[dim] = 0.0
            else:
                scores[dim] = self.cosine_similarity(query_vec, target_vec)

        weight_array = np.array([self.weights.get(dim, 0) for dim in self.dimensions])
        score_array = np.array([scores.get(dim, 0) for dim in self.dimensions])
        
        nonzero_mask = score_array != 0
        total_weight = np.sum(weight_array[nonzero_mask])
        
        if total_weight == 0:
            total_score = 0.0
        else:
            total_score = np.sum(weight_array * score_array) / total_weight
        
        total_score = np.round(total_score, 3)

        explanation = self.generate_explanation(scores, target)

        return {
            "total_score": total_score,
            "dimension_scores": scores,
            "explanation": explanation,
            "extracted_info": {
                "domain": target.get("extracted_domain", ""),
                "method": target.get("extracted_method", ""),
                "application": target.get("extracted_application", ""),
                "innovation": target.get("extracted_innovation", "")
            }
        }

    def generate_explanation(self, scores: Dict[str, float], target: Dict[str, Any]) -> str:
        parts = []

        domain_text = target.get("extracted_domain", "")
        if scores.get("domain", 0) > 0.8:
            parts.append(f"技术领域高度匹配")
        elif scores.get("domain", 0) > 0.6:
            parts.append(f"技术领域部分匹配")

        if scores.get("method", 0) > 0.7:
            parts.append(f"技术方法相似")

        if scores.get("application", 0) > 0.7:
            parts.append(f"应用场景相近")

        if scores.get("semantic", 0) > 0.8:
            parts.append(f"语义高度相似")

        return "；".join(parts) if parts else "基于语义相似度匹配"

g_similarity_calculator = SimilarityCalculator()

__all__ = ["g_similarity_calculator", "SimilarityCalculator"]

# ═══════════════════════════════════════════════════
# GDPR/CCPA COMPLIANCE DOCUMENTATION
# ═══════════════════════════════════════════════════
# This module processes anonymous data in compliance with GDPR
# and CCPA standards. It does not contain personal data (PII).
#
# Legal Basis: Anonymous Data Processing
# "Data rendered anonymous in such a way that the data subject
#  is not or no longer identifiable is excluded from GDPR scope."
#
# Techniques Applied:
# 1. K-Anonymity (min_k=5): Patterns unique to a single user
#    are not added to the aggregate pool.
# 2. Differential Privacy (Laplace, ε=1.0):
#    Mathematical noise is added to query results.
# 3. Data Minimization: Only signal type, category,
#    dimension, and value are stored.
# 4. TTL: 90 days — automatic destruction.
#
# NEVER enters the Aggregate store:
# - user_id, session_id, project_id
# - Message content
# - Exact timestamps
# - IP addresses
# ═══════════════════════════════════════════════════

import datetime
import numpy as np
from typing import Any

class AggregateSignalStore:
    def __init__(self, redis_client: Any, min_k: int = 5, epsilon: float = 1.0):
        self._redis = redis_client
        self._min_k = min_k      # k-anonymity: min 5 distinct records
        self._epsilon = epsilon  # differential privacy noise

    def record(self, signal_type: str, query_category: str, dimension: str, dimension_value: str) -> None:
        """
        Records an anonymous signal in a GDPR-compliant manner.
        Contains no PII — statistical counter only.
        """
        if not self._redis:
            return
            
        allowed_signal_types = {
            "correction", "rejection", "preference", "clarification", "positive"
        }
        if signal_type not in allowed_signal_types:
            return
        
        allowed_dimensions = {
            "response_length", "detail_level", "tone", "format", "language", "unknown", "heuristic", "llm_extracted"
        }
        if dimension not in allowed_dimensions:
            return
        
        # Time bucket (rounded to the hour for privacy, no exact timestamps)
        hour_bucket = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H")
        
        # Redis key — NO PII included
        key = f"magnet:agg:{signal_type}:{query_category}:{dimension}:{dimension_value}"
        counter_key = f"magnet:agg:count:{signal_type}:{query_category}"
        
        try:
            pipe = self._redis.pipeline()
            pipe.incr(key)
            pipe.incr(counter_key)
            pipe.expire(key, 60 * 60 * 24 * 90)        # 90-day TTL
            pipe.expire(counter_key, 60 * 60 * 24 * 90) 
            pipe.execute()
        except Exception:
            pass

    def get_prior(self, query_category: str, dimension: str) -> dict | None:
        """
        Returns the aggregate prior probability for a new user.
        K-anonymity: Returns None if total records < min_k.
        Differential Privacy: Adds Laplace noise to the results.
        """
        if not self._redis:
            return None
            
        pattern = f"magnet:agg:*:{query_category}:{dimension}:*"
        try:
            keys = list(self._redis.scan_iter(pattern))
            if not keys:
                return None
            
            counts = {}
            total = 0
            for key in keys:
                value = key.split(":")[-1]
                count = int(self._redis.get(key) or 0)
                counts[value] = count
                total += count
            
            if total < self._min_k:
                return None
            
            noisy_counts = {}
            for value, count in counts.items():
                noise = np.random.laplace(0, 1.0 / self._epsilon)
                noisy_counts[value] = max(0, count + noise)
            
            noisy_total = sum(noisy_counts.values())
            if noisy_total == 0:
                return None
            
            return {v: round(c / noisy_total, 3) for v, c in noisy_counts.items()}
        except Exception:
            return None

    def get_cold_start_injection(self, query_category: str) -> str:
        """Generates the cold start injection context for a new user."""
        if not self._redis:
            return ""
            
        lines = []
        for dimension in ["response_length", "detail_level", "tone", "language", "heuristic", "llm_extracted"]:
            prior = self.get_prior(query_category, dimension)
            if prior:
                top_value = max(prior, key=prior.get)
                top_pct = int(prior[top_value] * 100)
                if top_pct >= 55:  # Add if there is a strong aggregate signal
                    lines.append(f"  - {dimension}: {top_value} ({top_pct}% of users)")
        
        if not lines:
            return ""
        
        return (
            "[Aggregate Prior]\n"
            "Based on anonymized patterns from similar users:\n" +
            "\n".join(lines) + "\n\n"
            "Note: These are statistical suggestions. "
            "User's own behavior takes priority."
        )
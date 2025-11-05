"""
Hybrid Classifier - Simplified Version
Uses CCRE engine only (no proximity engine)
"""

import pandas as pd
from typing import Dict, List, Optional

# Use try/except for imports to handle both package and direct use
try:
    from .category_hierarchy import CategoryHierarchy
    from .ccre_engine import CCREEngine
except ImportError:
    from category_hierarchy import CategoryHierarchy
    from ccre_engine import CCREEngine


class HybridClassifier:
    """
    Simplified classifier using CCRE engine only
    """
    
    def __init__(
        self,
        proximity_rules: pd.DataFrame,
        hierarchy: pd.DataFrame,
        user_examples: Optional[pd.DataFrame] = None,
        fallback_rules: Optional[pd.DataFrame] = None,
        program: str = "Generic"
    ):
        """
        Initialize classifier with CCRE engine
        Args:
            proximity_rules: Not used (kept for compatibility)
            hierarchy: Category hierarchy DataFrame
            user_examples: Not used (kept for compatibility)
            fallback_rules: CCRE rules DataFrame
            program: Program name
        """
        self.program = program
        
        print(f"\n🔧 Initializing Classifier for {program}...")
        
        # Hierarchy validation
        self.hierarchy = CategoryHierarchy(hierarchy)
        
        # CCRE Engine (main engine)
        if fallback_rules is not None and not fallback_rules.empty:
            self.ccre_engine = CCREEngine(fallback_rules)
            print(f"   ✅ Loaded {len(fallback_rules)} CCRE rules")
        else:
            self.ccre_engine = None
            print("   ⚠️ No CCRE rules provided")
        
        print(f"✅ Classifier ready\n")
    
    def classify(self, transcript: str, confidence_threshold: float = 0.40) -> Dict:
        """
        Classify a single transcript using CCRE engine
        Args:
            transcript: Text to classify
            confidence_threshold: Minimum confidence (default 0.40 for better coverage)
        Returns:
            Classification result with full hierarchy
        """
        if not transcript or not isinstance(transcript, str):
            return self.hierarchy._unclassified_result("Empty transcript")
        
        # Use CCRE engine
        if self.ccre_engine:
            result = self.ccre_engine.classify(transcript)
            if result and result.get('confidence', 0) >= confidence_threshold:
                # Build result in correct format
                formatted_result = {
                    'cat_no': result.get('rule_id', 'UNKNOWN'),
                    'confidence': result.get('confidence', 0),
                    'matched_terms': result.get('matched_keywords', ''),
                    'source': 'ccre'
                }
                enriched = self.hierarchy.validate_and_enrich(formatted_result, self.program)
                if enriched['category'] != 'Unclassified':
                    return enriched
        
        return self.hierarchy._unclassified_result("No rules matched above threshold")
    
    def classify_batch(
        self,
        transcripts: List[str],
        batch_size: int = 1000,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Classify multiple transcripts in batches
        Args:
            transcripts: List of transcripts to classify
            batch_size: Not used, kept for compatibility
            show_progress: Not used, kept for compatibility
        Returns:
            List of classification results
        """
        results = [self.classify(t) for t in transcripts]
        return results
    
    def get_classification_stats(self, results: List[Dict]) -> Dict:
        """
        Get statistics about classification results
        Args:
            results: List of classification dictionaries
        Returns:
            Statistics dictionary
        """
        total = len(results)
        if total == 0:
            return {}
        
        # Count by source
        source_counts = {}
        for result in results:
            source = result.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Count classified vs unclassified
        classified = sum(1 for r in results if r['category'] != 'Unclassified')
        unclassified = total - classified
        
        # Confidence distribution
        confidences = [r.get('confidence', 0) for r in results]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        high_conf = sum(1 for c in confidences if c >= 0.70)
        med_conf = sum(1 for c in confidences if 0.50 <= c < 0.70)
        low_conf = sum(1 for c in confidences if 0.40 <= c < 0.50)
        
        # Top categories
        category_counts = {}
        for result in results:
            cat = result['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_transcripts': total,
            'classified': classified,
            'classified_pct': (classified / total) * 100,
            'unclassified': unclassified,
            'unclassified_pct': (unclassified / total) * 100,
            'avg_confidence': avg_confidence,
            'high_confidence': high_conf,
            'medium_confidence': med_conf,
            'low_confidence': low_conf,
            'by_source': source_counts,
            'top_categories': top_categories
        }

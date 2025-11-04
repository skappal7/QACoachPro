"""
Hybrid Classifier
Orchestrates 3-pass classification system:
1. Proximity Engine (highest accuracy)
2. User Examples (learned patterns)
3. Keyword Fallback (safety net)
"""

import pandas as pd
from typing import Dict, List, Optional
from .proximity_engine import ProximityEngine
from .category_hierarchy import CategoryHierarchy
from .user_examples import UserExamplesEngine
from .ccre_engine import CCREEngine


class HybridClassifier:
    """
    3-pass hybrid classification system with hierarchy validation
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
        Initialize hybrid classifier
        Args:
            proximity_rules: Proximity rules DataFrame
            hierarchy: Category hierarchy DataFrame
            user_examples: Optional user examples for fuzzy matching
            fallback_rules: Optional fallback keyword rules
            program: Program name for filtering
        """
        self.program = program
        
        # Initialize engines
        print(f"\n🔧 Initializing Hybrid Classifier for {program}...")
        
        # Pass 1: Proximity Engine
        if not proximity_rules.empty:
            self.proximity_engine = ProximityEngine(proximity_rules)
        else:
            self.proximity_engine = None
            print("   ⚠️ No proximity rules - Pass 1 disabled")
        
        # Hierarchy validation
        self.hierarchy = CategoryHierarchy(hierarchy)
        
        # Pass 2: User Examples Engine
        if user_examples is not None and not user_examples.empty:
            self.examples_engine = UserExamplesEngine(user_examples)
        else:
            self.examples_engine = None
            print("   ℹ️ No user examples - Pass 2 disabled")
        
        # Pass 3: Keyword Fallback
        if fallback_rules is not None and not fallback_rules.empty:
            self.fallback_engine = CCREEngine(fallback_rules)
        else:
            self.fallback_engine = None
            print("   ℹ️ No fallback rules - Pass 3 disabled")
        
        print(f"✅ Hybrid Classifier ready\n")
    
    def classify(self, transcript: str, confidence_threshold: float = 0.60) -> Dict:
        """
        Classify a single transcript using 3-pass system
        Args:
            transcript: Text to classify
            confidence_threshold: Minimum confidence to accept (default 0.60)
        Returns:
            Classification result with full hierarchy
        """
        if not transcript or not isinstance(transcript, str):
            return self.hierarchy._unclassified_result("Empty transcript")
        
        # Pass 1: Proximity matching (highest accuracy: 0.90-0.98)
        if self.proximity_engine:
            result = self.proximity_engine.classify(transcript, self.program)
            if result and result.get('confidence', 0) >= 0.85:
                # High confidence proximity match - use it
                enriched = self.hierarchy.validate_and_enrich(result, self.program)
                if enriched['category'] != 'Unclassified':
                    return enriched
        
        # Pass 2: User examples (learned patterns: 0.80-0.90)
        if self.examples_engine:
            result = self.examples_engine.fuzzy_match(transcript, threshold=0.85)
            if result and result.get('confidence', 0) >= 0.80:
                # Good fuzzy match - use it
                enriched = self.hierarchy.validate_and_enrich(result, self.program)
                if enriched['category'] != 'Unclassified':
                    return enriched
        
        # Pass 3: Keyword fallback (generic: 0.60-0.80)
        if self.fallback_engine:
            result = self.fallback_engine.classify(transcript)
            if result and result.get('confidence', 0) >= confidence_threshold:
                enriched = self.hierarchy.validate_and_enrich(result, self.program)
                if enriched['category'] != 'Unclassified':
                    return enriched
        
        # No match in any pass
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
            batch_size: Process in batches for memory efficiency
            show_progress: Print progress updates
        Returns:
            List of classification results
        """
        results = []
        total = len(transcripts)
        
        for i in range(0, total, batch_size):
            batch = transcripts[i:i+batch_size]
            batch_results = [self.classify(t) for t in batch]
            results.extend(batch_results)
            
            if show_progress:
                processed = min(i + batch_size, total)
                pct = (processed / total) * 100
                print(f"   Processed {processed:,}/{total:,} ({pct:.1f}%)")
        
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
        
        high_conf = sum(1 for c in confidences if c >= 0.85)
        med_conf = sum(1 for c in confidences if 0.70 <= c < 0.85)
        low_conf = sum(1 for c in confidences if 0.50 <= c < 0.70)
        
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
    
    def explain_classification(self, result: Dict) -> str:
        """
        Generate human-readable explanation of classification
        Args:
            result: Classification result dictionary
        Returns:
            Explanation string
        """
        if result['category'] == 'Unclassified':
            reason = result.get('reason', 'No matching rules found')
            return f"❌ Unclassified: {reason}"
        
        source = result.get('source', 'unknown')
        confidence = result.get('confidence', 0)
        matched = result.get('matched_keywords', '')
        
        explanation = f"✅ Classified as: {result['category']} > {result['subcategory']}"
        
        if result.get('tertiary'):
            explanation += f" > {result['tertiary']}"
        if result.get('quaternary'):
            explanation += f" > {result['quaternary']}"
        
        explanation += f"\n   Source: {source}"
        explanation += f"\n   Confidence: {confidence:.2%}"
        
        if matched:
            explanation += f"\n   Matched: {matched[:100]}"
        
        return explanation

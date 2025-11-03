"""
User Examples Module
Fuzzy phrase matching for user-provided examples
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
from difflib import SequenceMatcher


class UserExamplesEngine:
    """Fuzzy phrase matching engine for user examples"""
    
    def __init__(self, examples_df: pd.DataFrame, similarity_threshold: float = 0.85):
        """
        Initialize with user examples
        Args:
            examples_df: DataFrame with columns: example_phrase, category, subcategory, tertiary_category, confidence
            similarity_threshold: Minimum similarity score (0.85 = 85%)
        """
        self.examples_df = examples_df.copy()
        self.similarity_threshold = similarity_threshold
        self._preprocess_examples()
    
    def _preprocess_examples(self):
        """Preprocess examples for faster matching"""
        self.examples = []
        
        for _, row in self.examples_df.iterrows():
            phrase = str(row['example_phrase']).lower().strip()
            
            self.examples.append({
                'phrase': phrase,
                'phrase_normalized': ' '.join(phrase.split()),  # Clean whitespace
                'category': str(row['category']).strip(),
                'subcategory': str(row.get('subcategory', '')).strip() if pd.notna(row.get('subcategory')) else None,
                'tertiary_category': str(row.get('tertiary_category', '')).strip() if pd.notna(row.get('tertiary_category')) else None,
                'confidence': float(row.get('confidence', 0.95))
            })
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using SequenceMatcher
        Args:
            text1, text2: Texts to compare
        Returns:
            Similarity score (0.0 to 1.0)
        """
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def fuzzy_match(self, transcript: str) -> Optional[Dict]:
        """
        Find best matching user example using fuzzy matching
        Args:
            transcript: Transcript text
        Returns:
            Match dict with category info or None
        """
        if not self.examples:
            return None
        
        transcript_normalized = ' '.join(transcript.lower().strip().split())
        
        best_match = None
        best_score = 0.0
        
        for example in self.examples:
            # Check if phrase is substring (fast path)
            if example['phrase_normalized'] in transcript_normalized:
                similarity = 1.0
            else:
                # Fuzzy match
                similarity = self.calculate_similarity(transcript_normalized, example['phrase_normalized'])
            
            if similarity >= self.similarity_threshold and similarity > best_score:
                best_score = similarity
                best_match = {
                    'category': example['category'],
                    'subcategory': example['subcategory'],
                    'tertiary_category': example['tertiary_category'],
                    'confidence': example['confidence'] * similarity,  # Adjust confidence by similarity
                    'matched_phrase': example['phrase'],
                    'similarity': similarity,
                    'source': 'user_example'
                }
        
        return best_match
    
    def batch_match(self, transcripts: List[str]) -> List[Optional[Dict]]:
        """
        Match multiple transcripts
        Args:
            transcripts: List of transcript texts
        Returns:
            List of match dicts (None if no match)
        """
        return [self.fuzzy_match(t) for t in transcripts]
    
    def get_stats(self) -> Dict:
        """Get statistics about loaded examples"""
        if not self.examples:
            return {
                "total_examples": 0,
                "unique_categories": 0,
                "avg_phrase_length": 0
            }
        
        categories = set(ex['category'] for ex in self.examples)
        avg_length = sum(len(ex['phrase'].split()) for ex in self.examples) / len(self.examples)
        
        return {
            "total_examples": len(self.examples),
            "unique_categories": len(categories),
            "avg_phrase_length": round(avg_length, 1),
            "similarity_threshold": self.similarity_threshold
        }

"""
Feedback Manager Module
Sampling-based feedback collection and learning
"""

import random
from typing import Dict, List
import pandas as pd


class FeedbackManager:
    """Manages sampling and feedback collection"""
    
    def __init__(self, classified_df: pd.DataFrame, sample_size: int = 50):
        """
        Initialize feedback manager
        Args:
            classified_df: Classified results dataframe
            sample_size: Number of samples to show user
        """
        self.df = classified_df.copy()
        self.sample_size = sample_size
        self.samples = None
    
    def generate_stratified_sample(self) -> pd.DataFrame:
        """
        Generate stratified random sample
        Strategy: Sample across confidence levels to ensure diversity
        Returns:
            Sample dataframe
        """
        if len(self.df) <= self.sample_size:
            self.samples = self.df.copy()
            return self.samples
        
        # Stratify by confidence level
        high_conf = self.df[self.df['confidence'] >= 0.7]
        med_conf = self.df[(self.df['confidence'] >= 0.5) & (self.df['confidence'] < 0.7)]
        low_conf = self.df[self.df['confidence'] < 0.5]
        
        # Calculate sample sizes (proportional)
        total = len(self.df)
        n_high = int((len(high_conf) / total) * self.sample_size)
        n_med = int((len(med_conf) / total) * self.sample_size)
        n_low = self.sample_size - n_high - n_med
        
        samples = []
        
        # Sample from each group
        if len(high_conf) > 0:
            samples.append(high_conf.sample(min(n_high, len(high_conf)), random_state=42))
        
        if len(med_conf) > 0:
            samples.append(med_conf.sample(min(n_med, len(med_conf)), random_state=42))
        
        if len(low_conf) > 0:
            samples.append(low_conf.sample(min(n_low, len(low_conf)), random_state=42))
        
        self.samples = pd.concat(samples, ignore_index=True) if samples else self.df.head(self.sample_size)
        
        # Shuffle
        self.samples = self.samples.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return self.samples
    
    def generate_low_confidence_sample(self, threshold: float = 0.6) -> pd.DataFrame:
        """
        Generate sample focusing on low confidence cases
        Args:
            threshold: Confidence threshold
        Returns:
            Sample dataframe
        """
        low_conf = self.df[self.df['confidence'] < threshold].copy()
        
        if len(low_conf) == 0:
            # If no low confidence, use stratified approach
            return self.generate_stratified_sample()
        
        # Sort by confidence ascending (worst first)
        low_conf = low_conf.sort_values('confidence')
        
        self.samples = low_conf.head(min(self.sample_size, len(low_conf)))
        
        return self.samples
    
    def process_corrections(self, corrections: List[Dict]) -> pd.DataFrame:
        """
        Process user corrections into user examples format
        Args:
            corrections: List of dicts with: transcript_id, correct_category, correct_subcategory, correct_tertiary
        Returns:
            DataFrame in user_examples format
        """
        examples = []
        
        for correction in corrections:
            # Find original transcript
            transcript_id = correction.get('transcript_id')
            row = self.df[self.df['transcript_id'] == transcript_id]
            
            if len(row) > 0:
                examples.append({
                    'example_phrase': row.iloc[0]['redacted_transcript'][:200],  # First 200 chars
                    'category': correction['correct_category'],
                    'subcategory': correction.get('correct_subcategory'),
                    'tertiary_category': correction.get('correct_tertiary'),
                    'confidence': 0.95,
                    'source': 'feedback'
                })
        
        return pd.DataFrame(examples) if examples else pd.DataFrame()
    
    def get_correction_stats(self, corrections: List[Dict]) -> Dict:
        """
        Calculate statistics from corrections
        Args:
            corrections: List of correction dicts
        Returns:
            Stats dict
        """
        if not corrections:
            return {
                "total_corrections": 0,
                "correction_rate": 0.0
            }
        
        total_corrections = len(corrections)
        correction_rate = (total_corrections / len(self.samples)) * 100 if self.samples is not None else 0
        
        # Category distribution
        categories = {}
        for corr in corrections:
            cat = corr['correct_category']
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "total_corrections": total_corrections,
            "correction_rate": round(correction_rate, 1),
            "categories_corrected": categories,
            "top_correction": max(categories.items(), key=lambda x: x[1])[0] if categories else None
        }

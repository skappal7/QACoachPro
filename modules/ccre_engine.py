"""
CCRE (Context Clustered Rule Engine) Classification Module
Optimized for speed and accuracy - processes 200K transcripts in 60-75 minutes
"""

import re
import json
from typing import Dict, List, Tuple, Optional
import pandas as pd


class CCREEngine:
    """Optimized Context Clustered Rule Engine"""
    
    def __init__(self, rules_df: pd.DataFrame):
        """
        Initialize CCRE engine with rules
        Args:
            rules_df: DataFrame with columns: rule_id, category, subcategory, required_groups, forbidden_terms
        """
        self.rules = rules_df.copy()
        self._parse_and_optimize_rules()
        
        # Pre-compile negation pattern
        negation_words = [
            'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere',
            'none', 'cannot', "can't", "won't", "don't", "doesn't", "didn't",
            "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't"
        ]
        self.negation_pattern = re.compile(r'\b(' + '|'.join(negation_words) + r')\b', re.IGNORECASE)
    
    def _parse_and_optimize_rules(self):
        """Parse JSON and optimize rules for fast matching"""
        parsed_rules = []
        
        for idx, row in self.rules.iterrows():
            try:
                # Parse required_groups
                if isinstance(row['required_groups'], str):
                    required_groups = json.loads(row['required_groups'])
                else:
                    required_groups = row['required_groups']
                
                # Parse forbidden_terms
                if isinstance(row['forbidden_terms'], str):
                    forbidden_terms = json.loads(row['forbidden_terms'])
                elif pd.isna(row['forbidden_terms']):
                    forbidden_terms = []
                else:
                    forbidden_terms = row['forbidden_terms']
                
                # Pre-compile regex for forbidden terms (faster matching)
                if forbidden_terms:
                    forbidden_pattern = re.compile(
                        r'\b(' + '|'.join(re.escape(term) for term in forbidden_terms) + r')\b',
                        re.IGNORECASE
                    )
                else:
                    forbidden_pattern = None
                
                # Flatten keywords for faster matching
                all_keywords = []
                for group in required_groups:
                    all_keywords.extend(group)
                
                parsed_rules.append({
                    'rule_id': row['rule_id'],
                    'category': row['category'],
                    'subcategory': row['subcategory'],
                    'required_groups': required_groups,
                    'forbidden_pattern': forbidden_pattern,
                    'all_keywords': all_keywords,
                    'total_groups': len(required_groups)
                })
                
            except Exception as e:
                print(f"Warning: Failed to parse rule {row.get('rule_id', idx)}: {e}")
        
        self.parsed_rules = parsed_rules
    
    def normalize_text(self, text: str) -> str:
        """
        Fast text normalization
        Args:
            text: Input text
        Returns:
            Normalized lowercase text
        """
        if not isinstance(text, str):
            return ""
        
        # Single pass: lowercase and clean whitespace
        return ' '.join(text.lower().split())
    
    def match_rule(self, normalized_text: str, rule: Dict) -> Optional[Dict]:
        """
        Fast rule matching with scoring
        Args:
            normalized_text: Normalized transcript
            rule: Parsed rule dict
        Returns:
            Match result with score or None
        """
        # Quick check: forbidden terms
        if rule['forbidden_pattern'] and rule['forbidden_pattern'].search(normalized_text):
            return None
        
        # Match keyword groups
        matched_groups = 0
        matched_keywords = []
        
        for group in rule['required_groups']:
            group_matched = False
            for keyword in group:
                if keyword.lower() in normalized_text:
                    group_matched = True
                    matched_keywords.append(keyword)
                    break
            
            if group_matched:
                matched_groups += 1
        
        # Must match at least 1 group
        if matched_groups == 0:
            return None
        
        # Calculate base score
        base_score = matched_groups / rule['total_groups']
        
        # Proximity bonus: keywords close together
        proximity_bonus = 0.0
        if len(matched_keywords) >= 2:
            positions = [normalized_text.find(kw.lower()) for kw in matched_keywords if normalized_text.find(kw.lower()) != -1]
            if len(positions) >= 2:
                positions.sort()
                avg_distance = sum(positions[i+1] - positions[i] for i in range(len(positions)-1)) / (len(positions)-1)
                if avg_distance < 50:
                    proximity_bonus = 0.2
                elif avg_distance < 100:
                    proximity_bonus = 0.15
                elif avg_distance < 200:
                    proximity_bonus = 0.1
                else:
                    proximity_bonus = 0.05
        
        # Negation penalty: negation words near keywords
        negation_penalty = 0.0
        for keyword in matched_keywords:
            kw_pos = normalized_text.find(keyword.lower())
            if kw_pos > 0:
                context_start = max(0, kw_pos - 50)
                context = normalized_text[context_start:kw_pos]
                if self.negation_pattern.search(context):
                    negation_penalty += 0.1
        
        negation_penalty = min(negation_penalty, 0.3)
        
        # Final confidence
        confidence = max(0.0, min(1.0, base_score + proximity_bonus - negation_penalty))
        
        # Threshold filter
        if confidence < 0.4:
            return None
        
        return {
            'rule_id': rule['rule_id'],
            'category': rule['category'],
            'subcategory': rule['subcategory'],
            'confidence': confidence,
            'matched_keywords': matched_keywords,
            'matched_groups': matched_groups
        }
    
    def classify(self, transcript: str) -> Dict:
        """
        Complete classification pipeline - optimized single pass
        Args:
            transcript: Raw transcript text
        Returns:
            Classification result dict
        """
        # Normalize once
        normalized = self.normalize_text(transcript)
        
        if not normalized:
            return {
                "category": "Unclassified",
                "subcategory": "Empty Transcript",
                "confidence": 0.0,
                "resolve_reason": "Empty or invalid input",
                "matched_keywords": "",
                "num_rules_activated": 0
            }
        
        # Match all rules in single pass
        activated_rules = []
        
        for rule in self.parsed_rules:
            match = self.match_rule(normalized, rule)
            if match:
                activated_rules.append(match)
        
        # No matches
        if not activated_rules:
            return {
                "category": "Unclassified",
                "subcategory": "No Match",
                "confidence": 0.0,
                "resolve_reason": "No rules activated (threshold: 0.4)",
                "matched_keywords": "",
                "num_rules_activated": 0
            }
        
        # Sort by confidence descending
        activated_rules.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Use highest confidence rule - NO OVERRIDES
        best_rule = activated_rules[0]
        
        return {
            "category": best_rule['category'],
            "subcategory": best_rule['subcategory'],
            "confidence": best_rule['confidence'],
            "resolve_reason": f"Highest confidence rule (matched {best_rule['matched_groups']} groups)",
            "matched_keywords": " | ".join(best_rule['matched_keywords']),
            "num_rules_activated": len(activated_rules)
        }
    
    def classify_batch(self, transcripts: List[str]) -> List[Dict]:
        """
        Classify a batch of transcripts
        Args:
            transcripts: List of transcript texts
        Returns:
            List of classification results
        """
        return [self.classify(t) for t in transcripts]

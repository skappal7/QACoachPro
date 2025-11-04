"""
Proximity Matching Engine
Handles NEAR, AND, OR, NOT operators with word distance constraints
Optimized with keyword indexing and early termination
"""

import re
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class ProximityEngine:
    """Fast proximity-based rule matching with optimization"""
    
    def __init__(self, rules_df: pd.DataFrame):
        """
        Initialize proximity engine with rules
        Args:
            rules_df: DataFrame with columns from proximity_rules.csv
        """
        self.rules = rules_df.copy()
        self.total_rules = len(rules_df)
        self._parse_and_index_rules()
    
    def _parse_and_index_rules(self):
        """Parse rules and build inverted index for fast lookup"""
        self.parsed_rules = []
        self.keyword_index = defaultdict(set)  # keyword -> set of rule indices
        
        for idx, row in self.rules.iterrows():
            try:
                # Parse terms (handle None values)
                terms = []
                if pd.notna(row.get('Term1')):
                    terms.append(str(row['Term1']).lower().strip())
                if pd.notna(row.get('Term2')):
                    terms.append(str(row['Term2']).lower().strip())
                if pd.notna(row.get('Term3')):
                    terms.append(str(row['Term3']).lower().strip())
                
                if not terms:
                    continue
                
                # Parse operators
                operators = []
                if pd.notna(row.get('Operator1')):
                    operators.append(str(row['Operator1']).upper())
                if pd.notna(row.get('Operator2')):
                    operators.append(str(row['Operator2']).upper())
                
                # Get proximity constraints
                proximity = float(row.get('Proximity', 0)) if pd.notna(row.get('Proximity')) else 0
                within_words = float(row.get('WithinWords', 0)) if pd.notna(row.get('WithinWords')) else 0
                
                rule_obj = {
                    'rule_id': idx,
                    'cat_no': row.get('CatNo'),
                    'program': row.get('Program'),
                    'terms': terms,
                    'operators': operators,
                    'proximity': int(proximity),
                    'within_words': int(within_words),
                    'sent_by': row.get('SentBy', 'ANY')
                }
                
                self.parsed_rules.append(rule_obj)
                
                # Build inverted index - index all terms for fast candidate selection
                for term in terms:
                    # Index first word of multi-word phrases
                    first_word = term.split()[0] if ' ' in term else term
                    self.keyword_index[first_word].add(len(self.parsed_rules) - 1)
                
            except Exception as e:
                # Skip malformed rules
                continue
        
        print(f"✅ Parsed {len(self.parsed_rules):,} proximity rules")
        print(f"✅ Built keyword index with {len(self.keyword_index):,} unique keywords")
    
    def _tokenize(self, text: str) -> List[str]:
        """Fast tokenization"""
        return text.lower().split()
    
    def _find_term_positions(self, tokens: List[str], term: str) -> List[int]:
        """Find all positions where term appears in tokens"""
        positions = []
        term_tokens = term.split()
        term_len = len(term_tokens)
        
        for i in range(len(tokens) - term_len + 1):
            # Check if multi-word phrase matches
            if tokens[i:i+term_len] == term_tokens:
                positions.append(i)
        
        return positions
    
    def _check_proximity(self, tokens: List[str], rule: Dict) -> Tuple[bool, float]:
        """
        Check if rule matches with proximity constraints
        Returns: (matched, confidence_score)
        """
        terms = rule['terms']
        operators = rule['operators']
        
        if len(terms) == 1:
            # Single term - just check if present
            positions = self._find_term_positions(tokens, terms[0])
            if positions:
                return True, 0.95
            return False, 0.0
        
        # Find positions of first term
        term1_positions = self._find_term_positions(tokens, terms[0])
        if not term1_positions:
            return False, 0.0
        
        # Check each subsequent term with operator
        for term_idx in range(1, len(terms)):
            term = terms[term_idx]
            operator = operators[term_idx - 1] if term_idx - 1 < len(operators) else 'NEAR'
            
            matched = False
            
            if operator == 'NEAR':
                # Check proximity constraint
                max_distance = rule['within_words'] if rule['within_words'] > 0 else rule['proximity']
                if max_distance == 0:
                    max_distance = 5  # Default: within 5 words
                
                for pos1 in term1_positions:
                    # Search within distance
                    search_end = min(pos1 + max_distance + len(term.split()) + 5, len(tokens))
                    search_tokens = tokens[pos1:search_end]
                    
                    if term in ' '.join(search_tokens):
                        matched = True
                        # Update positions for next term check
                        term1_positions = self._find_term_positions(tokens, term)
                        break
            
            elif operator == 'AND':
                # Term must exist anywhere in transcript
                if self._find_term_positions(tokens, term):
                    matched = True
            
            elif operator == 'OR':
                # Either term matches
                matched = True  # Already matched term1
            
            elif operator == 'NOT':
                # Term must NOT exist
                if not self._find_term_positions(tokens, term):
                    matched = True
            
            if not matched:
                return False, 0.0
        
        # All terms matched with constraints
        confidence = 0.90 + (0.05 if len(terms) >= 3 else 0.0)
        return True, min(confidence, 0.98)
    
    def classify(self, transcript: str, program: Optional[str] = None) -> Optional[Dict]:
        """
        Classify transcript using proximity rules
        Args:
            transcript: Text to classify
            program: Optional program filter (e.g., "Capital One")
        Returns:
            Classification result or None
        """
        if not transcript or not isinstance(transcript, str):
            return None
        
        # Tokenize once
        tokens = self._tokenize(transcript)
        if not tokens:
            return None
        
        # Get candidate rules using inverted index (OPTIMIZATION 1)
        candidate_indices = set()
        for token in set(tokens):  # Use set to avoid duplicates
            if token in self.keyword_index:
                candidate_indices.update(self.keyword_index[token])
        
        if not candidate_indices:
            return None
        
        # Filter by program if specified (OPTIMIZATION 2)
        candidates = []
        for idx in candidate_indices:
            rule = self.parsed_rules[idx]
            if program and rule['program'] != program:
                continue
            candidates.append(rule)
        
        if not candidates:
            return None
        
        # Check candidates and find best match
        best_match = None
        best_confidence = 0.0
        
        for rule in candidates:
            matched, confidence = self._check_proximity(tokens, rule)
            
            if matched:
                # Early termination for high-confidence matches (OPTIMIZATION 3)
                if confidence >= 0.95:
                    return {
                        'cat_no': rule['cat_no'],
                        'confidence': confidence,
                        'matched_terms': ' | '.join(rule['terms']),
                        'source': 'proximity',
                        'rule_id': rule['rule_id']
                    }
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = {
                        'cat_no': rule['cat_no'],
                        'confidence': confidence,
                        'matched_terms': ' | '.join(rule['terms']),
                        'source': 'proximity',
                        'rule_id': rule['rule_id']
                    }
        
        return best_match
    
    def classify_batch(self, transcripts: List[str], program: Optional[str] = None) -> List[Optional[Dict]]:
        """Classify multiple transcripts"""
        return [self.classify(t, program) for t in transcripts]
    
    def get_stats(self, program: Optional[str] = None) -> Dict:
        """Get statistics about loaded rules"""
        if program:
            program_rules = [r for r in self.parsed_rules if r['program'] == program]
            return {
                'total_rules': len(program_rules),
                'program': program,
                'unique_keywords': len(set(
                    word for rule in program_rules for term in rule['terms'] for word in term.split()
                ))
            }
        else:
            return {
                'total_rules': len(self.parsed_rules),
                'programs': len(set(r['program'] for r in self.parsed_rules)),
                'unique_keywords': len(self.keyword_index)
            }

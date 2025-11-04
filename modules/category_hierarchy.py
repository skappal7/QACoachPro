"""
Category Hierarchy Module
Validates and enriches classifications with 4-level hierarchy (L1-L4)
"""

import pandas as pd
from typing import Dict, Optional


class CategoryHierarchy:
    """Manages 4-level category hierarchy and validation"""
    
    def __init__(self, hierarchy_df: pd.DataFrame):
        """
        Initialize with hierarchy data
        Args:
            hierarchy_df: DataFrame from category_hierarchy.csv
                         Columns: Program, CatNo, L1, L2, L3, L4
        """
        self.hierarchy = hierarchy_df.copy()
        self._build_lookup()
    
    def _build_lookup(self):
        """Build fast lookup dictionary: (program, cat_no) -> hierarchy"""
        self.lookup = {}
        
        for _, row in self.hierarchy.iterrows():
            key = (row['Program'], row['CatNo'])
            self.lookup[key] = {
                'L1': row['L1'],
                'L2': row['L2'],
                'L3': row['L3'],
                'L4': row['L4']
            }
        
        print(f"✅ Built hierarchy lookup with {len(self.lookup):,} categories")
    
    def get_hierarchy(self, cat_no: str, program: str) -> Optional[Dict]:
        """
        Get full hierarchy for a category number
        Args:
            cat_no: Category number (e.g., "Cat1")
            program: Program name (e.g., "Capital One")
        Returns:
            Dictionary with L1, L2, L3, L4 or None if not found
        """
        return self.lookup.get((program, cat_no))
    
    def validate_and_enrich(self, classification: Dict, program: str) -> Dict:
        """
        Validate classification and enrich with full hierarchy
        Args:
            classification: Result from proximity/examples/keyword engine
                          Must have 'cat_no' field
            program: Program name
        Returns:
            Enriched classification with hierarchy or unclassified result
        """
        if not classification or 'cat_no' not in classification:
            return self._unclassified_result("No classification provided")
        
        cat_no = classification['cat_no']
        hierarchy = self.get_hierarchy(cat_no, program)
        
        if not hierarchy:
            return self._unclassified_result(f"Category {cat_no} not found in hierarchy")
        
        # Enrich with full hierarchy
        return {
            'category': hierarchy['L1'],
            'subcategory': hierarchy['L2'],
            'tertiary': hierarchy['L3'],
            'quaternary': hierarchy['L4'],
            'cat_no': cat_no,
            'confidence': classification.get('confidence', 0.0),
            'matched_keywords': classification.get('matched_terms', ''),
            'source': classification.get('source', 'unknown'),
            'program': program
        }
    
    def _unclassified_result(self, reason: str = "") -> Dict:
        """Return standard unclassified result"""
        return {
            'category': 'Unclassified',
            'subcategory': 'No Match',
            'tertiary': '',
            'quaternary': '',
            'cat_no': 'UNCLASS',
            'confidence': 0.0,
            'matched_keywords': '',
            'source': 'none',
            'program': '',
            'reason': reason
        }
    
    def get_categories_for_program(self, program: str) -> pd.DataFrame:
        """Get all categories for a specific program"""
        return self.hierarchy[self.hierarchy['Program'] == program].copy()
    
    def get_all_programs(self) -> list:
        """Get list of all available programs"""
        return sorted(self.hierarchy['Program'].unique().tolist())
    
    def get_stats(self, program: Optional[str] = None) -> Dict:
        """Get hierarchy statistics"""
        if program:
            df = self.hierarchy[self.hierarchy['Program'] == program]
            return {
                'program': program,
                'total_categories': len(df),
                'unique_l1': df['L1'].nunique(),
                'unique_l2': df['L2'].nunique(),
                'unique_l3': df['L3'].nunique(),
                'unique_l4': df['L4'].nunique()
            }
        else:
            return {
                'total_programs': self.hierarchy['Program'].nunique(),
                'total_categories': len(self.hierarchy),
                'unique_l1': self.hierarchy['L1'].nunique(),
                'unique_l2': self.hierarchy['L2'].nunique(),
                'unique_l3': self.hierarchy['L3'].nunique(),
                'unique_l4': self.hierarchy['L4'].nunique()
            }

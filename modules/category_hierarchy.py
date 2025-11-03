"""
Category Hierarchy Module
Manages and validates 3-level category hierarchy
"""

from typing import Dict, List, Set, Tuple, Optional
import pandas as pd


class CategoryHierarchy:
    """Manages 3-level category hierarchy validation"""
    
    def __init__(self, hierarchy_df: pd.DataFrame):
        """
        Initialize hierarchy from dataframe
        Args:
            hierarchy_df: DataFrame with columns: category, subcategory, tertiary_category
        """
        self.df = hierarchy_df.copy()
        self._build_hierarchy_tree()
    
    def _build_hierarchy_tree(self):
        """Build fast lookup structures for validation"""
        self.hierarchy = {}
        
        for _, row in self.df.iterrows():
            cat = str(row['category']).strip()
            subcat = str(row.get('subcategory', '')).strip() if pd.notna(row.get('subcategory')) else None
            tertiary = str(row.get('tertiary_category', '')).strip() if pd.notna(row.get('tertiary_category')) else None
            
            if cat not in self.hierarchy:
                self.hierarchy[cat] = {}
            
            if subcat:
                if subcat not in self.hierarchy[cat]:
                    self.hierarchy[cat][subcat] = set()
                
                if tertiary:
                    self.hierarchy[cat][subcat].add(tertiary)
        
        # Build flat lists for quick checks
        self.valid_categories = set(self.hierarchy.keys())
        self.valid_pairs = set()
        self.valid_triples = set()
        
        for cat, subcats in self.hierarchy.items():
            for subcat, tertiaries in subcats.items():
                self.valid_pairs.add((cat, subcat))
                for tertiary in tertiaries:
                    self.valid_triples.add((cat, subcat, tertiary))
    
    def validate(self, category: str, subcategory: Optional[str] = None, 
                 tertiary_category: Optional[str] = None) -> Tuple[bool, str]:
        """
        Validate if category path exists in hierarchy
        Args:
            category: Parent category
            subcategory: Optional subcategory
            tertiary_category: Optional tertiary category
        Returns:
            (is_valid, error_message)
        """
        category = category.strip() if category else ""
        subcategory = subcategory.strip() if subcategory else None
        tertiary_category = tertiary_category.strip() if tertiary_category else None
        
        # Check category exists
        if category not in self.valid_categories:
            return False, f"Invalid category: '{category}'"
        
        # If only category provided, it's valid
        if not subcategory:
            return True, ""
        
        # Check category-subcategory pair
        if (category, subcategory) not in self.valid_pairs:
            return False, f"Invalid subcategory '{subcategory}' for category '{category}'"
        
        # If no tertiary provided, pair is valid
        if not tertiary_category:
            return True, ""
        
        # Check full triple
        if (category, subcategory, tertiary_category) not in self.valid_triples:
            return False, f"Invalid tertiary '{tertiary_category}' for {category}/{subcategory}"
        
        return True, ""
    
    def get_subcategories(self, category: str) -> List[str]:
        """Get all valid subcategories for a category"""
        if category in self.hierarchy:
            return list(self.hierarchy[category].keys())
        return []
    
    def get_tertiary_categories(self, category: str, subcategory: str) -> List[str]:
        """Get all valid tertiary categories for a category/subcategory pair"""
        if category in self.hierarchy and subcategory in self.hierarchy[category]:
            return list(self.hierarchy[category][subcategory])
        return []
    
    def get_all_categories(self) -> List[str]:
        """Get all valid parent categories"""
        return list(self.valid_categories)
    
    def find_closest_valid(self, category: str, subcategory: Optional[str] = None,
                          tertiary_category: Optional[str] = None) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Find closest valid category path
        Args:
            category, subcategory, tertiary_category
        Returns:
            (valid_category, valid_subcategory, valid_tertiary)
        """
        # Try exact match first
        is_valid, _ = self.validate(category, subcategory, tertiary_category)
        if is_valid:
            return category, subcategory, tertiary_category
        
        # Try category + subcategory
        if subcategory:
            is_valid, _ = self.validate(category, subcategory)
            if is_valid:
                return category, subcategory, None
        
        # Try just category
        if category in self.valid_categories:
            return category, None, None
        
        # Return unclassified
        return "Unclassified", "Invalid Hierarchy", None
    
    def get_hierarchy_stats(self) -> Dict:
        """Get hierarchy statistics"""
        total_categories = len(self.valid_categories)
        total_subcategories = len(self.valid_pairs)
        total_tertiary = len(self.valid_triples)
        
        return {
            "total_categories": total_categories,
            "total_subcategories": total_subcategories,
            "total_tertiary": total_tertiary,
            "total_paths": total_tertiary if total_tertiary > 0 else total_subcategories
        }

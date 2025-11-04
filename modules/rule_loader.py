"""
Rule Loader Module
Loads and manages rules from multiple sources with program filtering
"""

import pandas as pd
from typing import Optional, Tuple, Dict
import os


class RuleLoader:
    """Loads rules from GitHub CSVs and merges with custom uploads"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize rule loader
        Args:
            data_dir: Directory containing CSV files
        """
        self.data_dir = data_dir
        self.proximity_rules_path = os.path.join(data_dir, "proximity_rules.csv")
        self.hierarchy_path = os.path.join(data_dir, "category_hierarchy.csv")
        self.default_rules_path = os.path.join(data_dir, "default_rules.csv")
        
        # Load all rules once
        self._load_all_rules()
    
    def _load_all_rules(self):
        """Load all rules from CSV files"""
        print("📁 Loading rules from GitHub...")
        
        # Load proximity rules
        if os.path.exists(self.proximity_rules_path):
            self.all_proximity_rules = pd.read_csv(self.proximity_rules_path)
            print(f"   ✅ Loaded {len(self.all_proximity_rules):,} proximity rules")
        else:
            print(f"   ⚠️ proximity_rules.csv not found")
            self.all_proximity_rules = pd.DataFrame()
        
        # Load hierarchy
        if os.path.exists(self.hierarchy_path):
            self.all_hierarchy = pd.read_csv(self.hierarchy_path)
            print(f"   ✅ Loaded {len(self.all_hierarchy):,} hierarchy definitions")
        else:
            print(f"   ⚠️ category_hierarchy.csv not found")
            self.all_hierarchy = pd.DataFrame()
        
        # Load default fallback rules (if exists)
        if os.path.exists(self.default_rules_path):
            self.default_rules = pd.read_csv(self.default_rules_path)
            print(f"   ✅ Loaded {len(self.default_rules):,} default fallback rules")
        else:
            self.default_rules = pd.DataFrame()
    
    def get_program_rules(self, program: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get rules filtered by program
        Args:
            program: Program name (e.g., "Capital One", "PNC")
        Returns:
            Tuple of (proximity_rules_df, hierarchy_df)
        """
        if self.all_proximity_rules.empty or self.all_hierarchy.empty:
            print(f"⚠️ No rules loaded")
            return pd.DataFrame(), pd.DataFrame()
        
        # Filter by program
        proximity = self.all_proximity_rules[
            self.all_proximity_rules['Program'] == program
        ].copy()
        
        hierarchy = self.all_hierarchy[
            self.all_hierarchy['Program'] == program
        ].copy()
        
        print(f"✅ Filtered to {program}: {len(proximity):,} rules, {len(hierarchy):,} categories")
        
        return proximity, hierarchy
    
    def get_all_programs(self) -> list:
        """Get list of all available programs"""
        if self.all_hierarchy.empty:
            return []
        return sorted(self.all_hierarchy['Program'].unique().tolist())
    
    def merge_custom_rules(
        self,
        custom_proximity: Optional[pd.DataFrame],
        custom_hierarchy: Optional[pd.DataFrame],
        base_proximity: pd.DataFrame,
        base_hierarchy: pd.DataFrame,
        mode: str = "extend"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Merge custom uploaded rules with base rules
        Args:
            custom_proximity: Custom proximity rules DataFrame
            custom_hierarchy: Custom hierarchy DataFrame
            base_proximity: Base proximity rules
            base_hierarchy: Base hierarchy
            mode: "extend" (add to base) or "replace" (use only custom)
        Returns:
            Tuple of (merged_proximity, merged_hierarchy)
        """
        if mode == "replace":
            # Use only custom rules
            print("🔄 Using ONLY custom rules (replace mode)")
            return (
                custom_proximity if custom_proximity is not None else pd.DataFrame(),
                custom_hierarchy if custom_hierarchy is not None else pd.DataFrame()
            )
        
        elif mode == "extend":
            # Merge custom with base
            print("🔄 Extending base rules with custom rules")
            
            proximity = base_proximity.copy()
            hierarchy = base_hierarchy.copy()
            
            if custom_proximity is not None and not custom_proximity.empty:
                proximity = pd.concat([base_proximity, custom_proximity], ignore_index=True)
                print(f"   ✅ Added {len(custom_proximity):,} custom proximity rules")
            
            if custom_hierarchy is not None and not custom_hierarchy.empty:
                # Remove duplicates (prefer custom)
                custom_cat_nos = set(custom_hierarchy['CatNo'].unique())
                hierarchy = hierarchy[~hierarchy['CatNo'].isin(custom_cat_nos)]
                hierarchy = pd.concat([hierarchy, custom_hierarchy], ignore_index=True)
                print(f"   ✅ Added {len(custom_hierarchy):,} custom categories")
            
            return proximity, hierarchy
        
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'extend' or 'replace'")
    
    def validate_custom_upload(
        self,
        proximity_df: Optional[pd.DataFrame],
        hierarchy_df: Optional[pd.DataFrame]
    ) -> Tuple[bool, str]:
        """
        Validate custom uploaded rules
        Args:
            proximity_df: Custom proximity rules
            hierarchy_df: Custom hierarchy
        Returns:
            Tuple of (is_valid, error_message)
        """
        errors = []
        
        # Validate proximity rules format
        if proximity_df is not None and not proximity_df.empty:
            required_cols = ['Program', 'CatNo', 'Term1']
            missing = [col for col in required_cols if col not in proximity_df.columns]
            if missing:
                errors.append(f"Proximity rules missing columns: {missing}")
        
        # Validate hierarchy format
        if hierarchy_df is not None and not hierarchy_df.empty:
            required_cols = ['Program', 'CatNo', 'L1', 'L2', 'L3', 'L4']
            missing = [col for col in required_cols if col not in hierarchy_df.columns]
            if missing:
                errors.append(f"Hierarchy missing columns: {missing}")
        
        # Check if they have matching CatNos
        if (proximity_df is not None and not proximity_df.empty and 
            hierarchy_df is not None and not hierarchy_df.empty):
            prox_cats = set(proximity_df['CatNo'].unique())
            hier_cats = set(hierarchy_df['CatNo'].unique())
            
            missing_in_hier = prox_cats - hier_cats
            if missing_in_hier:
                errors.append(f"Rules reference categories not in hierarchy: {list(missing_in_hier)[:5]}")
        
        if errors:
            return False, "; ".join(errors)
        
        return True, "Valid"
    
    def get_default_fallback_rules(self) -> pd.DataFrame:
        """Get default fallback rules (your original 1,990 rules)"""
        return self.default_rules.copy()
    
    def get_stats(self) -> Dict:
        """Get statistics about loaded rules"""
        return {
            'total_proximity_rules': len(self.all_proximity_rules),
            'total_categories': len(self.all_hierarchy),
            'total_programs': self.all_hierarchy['Program'].nunique() if not self.all_hierarchy.empty else 0,
            'programs': self.get_all_programs(),
            'has_default_rules': not self.default_rules.empty
        }

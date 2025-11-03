"""
Supabase Client Helper Module
Handles all Supabase interactions: auth, caching, data isolation
"""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import streamlit as st
import pandas as pd
from supabase import create_client, Client


class SupabaseManager:
    """Manages Supabase authentication and caching"""
    
    def __init__(self, url: str, key: str):
        """Initialize Supabase client"""
        self.client: Client = create_client(url, key)
        self.user_id: Optional[str] = None
        
    def sign_in(self, email: str, password: str) -> Dict[str, Any]:
        """
        Sign in user with email/password
        Returns: {success: bool, user: dict, error: str}
        """
        try:
            response = self.client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            
            self.user_id = response.user.id
            
            # Update last_login in users table
            self.client.table("users").update({
                "last_login": datetime.now().isoformat()
            }).eq("id", self.user_id).execute()
            
            # Get username
            user_data = self.client.table("users").select("username").eq("id", self.user_id).execute()
            username = user_data.data[0]["username"] if user_data.data else email
            
            return {
                "success": True,
                "user": {
                    "id": self.user_id,
                    "email": response.user.email,
                    "username": username
                },
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "user": None,
                "error": str(e)
            }
    
    def sign_out(self) -> bool:
        """Sign out current user"""
        try:
            self.client.auth.sign_out()
            self.user_id = None
            return True
        except:
            return False
    
    def get_session(self) -> Optional[Dict]:
        """Get current session"""
        try:
            session = self.client.auth.get_session()
            if session:
                self.user_id = session.user.id
            return session
        except:
            return None
    
    @staticmethod
    def hash_text(text: str) -> str:
        """Generate MD5 hash for caching keys"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def cache_classification(
        self, 
        transcript_hash: str, 
        category: str, 
        subcategory: str, 
        confidence: float,
        matched_keywords: str,
        resolve_reason: str
    ) -> bool:
        """
        Cache classification result (7-day TTL)
        Uses upsert to avoid duplicate key errors
        Returns: Success status
        """
        try:
            data = {
                "user_id": self.user_id,
                "transcript_hash": transcript_hash,
                "category": category,
                "subcategory": subcategory,
                "confidence": confidence,
                "matched_keywords": matched_keywords,
                "resolve_reason": resolve_reason,
                "created_at": datetime.now().isoformat(),
                "expires_at": (datetime.now() + timedelta(days=7)).isoformat()
            }
            
            # Check if exists first
            existing = self.client.table("classification_cache").select("id").eq(
                "user_id", self.user_id
            ).eq("transcript_hash", transcript_hash).execute()
            
            if existing.data:
                # Update existing
                self.client.table("classification_cache").update(data).eq(
                    "user_id", self.user_id
                ).eq("transcript_hash", transcript_hash).execute()
            else:
                # Insert new
                self.client.table("classification_cache").insert(data).execute()
            
            return True
        except Exception as e:
            # Silently fail - caching is not critical
            return False
    
    def get_cached_classification(self, transcript_hash: str) -> Optional[Dict]:
        """
        Retrieve cached classification result
        Returns: Classification dict or None if not found/expired
        """
        try:
            result = self.client.table("classification_cache").select(
                "category, subcategory, confidence, matched_keywords, resolve_reason"
            ).eq("user_id", self.user_id).eq("transcript_hash", transcript_hash).execute()
            
            if result.data:
                return result.data[0]
            return None
        except:
            return None
    
    def cache_coaching(
        self, 
        agent_name: str, 
        prompt_hash: str, 
        coaching_json: Dict,
        model_used: str
    ) -> bool:
        """
        Cache coaching result (7-day TTL)
        Uses upsert to avoid duplicate key errors
        Returns: Success status
        """
        try:
            data = {
                "user_id": self.user_id,
                "agent_name": agent_name,
                "prompt_hash": prompt_hash,
                "coaching_json": coaching_json,
                "model_used": model_used,
                "created_at": datetime.now().isoformat(),
                "expires_at": (datetime.now() + timedelta(days=7)).isoformat()
            }
            
            # Check if exists first
            existing = self.client.table("coaching_cache").select("id").eq(
                "user_id", self.user_id
            ).eq("agent_name", agent_name).eq("prompt_hash", prompt_hash).execute()
            
            if existing.data:
                # Update existing
                self.client.table("coaching_cache").update(data).eq(
                    "user_id", self.user_id
                ).eq("agent_name", agent_name).eq("prompt_hash", prompt_hash).execute()
            else:
                # Insert new
                self.client.table("coaching_cache").insert(data).execute()
            
            return True
        except Exception as e:
            # Silently fail - caching is not critical
            return False
    
    def get_cached_coaching(self, agent_name: str, prompt_hash: str) -> Optional[Dict]:
        """
        Retrieve cached coaching result
        Returns: Coaching dict or None if not found/expired
        """
        try:
            result = self.client.table("coaching_cache").select(
                "coaching_json, model_used, created_at"
            ).eq("user_id", self.user_id).eq(
                "agent_name", agent_name
            ).eq("prompt_hash", prompt_hash).execute()
            
            if result.data:
                return result.data[0]
            return None
        except:
            return None
    
    def log_upload(self, filename: str, row_count: int, processing_time: float) -> bool:
        """Log upload history"""
        try:
            self.client.table("upload_history").insert({
                "user_id": self.user_id,
                "filename": filename,
                "row_count": row_count,
                "processing_time_seconds": processing_time,
                "created_at": datetime.now().isoformat()
            }).execute()
            return True
        except:
            return False
    
    def get_upload_history(self, limit: int = 10) -> List[Dict]:
        """Get user's upload history"""
        try:
            result = self.client.table("upload_history").select(
                "filename, row_count, processing_time_seconds, created_at"
            ).eq("user_id", self.user_id).order(
                "created_at", desc=True
            ).limit(limit).execute()
            
            return result.data if result.data else []
        except:
            return []
    
    def cleanup_expired_cache(self) -> Dict[str, int]:
        """
        Clean up expired cache entries
        Returns: Count of deleted records
        """
        try:
            # Delete expired classifications
            class_result = self.client.table("classification_cache").delete().lt(
                "expires_at", datetime.now().isoformat()
            ).execute()
            
            # Delete expired coaching
            coach_result = self.client.table("coaching_cache").delete().lt(
                "expires_at", datetime.now().isoformat()
            ).execute()
            
            return {
                "classifications": len(class_result.data) if class_result.data else 0,
                "coaching": len(coach_result.data) if coach_result.data else 0
            }
        except Exception as e:
            st.warning(f"Cache cleanup failed: {str(e)}")
            return {"classifications": 0, "coaching": 0}
    
    # ==========================================
    # Category Hierarchy Methods
    # ==========================================
    
    def upload_hierarchy(self, hierarchy_df) -> bool:
        """
        Upload category hierarchy to Supabase
        Args:
            hierarchy_df: DataFrame with category, subcategory, tertiary_category
        Returns:
            Success status
        """
        try:
            # Clear existing hierarchy for this user
            self.client.table("category_hierarchy").delete().eq("user_id", self.user_id).execute()
            
            # Insert new hierarchy
            records = []
            for _, row in hierarchy_df.iterrows():
                records.append({
                    "user_id": self.user_id,
                    "category": str(row['category']).strip(),
                    "subcategory": str(row.get('subcategory', '')).strip() if pd.notna(row.get('subcategory')) else None,
                    "tertiary_category": str(row.get('tertiary_category', '')).strip() if pd.notna(row.get('tertiary_category')) else None
                })
            
            if records:
                self.client.table("category_hierarchy").insert(records).execute()
            
            return True
        except Exception as e:
            st.error(f"Failed to upload hierarchy: {str(e)}")
            return False
    
    def get_hierarchy(self):
        """
        Retrieve user's category hierarchy
        Returns:
            DataFrame or None
        """
        try:
            result = self.client.table("category_hierarchy").select(
                "category, subcategory, tertiary_category"
            ).eq("user_id", self.user_id).execute()
            
            if result.data:
                import pandas as pd
                return pd.DataFrame(result.data)
            return None
        except:
            return None
    
    # ==========================================
    # User Examples Methods
    # ==========================================
    
    def upload_user_examples(self, examples_df) -> bool:
        """
        Upload user examples to Supabase
        Args:
            examples_df: DataFrame with example_phrase, category, subcategory, tertiary_category
        Returns:
            Success status
        """
        try:
            records = []
            for _, row in examples_df.iterrows():
                records.append({
                    "user_id": self.user_id,
                    "example_phrase": str(row['example_phrase']).strip(),
                    "category": str(row['category']).strip(),
                    "subcategory": str(row.get('subcategory', '')).strip() if pd.notna(row.get('subcategory')) else None,
                    "tertiary_category": str(row.get('tertiary_category', '')).strip() if pd.notna(row.get('tertiary_category')) else None,
                    "confidence": float(row.get('confidence', 0.95)),
                    "source": str(row.get('source', 'manual'))
                })
            
            if records:
                self.client.table("user_examples").insert(records).execute()
            
            return True
        except Exception as e:
            st.error(f"Failed to upload examples: {str(e)}")
            return False
    
    def get_user_examples(self):
        """
        Retrieve user's examples
        Returns:
            DataFrame or None
        """
        try:
            result = self.client.table("user_examples").select(
                "example_phrase, category, subcategory, tertiary_category, confidence, source"
            ).eq("user_id", self.user_id).execute()
            
            if result.data:
                import pandas as pd
                return pd.DataFrame(result.data)
            return None
        except:
            return None
    
    def delete_user_example(self, example_phrase: str) -> bool:
        """Delete a specific user example"""
        try:
            self.client.table("user_examples").delete().eq(
                "user_id", self.user_id
            ).eq("example_phrase", example_phrase).execute()
            return True
        except:
            return False
    
    def clear_all_user_examples(self) -> bool:
        """Clear all user examples"""
        try:
            self.client.table("user_examples").delete().eq("user_id", self.user_id).execute()
            return True
        except:
            return False


def init_supabase() -> SupabaseManager:
    """Initialize Supabase client from secrets"""
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return SupabaseManager(url, key)
    except Exception as e:
        st.error(f"❌ Supabase initialization failed: {str(e)}")
        st.stop()

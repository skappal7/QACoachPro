"""
LLM Coaching Module
Generates coaching insights using OpenRouter or Local LLM
"""

import json
import hashlib
import requests
from typing import Dict, List, Optional, Tuple
import pandas as pd
import streamlit as st
from modules.supabase_client import SupabaseManager


# OpenRouter free models with context windows
OPENROUTER_MODELS = {
    "deepseek/deepseek-chat-v3.1:free": 64000,
    "deepseek/deepseek-r1-distill-llama-70b:free": 32000,
    "meta-llama/llama-3.3-70b-instruct:free": 8000,
    "qwen/qwen2.5-vl-32b-instruct:free": 32000,
    "qwen/qwen3-235b-a22b:free": 32000,
    "mistralai/mistral-7b-instruct:free": 8000,
    "openchat/openchat-7b:free": 8000,
    "gryphe/mythomax-l2-13b:free": 8000,
    "openai/gpt-oss-20b:free": 8000,
    "meta-llama/llama-4-maverick:free": 16000,
    "moonshotai/kimi-vl-a3b-thinking:free": 32000,
    "moonshotai/kimi-k2:free": 16000,
}


class CoachingEngine:
    """LLM-powered coaching insights generator"""
    
    def __init__(
        self, 
        api_key: str,
        provider: str = "openrouter",
        local_endpoint: Optional[str] = None
    ):
        """
        Initialize coaching engine
        Args:
            api_key: OpenRouter API key
            provider: "openrouter" or "local"
            local_endpoint: Local LLM endpoint URL
        """
        self.api_key = api_key
        self.provider = provider
        
        if provider == "openrouter":
            self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
        else:
            self.endpoint = local_endpoint
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count (1 token ≈ 4 characters)"""
        return len(text) // 4
    
    @staticmethod
    def hash_prompt(prompt: str) -> str:
        """Generate hash for prompt caching"""
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def prepare_agent_data(
        self, 
        df: pd.DataFrame, 
        agent_name: str,
        max_transcripts: int = 5
    ) -> Dict:
        """
        Prepare agent data for coaching
        Args:
            df: Classified transcripts dataframe
            agent_name: Agent name
            max_transcripts: Maximum number of sample transcripts
        Returns:
            Agent data dict
        """
        # Filter agent transcripts
        agent_df = df[df['agent_name'] == agent_name].copy()
        
        if len(agent_df) == 0:
            return None
        
        # Calculate metrics
        metrics = {
            "total_calls": len(agent_df),
            "avg_confidence": round(agent_df['confidence'].mean(), 3),
            "unique_categories": agent_df['category'].nunique(),
            "unique_subcategories": agent_df['subcategory'].nunique(),
        }
        
        # Category distribution
        category_dist = agent_df['category'].value_counts().head(10).to_dict()
        
        # Sample diverse transcripts
        sample_transcripts = []
        
        # Try to get one from each top category
        top_categories = agent_df['category'].value_counts().head(5).index.tolist()
        
        for cat in top_categories:
            cat_df = agent_df[agent_df['category'] == cat]
            if len(cat_df) > 0:
                sample = cat_df.iloc[0]
                sample_transcripts.append({
                    "transcript": sample['redacted_transcript'][:500],  # Truncate long transcripts
                    "category": sample['category'],
                    "subcategory": sample['subcategory'],
                    "confidence": sample['confidence']
                })
            
            if len(sample_transcripts) >= max_transcripts:
                break
        
        return {
            "agent_name": agent_name,
            "metrics": metrics,
            "category_distribution": category_dist,
            "sample_transcripts": sample_transcripts
        }
    
    def build_coaching_prompt(self, agent_data: Dict, context_limit: int) -> Tuple[str, bool]:
        """
        Build coaching prompt with token awareness
        Args:
            agent_data: Agent data dict
            context_limit: Model context window size
        Returns:
            (prompt, was_truncated)
        """
        # Reserve 30% for response
        available_tokens = int(context_limit * 0.7)
        
        # Base prompt template
        base_prompt = f"""You are an expert contact center coach analyzing agent performance data.

Agent: {agent_data['agent_name']}

Performance Metrics:
- Total Calls: {agent_data['metrics']['total_calls']}
- Average Confidence: {agent_data['metrics']['avg_confidence']}
- Unique Categories: {agent_data['metrics']['unique_categories']}
- Unique Subcategories: {agent_data['metrics']['unique_subcategories']}

Category Distribution:
{json.dumps(agent_data['category_distribution'], indent=2)}

Sample Transcripts (PII Redacted):
"""
        
        # Add sample transcripts
        transcripts_text = ""
        for i, t in enumerate(agent_data['sample_transcripts'], 1):
            transcripts_text += f"""
Transcript {i}:
Category: {t['category']} → {t['subcategory']} (Confidence: {t['confidence']:.3f})
Text: {t['transcript']}

"""
        
        full_prompt = base_prompt + transcripts_text + """

Based on this data, provide coaching insights in JSON format:

{
  "root_cause": "Primary issue or pattern identified",
  "coaching_points": [
    "Actionable coaching point 1",
    "Actionable coaching point 2",
    "Actionable coaching point 3"
  ],
  "sample_script": "Example of an improved response",
  "kpi_recommendations": [
    "Recommendation 1",
    "Recommendation 2"
  ],
  "strengths": [
    "Positive aspect 1",
    "Positive aspect 2"
  ],
  "priority": "High/Medium/Low"
}

Respond ONLY with valid JSON, no additional text."""
        
        # Check token count
        estimated_tokens = self.estimate_tokens(full_prompt)
        
        if estimated_tokens > available_tokens:
            # Truncate transcripts
            truncated_count = min(3, len(agent_data['sample_transcripts']))
            truncated_transcripts = agent_data['sample_transcripts'][:truncated_count]
            
            transcripts_text = "[... truncated for context limit ...]\n\n"
            for i, t in enumerate(truncated_transcripts, 1):
                transcripts_text += f"""
Transcript {i}:
Category: {t['category']} → {t['subcategory']}
Text: {t['transcript'][:200]}...

"""
            
            full_prompt = base_prompt + transcripts_text + """

Based on this data, provide coaching insights in JSON format (same format as above).
Respond ONLY with valid JSON."""
            
            return full_prompt, True
        
        return full_prompt, False
    
    def call_llm(
        self, 
        prompt: str, 
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Dict:
        """
        Call LLM API
        Args:
            prompt: Prompt text
            model: Model name
            temperature: Temperature parameter
            max_tokens: Max response tokens
        Returns:
            Response dict
        """
        headers = {
            "Content-Type": "application/json",
        }
        
        if self.provider == "openrouter":
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            raise Exception("Request timed out after 60 seconds")
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
    
    def parse_coaching_response(self, response: Dict) -> Dict:
        """
        Parse LLM response and extract coaching JSON
        Args:
            response: Raw LLM response
        Returns:
            Parsed coaching dict
        """
        try:
            content = response['choices'][0]['message']['content']
            
            # Try to parse as JSON
            try:
                coaching = json.loads(content)
                return coaching
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                import re
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                if json_match:
                    coaching = json.loads(json_match.group(1))
                    return coaching
                else:
                    # Fallback: return raw content
                    return {
                        "raw_response": content,
                        "parse_error": "Could not extract JSON"
                    }
        except Exception as e:
            return {
                "error": f"Failed to parse response: {str(e)}",
                "raw_response": str(response)
            }
    
    def generate_coaching(
        self,
        df: pd.DataFrame,
        agent_name: str,
        model: str,
        supabase: Optional[SupabaseManager] = None
    ) -> Dict:
        """
        Generate coaching for an agent
        Args:
            df: Classified transcripts dataframe
            agent_name: Agent name
            model: LLM model to use
            supabase: Supabase manager for caching
        Returns:
            Coaching result dict
        """
        # Prepare agent data
        agent_data = self.prepare_agent_data(df, agent_name)
        
        if agent_data is None:
            return {
                "error": f"No data found for agent: {agent_name}"
            }
        
        # Get model context limit
        context_limit = OPENROUTER_MODELS.get(model, 8000)
        
        # Build prompt
        prompt, was_truncated = self.build_coaching_prompt(agent_data, context_limit)
        prompt_hash = self.hash_prompt(prompt)
        
        # Check cache first
        if supabase:
            cached = supabase.get_cached_coaching(agent_name, prompt_hash)
            if cached:
                return {
                    "agent_name": agent_name,
                    "coaching": cached['coaching_json'],
                    "model_used": cached['model_used'],
                    "cached": True,
                    "cached_at": cached['created_at'],
                    "was_truncated": was_truncated
                }
        
        # Call LLM
        try:
            response = self.call_llm(prompt, model)
            coaching = self.parse_coaching_response(response)
            
            # Cache result
            if supabase and 'error' not in coaching:
                supabase.cache_coaching(agent_name, prompt_hash, coaching, model)
            
            return {
                "agent_name": agent_name,
                "coaching": coaching,
                "model_used": model,
                "cached": False,
                "was_truncated": was_truncated
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "agent_name": agent_name
            }


def render_coaching_interface(df: pd.DataFrame, supabase: SupabaseManager):
    """
    Render coaching interface in Streamlit
    Args:
        df: Classified transcripts dataframe
        supabase: Supabase manager
    """
    st.header("🎓 LLM Coaching Generator")
    
    # Check for agent data
    if 'agent_name' not in df.columns:
        st.error("❌ No agent data available. LLM coaching requires agent_name column.")
        return
    
    # Get unique agents
    agents = df['agent_name'].dropna().unique().tolist()
    
    if len(agents) == 0:
        st.warning("⚠️ No valid agent names found in data")
        return
    
    st.info(f"📊 Found {len(agents)} unique agents in dataset")
    
    # Configuration
    st.subheader("⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        provider = st.radio(
            "LLM Provider",
            ["OpenRouter", "Local LLM"],
            help="Choose between OpenRouter API or local LLM"
        )
    
    with col2:
        if provider == "OpenRouter":
            # Get API key from secrets or user input
            api_key = st.secrets.get("OPENROUTER_API_KEY", "")
            if not api_key:
                api_key = st.text_input(
                    "OpenRouter API Key",
                    type="password",
                    help="Get your free API key from openrouter.ai"
                )
            
            if api_key:
                model = st.selectbox(
                    "Select Model",
                    list(OPENROUTER_MODELS.keys()),
                    help="All models are free tier"
                )
                
                context = OPENROUTER_MODELS[model]
                st.caption(f"Context window: {context:,} tokens")
            else:
                st.warning("⚠️ Please provide OpenRouter API key")
                return
        else:
            local_endpoint = st.text_input(
                "Local LLM Endpoint",
                value="http://localhost:1234/v1/chat/completions",
                help="LM Studio or Ollama endpoint"
            )
            
            model = st.text_input(
                "Model Name",
                value="local-model",
                help="Name of your local model"
            )
            
            api_key = ""  # Not needed for local
            context = 8000  # Default context
    
    st.markdown("---")
    
    # Agent selection
    st.subheader("👥 Select Agents")
    st.caption("Maximum 5 agents per batch for performance")
    
    selected_agents = st.multiselect(
        "Agents",
        agents,
        max_selections=5,
        help="Select up to 5 agents"
    )
    
    if not selected_agents:
        st.info("ℹ️ Please select at least one agent")
        return
    
    # Generate coaching button
    if st.button("🚀 Generate Coaching", type="primary", use_container_width=True):
        
        # Initialize coaching engine
        if provider == "OpenRouter":
            engine = CoachingEngine(api_key, provider="openrouter")
        else:
            engine = CoachingEngine(api_key, provider="local", local_endpoint=local_endpoint)
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        for i, agent in enumerate(selected_agents):
            status_text.text(f"Generating coaching for {agent}... ({i+1}/{len(selected_agents)})")
            progress_bar.progress((i) / len(selected_agents))
            
            result = engine.generate_coaching(df, agent, model, supabase)
            results.append(result)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Coaching generation complete!")
        
        st.markdown("---")
        
        # Display results
        for result in results:
            if 'error' in result:
                st.error(f"❌ {result['agent_name']}: {result['error']}")
                continue
            
            with st.expander(f"🎓 Coaching for {result['agent_name']}", expanded=True):
                
                # Show cache status
                if result.get('cached', False):
                    st.success(f"⚡ Loaded from cache (generated {result['cached_at'][:10]})")
                else:
                    st.info(f"🆕 Newly generated using {result['model_used']}")
                
                if result.get('was_truncated', False):
                    st.warning("⚠️ Prompt was truncated to fit context window")
                
                coaching = result['coaching']
                
                # Check for parse errors
                if 'parse_error' in coaching:
                    st.warning("⚠️ Could not parse structured JSON. Showing raw response:")
                    st.text(coaching.get('raw_response', ''))
                    continue
                
                if 'error' in coaching:
                    st.error(f"❌ {coaching['error']}")
                    continue
                
                # Display structured coaching
                st.markdown(f"### 🎯 Root Cause")
                st.info(coaching.get('root_cause', 'N/A'))
                
                st.markdown("### 📝 Coaching Points")
                for i, point in enumerate(coaching.get('coaching_points', []), 1):
                    st.markdown(f"{i}. {point}")
                
                st.markdown("### 💬 Sample Script")
                st.success(coaching.get('sample_script', 'N/A'))
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 💪 Strengths")
                    for strength in coaching.get('strengths', []):
                        st.markdown(f"✅ {strength}")
                
                with col2:
                    st.markdown("### 📊 KPI Recommendations")
                    for rec in coaching.get('kpi_recommendations', []):
                        st.markdown(f"📈 {rec}")
                
                priority = coaching.get('priority', 'Medium')
                if priority == "High":
                    st.error(f"🚨 Priority: {priority}")
                elif priority == "Low":
                    st.success(f"✅ Priority: {priority}")
                else:
                    st.warning(f"⚠️ Priority: {priority}")

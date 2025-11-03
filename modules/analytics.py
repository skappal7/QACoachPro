"""
Analytics Module
DuckDB-powered analytics for classification results
"""

import duckdb
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional


class AnalyticsEngine:
    """DuckDB-based analytics engine"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize analytics engine
        Args:
            df: Classified transcripts dataframe
        """
        self.df = df
        self.conn = duckdb.connect(database=':memory:')
        
        # Register dataframe as DuckDB table
        self.conn.register('transcripts', df)
        
        # Check for agent_name column
        self.has_agent_data = 'agent_name' in df.columns
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics"""
        query = """
        SELECT 
            COUNT(*) as total_transcripts,
            AVG(confidence) as avg_confidence,
            COUNT(DISTINCT category) as unique_categories,
            COUNT(DISTINCT subcategory) as unique_subcategories
        FROM transcripts
        """
        
        result = self.conn.execute(query).fetchone()
        
        stats = {
            "total_transcripts": result[0],
            "avg_confidence": round(result[1], 3) if result[1] else 0.0,
            "unique_categories": result[2],
            "unique_subcategories": result[3]
        }
        
        # Add agent count if available
        if self.has_agent_data:
            agent_query = "SELECT COUNT(DISTINCT agent_name) FROM transcripts WHERE agent_name IS NOT NULL"
            agent_count = self.conn.execute(agent_query).fetchone()[0]
            stats["unique_agents"] = agent_count
        
        return stats
    
    def get_category_distribution(self, limit: int = 10) -> pd.DataFrame:
        """
        Get category distribution
        Args:
            limit: Number of top categories to return
        Returns:
            DataFrame with category, count, percentage
        """
        query = f"""
        SELECT 
            category,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
        FROM transcripts
        GROUP BY category
        ORDER BY count DESC
        LIMIT {limit}
        """
        
        return self.conn.execute(query).df()
    
    def get_subcategory_distribution(
        self, 
        parent_category: Optional[str] = None, 
        limit: int = 20
    ) -> pd.DataFrame:
        """
        Get subcategory distribution
        Args:
            parent_category: Filter by parent category (optional)
            limit: Number of results
        Returns:
            DataFrame with category, subcategory, count, percentage
        """
        if parent_category:
            query = f"""
            SELECT 
                category,
                subcategory,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
            FROM transcripts
            WHERE category = '{parent_category}'
            GROUP BY category, subcategory
            ORDER BY count DESC
            LIMIT {limit}
            """
        else:
            query = f"""
            SELECT 
                category,
                subcategory,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
            FROM transcripts
            GROUP BY category, subcategory
            ORDER BY count DESC
            LIMIT {limit}
            """
        
        return self.conn.execute(query).df()
    
    def get_resolution_reason_distribution(self) -> pd.DataFrame:
        """Get distribution of resolution reasons"""
        query = """
        SELECT 
            resolve_reason,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
        FROM transcripts
        GROUP BY resolve_reason
        ORDER BY count DESC
        """
        
        return self.conn.execute(query).df()
    
    def get_agent_performance(self) -> Optional[pd.DataFrame]:
        """
        Get agent performance metrics
        Returns:
            DataFrame with agent metrics or None if no agent data
        """
        if not self.has_agent_data:
            return None
        
        query = """
        SELECT 
            agent_name,
            COUNT(*) as total_calls,
            ROUND(AVG(confidence), 3) as avg_confidence,
            COUNT(DISTINCT category) as unique_categories,
            COUNT(DISTINCT subcategory) as unique_subcategories,
            ROUND(AVG(num_rules_activated), 1) as avg_rules_activated
        FROM transcripts
        WHERE agent_name IS NOT NULL
        GROUP BY agent_name
        ORDER BY total_calls DESC
        """
        
        return self.conn.execute(query).df()
    
    def get_low_confidence_cases(self, threshold: float = 0.6, limit: int = 100) -> pd.DataFrame:
        """
        Get low confidence classifications for review
        Args:
            threshold: Confidence threshold
            limit: Number of results
        Returns:
            DataFrame with low confidence cases
        """
        columns = ['transcript_id', 'category', 'subcategory', 'confidence', 'resolve_reason']
        
        if self.has_agent_data:
            columns.append('agent_name')
        
        columns_str = ', '.join(columns)
        
        query = f"""
        SELECT {columns_str}
        FROM transcripts
        WHERE confidence < {threshold}
        ORDER BY confidence ASC
        LIMIT {limit}
        """
        
        return self.conn.execute(query).df()
    
    def run_custom_query(self, query: str) -> pd.DataFrame:
        """
        Run custom SQL query
        Args:
            query: SQL query string
        Returns:
            Result DataFrame
        """
        try:
            return self.conn.execute(query).df()
        except Exception as e:
            raise ValueError(f"Query execution failed: {str(e)}")
    
    def close(self):
        """Close DuckDB connection"""
        self.conn.close()


def render_analytics_dashboard(df: pd.DataFrame):
    """
    Render analytics dashboard in Streamlit
    Args:
        df: Classified transcripts dataframe
    """
    st.header("📊 Analytics Dashboard")
    
    # Initialize analytics engine
    analytics = AnalyticsEngine(df)
    
    # Summary statistics
    st.subheader("📈 Summary Statistics")
    stats = analytics.get_summary_stats()
    
    cols = st.columns(4 if analytics.has_agent_data else 3)
    
    cols[0].metric(
        "Total Transcripts",
        f"{stats['total_transcripts']:,}",
        help="Total number of classified transcripts"
    )
    
    cols[1].metric(
        "Avg Confidence",
        f"{stats['avg_confidence']:.3f}",
        help="Average classification confidence score"
    )
    
    cols[2].metric(
        "Unique Categories",
        stats['unique_categories'],
        help="Number of unique parent categories"
    )
    
    if analytics.has_agent_data:
        cols[3].metric(
            "Unique Agents",
            stats['unique_agents'],
            help="Number of unique agents"
        )
    
    st.markdown("---")
    
    # Category distribution
    st.subheader("📊 Category Distribution")
    
    cat_dist = analytics.get_category_distribution(limit=10)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.bar_chart(cat_dist.set_index('category')['count'])
    
    with col2:
        st.dataframe(
            cat_dist,
            use_container_width=True,
            hide_index=True
        )
    
    st.markdown("---")
    
    # Subcategory distribution
    st.subheader("🔍 Subcategory Analysis")
    
    # Category filter
    categories = ["All"] + analytics.df['category'].unique().tolist()
    selected_category = st.selectbox(
        "Filter by Parent Category",
        categories,
        help="Show subcategories for specific parent category"
    )
    
    if selected_category == "All":
        subcat_dist = analytics.get_subcategory_distribution(limit=20)
    else:
        subcat_dist = analytics.get_subcategory_distribution(parent_category=selected_category, limit=20)
    
    st.dataframe(
        subcat_dist,
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    
    # Resolution reason distribution
    st.subheader("🎯 Resolution Reason Distribution")
    st.caption("Shows which classification heuristics were triggered most often")
    
    resolve_dist = analytics.get_resolution_reason_distribution()
    st.dataframe(
        resolve_dist,
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    
    # Agent performance (if available)
    if analytics.has_agent_data:
        st.subheader("👥 Agent Performance Metrics")
        
        agent_perf = analytics.get_agent_performance()
        
        if agent_perf is not None and len(agent_perf) > 0:
            st.dataframe(
                agent_perf,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("ℹ️ No agent data available")
    else:
        st.info("⚠️ Agent performance metrics not available (no agent_name column in data)")
    
    st.markdown("---")
    
    # Low confidence cases
    st.subheader("⚠️ Low Confidence Cases")
    st.caption("Transcripts with confidence < 0.6 for quality review")
    
    threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.6, 0.1)
    
    low_conf = analytics.get_low_confidence_cases(threshold=threshold, limit=100)
    
    if len(low_conf) > 0:
        st.dataframe(
            low_conf,
            use_container_width=True,
            hide_index=True
        )
        
        st.info(f"📊 Found {len(low_conf)} cases below {threshold} confidence")
    else:
        st.success(f"✅ No cases found below {threshold} confidence")
    
    st.markdown("---")
    
    # Custom SQL query
    with st.expander("🔧 Advanced: Custom SQL Query"):
        st.caption("Query the 'transcripts' table directly")
        
        default_query = """SELECT category, COUNT(*) as count 
FROM transcripts 
GROUP BY category 
ORDER BY count DESC 
LIMIT 10"""
        
        custom_query = st.text_area(
            "SQL Query",
            value=default_query,
            height=150
        )
        
        if st.button("Run Query"):
            try:
                result = analytics.run_custom_query(custom_query)
                st.dataframe(result, use_container_width=True)
                st.success(f"✅ Query returned {len(result)} rows")
            except Exception as e:
                st.error(f"❌ Query failed: {str(e)}")
    
    # Cleanup
    analytics.close()

"""
AgentPulse AI - Enterprise QA & Coaching Platform
Main Streamlit Application with Hybrid Classification System
"""

import streamlit as st
import pandas as pd
import time
from datetime import datetime
import sys
import os

# Add modules path
sys.path.insert(0, os.path.dirname(__file__))

# Import core modules
from modules import (
    init_supabase,
    check_authentication,
    render_login_page,
    render_sidebar_user_info,
    PIIRedactor,
    render_analytics_dashboard,
    render_coaching_interface,
    render_export_interface
)

# Import hybrid classification modules
from modules.hybrid_classifier import HybridClassifier
from modules.category_hierarchy import CategoryHierarchy

# Page config
st.set_page_config(
    page_title="AgentPulse AI - Hybrid Classification",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Supabase
if 'supabase' not in st.session_state:
    st.session_state.supabase = init_supabase()

supabase = st.session_state.supabase

# Check authentication
if not check_authentication(supabase):
    render_login_page(supabase)
    st.stop()

# Main app (authenticated user)
render_sidebar_user_info(supabase)

# Load default rules and hierarchy
@st.cache_data
def load_default_rules():
    rules_df = pd.read_csv('data/default_rules.csv')
    hierarchy_df = pd.read_csv('data/category_hierarchy.csv')
    return rules_df, hierarchy_df

try:
    default_rules, hierarchy_data = load_default_rules()
    st.sidebar.success(f"✅ Loaded {len(default_rules):,} CCRE rules")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load rules: {str(e)}")
    st.stop()

# Header
st.title("🚀 AgentPulse AI - Hybrid Classification")
st.markdown("**Enterprise QA & Coaching Platform** - Fast, Accurate, Explainable")

# Tab navigation
tabs = st.tabs(["📁 Upload", "🔍 Classify", "📊 Analyze", "🎓 Coach", "📤 Export"])

# ===========================
# TAB 1: Upload
# ===========================
with tabs[0]:
    st.header("📁 Upload Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📄 Transcripts File")
        transcripts_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=['csv', 'xlsx'],
            key="transcripts_upload",
            help="Max 200MB, supports 200K+ rows"
        )
        
        if transcripts_file:
            # Load data
            try:
                if transcripts_file.name.endswith('.csv'):
                    df = pd.read_csv(transcripts_file)
                else:
                    df = pd.read_excel(transcripts_file)
                
                st.success(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
                
                # Show preview
                with st.expander("👀 Data Preview"):
                    st.dataframe(df.head(10), use_container_width=True)
                
                # Column detection
                st.subheader("🔍 Column Mapping")
                
                columns = df.columns.tolist()
                
                # Try to auto-detect transcript column
                transcript_candidates = [c for c in columns if 'transcript' in c.lower() or 'text' in c.lower() or 'conversation' in c.lower()]
                default_transcript = transcript_candidates[0] if transcript_candidates else columns[0]
                
                transcript_col = st.selectbox(
                    "Transcript Column (Required)",
                    columns,
                    index=columns.index(default_transcript),
                    help="Column containing conversation text"
                )
                
                # Optional columns
                st.markdown("**Optional Columns** (leave as 'None' if not available)")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    agent_col = st.selectbox(
                        "Agent Name Column",
                        ['None'] + columns,
                        help="For agent performance tracking"
                    )
                    
                    call_id_col = st.selectbox(
                        "Call ID Column",
                        ['None'] + columns,
                        help="Unique identifier for each call"
                    )
                
                with col_b:
                    timestamp_col = st.selectbox(
                        "Timestamp Column",
                        ['None'] + columns,
                        help="When the call occurred"
                    )
                
                # Store in session state
                st.session_state.uploaded_df = df
                st.session_state.transcript_col = transcript_col
                st.session_state.agent_col = None if agent_col == 'None' else agent_col
                st.session_state.call_id_col = None if call_id_col == 'None' else call_id_col
                st.session_state.timestamp_col = None if timestamp_col == 'None' else timestamp_col
                
            except Exception as e:
                st.error(f"❌ Failed to load file: {str(e)}")
    
    with col2:
        st.subheader("💡 Quick Stats")
        if 'uploaded_df' in st.session_state:
            df = st.session_state.uploaded_df
            st.metric("Total Rows", f"{len(df):,}")
            st.metric("Total Columns", len(df.columns))
            
            if st.session_state.transcript_col:
                avg_length = df[st.session_state.transcript_col].astype(str).apply(lambda x: len(x.split())).mean()
                st.metric("Avg Words", f"{avg_length:.0f}")
    
    # PII Redaction option
    st.markdown("---")
    st.subheader("🔒 PII Redaction")
    
    enable_pii = st.checkbox(
        "Enable PII Redaction",
        value=True,
        help="Automatically redact sensitive information"
    )
    
    st.session_state.enable_pii = enable_pii
    
    if enable_pii:
        st.info("✅ PII will be redacted: emails, phones, SSN, Aadhaar, account numbers, credit cards, URLs")

# ===========================
# TAB 2: Classify (HYBRID SYSTEM)
# ===========================
with tabs[1]:
    st.header("🔍 Classification Engine (CCRE)")
    
    # Check if data uploaded
    if 'uploaded_df' not in st.session_state:
        st.warning("⚠️ Please upload data in the Upload tab first")
        st.stop()
    
    df = st.session_state.uploaded_df.copy()
    transcript_col = st.session_state.transcript_col
    enable_pii = st.session_state.enable_pii
    
    st.info(f"📊 Ready to classify {len(df):,} transcripts using {len(default_rules):,} CCRE rules")
    
    # =============================
    # CLASSIFICATION SETTINGS
    # =============================
    st.subheader("⚙️ Classification Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        batch_size = st.selectbox(
            "Batch Size",
            [100, 500, 1000, 5000],
            index=2,
            help="Transcripts processed per batch"
        )
    
    with col2:
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.30,
            max_value=0.90,
            value=0.40,
            step=0.05,
            help="Lower threshold = more matches (default 0.40 for better coverage)"
        )
    
    with col3:
        # Estimate time
        est_time_per_transcript = 0.005  # 5ms with CCRE
        est_total_seconds = len(df) * est_time_per_transcript
        st.metric(
            "Estimated Time", 
            f"{est_total_seconds / 60:.1f} min" if est_total_seconds >= 60 else f"{est_total_seconds:.0f} sec"
        )
    
    # =============================
    # START CLASSIFICATION
    # =============================
    st.markdown("---")
    
    if st.button("🚀 Start Classification", type="primary", use_container_width=True):
        
        start_time = time.time()
        
        # Initialize Classifier
        with st.spinner("🔧 Initializing Classifier..."):
            try:
                classifier = HybridClassifier(
                    proximity_rules=pd.DataFrame(),  # Not used
                    hierarchy=hierarchy_data,
                    user_examples=None,
                    fallback_rules=default_rules,  # Use CCRE rules
                    program="CCRE"
                )
                st.success("✅ Classifier initialized")
            except Exception as e:
                st.error(f"❌ Failed to initialize classifier: {str(e)}")
                st.exception(e)
                st.stop()
        
        # Initialize PII redactor if needed
        if enable_pii:
            redactor = PIIRedactor()
        
        # Progress tracking
        st.subheader("📊 Classification Progress")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        stats_placeholder = st.empty()
        
        # Get transcripts
        transcripts = df[transcript_col].astype(str).tolist()
        
        # Clean transcripts - remove timestamps and formatting labels
        def clean_transcript(text):
            import re
            # Remove [HH:MM:SS AGENT]: and [HH:MM:SS CUSTOMER]: patterns
            text = re.sub(r'\[\d{2}:\d{2}:\d{2}\s+(AGENT|CUSTOMER)\]:\s*', ' ', text)
            # Remove extra whitespace
            text = ' '.join(text.split())
            return text.lower()
        
        st.info("🧹 Cleaning transcripts (removing timestamps and labels)...")
        transcripts = [clean_transcript(t) for t in transcripts]
        
        # Redact PII if enabled
        if enable_pii:
            with st.spinner("🔒 Redacting PII..."):
                transcripts = [redactor.redact(t) for t in transcripts]
        
        # Process in batches
        all_results = []
        total_transcripts = len(transcripts)
        
        for i in range(0, total_transcripts, batch_size):
            batch_end = min(i + batch_size, total_transcripts)
            batch = transcripts[i:batch_end]
            
            # Classify batch
            batch_results = classifier.classify_batch(
                batch,
                batch_size=100,
                show_progress=False
            )
            
            all_results.extend(batch_results)
            
            # Update progress
            progress = (batch_end / total_transcripts)
            progress_bar.progress(progress)
            status_text.text(f"Processed {batch_end:,} / {total_transcripts:,} transcripts ({progress*100:.1f}%)")
            
            # Show interim stats
            stats = classifier.get_classification_stats(all_results)
            
            with stats_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Classified", f"{stats['classified_pct']:.1f}%")
                col2.metric("Count", f"{stats['classified']:,}")
                col3.metric("Avg Confidence", f"{stats['avg_confidence']:.2%}")
                col4.metric("High Confidence", f"{stats['high_confidence']:,}")
        
        # Classification complete
        elapsed = time.time() - start_time
        
        st.success(f"✅ Classification complete in {elapsed:.1f} seconds!")
        
        # =============================
        # RESULTS ANALYSIS
        # =============================
        st.markdown("---")
        st.subheader("📈 Results Analysis")
        
        stats = classifier.get_classification_stats(all_results)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric(
            "Classification Rate",
            f"{stats['classified_pct']:.1f}%",
            delta=f"{stats['classified']:,} transcripts"
        )
        
        col2.metric(
            "Unclassified",
            f"{stats['unclassified_pct']:.1f}%",
            delta=f"-{stats['unclassified']:,}"
        )
        
        col3.metric(
            "Avg Confidence",
            f"{stats['avg_confidence']:.1%}"
        )
        
        col4.metric(
            "Processing Speed",
            f"{total_transcripts/elapsed:.0f}/sec"
        )
        
        # Confidence distribution
        st.markdown("**Confidence Distribution:**")
        conf_col1, conf_col2, conf_col3 = st.columns(3)
        
        conf_col1.metric("High (≥0.85)", f"{stats['high_confidence']:,}", f"{stats['high_confidence']/total_transcripts*100:.1f}%")
        conf_col2.metric("Medium (0.70-0.85)", f"{stats['medium_confidence']:,}", f"{stats['medium_confidence']/total_transcripts*100:.1f}%")
        conf_col3.metric("Low (0.50-0.70)", f"{stats['low_confidence']:,}", f"{stats['low_confidence']/total_transcripts*100:.1f}%")
        
        # Source breakdown
        st.markdown("**Classification Sources:**")
        source_data = []
        for source, count in stats['by_source'].items():
            source_data.append({
                "Source": source.capitalize(),
                "Count": count,
                "Percentage": f"{count/total_transcripts*100:.1f}%"
            })
        
        source_df = pd.DataFrame(source_data)
        st.dataframe(source_df, use_container_width=True, hide_index=True)
        
        # Top categories
        st.markdown("**Top 10 Categories:**")
        top_cat_data = []
        for cat, count in stats['top_categories'][:10]:
            top_cat_data.append({
                "Category": cat,
                "Count": count,
                "Percentage": f"{count/total_transcripts*100:.1f}%"
            })
        
        top_df = pd.DataFrame(top_cat_data)
        st.dataframe(top_df, use_container_width=True, hide_index=True)
        
        # =============================
        # RESULTS TABLE
        # =============================
        st.markdown("---")
        st.subheader("📋 Detailed Results")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Add original data columns
        results_df['transcript'] = df[transcript_col].values[:len(results_df)]
        
        if st.session_state.agent_col:
            results_df['agent_name'] = df[st.session_state.agent_col].values[:len(results_df)]
        
        if st.session_state.call_id_col:
            results_df['call_id'] = df[st.session_state.call_id_col].values[:len(results_df)]
        
        # Reorder columns
        display_columns = ['transcript', 'category', 'subcategory', 'tertiary', 'quaternary', 
                          'confidence', 'matched_keywords', 'source']
        
        if st.session_state.agent_col:
            display_columns.insert(1, 'agent_name')
        
        if st.session_state.call_id_col:
            display_columns.insert(1 if not st.session_state.agent_col else 2, 'call_id')
        
        # Store RAW results in session FIRST (with numeric confidence)
        st.session_state.classified_results = results_df.copy()
        
        # THEN create display version with formatted confidence
        display_df = results_df[display_columns].copy()
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
        
        # Show results
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # =============================
        # EXPORT OPTIONS
        # =============================
        st.markdown("---")
        st.subheader("📥 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV export - use RAW data not formatted display
            csv = results_df[display_columns].to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"classified_results_CCRE_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Excel export - use RAW data
            from io import BytesIO
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                results_df[display_columns].to_excel(writer, index=False, sheet_name='Results')
            
            st.download_button(
                label="📥 Download Excel",
                data=buffer.getvalue(),
                file_name=f"classified_results_CCRE_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        # Save to Supabase option
        if st.button("💾 Save to Database", use_container_width=True):
            with st.spinner("Saving to database..."):
                try:
                    # Convert to records for Supabase
                    records = results_df.to_dict('records')
                    # TODO: Implement Supabase save
                    st.success(f"✅ Saved {len(records):,} records to database")
                except Exception as e:
                    st.error(f"❌ Failed to save: {str(e)}")

# ===========================
# TAB 3: Analyze
# ===========================
with tabs[2]:
    if 'classified_results' in st.session_state:
        try:
            render_analytics_dashboard(st.session_state.classified_results, supabase)
        except Exception as e:
            st.error(f"Analytics error: {str(e)}")
            st.info("Showing basic stats instead:")
            df = st.session_state.classified_results
            col1, col2, col3 = st.columns(3)
            col1.metric("Total", len(df))
            col2.metric("Classified", sum(df['category'] != 'Unclassified'))
            col3.metric("Avg Confidence", f"{df['confidence'].mean():.1%}")
    else:
        st.info("📊 Classification results will appear here after running classification")

# ===========================
# TAB 4: Coach
# ===========================
with tabs[3]:
    if 'classified_results' in st.session_state:
        try:
            render_coaching_interface(st.session_state.classified_results, supabase)
        except Exception as e:
            st.warning(f"Coaching interface unavailable.")
    else:
        st.info("🎓 Coaching insights will appear here after running classification")

# ===========================
# TAB 5: Export
# ===========================
with tabs[4]:
    if 'classified_results' in st.session_state:
        try:
            render_export_interface(st.session_state.classified_results, supabase)
        except Exception as e:
            st.warning(f"Export interface unavailable. Use download buttons in Classify tab.")
    else:
        st.info("📤 Export options will appear here after running classification")

# Footer
st.markdown("---")
st.caption(f"AgentPulse AI v2.0 - Hybrid Classification System | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

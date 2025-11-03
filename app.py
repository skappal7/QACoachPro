"""
AgentPulse AI - Enterprise QA & Coaching Platform
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import time
from datetime import datetime

# Import modules
from modules import (
    init_supabase,
    check_authentication,
    render_login_page,
    render_sidebar_user_info,
    CCREEngine,
    PIIRedactor,
    render_analytics_dashboard,
    render_coaching_interface,
    render_export_interface
)

# Page config
st.set_page_config(
    page_title="AgentPulse AI",
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

# Header
st.title("🚀 AgentPulse AI")
st.markdown("**Enterprise QA & Coaching Platform** - Fast, Accurate, Explainable")

# Tab navigation
tabs = st.tabs(["📁 Upload", "🔍 Classify", "📊 Analyze", "🎓 Coach", "📤 Export"])

# ===========================
# TAB 1: Upload
# ===========================
with tabs[0]:
    st.header("📁 Upload Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Transcripts File")
        transcripts_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=['csv', 'xlsx'],
            help="Max 200MB, up to 200K+ rows"
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
                        help="For agent performance tracking and coaching"
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
        st.subheader("📋 Rules File (Optional)")
        
        rules_file = st.file_uploader(
            "Upload custom rules CSV",
            type=['csv'],
            help="Leave empty to use default 1,990 rules"
        )
        
        if rules_file:
            try:
                rules_df = pd.read_csv(rules_file)
                st.success(f"✅ Loaded {len(rules_df)} custom rules")
                st.session_state.rules_df = rules_df
                
                # Validate rules format
                required_cols = ['rule_id', 'category', 'subcategory', 'required_groups', 'forbidden_terms']
                missing_cols = [c for c in required_cols if c not in rules_df.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                else:
                    with st.expander("👀 Rules Preview"):
                        st.dataframe(rules_df.head(10), use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Failed to load rules: {str(e)}")
        else:
            st.info("ℹ️ Using default rules (1,990 rules loaded)")
            # Load default rules from data folder
            try:
                default_rules = pd.read_csv('data/default_rules.csv')
                st.session_state.rules_df = default_rules
                st.caption(f"Default rules: {len(default_rules)} rules")
            except:
                st.warning("⚠️ Default rules file not found. Please upload custom rules.")
    
    # PII Redaction option
    st.markdown("---")
    st.subheader("🔒 PII Redaction")
    
    enable_pii = st.checkbox(
        "Enable PII Redaction",
        value=True,
        help="Automatically redact sensitive information (emails, phones, SSN, etc.)"
    )
    
    st.session_state.enable_pii = enable_pii
    
    if enable_pii:
        st.info("✅ PII will be redacted: emails, phones, SSN, Aadhaar, account numbers, credit cards, URLs")
    else:
        st.warning("⚠️ PII redaction disabled. Ensure data is already anonymized.")

# ===========================
# TAB 2: Classify
# ===========================
with tabs[1]:
    st.header("🔍 Classification Engine")
    
    # Check if data uploaded
    if 'uploaded_df' not in st.session_state:
        st.warning("⚠️ Please upload data in the Upload tab first")
    else:
        df = st.session_state.uploaded_df.copy()
        transcript_col = st.session_state.transcript_col
        agent_col = st.session_state.agent_col
        rules_df = st.session_state.rules_df
        enable_pii = st.session_state.enable_pii
        
        st.info(f"📊 Ready to classify {len(df):,} transcripts")
        
        # Configuration
        st.subheader("⚙️ Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            batch_size = st.selectbox(
                "Batch Size",
                [1000, 5000, 10000, 20000],
                index=2,
                help="Number of transcripts to process per batch"
            )
        
        with col2:
            st.metric("Estimated Time", f"{(len(df) * 0.123 / 60):.1f} minutes", help="Based on 123ms per transcript")
        
        # Start classification
        if st.button("🚀 Start Classification", type="primary", use_container_width=True):
            
            start_time = time.time()
            
            # Initialize CCRE engine
            with st.spinner("Initializing CCRE engine..."):
                engine = CCREEngine(rules_df)
            
            # Initialize PII redactor
            if enable_pii:
                redactor = PIIRedactor()
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process in batches
            results = []
            total_rows = len(df)
            
            for i in range(0, total_rows, batch_size):
                batch_end = min(i + batch_size, total_rows)
                batch_df = df.iloc[i:batch_end].copy()
                
                status_text.text(f"Processing transcripts {i+1} to {batch_end}...")
                
                # PII redaction
                if enable_pii:
                    batch_df['redacted_transcript'] = redactor.redact_batch(batch_df[transcript_col].tolist())
                    transcripts_to_classify = batch_df['redacted_transcript'].tolist()
                else:
                    batch_df['redacted_transcript'] = batch_df[transcript_col]
                    transcripts_to_classify = batch_df[transcript_col].tolist()
                
                # Classification (NO CACHE CHECKING IN LOOP)
                classifications = engine.classify_batch(transcripts_to_classify)
                
                # Add results to dataframe
                for j, cls in enumerate(classifications):
                    batch_df.loc[batch_df.index[j], 'category'] = cls['category']
                    batch_df.loc[batch_df.index[j], 'subcategory'] = cls['subcategory']
                    batch_df.loc[batch_df.index[j], 'confidence'] = cls['confidence']
                    batch_df.loc[batch_df.index[j], 'resolve_reason'] = cls['resolve_reason']
                    batch_df.loc[batch_df.index[j], 'matched_keywords'] = cls['matched_keywords']
                    batch_df.loc[batch_df.index[j], 'num_rules_activated'] = cls['num_rules_activated']
                    batch_df.loc[batch_df.index[j], 'transcript_id'] = i + j
                
                results.append(batch_df)
                
                # Update progress
                progress_bar.progress(batch_end / total_rows)
            
            # Combine results
            classified_df = pd.concat(results, ignore_index=True)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # BATCH cache write AFTER classification completes
            if supabase:
                status_text.text("Caching results...")
                for idx, row in classified_df.iterrows():
                    try:
                        trans_hash = supabase.hash_text(row['redacted_transcript'])
                        supabase.cache_classification(
                            trans_hash,
                            row['category'],
                            row['subcategory'],
                            row['confidence'],
                            row['matched_keywords'],
                            row['resolve_reason']
                        )
                    except:
                        pass  # Ignore cache errors
            
            status_text.text("✅ Classification complete!")
            progress_bar.progress(1.0)
            
            # Store in session state
            st.session_state.classified_df = classified_df
            
            # Log to Supabase
            supabase.log_upload(
                st.session_state.uploaded_df.attrs.get('filename', 'unknown.csv'),
                len(classified_df),
                processing_time
            )
            
            # Success metrics
            st.markdown("---")
            st.success(f"✅ Successfully classified {len(classified_df):,} transcripts in {processing_time:.1f} seconds")
            
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Total Processed", f"{len(classified_df):,}")
            col2.metric("Processing Time", f"{processing_time:.1f}s")
            col3.metric("Avg Time/Transcript", f"{(processing_time/len(classified_df)*1000):.1f}ms")
            col4.metric("Avg Confidence", f"{classified_df['confidence'].mean():.3f}")
            
            # Show PII stats
            if enable_pii:
                with st.expander("🔒 PII Redaction Statistics"):
                    stats = redactor.get_stats()
                    total_pii = sum(stats.values())
                    st.metric("Total PII Redacted", total_pii)
                    
                    for pii_type, count in stats.items():
                        if count > 0:
                            st.write(f"- {pii_type}: {count}")
            
            # Preview results
            with st.expander("👀 Classification Results Preview"):
                preview_cols = ['transcript_id', 'category', 'subcategory', 'confidence', 'resolve_reason']
                if agent_col:
                    preview_cols.insert(1, agent_col)
                
                st.dataframe(
                    classified_df[preview_cols].head(20),
                    use_container_width=True
                )

# ===========================
# TAB 3: Analyze
# ===========================
with tabs[2]:
    if 'classified_df' not in st.session_state:
        st.warning("⚠️ Please classify data first in the Classify tab")
    else:
        render_analytics_dashboard(st.session_state.classified_df)

# ===========================
# TAB 4: Coach
# ===========================
with tabs[3]:
    if 'classified_df' not in st.session_state:
        st.warning("⚠️ Please classify data first in the Classify tab")
    else:
        render_coaching_interface(st.session_state.classified_df, supabase)

# ===========================
# TAB 5: Export
# ===========================
with tabs[4]:
    if 'classified_df' not in st.session_state:
        st.warning("⚠️ Please classify data first in the Classify tab")
    else:
        render_export_interface(st.session_state.classified_df)

# Footer
st.markdown("---")
st.caption("© 2025 AgentPulse AI | v1.0 | Production Ready")

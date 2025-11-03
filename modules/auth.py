"""
Authentication Module
Glassmorphic login interface with Supabase backend
"""

import streamlit as st
from modules.supabase_client import SupabaseManager


def inject_glassmorphic_css():
    """Inject glassmorphic login CSS"""
    st.markdown("""
    <style>
    /* Hide Streamlit default elements on login page */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Full-screen background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-attachment: fixed;
    }
    
    /* Glassmorphic login container */
    .login-container {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 400px;
        padding: 40px;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    
    /* Login title */
    .login-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    
    .login-subtitle {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.9);
        text-align: center;
        margin-bottom: 30px;
        font-weight: 300;
    }
    
    /* Input fields */
    .stTextInput input {
        background: rgba(255, 255, 255, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 10px !important;
        color: white !important;
        padding: 12px !important;
        font-size: 16px !important;
    }
    
    .stTextInput input::placeholder {
        color: rgba(255, 255, 255, 0.6) !important;
    }
    
    .stTextInput label {
        color: white !important;
        font-weight: 500 !important;
        font-size: 14px !important;
    }
    
    /* Login button */
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Error/Success messages */
    .stAlert {
        background: rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)


def render_login_page(supabase: SupabaseManager):
    """Render glassmorphic login page"""
    
    inject_glassmorphic_css()
    
    # Create centered container
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="login-container">
            <div class="login-title">🚀 AgentPulse AI</div>
            <div class="login-subtitle">Enterprise QA & Coaching Platform</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
        
        # Login form
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("Email", placeholder="admin@agentpulse.ai")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            remember = st.checkbox("Remember me")
            
            submit = st.form_submit_button("🔐 Sign In")
            
            if submit:
                if not email or not password:
                    st.error("❌ Please enter both email and password")
                else:
                    with st.spinner("Authenticating..."):
                        result = supabase.sign_in(email, password)
                        
                        if result["success"]:
                            # Store session in Streamlit state
                            st.session_state.authenticated = True
                            st.session_state.user_id = result["user"]["id"]
                            st.session_state.user_email = result["user"]["email"]
                            st.session_state.username = result["user"]["username"]
                            st.session_state.remember_me = remember
                            
                            st.success(f"✅ Welcome back, {result['user']['username']}!")
                            st.rerun()
                        else:
                            st.error(f"❌ Authentication failed: {result['error']}")


def check_authentication(supabase: SupabaseManager) -> bool:
    """
    Check if user is authenticated
    Returns: True if authenticated, False otherwise
    """
    
    # Check session state first
    if st.session_state.get("authenticated", False):
        # Verify session is still valid
        session = supabase.get_session()
        if session:
            return True
        else:
            # Session expired, clear state
            st.session_state.authenticated = False
            return False
    
    # Check for existing Supabase session
    session = supabase.get_session()
    if session:
        # Restore session state
        st.session_state.authenticated = True
        st.session_state.user_id = session.user.id
        st.session_state.user_email = session.user.email
        
        # Get username from database
        try:
            user_data = supabase.client.table("users").select("username").eq(
                "id", session.user.id
            ).execute()
            st.session_state.username = user_data.data[0]["username"] if user_data.data else session.user.email
        except:
            st.session_state.username = session.user.email
        
        return True
    
    return False


def logout(supabase: SupabaseManager):
    """Logout user and clear session"""
    supabase.sign_out()
    
    # Clear all session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    st.rerun()


def render_sidebar_user_info(supabase: SupabaseManager):
    """Render user info in sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 👤 User Profile")
        st.markdown(f"**Username:** {st.session_state.username}")
        st.markdown(f"**Email:** {st.session_state.user_email}")
        
        if st.button("🚪 Logout", use_container_width=True):
            logout(supabase)
        
        st.markdown("---")
        
        # Upload history
        with st.expander("📁 Recent Uploads"):
            history = supabase.get_upload_history(limit=5)
            if history:
                for item in history:
                    st.markdown(f"""
                    **{item['filename']}**  
                    Rows: {item['row_count']:,} | Time: {item['processing_time_seconds']:.1f}s  
                    {item['created_at'][:10]}
                    """)
            else:
                st.info("No uploads yet")
        
        # About section
        with st.expander("ℹ️ About"):
            st.markdown("""
            **AgentPulse AI v1.0**
            
            Enterprise QA & Coaching Platform
            
            - 🚀 200K transcripts in 60-75 min
            - 🎯 96% classification accuracy
            - 💡 Explainable AI decisions
            - 🔒 Secure & isolated data
            
            © 2025 AgentPulse AI
            """)

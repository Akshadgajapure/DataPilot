import streamlit as st

def inject_custom_css():
    """
    Safely injects custom CSS for a premium, Enterprise Dark Mode look.
    Avoids clobbering Streamlit's internal layout grid.
    """
    custom_css = """
    <style>
    /* 1. Global Typography */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    p, h1, h2, h3, h4, h5, h6, span, div {
        font-family: 'Outfit', sans-serif;
    }

    /* 2. Hide Streamlit Watermarks & Clutter */
    #MainMenu {display: none;}
    header[data-testid="stHeader"] {display: none;}
    footer {display: none;}
    
    /* 2.5 Dynamic 3D App Background */
    div[data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 15% 50%, #0F172A, #020617 50%, #0F172A 100%);
        background-attachment: fixed;
    }

    /* 3. Metric Cards 3D Styling (Floating Glassmorphism) */
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 10px 20px -5px rgba(0, 0, 0, 0.5), inset 0 2px 4px rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        position: relative;
        overflow: hidden;
    }
    
    /* Glow effect line at top of metric cards */
    div[data-testid="metric-container"]::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, #4F46E5, transparent);
        opacity: 0.5;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.6), 0 10px 10px -5px rgba(99, 102, 241, 0.3), inset 0 2px 4px rgba(255, 255, 255, 0.1);
        border-color: rgba(99, 102, 241, 0.5);
    }
    
    /* 4. True 3D Button Styling */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: none;
        padding: 0.6rem 1.2rem;
        transition: all 0.15s ease;
        background: linear-gradient(180deg, #334155 0%, #1E293B 100%);
        color: white;
        box-shadow: 0 4px 0 #0F172A, 0 8px 15px rgba(0,0,0,0.4);
    }
    
    .stButton > button:active {
        transform: translateY(4px);
        box-shadow: 0 0 0 #0F172A, 0 4px 6px rgba(0,0,0,0.3);
    }

    /* Primary 3D Button override */
    .stButton > button[kind="primary"] {
        background: linear-gradient(180deg, #6366F1 0%, #4338CA 100%);
        box-shadow: 0 4px 0 #312E81, 0 8px 15px rgba(99, 102, 241, 0.4);
    }
    
    .stButton > button[kind="primary"]:active {
        box-shadow: 0 0 0 #312E81, 0 4px 6px rgba(99, 102, 241, 0.3);
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(180deg, #818CF8 0%, #4F46E5 100%);
    }

    /* 5. DataFrame / Table Styling (Floating Depth) */
    div[data-testid="stDataFrame"] {
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        background: #0F172A;
        box-shadow: 0 15px 30px -5px rgba(0, 0, 0, 0.5), inset 0 2px 5px rgba(255,255,255,0.02);
        overflow: hidden;
    }

    /* 6. File Uploader Styling (3D Neumorphic) */
    section[data-testid="stFileUploadDropzone"] {
        background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
        border: 2px dashed #475569;
        border-radius: 20px;
        padding: 4rem;
        box-shadow: inset 0 5px 15px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    
    section[data-testid="stFileUploadDropzone"]:hover {
        border-color: #6366F1;
        background: linear-gradient(145deg, #1E293B 0%, #171E30 100%);
        box-shadow: inset 0 5px 20px rgba(99, 102, 241, 0.2);
        transform: scale(1.01);
    }

    /* 7. Sidebar Navigation Links (Pill-shaped 3D) */
    div[data-testid="stSidebarNav"] li div a {
        border-radius: 12px;
        margin: 4px 12px;
        transition: all 0.2s ease;
    }
    
    div[data-testid="stSidebarNav"] li div a:hover {
        background: linear-gradient(90deg, #1E293B 0%, #334155 100%);
        box-shadow: 3px 3px 10px rgba(0,0,0,0.2), inset 1px 1px 2px rgba(255,255,255,0.05);
        transform: translateX(4px);
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def sidebar_brand():
    """Injects a logo/brand header at the top of the sidebar."""
    import os
    logo_path = os.path.join(os.path.dirname(__file__), "..", "assets", "logo.png")
    
    # Use columns to center the image slightly
    col1, col2, col3 = st.sidebar.columns([1, 6, 1])
    with col2:
        if os.path.exists(logo_path):
            st.image(logo_path, use_container_width=True)
        else:
            st.markdown("### AI Data Analyst")
    
    st.sidebar.markdown("<br>", unsafe_allow_html=True)

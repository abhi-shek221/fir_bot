import streamlit as st
import pandas as pd
from datetime import datetime
from ai_model import get_sections_and_analysis, generate_fir_structure

# Load the dataset
df = pd.read_csv('fir_sections.csv')

def main():
    st.set_page_config(page_title="FIR Assistant", layout="wide")
    
    # Custom CSS for a more dynamic and visually appealing design
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #f0f2f6;
        color: #333;
    }
    .main {
        padding: 2rem;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #f9f9f9;
        border: 1px solid #ddd;
        border-radius: 5px;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 700;
    }
    .section-card {
        background-color: #f1f8e9;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .analysis-card {
        background-color: #e8f5e9;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🚔 FIR Assistant")
    
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    
    if st.session_state.page == 'home':
        st.write("Welcome to the FIR Assistant. This tool helps in determining appropriate sections and generating FIR structures based on case descriptions.")
        if st.button("📝 Start New FIR"):
            st.session_state.page = 'query'
    
    elif st.session_state.page == 'query':
        st.header("📋 Case Information")
        
        col1, col2 = st.columns(2)
        with col1:
            date_of_incident = st.date_input("Date of Incident", datetime.now())
            time_of_incident = st.time_input("Time of Incident", datetime.now().time())
        with col2:
            place_of_occurrence = st.text_input("Place of Occurrence")
        
        nature_of_offense = st.text_area("Nature of the Offense")
        
        st.subheader("👤 Complainant Details")
        complainant_name = st.text_input("Complainant Name")
        complainant_contact = st.text_input("Complainant Contact")
        
        st.subheader("🕵️ Accused Details (if known)")
        accused_name = st.text_input("Accused Name (if known)")
        accused_description = st.text_area("Accused Description")
        
        st.subheader("📝 Case Description")
        case_description = st.text_area("Enter the detailed facts of the case:")
        
        if st.button("🔍 Generate FIR"):
            if case_description:
                user_inputs = f"""
                Date of Incident: {date_of_incident}
                Time of Incident: {time_of_incident}
                Place of Occurrence: {place_of_occurrence}
                Nature of Offense: {nature_of_offense}
                Complainant Name: {complainant_name}
                Complainant Contact: {complainant_contact}
                Accused Name: {accused_name}
                Accused Description: {accused_description}
                """
                
                sections, analysis = get_sections_and_analysis(case_description)
                fir_structure = generate_fir_structure(case_description, sections, user_inputs)
                
                st.header("📄 Generated FIR Structure")
                st.markdown(f'<div class="section-card">{fir_structure}</div>', unsafe_allow_html=True)
                
                st.header("📚 Relevant Sections")
                for section, description in sections.items():
                    st.markdown(f'<div class="section-card"><h3>Section: {section}</h3><p>{description}</p></div>', unsafe_allow_html=True)
                
                st.header("🔎 Case Analysis")
                st.markdown(f'<div class="analysis-card">{analysis}</div>', unsafe_allow_html=True)
            else:
                st.warning("Please enter a case description.")
        
        if st.button("🏠 Back to Home"):
            st.session_state.page = 'home'
            st.experimental_rerun()

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import os
import re
import speech_recognition as sr

# Load the dataset
df = pd.read_csv('fir_sections.csv')

# Preprocess section numbers
df['Section_Number'] = df['Section'].apply(lambda x: re.findall(r'\d+', str(x))[0] if re.findall(r'\d+', str(x)) else '')

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Description'])

# Configure the Groq client
client = Groq(api_key="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")

# Initialize speech recognizer
recognizer = sr.Recognizer()

def get_relevant_sections(case_description, top_n=10):
    case_vector = vectorizer.transform([case_description])
    cosine_similarities = cosine_similarity(case_vector, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    relevant_sections = {}
    for index in top_indices:
        section = df.iloc[index]['Section']
        section_number = df.iloc[index]['Section_Number']
        description = df.iloc[index]['Description']
        relevant_sections[f"{section} ({section_number})"] = description
    return relevant_sections

def get_sections_and_analysis(case_description):
    relevant_sections = get_relevant_sections(case_description)
    
    sections_prompt = f"""
    Based on the following case description, provide a comprehensive list of all possible relevant sections and acts that could be applicable. For each section, provide a brief explanation of why it might be relevant to the case.

    Case Description: {case_description}

    Relevant Sections from TF-IDF:
    {relevant_sections}

    Please provide a detailed analysis, considering:
    1. All possible offenses described in the case
    2. Any aggravating factors
    3. Potential secondary offenses
    4. Relevant procedural sections

    Format your response as:
    Section: [Section Number]
    Act: [Act Name]
    Relevance: [Brief explanation]

    After listing all potential sections, provide a brief analysis of the case and suggest next steps for the investigation.
    """
    
    completion = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "You are a legal assistant specializing in Indian law."},
            {"role": "user", "content": sections_prompt}
        ],
        temperature=0.5,
        max_tokens=2000
    )
    return relevant_sections, completion.choices[0].message.content

def generate_fir_structure(case_description, relevant_sections, user_inputs):
    prompt = f"""
    Based on the following case description, relevant sections, and user inputs, generate a detailed FIR (First Information Report) structure:

    Case Description: {case_description}

    Relevant Sections: {relevant_sections}

    User Inputs:
    {user_inputs}

    Please provide a comprehensive FIR structure including:
    1. FIR Number and Registration Date
    2. Date and Time of the Incident
    3. Place of Occurrence
    4. Nature of the Offense
    5. Details of the Complainant
    6. Details of the Accused (if known)
    7. Brief Facts of the Case
    8. Sections Applied
    9. Action Taken
    10. Investigating Officer Details

    Also, provide a brief analysis of the case and suggest next steps for the investigation.
    """
    
    completion = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "You are a legal assistant specializing in Indian law."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=2000
    )
    return completion.choices[0].message.content

def recognize_speech():
    with sr.Microphone() as source:
        st.write("Speak now...")
        audio = recognizer.listen(source)
        
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.error("Sorry, I couldn't understand that.")
        return None
    except sr.RequestError:
        st.error("Sorry, there was an error with the speech recognition service.")
        return None

def main():
    st.set_page_config(page_title="FIR Assistant", layout="wide")
    
    # Custom CSS (same as before)
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    :root {
        --primary-color: #4F46E5;
        --secondary-color: #10B981;
        --background-color: #F3F4F6;
        --text-color: #1F2937;
    }
    
    body {
        font-family: 'Inter', sans-serif;
        background-color: var(--background-color);
        color: var(--text-color);
    }
    .main {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    .stButton > button {
        background-color: var(--primary-color);
        color: white;
        font-weight: 600;
        padding: 0.5rem 1rem;
        border-radius: 0.375rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #4338CA;
        transform: translateY(-2px);
    }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: white;
        border: 1px solid #D1D5DB;
        border-radius: 0.375rem;
        padding: 0.5rem;
    }
    h1, h2, h3 {
        color: var(--text-color);
        font-weight: 700;
    }
    .card {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .section-card {
        background-color: #EEF2FF;
        border-left: 4px solid var(--primary-color);
    }
    .analysis-card {
        background-color: #ECFDF5;
        border-left: 4px solid var(--secondary-color);
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
        
        with st.form("fir_form"):
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
            
            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("🔍 Generate FIR")
            with col2:
                voice_input = st.form_submit_button("🎤 Voice Input")
        
        if voice_input:
            speech_text = recognize_speech()
            if speech_text:
                st.session_state.case_description = speech_text
                st.success("Speech recognized! Click 'Generate FIR' to proceed.")
            else:
                st.warning("Please try speaking again.")
        
        if submitted:
            case_description = st.session_state.get('case_description', case_description)
            if case_description:
                with st.spinner("Analyzing case and generating FIR..."):
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
                    st.markdown(f'<div class="card">{fir_structure}</div>', unsafe_allow_html=True)
                    
                    st.header("📚 Relevant Sections")
                    for section, description in sections.items():
                        st.markdown(f'<div class="card section-card"><h3>Section: {section}</h3><p>{description}</p></div>', unsafe_allow_html=True)
                    
                    st.header("🔎 Case Analysis")
                    st.markdown(f'<div class="card analysis-card">{analysis}</div>', unsafe_allow_html=True)
            else:
                st.warning("Please enter a case description or use voice input.")
        
        if st.button("🏠 Back to Home"):
            st.session_state.page = 'home'
            st.experimental_rerun()

if __name__ == "__main__":
    main()
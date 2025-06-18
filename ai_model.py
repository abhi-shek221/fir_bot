import os
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load the dataset
df = pd.read_csv('fir_sections.csv')

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Description'])

# Configure the Groq client
client = Groq(api_key="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")


def get_relevant_sections(case_description):
    case_vector = vectorizer.transform([case_description])
    cosine_similarities = cosine_similarity(case_vector, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[-5:][::-1]  # Get top 5 most similar sections
    relevant_sections = {}
    for index in top_indices:
        relevant_sections[df.iloc[index]['Section']] = df.iloc[index]['Description']
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
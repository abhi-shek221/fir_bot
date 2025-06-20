# FIR Assistant

FIR Assistant is a Streamlit web application designed to assist legal professionals and users in generating detailed First Information Report (FIR) structures based on case descriptions. The app leverages natural language processing techniques and the Groq API to identify relevant legal sections and provide comprehensive case analysis.

## Features

- Input detailed case descriptions and receive a list of relevant legal sections and acts.
- Generate a structured FIR including details such as FIR number, registration date, incident details, complainant and accused information, and more.
- Provides case analysis and suggested next steps for investigation.
- Uses TF-IDF vectorization and cosine similarity to find relevant sections from a dataset.
- Integrates with Groq API for advanced language model completions.
- User-friendly interface built with Streamlit and styled with custom CSS.

## Dependencies

- Python 3.x
- streamlit
- pandas
- scikit-learn
- groq

## Installation

1. Clone the repository.
2. Install the required Python packages:

```bash
pip install streamlit pandas scikit-learn groq
```

3. Ensure the dataset file `fir_sections.csv` is present in the project directory.
4. Set your Groq API key in the code or environment variable as needed.

## Usage

Run the Streamlit app with:

```bash
streamlit run fir_assistant.py
```

Follow the on-screen instructions to input case details and generate FIR structures.

## License

This project is licensed under the MIT License.
#   f i r _ b o t  
 
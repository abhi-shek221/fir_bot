import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

# Import the main file
import fir_ass_speech

class TestFIRAssistant(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a mock DataFrame for testing
        data = {
            'Section': ['Section 302', 'Section 307', 'Section 354'],
            'Section_Number': ['302', '307', '354'],
            'Description': ['Murder', 'Attempt to murder', 'Assault on woman'],
        }
        cls.mock_df = pd.DataFrame(data)
        # Set the df in the main file
        fir_ass_speech.df = cls.mock_df

    def setUp(self):
        # Reset Streamlit session state before each test
        fir_ass_speech.st.session_state = {}

    @patch('fir_ass_speech.vectorizer.transform')
    @patch('fir_ass_speech.cosine_similarity')
    def test_get_relevant_sections(self, mock_cosine_similarity, mock_transform):
        mock_transform.return_value = np.array([[1, 0, 0]])
        mock_cosine_similarity.return_value = np.array([[0.8, 0.6, 0.4]])
        
        case_description = "A man was found dead with multiple stab wounds."
        relevant_sections = fir_ass_speech.get_relevant_sections(case_description)
        
        self.assertIn('Section 302 (302)', relevant_sections)
        self.assertEqual(relevant_sections['Section 302 (302)'], 'Murder')

    @patch('fir_ass_speech.client.chat.completions.create')
    @patch('fir_ass_speech.get_relevant_sections')
    def test_get_sections_and_analysis(self, mock_get_relevant_sections, mock_create):
        mock_get_relevant_sections.return_value = {"Section 302 (302)": "Murder"}
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Mock analysis"
        mock_create.return_value = mock_response

        case_description = "A woman was assaulted in a public place."
        sections, analysis = fir_ass_speech.get_sections_and_analysis(case_description)

        self.assertIsInstance(sections, dict)
        self.assertEqual(analysis, "Mock analysis")

    @patch('fir_ass_speech.client.chat.completions.create')
    def test_generate_fir_structure(self, mock_create):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Mock FIR structure"
        mock_create.return_value = mock_response

        case_description = "A theft occurred at a jewelry store."
        relevant_sections = {"Section 379 (379)": "Punishment for theft"}
        user_inputs = "Mock user inputs"

        fir_structure = fir_ass_speech.generate_fir_structure(case_description, relevant_sections, user_inputs)

        self.assertEqual(fir_structure, "Mock FIR structure")

    @patch('fir_ass_speech.recognizer.recognize_google')
    @patch('fir_ass_speech.recognizer.listen')
    def test_recognize_speech(self, mock_listen, mock_recognize_google):
        mock_audio = MagicMock()
        mock_listen.return_value = mock_audio
        mock_recognize_google.return_value = "Test speech input"

        with patch('fir_ass_speech.st.write'):  # Mock st.write to avoid Streamlit-related errors
            result = fir_ass_speech.recognize_speech()

        self.assertEqual(result, "Test speech input")

    @patch('streamlit.text_input')
    @patch('streamlit.text_area')
    @patch('streamlit.date_input')
    @patch('streamlit.time_input')
    @patch('streamlit.form')
    @patch('fir_ass_speech.get_sections_and_analysis')
    @patch('fir_ass_speech.generate_fir_structure')
    def test_main_flow(self, mock_generate_fir, mock_get_sections, mock_form, mock_time, mock_date, mock_text_area, mock_text_input):
        # Mock form context manager
        mock_form.return_value.__enter__.return_value = None
        mock_form.return_value.__exit__.return_value = None

        # Mock user inputs
        mock_text_input.side_effect = ["Test Place", "Test Complainant", "Test Contact", "Test Accused"]
        mock_text_area.side_effect = ["Test Nature", "Test Description", "Test Case Description"]
        mock_date.return_value = datetime(2023, 1, 1)
        mock_time.return_value = datetime(2023, 1, 1, 12, 0, 0).time()

        # Mock analysis results
        mock_get_sections.return_value = ({"Section 1 (1)": "Description 1"}, "Test Analysis")
        mock_generate_fir.return_value = "Test FIR Structure"

        # Run the main function
        with patch('streamlit.session_state', {'page': 'query'}):
            with patch('streamlit.form_submit_button') as mock_submit:
                mock_submit.return_value = True
                fir_ass_speech.main()

        # Assert that the main flow executed correctly
        mock_get_sections.assert_called_once()
        mock_generate_fir.assert_called_once()

if __name__ == '__main__':
    unittest.main()
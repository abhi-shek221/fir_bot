import unittest
from fir_assistant import get_relevant_sections, get_sections_and_analysis, generate_fir_structure

class TestFIRAssistant(unittest.TestCase):

    def setUp(self):
        self.sample_case = """
        On June 15, 2024, at approximately 10:30 PM, Mr. Rahul Sharma reported that his car, 
        a white Toyota Corolla (license plate: MH02AB1234), was stolen from the parking lot 
        of Central Mall in Mumbai. Mr. Sharma stated that he had parked his car at 7:00 PM 
        and upon returning at 10:15 PM, he found it missing. There were no witnesses to the theft.
        """

    def test_get_relevant_sections(self):
        relevant_sections = get_relevant_sections(self.sample_case)
        self.assertIsInstance(relevant_sections, dict)
        self.assertTrue(len(relevant_sections) > 0)
        # Check if any theft-related section is included
        theft_related = any('theft' in desc.lower() or 'stolen' in desc.lower() for desc in relevant_sections.values())
        self.assertTrue(theft_related, "No theft-related section found in relevant sections")

    def test_get_sections_and_analysis(self):
        sections, analysis = get_sections_and_analysis(self.sample_case)
        self.assertIsInstance(sections, dict)
        self.assertIsInstance(analysis, str)
        self.assertTrue(len(analysis) > 0)
        self.assertTrue('stolen' in analysis.lower() or 'theft' in analysis.lower())
        self.assertTrue('car' in analysis.lower())

    def test_generate_fir_structure(self):
        user_inputs = """
        Date of Incident: 2024-06-15
        Time of Incident: 22:30:00
        Place of Occurrence: Central Mall parking lot, Mumbai
        Nature of Offense: Car Theft
        Complainant Name: Rahul Sharma
        Complainant Contact: +91 9876543210
        Accused Name: Unknown
        Accused Description: No description available
        """
        fir_structure = generate_fir_structure(self.sample_case, {"379 IPC (379)": "Punishment for theft"}, user_inputs)
        print("Generated FIR structure:", fir_structure)
        self.assertIsInstance(fir_structure, str)
        self.assertTrue(len(fir_structure) > 0)
        self.assertTrue('FIR Number' in fir_structure)
        self.assertTrue('Date and Time of the Incident' in fir_structure)
        self.assertTrue('Place of Occurrence' in fir_structure)
        self.assertTrue('Sections Applied' in fir_structure)

if __name__ == '__main__':
    unittest.main()
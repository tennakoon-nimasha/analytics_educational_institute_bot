import os
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Configuration
PDF_DIR = "pdfs"
OUTPUT_DIR = "output_markdown"
GCP_API_KEY = os.getenv("GCP_API_KEY")

genai.configure(api_key=GCP_API_KEY)

def save_text_to_md(text, filename):
    """Saves the given text to a markdown file.

    Args:
        text (str): The text to save.
        filename (str): The name of the markdown file to create or overwrite.
    """
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(text)
        print(f"Text saved to {filename} successfully.")
    except Exception as e:
        print(f"Error saving to {filename}: {e}")

EXTRACTION_PROMPT = """
Extract all the data from the PDF file and organize the information in a markdown format. The mardkdown should include all the information in the PDF, including tables, charts, and any other relevant data. The markdown should be structured in a way that makes it easy to read and understand.
Output Format:markdown
- Use headings and subheadings to organize the information. 
- Use bullet points and numbered lists to present information clearly.
- Include tables and charts in markdown format where applicable.

"""

def extract_financial_data_from_pdf(pdf_path):
    """Extract financial data from PDF using Gemini API"""
    try:
        print(f"Uploading file: {pdf_path}")
        
        # Upload the PDF using the Gemini File API
        file_part = genai.upload_file(pdf_path)
        
        # Get a model instance with the appropriate configuration
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={
                "temperature": 0.0,  # Use a low temperature for factual extraction
                "top_p": 0.95,
                "top_k": 0,
                "max_output_tokens": 4096,  # Increase token limit for detailed extraction
            }
        )
        
        # Call Gemini API with the uploaded file and prompt
        response = model.generate_content(
            [file_part, EXTRACTION_PROMPT],
            stream=False
        )
        
        # Get the response text
        if hasattr(response, 'text'):
            result_text = response.text
            file_name = os.path.splitext(os.path.basename(pdf_path))[0] + ".md"
            output_path = os.path.join(OUTPUT_DIR, file_name)   
            save_text_to_md(result_text, output_path)
            print(f"Extraction completed for {pdf_path}.")
    
    except Exception as e:
        print(f"Error extracting data from {pdf_path}: {e}")
def main():
    """Main function to extract data from all PDF files in the directory."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        extract_financial_data_from_pdf(pdf_path)

if __name__ == "__main__":
    main()

import tkinter as tk
from tkinter import filedialog
import sys

temperature_G = 0  # 0 to 2
MAX = 10         # corpus token
IMG_MAX = 10
CHUNK_MAX = 1000

# Define the system prompt (unchanged)
general_medrag_system = '''
    DO NOT DISCUSS FILES WITHIN **CORPUS**

    You are a helpful medical expert, and your task is to evaluate a medical report and/or image. 
    Please first think step-by-step describing relevant findings in the critical image(s) and report(s). 
    Organize your output with json(section[str, str]) and list the number of images received (may be 0). 
    Your responses will be used for research purposes only, so please have a definite answer.
'''

# Updated template using Python string formatting
general_medrag = '''
Here are the background knowledge documents, do not reference this data in your response:
<CORPUS>
{context}                  
</CORPUS>

Here is the question and critical information used in your response. Overweigh the text:
{question}

Here are the potential choices. Give each a percentile confidence:
{options}

Please think step-by-step and generate your diagnosis in json (must be 100+ tokens if images are present):
'''

# Define options (for testing)
"""
options = {
    '0': 'No Abnormalities',
    'LC': 'Lung Cancer',
    'PN': 'Pneumonia',
    'PNX': 'Pneumothorax',
    'PE': 'Pleural Effusion',
    'MM': 'Mediastinal Mass',
    'HL': 'Hilar Lymphadenopathy',
    'F': 'Rib Fracture',
    'PNP': 'Pneumoperitoneum',
    'A': 'Atelectasis'
}
"""

options = {
    '': 'Null',
}

# Convert options dict to string format for the template
options_text = '\n'.join([f"{key}. {value}" for key, value in options.items()])

# Context for the general_medrag template (can be None if not applicable)
context = "Sample context information relevant to the question."

class TemplateRun:
    def __init__(self):
        # Initialize the Tkinter root window
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the root window

    def text_files(self):
        # Prompt user to select multiple query text files
        text_file_paths = filedialog.askopenfilenames(
            title="Select Query Text Files",
            filetypes=[("Text files", "*.txt")]
        )

        # Check if any files were selected
        if not text_file_paths:
            print("No files selected. Exiting the program.")
            sys.exit()

        # Read and concatenate the contents of all selected files into a single 'question' string
        questions = []
        for file_path in text_file_paths:
            try:
                with open(file_path, 'r') as file:
                    content = file.read().strip()
                    questions.append(content)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        # Join all questions with a separator (e.g., two newlines)
        question = "\n\n".join(questions)
        return question

    def image_files(self):
        # Prompt user to select image files
        image_file_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
        )

        if not image_file_paths:
            print("No images selected.")
            sys.exit(1)
    
        return image_file_paths

    # Render the general_medrag template using Python's string formatting
    def rendered_medrag(self, question):
        return general_medrag.format(
            context=context,
            question=question,
            options=options_text
        )

    def run(self):
        # Get question text from text files
        question = self.text_files()

        # Optionally, get image files if needed
        # image_paths = self.image_files()  # Uncomment if images are required

        # Render the template
        rendered_prompt = self.rendered_medrag(question)

        # Print the rendered prompt
        print("\nRendered MedRAG Prompt:\n")
        print(rendered_prompt)

if __name__ == "__main__":
    template_runner = TemplateRun()
    template_runner.run()

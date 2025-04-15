from liquid import Template
import tkinter as tk
from tkinter import filedialog

# Initialize Tkinter root window
root = tk.Tk()
root.withdraw()  # Hide the root window

# Prompt user to select multiple query text files
text_file_paths = filedialog.askopenfilenames(
    title="Select Query Text Files",
    filetypes=[("Text files", "*.txt")]
)

# Check if any files were selected
if not text_file_paths:
    print("No files selected. Exiting the program.")
    exit()

# Read and concatenate the contents of all selected files into the 'question' variable
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

# Define the templates and options
general_medrag_system = '''
    You are a helpful medical expert, and your task is to evaluate a linked medical report and image. 
    Please first think step-by-step describing relevant findings in the image and report, then choose an answer. 
    No need to discuss the previous reports used for formatting knowledge.
    Organize your output with json(section[str, str]) and list the number of images received. 
    Your responses will be used for research purposes only, so please have a definite answer.
'''

general_medrag = Template('''
Here are the relevant documents:
{{ context }}

Here is the question:
{{ question }}

Here are the potential choices:
{{ options }}

Please think step-by-step and generate your 100+ token diagnosis in json:
''')

# Define options (for testing)
options = {
    'A': 'Option A Description',
    'B': 'Option B Description',
    'C': 'Option C Description',
    'N/A': 'Null',
}

# Convert options dict to string format for the template
options_text = '\n'.join([f"{key}. {value}" for key, value in options.items()])

# If you have a context for the general_medrag template (can be None if not applicable)
context = "Sample context information relevant to the question."

# Render the general_medrag template
rendered_medrag = general_medrag.render({
    'context': context,
    'question': question,
    'options': options_text
})

print("\nRendered MedRAG Prompt:")
print(rendered_medrag)

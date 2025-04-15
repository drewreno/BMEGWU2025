from liquid import Template
import os

# Define the directory and file path
fdir = "corpus/question"
file_path = os.path.join(fdir, "query.txt")  

# Read the content of the file into the 'question' variable
with open(file_path, 'r') as file:
    question = file.read().strip()

# Define the templates and options
general_cot_system = ('''
    You are a helpful medical expert, and your task is to answer a multi-choice medical question. 
    Please first think step-by-step and then choose the answer from the provided options. 
    Organize your output in a json formatted as Dict{Str(explanation)}. 
    Your responses will be used for research purposes only, so please have a definite answer.
''')

general_cot = Template('''
Here is the question:
{{ question }}

{{ options }}

Please think step-by-step and generate your output in json:
''')

general_medrag_system = ('''
    You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents. 
    Please first think step-by-step and then choose the answer from the provided options. 
    Organize your output in a json formatted as Dict{"Str(explanation)}}. 
    Your responses will be used for research purposes only, so please have a definite answer.
''')

general_medrag = Template('''
Here are the relevant documents:
{{ context }}

Here is the question:
{{ question }}

Here are the potential choices:
{{ options }}

Please think step-by-step and generate your output in json:
''')

# Define options (for testing)
options = {
    'A': 'CAD',
    'B': 'else'
}

# Convert options dict to string format for the template
options_text = '\n'.join([f"{key}. {value}" for key, value in options.items()])

# Render the general_cot template
rendered_cot = general_cot.render({
    'question': question,
    'options': options_text
})

# Print the rendered CoT output
print("Rendered CoT Prompt:")
print(rendered_cot)

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

import os
from docx import Document
from bs4 import BeautifulSoup

def convert_html_to_docx(html_file, output_folder):
    with open(html_file, 'r') as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text()

    doc = Document()
    doc.add_paragraph(text)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_filename = os.path.splitext(os.path.basename(html_file))[0] + '.docx'
    output_path = os.path.join(output_folder, output_filename)

    doc.save(output_path)
    print(f"DOCX file saved at: {output_path}")

input_folder = 'html_source'
output_folder = 'docx_file'

# Get a list of HTML files in the input folder
html_files = [file for file in os.listdir(input_folder) if file.endswith('.html')]

# Convert each HTML file to DOCX
for html_file in html_files:
    html_path = os.path.join(input_folder, html_file)
    convert_html_to_docx(html_path, output_folder)

import os
import glob
import re

# Print the working directory
print("Current working directory:", os.getcwd())

# Get a list of all files in the working directory
files = os.listdir(".")
print("Files in working directory:", files)

site_files = os.listdir("_site")
print("Files in '_site' directory:", site_files)

# Process all HTML files in the _site/reference/ directory
html_files = glob.glob("_site/reference/*.html")
print(f"Found {len(html_files)} HTML files to process")

for html_file in html_files:
    print(f"Processing: {html_file}")

    with open(html_file, "r") as file:
        content = file.readlines()
    # If the inner content of the <h1> tag either:
    # - has a literal `.` in it, or
    # - doesn't start with a capital letter,
    # then add `()` to the end of the content of the <h1> tag
    for i, line in enumerate(content):
        # Use regex to find the h1 tag with potential whitespace variations
        h1_match = re.search(r'<h1\s+class="title">', line)
        if h1_match:
            # Extract the content of the <h1> tag
            start = h1_match.end()
            end = line.find("</h1>", start)
            h1_content = line[start:end].strip()

            # Check if the content meets the criteria
            if "." in h1_content or (h1_content and not h1_content[0].isupper()):
                # Modify the content
                h1_content += "()"

            # Replace the <h1> tag with the modified content
            content[i] = line[:start] + h1_content + line[end:]

    # Add a style attribute to the <h1> tag to use a monospace font for code-like appearance
    content = [
        line.replace(
            '<h1 class="title">',
            "<h1 class=\"title\" style=\"font-family: SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;\">",
        )
        for line in content
    ]

    with open(html_file, "w") as file:
        file.writelines(content)

print("Finished processing all files")

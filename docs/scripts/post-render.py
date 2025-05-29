import os
import glob

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

    # Replace <h1> tag with a styled version
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

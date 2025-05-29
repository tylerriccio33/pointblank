import os

# Print the working directory
print("Current working directory:", os.getcwd())

# Get a list of all files in the working directory
files = os.listdir(".")
print("Files in working directory:", files)

site_files = os.listdir("_site")
print("Files in '_site' directory:", site_files)

with open("_site/reference/Actions.html", "r") as file:
    content = file.readlines()

# Replace <h1> tag with a styled version
content = [
    line.replace(
        '<h1 class="title">Actions</h1>',
        "<h1 class=\"title\" style=\"font-family: SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;\">Actions</h1>",
    )
    for line in content
]

with open("_site/reference/Actions.html", "w") as file:
    file.writelines(content)

import os

# Print the working directory
print("Current working directory:", os.getcwd())

# Get a list of all files in the current directory
files = os.listdir("reference")
print("Files in 'reference' directory:", files)

with open("docs/reference/Actions.html", "r") as file:
    content = file.readlines()

# Replace `<h1>Actions</h1>` with `<h1><code>Actions</code></h1>`
content = [line.replace("<h1>Actions</h1>", "<h1><code>Actions</code></h1>") for line in content]

with open("docs/reference/Actions.html", "w") as file:
    file.writelines(content)

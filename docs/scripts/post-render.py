with open("reference/Actions.html", "r") as file:
    content = file.readlines()

# Replace `<h1>Actions</h1>` with `<h1><code>Actions</code></h1>`
content = [line.replace("<h1>Actions</h1>", "<h1><code>Actions</code></h1>") for line in content]

with open("reference/Actions.html", "w") as file:
    file.writelines(content)

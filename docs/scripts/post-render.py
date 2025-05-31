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

# Process all HTML files in the `_site/reference/` directory (except `index.html`)
# and apply the specified transformations
html_files = [f for f in glob.glob("_site/reference/*.html") if not f.endswith("index.html")]

print(f"Found {len(html_files)} HTML files to process")

for html_file in html_files:
    print(f"Processing: {html_file}")

    with open(html_file, "r") as file:
        content = file.readlines()

    # Determine the classification of each h1 tag based on its content
    classification_info = {}
    for i, line in enumerate(content):
        h1_match = re.search(r'<h1\s+class="title">(.*?)</h1>', line)
        if h1_match:
            original_h1_content = h1_match.group(1).strip()
            # Store classification based on original content
            if original_h1_content and original_h1_content[0].isupper():
                if "." in original_h1_content:
                    classification_info[i] = ("method", "steelblue", "#E3F2FF")
                else:
                    classification_info[i] = ("class", "darkgreen", "#E3FEE3")
            else:
                classification_info[i] = ("function", "darkorange", "#FFF1E0")

    # Remove the literal text `Validate.` from the h1 tag
    # TODO: Add line below stating the class name for the method
    content = [
        line.replace(
            '<h1 class="title">Validate.',
            '<h1 class="title">',
        )
        for line in content
    ]

    # If the inner content of the h1 tag either:
    # - has a literal `.` in it, or
    # - doesn't start with a capital letter,
    # then add `()` to the end of the content of the h1 tag
    for i, line in enumerate(content):
        # Use regex to find the h1 tag with potential whitespace variations
        h1_match = re.search(r'<h1\s+class="title">', line)
        if h1_match:
            # Extract the content of the h1 tag
            start = h1_match.end()
            end = line.find("</h1>", start)
            h1_content = line[start:end].strip()

            # Check if the content meets the criteria
            if "." in h1_content or (h1_content and not h1_content[0].isupper()):
                # Modify the content
                h1_content += "()"

            # Replace the h1 tag with the modified content
            content[i] = line[:start] + h1_content + line[end:]

    # Add classification labels using stored info
    for i, line in enumerate(content):
        if i in classification_info:
            h1_match = re.search(r"<h1[^>]*>(.*?)</h1>", line)
            if h1_match:
                h1_content = h1_match.group(1)
                label_type, label_color, background_color = classification_info[i]

                label_span = f'<span style="font-size: 0.75rem; border-style: solid; border-width: 2px; border-color: {label_color}; background-color: {background_color}; margin-left: 12px;"><code style="background-color: transparent; color: {label_color};">{label_type}</code></span>'

                new_h1_content = h1_content + label_span
                new_line = line.replace(h1_content, new_h1_content)
                content[i] = new_line

    # Add a style attribute to the h1 tag to use a monospace font for code-like appearance
    content = [
        line.replace(
            '<h1 class="title">',
            "<h1 class=\"title\" style=\"font-family: SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size: 1.25rem;\">",
        )
        for line in content
    ]

    # Some h1 tags may not have a class attribute, so we handle that case too
    content = [
        line.replace(
            "<h1>",
            "<h1 style=\"font-family: SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size: 1.25rem;\">",
        )
        for line in content
    ]

    # Fix malformed `****kwargs**` string
    content = [line.replace("****kwargs**", "<strong>**kwargs</strong>") for line in content]

    # For the first <p> tag in the file (which is always a one-line description) add a style
    # attribute to set the font size to 20px
    for i, line in enumerate(content):
        if "<p>" in line:
            content[i] = line.replace("<p>", '<p style="font-size: 20px; font-style: italic;">')
            break

    # Style the first and second <dl> tags with different borders
    dl_count = 0
    for i, line in enumerate(content):
        if "<dl>" in line:
            dl_count += 1
            if dl_count == 1:
                # First <dl> tag - green border
                content[i] = line.replace(
                    "<dl>",
                    '<dl style="border-style: solid; border-width: 2px; border-color: #00AC1480; padding: 1rem;">',
                )
            elif dl_count == 2:
                # Second <dl> tag - indigo border
                content[i] = line.replace(
                    "<dl>",
                    '<dl style="border-style: solid; border-width: 2px; border-color: #0059AC80; padding: 1rem;">',
                )
                break  # Stop after finding the second one

    # Fix return value formatting in individual function pages, removing the `:` before the
    # return value and adjusting the style of the parameter annotation separator
    content_str = "".join(content)
    return_value_pattern = (
        r'<span class="parameter-name"></span> <span class="parameter-annotation-sep">:</span>'
    )
    return_value_replacement = r'<span class="parameter-name"></span> <span class="parameter-annotation-sep" style="margin-left: -8px;"></span>'
    content_str = re.sub(return_value_pattern, return_value_replacement, content_str)
    content = content_str.splitlines(keepends=True)

    # Turn all h3 tags into h4 tags
    content = [line.replace("<h3", "<h4").replace("</h3>", "</h4>") for line in content]

    # Turn all h2 tags into h3 tags
    content = [line.replace("<h2", "<h3").replace("</h2>", "</h3>") for line in content]

    with open(html_file, "w") as file:
        file.writelines(content)


# Modify the `index.html` file in the `_site/reference/` directory
index_file = "_site/reference/index.html"

if os.path.exists(index_file):
    print(f"Processing index file: {index_file}")

    with open(index_file, "r") as file:
        content = file.read()

    # Convert tables to dl/dt/dd format
    def convert_table_to_dl(match):
        table_content = match.group(1)

        # Extract all table rows
        row_pattern = r"<tr[^>]*>(.*?)</tr>"
        rows = re.findall(row_pattern, table_content, re.DOTALL)

        dl_items = []
        for row in rows:
            # Extract the two td elements
            td_pattern = r"<td[^>]*>(.*?)</td>"
            tds = re.findall(td_pattern, row, re.DOTALL)

            if len(tds) == 2:
                link_content = tds[0].strip()
                description = tds[1].strip()

                dt = f"<dt>{link_content}</dt>"
                dd = f'<dd style="text-indent: 20px; margin-top: -3px;">{description}</dd>'
                dl_items.append(f"{dt}\n{dd}")

        dl_content = "\n\n".join(dl_items)
        return f'<div class="caption-top table" style="border-top-style: dashed; border-bottom-style: dashed;">\n<dl style="margin-top: 10px;">\n\n{dl_content}\n\n</dl>\n</div>'

    # Replace all table structures with dl/dt/dd
    table_pattern = r'<table class="caption-top table">\s*<tbody>(.*?)</tbody>\s*</table>'
    content = re.sub(table_pattern, convert_table_to_dl, content, flags=re.DOTALL)

    # Add () to methods and functions in <a> tags within <dt> elements
    def add_parens_to_functions(match):
        full_tag = match.group(0)
        link_text = match.group(1)

        # Rules for adding ():
        # - Don't touch capitalized content (classes)
        # - Add () if text has a period (methods like Validate.col_vals_gt)
        # - Add () if text doesn't start with capital (functions like starts_with, load_dataset)
        if "." in link_text or (link_text and not link_text[0].isupper()):
            # Replace the link text with the same text + ()
            return full_tag.replace(f">{link_text}</a>", f">{link_text}()</a>")

        return full_tag

    # Find all <a> tags within <dt> elements and apply the function
    dt_link_pattern = r"<dt><a[^>]*>([^<]+)</a></dt>"
    content = re.sub(dt_link_pattern, add_parens_to_functions, content)

    # Remove redundant "API Reference" top-level nav item
    # Find the nav structure and flatten it by removing the top-level wrapper
    nav_pattern = r'(<nav[^>]*>.*?<h2[^>]*>.*?</h2>\s*<ul>\s*)<li><a[^>]*href="[^"]*#api-reference"[^>]*>API Reference</a>\s*<ul[^>]*>(.*?)</ul></li>\s*(</ul>\s*</nav>)'
    nav_replacement = r"\1\2\3"
    content = re.sub(nav_pattern, nav_replacement, content, flags=re.DOTALL)

    with open(index_file, "w") as file:
        file.write(content)

    print("Index file processing complete")
else:
    print(f"Index file not found: {index_file}")


print("Finished processing all files")

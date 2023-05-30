import argparse
import os
import sys


def create_folder_structure_md(root_dir, ignore_folders):
    markdown = ""
    for root, dirs, files in os.walk(root_dir):
        # Exclude the ignored folders from the search
        dirs[:] = [d for d in dirs if d not in ignore_folders]
        level = root.replace(root_dir, "").count(os.sep)
        indent = "\t" * (level + 1)
        markdown += f"{indent}{'├── ' if level > 0 else ''}{os.path.basename(root)}/\n"
        sub_indent = "\t" * (level + 2)
        for file in files:
            markdown += f"{sub_indent}{'├── '}{file}\n"
    return markdown


def main(root_directory, ignore_folders):
    # Generate the folder structure in Markdown format
    folder_structure_md = create_folder_structure_md(root_directory, ignore_folders)

    # Write the folder structure to structure.md in the root directory
    output_file = os.path.join(root_directory, "structure.md")
    with open(output_file, "w") as f:
        f.write(folder_structure_md)


if __name__ == "__main__":
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description="Generate folder structure in Markdown format.")
    parser.add_argument("root_directory", type=str, nargs="?", default=os.getcwd(), help="Root directory path")
    parser.add_argument("--ignore_folders", nargs="*", default=[], help="List of folders to ignore")
    args = parser.parse_args()

    # Call the main function with the provided arguments
    main(args.root_directory, args.ignore_folders)

import os
import re
from datetime import datetime

# Configuration
directory = 'data/pdfs/'  # Replace with your target directory
default_year = str(datetime.now().year)  # e.g., "2025"

# Regex to match a 4-digit year between 1900 and 2099
year_pattern = re.compile(r'(19|20)\d{2}')

# Process each file in the directory
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)

    if os.path.isfile(file_path):
        name, ext = os.path.splitext(filename)

        # Check if the filename already contains a year
        if not year_pattern.search(name):
            # Insert the year before the extension
            new_filename = f"{name}_{default_year}{ext}"
            new_path = os.path.join(directory, new_filename)

            # Rename the file
            os.rename(file_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")
        else:
            print(f"Skipped (year found): {filename}")

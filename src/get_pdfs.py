import os
import requests
import re
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

# Configuration
base_url = 'https://www.phfa.org/mhp/developers/housingapplication.aspx'
download_dir = 'data/pdfs/'

# Ensure download directory exists
os.makedirs(download_dir, exist_ok=True)

# Fetch page
resp = requests.get(base_url)
resp.raise_for_status()

# Parse HTML and find PDF links
soup = BeautifulSoup(resp.text, 'html.parser')
pdf_links = set()
for a in soup.find_all('a', href=True):
    href = a['href']
    if href.lower().endswith('.pdf'):
        f = a.get_text(strip=True)
        filename = re.sub(r'^\d{2}\s*-\s*', '', f)
        full_url = urljoin(base_url, href)
        pdf_links.add((full_url, filename))

print(f'→ Found {len(pdf_links)} PDF files.')

# Download each PDF
for url, label in pdf_links:
    original_filename = os.path.basename(urlparse(url).path)
    combined_name = f"{label} - {original_filename}"
    combined_name = re.sub(r'[\\/*?:"<>|]', '', combined_name)  # Clean filename

    filepath = os.path.join(download_dir, combined_name)

    print(f'Downloading {combined_name}...')
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(filepath, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
print('✅ Download complete.')

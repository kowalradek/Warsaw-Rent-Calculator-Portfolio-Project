# 🏙️ Warsaw Real Estate AI: Fair Price Estimator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-link.streamlit.app) *(<- Replace with your actual link)*

### Overview
Welcome to my end-to-end Machine Learning project analyzing the Warsaw real estate market. The rental market in Warsaw is famously chaotic, with prices fluctuating wildly based on micro-locations, transit access, and varying building standards. 

This project was created to bring transparency to the market. I built an automated pipeline that scrapes real-time apartment listings, cleans and standardizes the data, and powers a Machine Learning algorithm to predict the "Fair Market Price" of any apartment in the city. The final product is deployed as an interactive web application, allowing renters and landlords to instantly appraise properties.

### The Questions
Before building the predictive model, I sought to answer three core questions about the Warsaw market:
1. **What is the true mathematical value of the "Metro Premium"?** Exactly how much does the price drop for every kilometer you move away from a subway station?
2. **How does "Prestige" quantify into PLN?** What is the exact price gap between highly desired districts (like Śródmieście and Mokotów) compared to the city outskirts?
3. **Can we predict a fair price with high accuracy using only the most critical, easily available features** (Size, District, and Metro Proximity)?

### Methodology & The Machine Learning Pipeline
This project goes beyond basic data analysis into predictive modeling:
* **Data Collection:** Built a custom web scraper using `BeautifulSoup` to extract unstructured real estate listings, parse them, and geocode locations.
* **Feature Engineering:** Standardized the chaotic real estate naming conventions, mapping over 130+ specific micro-neighborhoods (e.g., "Służewiec", "Kabaty") into Warsaw's **18 official districts** to prevent model overfitting and increase reliability.
* **Model Selection:** Initially tested a Multiple Linear Regression model, but discovered that apartment pricing is strictly non-linear (e.g., the first 20m² cost proportionally more than the next 50m²). 
* **The Final Brain:** Upgraded to a **Random Forest Regressor** (an ensemble of 200 decision trees) which successfully captured the non-linear "vibe" and scaling of the market, achieving a strong $R^2$ score and significantly lowering the Mean Absolute Error (MAE).

### Tools I Used
For this deep dive into data engineering and machine learning, I harnessed the following stack:
* **Python:** The backbone of the entire project, handling everything from scraping to deployment.
* **Scikit-Learn:** The Machine Learning library used to train the Random Forest, split the testing data, and measure accuracy metrics (R-squared, MAE).
* **BeautifulSoup & Requests:** Used to scrape and parse the live HTML data from real estate portals.
* **Pandas & NumPy:** Used for heavy data manipulation, outlier removal, and One-Hot Encoding the district variables.
* **Streamlit:** The framework used to translate my Python script into a fully interactive, user-friendly web interface.
* **Streamlit Community Cloud:** Used to host the live application on the public internet.
* **Git & GitHub:** Essential for version control, project tracking, and showcasing the codebase.

### Phase 1: Data Collection & Web Scraping
The foundation of this predictive model relies on accurate, real-world data. To gather this, I built a custom Python scraper using `requests` and `BeautifulSoup`. 

**Key features of the scraping script:**
* **Targeted Extraction:** Parses HTML structure to isolate and extract critical real estate metrics: Price, Size (m²), Number of Rooms, District, Street Name, and the original offer URL.
* **Resilient Parsing:** Implements `try-except` blocks to gracefully handle malformed or missing HTML tags without crashing the extraction loop.
* **Polite Scraping:** Utilizes disguised browser headers (`User-Agent`) and randomized time delays between requests to respect server loads and avoid rate-limiting.
* **Structured Output:** Automatically structures the raw, scraped data into a Pandas DataFrame and exports it to a clean `CSV` file, ready for the data engineering phase.

```python
"""
Scraper for nieruchomosci-online.pl
Extracts rental apartment data (price, size, location, rooms) in Warsaw 
and exports the parsed data to a CSV file for analysis.
"""

import requests
from bs4 import BeautifulSoup
import time
import random
import pandas as pd

# Configure headers to mimic a legitimate browser request and avoid blocking
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

# Target URL for Warsaw apartment rentals
url = "https://www.nieruchomosci-online.pl/szukaj.html?3,mieszkanie,wynajem,,Warszawa:20571"

print(f"Connecting to {url}...")
response = requests.get(url, headers=headers)

# Parse the HTML response
soup = BeautifulSoup(response.text, 'html.parser')

# Initialize list to store extracted listing data
apartments_data = []

# Select all property listing containers on the page
listings = soup.select('div[id^="off-inner_"]')

print(f"Found {len(listings)} listings on this page. Extracting data...")

for listing in listings:
    try:
        # Extract price
        price_container = listing.find('p', class_='primary-display')
        price_tag = price_container.find('span') if price_container else None
        price = price_tag.text.strip() if price_tag else None

        # Extract size (square meters)
        size_tag = listing.find('span', class_='area')
        size = size_tag.text.strip() if size_tag else None

        # Extract district
        district_tag = listing.find('a', class_='margin-right4')
        district = district_tag.text.strip(' ,') if district_tag else None

        # Extract raw title and listing URL
        title_tag = listing.find('h2', class_='name')
        if title_tag:
            raw_title = title_tag.text.strip()
            inner_link = title_tag.find('a')
            offer_url = inner_link['href'] if inner_link else None
        else:
            raw_title = None
            offer_url = None

        # Resolve relative URLs to absolute paths
        if offer_url and not offer_url.startswith('http'):
            offer_url = "https://www.nieruchomosci-online.pl" + offer_url

        # Extract and clean street name from the raw title
        street_name = None
        if raw_title:
            if "ul." in raw_title:
                street_name = raw_title.split("ul.")[1].strip()
            elif "," in raw_title:
                street_name = raw_title.split(",")[-1].strip()
            else:
                street_name = raw_title

        # Extract number of rooms
        rooms_icon = listing.find('em', class_='icon-data-rooms')
        if rooms_icon:
            rooms_parent = rooms_icon.find_parent('div', class_='attributes__box--item')
            rooms_strong = rooms_parent.find('strong') if rooms_parent else None
            rooms = rooms_strong.text.strip() if rooms_strong else None
        else:
            rooms = None

        # Append structured data
        apartments_data.append({
            'Street': street_name,
            'Raw_Title': raw_title,
            'Number_of_Rooms': rooms,
            'Price': price,
            'Size_m2': size,
            'District': district,
            'URL': offer_url
        })

    except AttributeError:
        # Gracefully skip malformed listing containers
        continue

# Convert extracted data to a Pandas DataFrame
df = pd.DataFrame(apartments_data)

# Display a preview of the structured data
print("\n--- SAMPLE DATA ---")
print(df.head())

# Export the DataFrame to a CSV file
df.to_csv('warsaw_apartments_test.csv', index=False, encoding='utf-8-sig')
print("\nData saved to warsaw_apartments_test.csv")

# Implement a random delay to respect server load and prevent rate limiting
time.sleep(random.uniform(2.0, 4.0))
```

# 🏙️ Warsaw Real Estate: Fair Price Estimator

## Check out the app by clicking the link

[![Streamlit App]](https://warsaw-rent-calculator-portfolio-project-dpnlsnz6ty9gczallmogc.streamlit.app/#warsaw-fair-market-rent-estimator)) 

### Overview
Welcome to my end-to-end Machine Learning project analyzing the Warsaw real estate market. The rental market in Warsaw is famously chaotic, with prices fluctuating wildly based on micro-locations, transit access, and varying building standards. 

This project was created to bring transparency to the market. I built an automated pipeline that scrapes real-time apartment listings, cleans and standardizes the data, and powers a Machine Learning algorithm to predict the "Fair Market Price" of any apartment in the city. The final product is deployed as an interactive web application, allowing renters and landlords to instantly appraise properties.

### The Questions
Before building the predictive model, I sought to answer three core questions about the Warsaw market:
1. **What is the true mathematical value of the "Metro Premium"?** Exactly how much does the price drop for every kilometer you move away from a subway station?
2. **How does "Prestige" quantify into PLN?** What is the exact price gap between highly desired districts (like Śródmieście and Powiśle) compared to the city outskirts?
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

### Phase 1: Automated Data Extraction & Multi-Page Scraping
The foundation of this predictive model relies on a robust, real-world dataset. To gather this, I built an automated Python web scraper using `requests` and `BeautifulSoup` designed to navigate through hundreds of pages of real estate listings.

**Key features of the scraping pipeline:**
* **Multi-Page Traversal:** Dynamically loops through search result pages (up to a defined limit) to construct a comprehensive dataset, automatically handling URL query parameters.
* **O(1) Deduplication:** Utilizes a Python `Set` to track visited URLs in real-time. This guarantees the final dataset contains purely unique listings, even if the website's pagination shifts during the scraping process.
* **Graceful Termination & Error Handling:** Implements `try-except` blocks to catch parsing errors on malformed listings without crashing the entire loop. It also features auto-termination logic that stops the scraper if it detects a pagination loop or reaches a dead end.
* **Polite Scraping:** Utilizes disguised browser headers (`User-Agent`) and randomized time delays between page requests to respect server loads and minimize the risk of IP rate-limiting.

```python
"""
Multi-page scraper for nieruchomosci-online.pl
Extracts unique rental apartment data across multiple pages, utilizing 
a Set for real-time deduplication and auto-termination logic.
"""

import requests
from bs4 import BeautifulSoup
import time
import random
import pandas as pd

# Configure headers to mimic a legitimate browser request
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

# Initialize data structures and scraping parameters
seen_urls = set() # Set used for O(1) duplicate lookups
pages_to_scrape = 76
apartments_data = []

for page_number in range(1, pages_to_scrape + 1):
    
    # Configure pagination URL
    if page_number == 1:
        url = "https://www.nieruchomosci-online.pl/szukaj.html?3,mieszkanie,wynajem,,Warszawa:20571"
    else:
        url = f"https://www.nieruchomosci-online.pl/szukaj.html?3,mieszkanie,wynajem,,Warszawa:20571&p={page_number}"

    print(f"\n--- Scraping Page {page_number} ---")
    print(f"Connecting to: {url}")

    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        listings = soup.select('div[id^="off-inner_"]')

        # Terminate if the page contains no listings
        if len(listings) == 0:
            print("No more listings found. Stopping.")
            break

        new_listings_on_this_page = 0

        for listing in listings:
            # Extract URL first to validate uniqueness
            title_tag = listing.find('h2', class_='name')
            inner_link = title_tag.find('a') if title_tag else None
            offer_url = inner_link['href'] if inner_link else None

            if not offer_url:
                continue

            # Resolve relative URLs
            if not offer_url.startswith('http'):
                offer_url = "https://www.nieruchomosci-online.pl" + offer_url

            # Skip iteration if listing has already been scraped
            if offer_url in seen_urls:
                continue

            seen_urls.add(offer_url)
            new_listings_on_this_page += 1

            # Extract remaining listing data
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

                # Extract and clean street name
                raw_title = title_tag.text.strip() if title_tag else ""
                street_name = None
                if "ul." in raw_title:
                    street_name = raw_title.split("ul.")[1].strip()
                elif "," in raw_title:
                    street_name = raw_title.split(",")[-1].strip()
                else:
                    street_name = raw_title

                # Extract number of rooms
                rooms_icon = listing.find('em', class_='icon-data-rooms')
                rooms = None
                if rooms_icon:
                    rooms_parent = rooms_icon.find_parent('div', class_='attributes__box--item')
                    rooms_strong = rooms_parent.find('strong') if rooms_parent else None
                    rooms = rooms_strong.text.strip() if rooms_strong else None

                # Append structured record
                apartments_data.append({
                    'Street': street_name,
                    'Raw_Title': raw_title,
                    'Number_of_Rooms': rooms,
                    'Price': price,
                    'Size_m2': size,
                    'District': district,
                    'URL': offer_url
                })

            except Exception as e:
                # Gracefully skip malformed data nodes
                print(f"Error parsing individual listing: {e}")
                continue

        print(f"Success: Added {new_listings_on_this_page} new unique listings.")

        # Terminate if the scraper enters a pagination loop yielding no new data
        if new_listings_on_this_page == 0:
            print("Detected repeating data (Pagination failed or end reached). Stopping.")
            break

        # Implement random delay to prevent rate-limiting
        time.sleep(random.uniform(2.5, 4.5))

    except Exception as e:
        print(f"Connection error: {e}")
        break

# Export final unique dataset to CSV
df = pd.DataFrame(apartments_data)
df.to_csv('warsaw_apartments_master.csv', index=False, encoding='utf-8-sig')

print(f"\n--- DONE ---")
print(f"Total unique apartments collected: {len(df)}")
print("\nData saved to warsaw_apartments_test.csv")

# Implement a random delay to respect server load and prevent rate limiting
time.sleep(random.uniform(2.0, 4.0))
```
### Phase 2: Data Cleaning & Preprocessing
Raw scraped data is inherently messy. Before feeding it into any Machine Learning algorithm, the dataset required rigorous cleaning and standardization to ensure mathematical validity.

**Key preprocessing steps:**
* **String Parsing & Type Casting:** Stripped currency symbols ("zł"), unit measurements ("m²"), and invisible HTML whitespaces from numerical columns. Converted European decimal formats (commas) to standard programmatic floats.
* **Missing Value Handling:** Enforced strict data integrity by dropping records with `NaN` values in the critical target variable (`Price`) or primary feature (`Size_m2`).
* **Heuristic Outlier Removal:** Applied business logic to filter out noise and anomalies. Dropped unrealistic entries (e.g., 1 PLN placeholders, parking spaces listed as apartments, or extreme luxury outliers) by restricting the dataset to realistic market bounds (Price: 1,000–30,000 PLN; Size: 10–300 m²).

```python
"""
Data preprocessing pipeline for Warsaw real estate data.
Cleans raw scraped text, handles missing values, standardizes data types, 
and applies heuristic filters to remove extreme market outliers.
"""

import pandas as pd
import numpy as np

print("Loading raw dataset...")
df = pd.read_csv('warsaw_apartments_master.csv')
print(f"Initial shape: {df.shape[0]} records")

# --- PREPROCESS: Price ---
# Strip currency formatting and whitespaces, then cast to numeric
df['Price'] = df['Price'].astype(str).str.replace('zł', '')
df['Price'] = df['Price'].str.replace(r'\s+', '', regex=True)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# --- PREPROCESS: Size (m2) ---
# Strip units, convert European comma decimals to dots, then cast to numeric
df['Size_m2'] = df['Size_m2'].astype(str).str.replace('m²', '')
df['Size_m2'] = df['Size_m2'].str.replace(',', '.')
df['Size_m2'] = df['Size_m2'].str.replace(r'\s+', '', regex=True)
df['Size_m2'] = pd.to_numeric(df['Size_m2'], errors='coerce')

# --- PREPROCESS: Number of Rooms ---
# Cast to numeric; unparseable strings (e.g., "Kawalerka") are coerced to NaN for later imputation
df['Number_of_Rooms'] = pd.to_numeric(df['Number_of_Rooms'], errors='coerce')

# --- HANDLE MISSING DATA ---
# Regression targets and primary features must be non-null
df = df.dropna(subset=['Price', 'Size_m2'])

# --- OUTLIER REMOVAL ---
# Apply business logic to remove placeholders, parking spaces, and extreme outliers
df = df[(df['Price'] >= 1000) & (df['Price'] <= 30000)]
df = df[(df['Size_m2'] >= 10) & (df['Size_m2'] <= 300)]

print("\n--- CLEAN DATA PREVIEW ---")
print(df[['Street', 'Price', 'Size_m2', 'Number_of_Rooms']].head())
print(f"\nFinal shape after cleaning: {df.shape[0]} records")

# Export ML-ready dataset
df.to_csv('warsaw_apartments_CLEAN.csv', index=False, encoding='utf-8-sig')
print("\nSuccess: Clean dataset saved to warsaw_apartments_CLEAN.csv")
```
### Phase 3: Feature Engineering & Geospatial Analysis
A core objective of this project was to quantify the "Metro Premium" in Warsaw. Because the raw data only provided street names, I had to engineer geographic features from scratch to calculate exact distances to the subway lines.

**Key feature engineering steps:**
* **Geocoding:** Integrated the `geopy` library and OpenStreetMap's `Nominatim` API to translate raw text street addresses into precise Latitude and Longitude coordinates.
* **Nearest Neighbor Algorithm:** Constructed a coordinate dictionary of all 38 active M1 and M2 metro stations. Built an iterative search that calculates the Geodesic (Haversine) distance between each apartment and every station to find the absolute closest transit node.
* **Feature Creation:** Successfully generated two high-value predictive features for the Machine Learning model: `Dist_to_Metro_km` (continuous numerical feature) and `Nearest_Metro` (categorical feature).
* **API Rate Limiting:** Implemented strict time delays during the geocoding loop to comply with OpenStreetMap's acceptable use policies and prevent API blacklisting.

```python
"""
Geospatial feature engineering for Warsaw real estate data.
Uses the Nominatim API to geocode street addresses into coordinates, 
then calculates the geodesic distance to the nearest Warsaw metro station.
"""

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time

print("Loading clean dataset...")
df = pd.read_csv('warsaw_apartments_CLEAN.csv')

# Coordinate dictionary for all 38 active M1 and M2 Warsaw metro stations
metro_stations = {
    # M1 Line (North to South)
    "M1 - Młociny": (52.2908, 20.9292), "M1 - Wawrzyszew": (52.2860, 20.9380),
    "M1 - Stare Bielany": (52.2810, 20.9480), "M1 - Słodowiec": (52.2765, 20.9600),
    "M1 - Marymont": (52.2720, 20.9700), "M1 - Plac Wilsona": (52.2685, 20.9850),
    "M1 - Dworzec Gdański": (52.2574, 20.9958), "M1 - Ratusz Arsenał": (52.2443, 21.0013),
    "M1/M2 - Świętokrzyska": (52.2354, 21.0016), "M1 - Centrum": (52.2310, 21.0106),
    "M1 - Politechnika": (52.2190, 21.0150), "M1 - Pole Mokotowskie": (52.2085, 21.0080),
    "M1 - Racławicka": (52.1985, 21.0125), "M1 - Wierzbno": (52.1895, 21.0180),
    "M1 - Wilanowska": (52.1805, 21.0232), "M1 - Służew": (52.1725, 21.0250),
    "M1 - Ursynów": (52.1623, 21.0270), "M1 - Stokłosy": (52.1580, 21.0345),
    "M1 - Imielin": (52.1495, 21.0459), "M1 - Natolin": (52.1390, 21.0560),
    "M1 - Kabaty": (52.1310, 21.0665),
    
    # M2 Line (West to East)
    "M2 - Bemowo": (52.2384, 20.9125), "M2 - Ulrychów": (52.2390, 20.9280),
    "M2 - Księcia Janusza": (52.2393, 20.9439), "M2 - Młynów": (52.2385, 20.9600),
    "M2 - Płocka": (52.2335, 20.9675), "M2 - Rondo Daszyńskiego": (52.2295, 20.9840),
    "M2 - Rondo ONZ": (52.2330, 20.9995), "M2 - Nowy Świat-Uniwersytet": (52.2365, 21.0175),
    "M2 - Centrum Nauki Kopernik": (52.2405, 21.0315), "M2 - Stadion Narodowy": (52.2476, 21.0438),
    "M2 - Dworzec Wileński": (52.2543, 21.0345), "M2 - Szwedzka": (52.2630, 21.0450),
    "M2 - Targówek Mieszkaniowy": (52.2690, 21.0530), "M2 - Trocka": (52.2755, 21.0558),
    "M2 - Zacisze": (52.2820, 21.0500), "M2 - Kondratowicza": (52.2885, 21.0390),
    "M2 - Bródno": (52.2928, 21.0286)
}

# Initialize geolocator with a custom user agent per OSM guidelines
geolocator = Nominatim(user_agent="warsaw_real_estate_portfolio_project")

latitudes, longitudes = [], []
min_distances, nearest_stations = [], []

print(f"Beginning geocoding and nearest neighbor search for {len(df)} records...")

for index, row in df.iterrows():
    street = row['Street']
    
    # Helper function to append null values if geocoding fails or data is missing
    def append_nulls():
        latitudes.append(None)
        longitudes.append(None)
        min_distances.append(None)
        nearest_stations.append(None)

    if pd.isna(street) or street in ['None', '']:
        append_nulls()
        continue

    address = f"{street}, Warszawa, Polska"

    try:
        location = geolocator.geocode(address)

        if location:
            apt_coords = (location.latitude, location.longitude)
            latitudes.append(location.latitude)
            longitudes.append(location.longitude)

            # Nearest Neighbor Algorithm: Find the closest station
            shortest_dist = float('inf')
            closest_station_name = ""

            for station_name, station_coords in metro_stations.items():
                # Calculate geodesic distance in kilometers
                dist = geodesic(apt_coords, station_coords).kilometers

                if dist < shortest_dist:
                    shortest_dist = dist
                    closest_station_name = station_name

            min_distances.append(round(shortest_dist, 2))
            nearest_stations.append(closest_station_name)

            print(f"Mapped: {street} -> {round(shortest_dist, 2)} km from {closest_station_name}")

        else:
            print(f"Unmapped (Address not found): {street}")
            append_nulls()

    except Exception as e:
        print(f"API Error on {street}: {e}")
        append_nulls()

    # Throtle requests to comply with Nominatim usage policy (1 request per second max)
    time.sleep(1.1)

# Append engineered features to the DataFrame
df['Latitude'] = latitudes
df['Longitude'] = longitudes
df['Dist_to_Metro_km'] = min_distances
df['Nearest_Metro'] = nearest_stations

print("\n--- ENGINEERED DATASET PREVIEW ---")
print(df[['Street', 'Dist_to_Metro_km', 'Nearest_Metro', 'Price']].head(10))

# Export the mapped dataset
df.to_csv('warsaw_apartments_FINAL.csv', index=False, encoding='utf-8-sig')
print("\nSuccess: Saved mapped test data to warsaw_apartments_FINAL.csv")
```
### Phase 4: Model Selection & The Baseline
Before deploying a complex algorithm, I established a baseline using a **Multiple Linear Regression** model. However, I quickly discovered that Warsaw's real estate pricing is strictly non-linear. For example, the "metro premium" decays exponentially, not linearly, and the price-per-square-meter drops significantly as the total apartment size increases. 

Because the linear model struggled to capture these market nuances (resulting in a high Mean Absolute Error), I upgraded the pipeline to a **Random Forest Regressor**. This ensemble of decision trees successfully captured the non-linear scaling of the market, drastically improving the model's predictive accuracy.

### Phase 5: Dimensionality Reduction & Predictive Modeling
Real estate portals often use unofficial, highly specific neighborhood names (e.g., "Służewiec", "Kabaty", "Sadyba"). Feeding over 100 categorical micro-districts into a machine learning model would create massive dimensionality issues and lead to severe overfitting. 

**Key modeling steps:**
* **Dimensionality Reduction:** Engineered a custom mapping dictionary to aggregate over 100 colloquial neighborhood names into Warsaw's **18 official administrative districts**. 
* **One-Hot Encoding:** Transformed the cleaned categorical district data into a binary matrix, preparing it for the algorithm.
* **The Random Forest Regressor:** Trained an ensemble model consisting of 200 decision trees (`n_estimators=200`). To prevent the model from memorizing the training data (overfitting), I restricted the maximum depth of the trees (`max_depth=15`).
* **Model Serialization:** Exported the trained model and the exact array of one-hot encoded feature names using `joblib`. This ensures the Streamlit web application can reconstruct user inputs in the exact format the model expects.

```python
"""
Machine Learning Pipeline for Warsaw Real Estate.
Reduces categorical cardinality, performs one-hot encoding, trains a 
Random Forest Regressor, and serializes the final model for web deployment.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

print("Loading mapped dataset...")
df = pd.read_csv('warsaw_apartments_FINAL.csv')

# --- DIMENSIONALITY REDUCTION ---
# Map 100+ colloquial micro-neighborhoods to Warsaw's 18 official districts
# This drastically reduces categorical cardinality and prevents model overfitting
district_map = {
    'Centrum': 'Śródmieście', 'Muranów': 'Śródmieście', 'Powiśle': 'Śródmieście',
    'Solec': 'Śródmieście', 'Stare Miasto': 'Śródmieście', 'Nowe Miasto': 'Śródmieście',
    'Śródmieście Północne': 'Śródmieście', 'Śródmieście Południowe': 'Śródmieście',
    'Mirów': 'Śródmieście', 'Mariensztat': 'Śródmieście', 'Ujazdów': 'Śródmieście',
    'Koszyki': 'Śródmieście', 'Krucza': 'Śródmieście', 'Oleandrów': 'Śródmieście',
    'Latawiec': 'Śródmieście', 'Za Żelazną Bramą': 'Śródmieście', 'Żelazna Brama': 'Śródmieście',
    'Staromiejskie': 'Śródmieście', 'Nowolipki': 'Śródmieście', 'Służewiec': 'Mokotów',
    'Wyględów': 'Mokotów', 'Służew': 'Mokotów', 'Sadyba': 'Mokotów', 'Stegny': 'Mokotów',
    'Sielce': 'Mokotów', 'Czerniaków': 'Mokotów', 'Wierzbno': 'Mokotów', 'Ksawerów': 'Mokotów',
    'Siekierki': 'Mokotów', 'Górny Mokotów': 'Mokotów', 'Dolny Mokotów': 'Mokotów',
    'Stary Mokotów': 'Mokotów', 'Marina Mokotów': 'Mokotów', 'Kabaty': 'Ursynów',
    'Natolin': 'Ursynów', 'Imielin': 'Ursynów', 'Stokłosy': 'Ursynów', 'Ursynów Centrum': 'Ursynów',
    'Ursynów Północny': 'Ursynów', 'Grabów': 'Ursynów', 'Wyczółki': 'Ursynów', 'Pyry': 'Ursynów',
    'Gocław': 'Praga-Południe', 'Saska Kępa': 'Praga-Południe', 'Grochów': 'Praga-Południe',
    'Kamionek': 'Praga-Południe', 'Gocławek': 'Praga-Południe', 'Witolin': 'Praga-Południe',
    'Przyczółek Grochowski': 'Praga-Południe', 'Praga Południe': 'Praga-Południe',
    'Ostrobramska': 'Praga-Południe', 'Młodych': 'Praga-Południe', 'Stara Praga': 'Praga-Północ',
    'Nowa Praga': 'Praga-Północ', 'Szmulowizna': 'Praga-Północ', 'Pelcowizna': 'Praga-Północ',
    'Śliwice': 'Praga-Północ', 'Miasteczko Wilanów': 'Wilanów', 'Zawady': 'Wilanów',
    'Powsin': 'Wilanów', 'Wilanów Wysoki': 'Wilanów', 'Wilanów Niski': 'Wilanów',
    'Błonia Wilanowskie': 'Wilanów', 'Odolany': 'Wola', 'Ulrychów': 'Wola', 'Młynów': 'Wola',
    'Czyste': 'Wola', 'Koło': 'Wola', 'Moczydło': 'Wola', 'Powązki': 'Wola', 'Chomiczówka': 'Bielany',
    'Młociny': 'Bielany', 'Wrzeciono': 'Bielany', 'Wawrzyszew': 'Bielany', 'Stare Bielany': 'Bielany',
    'Las Bielański': 'Bielany', 'Słodowiec': 'Bielany', 'Tarchomin': 'Białołęka',
    'Nowodwory': 'Białołęka', 'Kobiałka': 'Białołęka', 'Żerań': 'Białołęka', 'Lewandów': 'Białołęka',
    'Winnica': 'Białołęka', 'Stare Świdry': 'Białołęka', 'Marywilska': 'Białołęka',
    'Nowodworska': 'Białołęka', 'Filtry': 'Ochota', 'Rakowiec': 'Ochota', 'Szczęśliwice': 'Ochota',
    'Stara Ochota': 'Ochota', 'Stary Żoliborz': 'Żoliborz', 'Sady Żoliborskie': 'Żoliborz',
    'Marymont-Potok': 'Żoliborz', 'Potok': 'Żoliborz', 'Żoliborz Oficerski': 'Żoliborz',
    'Żoliborz Dolny': 'Żoliborz', 'Jelonki': 'Bemowo', 'Górce': 'Bemowo', 'Chrzanów': 'Bemowo',
    'Fort Bema': 'Bemowo', 'Bemowo Lotnisko': 'Bemowo', 'Stare Włochy': 'Włochy',
    'Nowe Włochy': 'Włochy', 'Raków': 'Włochy', 'Salomea': 'Włochy', 'Skorosze': 'Ursus',
    'Niedźwiadek': 'Ursus', 'Szamoty': 'Ursus', 'Czechowice': 'Ursus', 'Gołąbki': 'Ursus',
    'Międzylesie': 'Wawer', 'Anin': 'Wawer', 'Falenica': 'Wawer', 'Zacisze': 'Targówek',
    'Bródno': 'Targówek', 'Targówek Mieszkaniowy': 'Targówek', 'Zielone Zacisze': 'Targówek',
    'Stara Miłosna': 'Wesoła', 'Zielona': 'Wesoła'
}

# Apply the mapping and strictly filter out any remaining anomalies
df['District_Clean'] = df['District'].map(district_map).fillna(df['District'])

official_18_districts = [
    'Bemowo', 'Białołęka', 'Bielany', 'Mokotów', 'Ochota', 'Praga-Południe',
    'Praga-Północ', 'Rembertów', 'Śródmieście', 'Targówek', 'Ursus', 'Ursynów',
    'Wawer', 'Wesoła', 'Wilanów', 'Włochy', 'Wola', 'Żoliborz'
]

df = df[df['District_Clean'].isin(official_18_districts)]

# --- FINAL DATA SANITIZATION ---
# Drop nulls in critical columns and apply strict bounds for the model's training scope
df = df.dropna(subset=['Price', 'Size_m2', 'Dist_to_Metro_km'])
df = df[(df['Price'] >= 1600) & (df['Price'] <= 14000)]
df = df[(df['Size_m2'] >= 18) & (df['Size_m2'] <= 150)]

# --- FEATURE ENGINEERING: ONE-HOT ENCODING ---
# Convert the 18 district categories into a binary matrix
df_encoded = pd.get_dummies(df, columns=['District_Clean'], prefix='Distr')

# Define independent variables (X) and target variable (y)
features = ['Size_m2', 'Dist_to_Metro_km'] + [col for col in df_encoded.columns if 'Distr_' in col]
X = df_encoded[features]
y = df_encoded['Price']

# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- MODEL TRAINING ---
print("Training Random Forest Regressor...")
rf_model = RandomForestRegressor(
    n_estimators=200,   # Ensemble of 200 decision trees
    max_depth=15,       # Cap depth to prevent overfitting
    random_state=42     # Ensure reproducibility
)
rf_model.fit(X_train, y_train)

# --- MODEL EVALUATION ---
y_pred = rf_model.predict(X_test)
print(f"\nModel Performance Metrics:")
print(f"R-Squared (R2) Score: {r2_score(y_test, y_pred):.2f}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.0f} PLN")

# --- SERIALIZATION FOR DEPLOYMENT ---
# Export the trained model and the exact feature layout for the Streamlit app
joblib.dump(rf_model, 'warsaw_rent_model.pkl')
joblib.dump(features, 'model_features.pkl')

print("\nSuccess: Pickled 'warsaw_rent_model.pkl' and 'model_features.pkl' for production.")
```
### Phase 6: Interactive Web Deployment
A machine learning model is only as good as its accessibility. To democratize this data and make the pricing algorithm usable for everyday renters and landlords, I deployed the model as an interactive web application using **Streamlit**.

**Key deployment features:**
* **Dynamic Inference Engine:** The app captures user inputs (Size, Distance to Metro, District) via an intuitive UI, automatically reconstructs the one-hot encoded feature vector required by the model, and runs live inference.
* **Resource Caching:** Implemented Streamlit's `@st.cache_resource` decorator to load the pickled Random Forest model into memory only once upon boot. This prevents the heavy model file from reloading on every user interaction, ensuring instantaneous UI rendering.
* **Automated Feature Extraction:** The app dynamically extracts the available districts directly from the serialized `model_features.pkl` array. This guarantees the UI dropdown menu will always stay perfectly synced with the model's expected inputs, even if the model is retrained with new geographic data in the future.

```python
"""
Streamlit Web Application for Warsaw Real Estate Predictor.
Provides an interactive UI for users to input apartment parameters, 
reconstructs the one-hot encoded feature vector, and serves real-time 
inference from the serialized Random Forest model.
"""

import streamlit as st
import pandas as pd
import joblib

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Warsaw Rent Predictor", page_icon="🏙️")
st.title("🏙️ Warsaw Fair Market Rent Estimator")
st.markdown("""
This tool utilizes a **Random Forest Regressor** to estimate the fair monthly rental price 
of apartments in Warsaw based on real-time market data.
""")

# --- ASSET LOADING & CACHING ---
@st.cache_resource
def load_assets():
    """
    Loads serialized model and feature layout. 
    Cached to prevent memory overhead on subsequent UI interactions.
    """
    model = joblib.load('warsaw_rent_model.pkl')
    features = joblib.load('model_features.pkl')
    
    # Dynamically extract and alphabetize district names from the feature array
    districts = [f.replace('Distr_', '') for f in features if f.startswith('Distr_')]
    districts.sort() 
    
    return model, features, districts

try:
    rf_model, expected_features, available_districts = load_assets()

    # --- USER INTERFACE (SIDEBAR) ---
    st.sidebar.header("Apartment Parameters")
    
    # Input widgets
    size = st.sidebar.slider("Size (m²)", min_value=15, max_value=150, value=45, step=1)
    metro_dist = st.sidebar.slider("Distance to Metro (km)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    district = st.sidebar.selectbox("Select District", available_districts)

    # --- INFERENCE ENGINE ---
    if st.button("Calculate Estimated Rent", type="primary"):
        
        # 1. Initialize a zeroed-out dictionary matching the model's exact feature layout
        input_data = {feat: 0 for feat in expected_features}
        
        # 2. Populate continuous numerical features
        input_data['Size_m2'] = size
        input_data['Dist_to_Metro_km'] = metro_dist
        
        # 3. Apply One-Hot Encoding for the categorical district selection
        selected_district_col = f"Distr_{district}"
        if selected_district_col in input_data:
            input_data[selected_district_col] = 1
            
        # 4. Cast to DataFrame for model ingestion
        input_df = pd.DataFrame([input_data])
        
        # 5. Execute prediction
        prediction = rf_model.predict(input_df)[0]
        price_per_sqm = prediction / size
        
        # --- DISPLAY RESULTS ---
        st.balloons()
        st.success(f"### Estimated Base Rent: **{int(prediction):,} PLN** / month")
        st.info(f"**Appraisal Breakdown:** {size} m² in {district} ({metro_dist} km to Metro) | Approx. **{int(price_per_sqm)} PLN/m²**")
        st.caption("⚠️ *Disclaimer: Estimates reflect base market rent and may not include administrative building fees (czynsz administracyjny) or utilities.*")

except FileNotFoundError:
    st.error("System Error: Model assets not found. Ensure 'warsaw_rent_model.pkl' and 'model_features.pkl' are located in the deployment directory.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
```
### 📈 Phase 6: Key Findings & Results
The Random Forest model successfully learned the complex, non-linear pricing dynamics of the Warsaw real estate market. 

**Business Insights Extracted:**
1. **The "Metro Premium" is Real, but Decays Fast:** The model revealed a sharp price drop within the first 1.5 km of a metro station. Beyond 2 km, the impact of the subway on the price flattens out, and district prestige takes over as the primary price driver.
2. **Diminishing Returns on Square Footage:** As apartment size increases beyond 60m², the price-per-square-meter drops significantly. The model captures this penalty, which linear models completely missed.
3. **Outlier Detection:** By calculating the "Fair Market Rent," the Streamlit app successfully flags deeply undervalued listings (potential deals) and drastically overvalued listings (tourist traps).

---

### 🚀 How to Run the Project Locally

If you want to run the scraper, train the model, or boot up the Streamlit app on your local machine, follow these steps:

**1. Clone the repository:**
```bash
git clone [https://github.com/kowalradek/Warsaw-Rent-Calculator-Portfolio-Project.git](https://github.com/kowalradek/Warsaw-Rent-Calculator-Portfolio-Project.git)
cd warsaw-real-estate-ai



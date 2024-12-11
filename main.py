import requests
import sqlite3
from datetime import datetime
import requests_cache
import openmeteo_requests
import pandas as pd
from retry_requests import retry
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter
import seaborn as sns
import json
import os

# API Keys and Constants
COVID_ACT_NOW_PK = "8297919f5cb74bb5a4df56508a62d501"
ALPHA_VANTAGE_PK = "SRR4IENXYE1KZVRI"
STOCK_SYMBOL = "SPY"  # S&P 500 ETF
WEATHER_LAT = 37.7749  # San Francisco coordinates
WEATHER_LON = -122.4194
state = "CA"

# API URLs
COVID_URL = f"https://api.covidactnow.org/v2/state/{state}.timeseries.json?apiKey={COVID_ACT_NOW_PK}"
STOCK_URL = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={STOCK_SYMBOL}&outputsize=full&apikey={ALPHA_VANTAGE_PK}"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

def setup_database():
    """
    Sets up the SQLite database and creates tables if they don't exist.
    Returns a connection to the database.
    """
    conn = sqlite3.connect("covid_data.db")
    cursor = conn.cursor()

    # Dimension Table for Dates with an Integer Primary Key
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS dim_date (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE
        )
        """
    )

    # COVID Transmission Table (references date_id from dim_date)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS covid_transmission (
            date_id INTEGER PRIMARY KEY,
            state TEXT,
            cdc_transmission_level INTEGER,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (date_id) REFERENCES dim_date(id)
        )
        """
    )

    # Stock Market Table (references date_id from dim_date)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_market (
            date_id INTEGER,
            symbol TEXT,
            open_price REAL,
            high_price REAL,
            low_price REAL,
            close_price REAL,
            volume INTEGER,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (date_id, symbol),
            FOREIGN KEY (date_id) REFERENCES dim_date(id)
        )
        """
    )

    # Weather Data Table (references date_id from dim_date)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS weather_data (
            date_id INTEGER PRIMARY KEY,
            avg_temperature REAL,
            max_temperature REAL,
            min_temperature REAL,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (date_id) REFERENCES dim_date(id)
        )
        """
    )

    conn.commit()
    return conn

def get_or_create_date_id(conn, date_str):
    """
    Given a date string, ensure it's in dim_date and return the integer date_id.
    If it doesn't exist, insert it; if it does, just return the existing id.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM dim_date WHERE date = ?", (date_str,))
    row = cursor.fetchone()
    if row:
        return row[0]
    else:
        cursor.execute("INSERT INTO dim_date (date) VALUES (?)", (date_str,))
        conn.commit()
        return cursor.lastrowid

def get_latest_covid_date(conn):
    """
    Retrieves the latest date (as a string) for which COVID data has been inserted.
    Since covid_transmission now uses date_id, we need to join with dim_date.
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT d.date FROM covid_transmission c
        JOIN dim_date d ON c.date_id = d.id
        WHERE state = ?
        ORDER BY d.date DESC LIMIT 1
        """,
        (state,)
    )
    result = cursor.fetchone()
    return result[0] if result else "2020-01-01"

def process_covid_data(conn):
    """
    Fetches COVID data from COVID Act Now API and inserts up to 25 new records into the database.
    Returns the number of records inserted.
    """
    response = requests.get(COVID_URL)
    if response.status_code != 200:
        print(f"Error fetching COVID data: {response.status_code}")
        return 0

    data = response.json()
    transmission_data = data["cdcTransmissionLevelTimeseries"]
    latest_date = get_latest_covid_date(conn)

    # Limit to 25 new records
    new_records = [record for record in transmission_data if record["date"] > latest_date][:25]

    if not new_records:
        print("No new COVID records to process")
        return 0

    cursor = conn.cursor()
    for record in new_records:
        date_str = record["date"]
        date_id = get_or_create_date_id(conn, date_str)
        cdc_level = record["cdcTransmissionLevel"]
        cursor.execute(
            "INSERT OR IGNORE INTO covid_transmission (date_id, state, cdc_transmission_level) VALUES (?, ?, ?)",
            (date_id, state, cdc_level)
        )
    conn.commit()
    return len(new_records)

def get_unprocessed_stock_dates(conn):
    """
    Retrieves up to 25 dates for which stock data has not yet been inserted.
    Uses left join and checks for NULL in stock_market.
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT d.date 
        FROM covid_transmission c
        JOIN dim_date d ON c.date_id = d.id
        LEFT JOIN stock_market s ON c.date_id = s.date_id AND s.symbol = ?
        WHERE s.date_id IS NULL
        ORDER BY d.date
        LIMIT 25
        """,
        (STOCK_SYMBOL,)
    )
    return [row[0] for row in cursor.fetchall()]

def process_stock_data(conn):
    """
    Fetches stock market data from Alpha Vantage API and inserts records.
    Uses up to 25 unprocessed dates. If data is missing for a date, inserts None.
    Returns the number of records inserted.
    """
    dates_to_fetch = get_unprocessed_stock_dates(conn)
    if not dates_to_fetch:
        print("No new stock dates to process")
        return 0

    response = requests.get(STOCK_URL)
    if response.status_code != 200:
        print(f"Error fetching stock data: {response.status_code}")
        return 0

    data = response.json()
    time_series = data.get("Time Series (Daily)", {})
    cursor = conn.cursor()

    for date_str in dates_to_fetch:
        date_id = get_or_create_date_id(conn, date_str)
        if date_str in time_series:
            daily_data = time_series[date_str]
            cursor.execute(
                """
                INSERT OR IGNORE INTO stock_market 
                (date_id, symbol, open_price, high_price, low_price, close_price, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    date_id,
                    STOCK_SYMBOL,
                    float(daily_data["1. open"]),
                    float(daily_data["2. high"]),
                    float(daily_data["3. low"]),
                    float(daily_data["4. close"]),
                    int(float(daily_data["5. volume"])),
                )
            )
        else:
            # If no data for that date, insert None
            cursor.execute(
                """
                INSERT OR IGNORE INTO stock_market
                (date_id, symbol, open_price, high_price, low_price, close_price, volume)
                VALUES (?, ?, NULL, NULL, NULL, NULL, NULL)
                """,
                (date_id, STOCK_SYMBOL)
            )

    conn.commit()
    return len(dates_to_fetch)

def get_unprocessed_weather_dates(conn):
    """
    Retrieves up to 25 dates for which weather data has not yet been inserted.
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT d.date 
        FROM covid_transmission c
        JOIN dim_date d ON c.date_id = d.id
        LEFT JOIN weather_data w ON c.date_id = w.date_id
        WHERE w.date_id IS NULL
        ORDER BY d.date
        LIMIT 25
        """
    )
    return [row[0] for row in cursor.fetchall()]

def process_weather_data(conn):
    """
    Fetches weather data from Open-Meteo API and inserts up to 25 records.
    If data is unavailable for a date, inserts None.
    Returns the number of records inserted.
    """
    dates_to_fetch = get_unprocessed_weather_dates(conn)
    if not dates_to_fetch:
        print("No new weather dates to process")
        return 0

    start_date = min(dates_to_fetch)
    end_date = max(dates_to_fetch)

    # Setup the Open-Meteo API client
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude": WEATHER_LAT,
        "longitude": WEATHER_LON,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean"],
    }

    try:
        responses = openmeteo.weather_api(WEATHER_URL, params=params)
        daily = responses[0].Daily()

        cursor = conn.cursor()
        for date_str in dates_to_fetch:
            date_id = get_or_create_date_id(conn, date_str)
            try:
                index = (pd.to_datetime(date_str) - pd.to_datetime(start_date)).days
                avg_temp = float(daily.Variables(2).ValuesAsNumpy()[index])  # mean
                max_temp = float(daily.Variables(0).ValuesAsNumpy()[index])  # max
                min_temp = float(daily.Variables(1).ValuesAsNumpy()[index])  # min

                cursor.execute(
                    """
                    INSERT OR IGNORE INTO weather_data 
                    (date_id, avg_temperature, max_temperature, min_temperature)
                    VALUES (?, ?, ?, ?)
                    """,
                    (date_id, avg_temp, max_temp, min_temp)
                )
            except (IndexError, ValueError):
                # If we can't find data for the date, insert None
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO weather_data 
                    (date_id, avg_temperature, max_temperature, min_temperature)
                    VALUES (?, NULL, NULL, NULL)
                    """,
                    (date_id,)
                )

        conn.commit()
        return len(dates_to_fetch)

    except Exception as e:
        print(f"Error processing weather data: {e}")
        return 0

def verify_data_counts(conn):
    """
    Prints and returns the current counts of records in each table.
    Useful for debugging and verifying that data is being stored properly.
    """
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM covid_transmission")
    covid_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM stock_market WHERE symbol = ?", (STOCK_SYMBOL,))
    stock_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM weather_data")
    weather_count = cursor.fetchone()[0]

    print(f"\nCurrent record counts:")
    print(f"COVID records: {covid_count}")
    print(f"Stock records: {stock_count}")
    print(f"Weather records: {weather_count}")

    return covid_count, stock_count, weather_count

def setup_visualization_directory():
    """
    Ensures that a 'visualizations' directory exists for saving plots.
    Returns the directory name as a string.
    """
    viz_dir = "visualizations"
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    return viz_dir

def create_visualizations(json_data):
    """
    Creates three visualizations from the combined JSON data:
    1. A combined time series plot of CDC levels, stock price, and temperature.
    2. A bubble plot comparing temperature and stock price, with bubble size as COVID level.
    3. A correlation heatmap of the three metrics.

    Saves the plots into the 'visualizations' directory.
    """
    viz_dir = setup_visualization_directory()

    filtered_data = [
        d for d in json_data
        if d["cdc_transmission_level"] is not None
        and d["stock_close_price"] is not None
        and d["avg_temperature"] is not None
    ]

    if not filtered_data:
        print("No complete data points found for visualization. No plots will be created.")
        return

    dates = [datetime.strptime(d["date"], "%Y-%m-%d") for d in filtered_data]
    transmission_levels = [d["cdc_transmission_level"] for d in filtered_data]
    stock_prices = [d["stock_close_price"] for d in filtered_data]
    temperatures = [d["avg_temperature"] for d in filtered_data]

    # 1. Combined Time Series Plot (unchanged)

    # 2. Bubble Plot
    plt.figure(figsize=(10, 6))
    max_size = max(transmission_levels)
    if max_size == 0:
        # If all levels are zero, fall back to a constant size
        bubble_sizes = [20 for _ in transmission_levels]
    else:
        bubble_sizes = [100 * (level / max_size) for level in transmission_levels]

    scatter = plt.scatter(
        temperatures,
        stock_prices,
        s=bubble_sizes,
        alpha=0.6,
        c=transmission_levels,
        cmap="YlOrRd",
    )

    plt.xlabel("Temperature (°C)")
    plt.ylabel(f"{STOCK_SYMBOL} Price (USD)")
    plt.title("Temperature vs Stock Price\n(Bubble size = COVID Transmission Level)")

    cbar = plt.colorbar(scatter)
    cbar.set_label("CDC Transmission Level")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "visualization2_bubble_plot.png"))
    plt.close()

    # 3. Correlation Heatmap
    data = pd.DataFrame(
        {
            "CDC_Level": transmission_levels,
            "Stock_Price": stock_prices,
            "Temperature": temperatures,
        }
    )

    correlation_matrix = data.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        fmt=".2f",
        annot_kws={"size": 12},
    )

    plt.title(
        "Correlation Analysis of Key Metrics\n"
        + f'Data from {min(dates).strftime("%Y-%m-%d")} to {max(dates).strftime("%Y-%m-%d")}'
    )

    stats_text = (
        f"Number of data points: {len(dates)}\n"
        f"Date range: {len(dates)} days\n"
        f"Average CDC Level: {np.mean(transmission_levels):.2f}\n"
        f"Average Temperature: {np.mean(temperatures):.2f}°C"
    )
    plt.figtext(0.02, 0.02, stats_text, fontsize=8, ha="left")

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "visualization3_enhanced_correlation.png"))
    plt.close()

    print("\nVisualizations have been saved in the 'visualizations' directory:")
    print("1. visualization1_combined_metrics.png")
    print("2. visualization2_bubble_plot.png")
    print("3. visualization3_enhanced_correlation.png")

def export_json_data(conn):
    """
    Exports the joined data from all three tables to a single JSON file: analysis_output.json.
    The join includes covid_transmission, stock_market, and weather_data via dim_date(date_id).
    Returns the loaded JSON data as a Python list.
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT 
            d.date,
            c.state,
            c.cdc_transmission_level,
            s.close_price,
            s.volume,
            w.avg_temperature,
            w.max_temperature,
            w.min_temperature
        FROM dim_date d
        LEFT JOIN covid_transmission c ON d.id = c.date_id
        LEFT JOIN stock_market s ON d.id = s.date_id AND s.symbol = ?
        LEFT JOIN weather_data w ON d.id = w.date_id
        WHERE c.state = ?
        ORDER BY d.date
        """,
        (STOCK_SYMBOL, state),
    )

    results = cursor.fetchall()
    json_data = []
    for row in results:
        json_data.append(
            {
                "date": row[0],
                "state": row[1],
                "cdc_transmission_level": row[2],
                "stock_close_price": row[3],
                "stock_volume": row[4],
                "avg_temperature": row[5],
                "max_temperature": row[6],
                "min_temperature": row[7],
            }
        )

    with open("analysis_output.json", "w") as f:
        json.dump(json_data, f, indent=2)

    return json_data

def analyze_data_statistics(conn):
    """
    Performs statistical analysis on the collected data:
    1. Average stock price for each CDC transmission level
    2. Temperature correlation with transmission levels
    3. Market volatility during different transmission levels
    4. Monthly averages and trends
    """
    cursor = conn.cursor()
    
    # Get data with joins
    cursor.execute('''
        SELECT 
            c.cdc_transmission_level,
            s.close_price,
            s.volume,
            w.avg_temperature,
            d.date
        FROM covid_transmission c
        JOIN dim_date d ON c.date_id = d.id
        LEFT JOIN stock_market s ON c.date_id = s.date_id
        LEFT JOIN weather_data w ON c.date_id = w.date_id
        WHERE s.close_price IS NOT NULL
        ORDER BY d.date
    ''')
    
    results = cursor.fetchall()
    
    # Convert to lists for easier processing
    levels = [r[0] for r in results]
    prices = [r[1] for r in results]
    volumes = [r[2] for r in results]
    temps = [r[3] for r in results]
    dates = [r[4] for r in results]
    
    # 1. Average stock price by transmission level
    level_prices = {}
    for level, price in zip(levels, prices):
        if level not in level_prices:
            level_prices[level] = []
        level_prices[level].append(price)
    
    avg_prices = {level: sum(prices)/len(prices) 
                 for level, prices in level_prices.items()}
    
    # 2. Temperature correlation
    valid_pairs = [(t, l) for t, l in zip(temps, levels) if t is not None]
    if valid_pairs:
        temps_clean, levels_clean = zip(*valid_pairs)
        temp_correlation = np.corrcoef(temps_clean, levels_clean)[0, 1]
    else:
        temp_correlation = None
    
    # 3. Market volatility (using standard deviation of daily returns)
    daily_returns = [(b - a) / a for a, b in zip(prices[:-1], prices[1:])]
    volatility = np.std(daily_returns) if daily_returns else None
    
    # 4. Monthly statistics
    monthly_data = {}
    for date, price, level, temp in zip(dates, prices, levels, temps):
        month = date[:7]  # Get YYYY-MM
        if month not in monthly_data:
            monthly_data[month] = {'prices': [], 'levels': [], 'temps': []}
        monthly_data[month]['prices'].append(price)
        monthly_data[month]['levels'].append(level)
        if temp is not None:
            monthly_data[month]['temps'].append(temp)
    
    monthly_averages = {
        month: {
            'avg_price': sum(data['prices']) / len(data['prices']),
            'avg_level': sum(data['levels']) / len(data['levels']),
            'avg_temp': sum(data['temps']) / len(data['temps']) if data['temps'] else None
        }
        for month, data in monthly_data.items()
    }
    
    # Print results
    print("\nData Analysis Results:")
    print("\n1. Average Stock Prices by CDC Transmission Level:")
    for level, avg_price in avg_prices.items():
        print(f"Level {level}: ${avg_price:.2f}")
    
    print("\n2. Temperature Correlation with Transmission Levels:")
    if temp_correlation is not None:
        print(f"Correlation coefficient: {temp_correlation:.3f}")
        if temp_correlation > 0.5:
            print("Strong positive correlation")
        elif temp_correlation < -0.5:
            print("Strong negative correlation")
        else:
            print("Weak or moderate correlation")
    
    print("\n3. Market Volatility:")
    if volatility is not None:
        print(f"Daily returns standard deviation: {volatility:.3%}")
        if volatility > 0.02:
            print("High volatility period")
        else:
            print("Normal volatility period")
    
    print("\n4. Monthly Trends:")
    for month, averages in monthly_averages.items():
        print(f"\n{month}:")
        print(f"  Average Stock Price: ${averages['avg_price']:.2f}")
        print(f"  Average CDC Level: {averages['avg_level']:.1f}")
        if averages['avg_temp'] is not None:
            print(f"  Average Temperature: {averages['avg_temp']:.1f}°C")
    
    return {
        'level_prices': avg_prices,
        'temp_correlation': temp_correlation,
        'volatility': volatility,
        'monthly_averages': monthly_averages
    }

def main():
    """
    Main execution of the script:
    1. Sets up the database and tables.
    2. Processes up to 25 COVID, stock, and weather records each.
    3. Verifies the data counts in the database.
    4. Exports data to JSON file.
    5. Attempts to create visualizations from the exported JSON data.
    """
    conn = setup_database()
    try:
        # Process 25 records from each API
        covid_records = process_covid_data(conn)
        print(f"\nProcessed {covid_records} COVID records")

        stock_records = process_stock_data(conn)
        print(f"Processed {stock_records} stock records")

        weather_records = process_weather_data(conn)
        print(f"Processed {weather_records} weather records")

        # Verify current counts
        covid_count, stock_count, weather_count = verify_data_counts(conn)
        
        # If we have enough data, perform analysis
        if covid_count >= 25 and stock_count >= 25 and weather_count >= 25:
            print("\nPerforming statistical analysis...")
            statistics = analyze_data_statistics(conn)
        
        # Export to JSON
        json_data = export_json_data(conn)
        print("\nData has been exported to 'analysis_output.json'.")

        # Create visualizations if we have some complete data
        create_visualizations(json_data)

    except Exception as e:
        print(f"Error in main execution: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()

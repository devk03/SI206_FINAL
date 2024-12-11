# COVID-19, Weather, and Stock Market Analysis Tool

A data analysis tool that explores the relationships between COVID-19 transmission levels in California, the performance of the S&P 500 (SPY), and weather patterns in San Francisco.

## Features

- **Data Collection**:
  - Collects data from three APIs:
    - **COVID-19 Transmission Data**: [COVID Act Now API](https://covidactnow.org/data-api)
    - **Stock Market Data**: [Alpha Vantage API](https://www.alphavantage.co/)
    - **Weather Data**: [Open-Meteo API](https://open-meteo.com/)
- **Data Storage**:
  - Stores data in a structured SQLite database.
  - Tables are linked using foreign keys for efficient querying.
- **Processing**:
  - Processes 25 records per run, requiring 4 runs to collect a complete dataset (100 records per table).
- **Visualization**:
  - Creates three visualizations to explore relationships between datasets.
- **Export**:
  - Exports combined data to a JSON file for further analysis.

## Prerequisites

- **Python**: Version 3.10 or higher
- **pip**: Python package installer

## Installation (Using pip)

1. Clone this repository:

   ```bash
   git clone [repository-url]
   cd [repository-name]
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the main script:
   ```bash
   python main.py
   ```

## Usage

1. **Data Collection**:  
   The script must be run multiple times (4 runs in total) to collect all records. Each run performs the following:

   - Fetches and processes 25 new records for each table.
   - Updates the SQLite database.
   - Displays the current record counts in the terminal.

2. **Output Generation**:  
   Once all tables reach 100 records, the script automatically:
   - Generates `analysis_output.json` with the combined dataset.
   - Creates visualizations showing trends and relationships.

## Output Files

- **Combined Dataset**:
  - `analysis_output.json`: A JSON file containing data from all three sources.
- **Visualizations**:  
  Generated in the `visualizations/` directory:
  1. `visualization1_combined_metrics.png`: A time series plot of all metrics.
  2. `visualization2_bubble_plot.png`: A bubble plot of temperature vs. stock price, with bubble size representing COVID-19 transmission levels.
  3. `visualization3_enhanced_correlation.png`: A correlation heatmap of the three metrics.

## Data Sources

- **COVID-19 Data**: [COVID Act Now API](https://covidactnow.org/data-api)
- **Stock Market Data**: [Alpha Vantage API](https://www.alphavantage.co/)
- **Weather Data**: [Open-Meteo API](https://open-meteo.com/)

## Project Structure

```
.
├── main.py               # Main script for data collection and processing
├── covid_data.db         # SQLite database storing all collected data
├── visualizations/       # Directory containing generated visualization plots
│   ├── visualization1_combined_metrics.png
│   ├── visualization2_bubble_plot.png
│   └── visualization3_enhanced_correlation.png
├── analysis_output.json  # Final combined dataset in JSON format
├── requirements.txt      # Dependency file for pip
└── README.md             # Project documentation
```

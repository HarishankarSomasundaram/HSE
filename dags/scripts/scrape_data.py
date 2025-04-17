import requests
from datetime import datetime

def scrape_hse_data():
    today = datetime.today().strftime('%d/%m/%Y')
    url = f"https://uec.hse.ie/uec/TGAR.php?EDDATE={today}"
    response = requests.get(url)
    with open('/opt/airflow/data/raw_data.html', 'w', encoding='utf-8') as f:
        f.write(response.text)

import requests
from datetime import datetime,timedelta
import logging
from bs4 import BeautifulSoup
from pymongo import MongoClient
import pandas as pd

def scrape_hse_data(**kwargs):
    logger = logging.getLogger(__name__)
    # today = datetime.today().strftime('%d/%m/%Y')
    # url = f"https://uec.hse.ie/uec/TGAR.php?EDDATE={today}"
    # logger.info(f"Scraping HSE data from: {url}")
    #
    # try:
    #     resp = requests.get(url)
    #     logger.info(f"Status code: {resp.status_code}")
    #     if resp.status_code != 200:
    #         logger.error(f"Failed to retrieve data: {resp.text}")
    #         return []
    #
    #     soup = BeautifulSoup(resp.text, 'html.parser')
    #     table = soup.find('table')
    #     if not table:
    #         logger.warning("No table found in HTML.")
    #         return []
    #
    #     rows = []
    #     for tr in table.find_all('tr'):
    #         cols = [td.get_text(strip=True) for td in tr.find_all(['td','th'])]
    #         if cols:
    #             rows.append({ f"col_{i}": v for i, v in enumerate(cols) })
    #     logger.info(f"Extracted {len(rows)} rows")
    #
    #     # Push rows to XCom for downstream tasks
    #     return rows
    client = MongoClient('mongodb://mongodb:27017/')
    db = client['HSE']
    coll = db['trolleys']

    last_record = coll.find_one(sort=[("date", -1)])
    if last_record and 'date' in last_record:
        start_date = last_record['date'] + timedelta(days=1)
    else:
        start_date = datetime.today()
    rows = []
    today = datetime.today()
    try:
        for i in range(5):  # 180 days of data
            date = start_date + timedelta(days=i)
            date = date.replace(hour=8, minute=0, second=0, microsecond=0)
            # Extract day, month, and year
            day = date.strftime('%d')  # Day as string 'DD'
            month = date.strftime('%m')  # Month as string 'MM'
            year = date.strftime('%Y')  # Year as string 'YYYY'
            print(f"Day: {day}, Month: {month}, Year: {year}")

            url = "https://uec.hse.ie/uec/TGAR.php?EDDATE=" + day + "%2F" + month + "%2F" + year
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table')
            df = pd.read_html(str(table))[0]
            df_title = df.dropna(subset=[df.columns[0]])
            region = ''
            region_index = 0
            trolleys = []
            for index in range(0, df_title.shape[0]):
                if len(region) > 0 and region_index > 0 and 'Total' not in region:
                    df_temp = df.iloc[region_index:df_title[('Unnamed: 0_level_0', 'Unnamed: 0_level_1')].index[index]]
                    for i in range(0, df_temp.shape[0]):
                        hospital = df_temp.iloc[i]['Unnamed: 1_level_0']['Unnamed: 1_level_1']
                        if type(hospital) is not float:
                            if not df_temp.iloc[i]['Daily Trolley count'][
                                       'ED Trolleys'] == hospital and 'Total' not in hospital:
                                row = {'region': region, 'hospital': hospital, 'date': date,
                                       'ED Trolleys': 0 if pd.isna(
                                           df_temp.iloc[i]['Daily Trolley count']['ED Trolleys']) else int(
                                           df_temp.iloc[i]['Daily Trolley count']['ED Trolleys']),
                                       'Ward Trolleys': 0 if pd.isna(
                                           df_temp.iloc[i]['Daily Trolley count']['Ward Trolleys']) else int(
                                           df_temp.iloc[i]['Daily Trolley count']['Ward Trolleys']),
                                       'Total Trolleys': 0 if pd.isna(
                                           df_temp.iloc[i]['Daily Trolley count']['Total']) else int(
                                           df_temp.iloc[i]['Daily Trolley count']['Total']),
                                       'Surge Capacity in Use (Full report @14:00)': 0 if pd.isna(
                                           df_temp.iloc[i]['Surge Capacity in Use (Full report @14:00)'][0]) else int(
                                           df_temp.iloc[i]['Surge Capacity in Use (Full report @14:00)'][0]),
                                       'Delayed Transfers of Care (As of Midnight)': 0 if pd.isna(
                                           df_temp.iloc[i]['Delayed Transfers of Care (As of Midnight)'][0]) else int(
                                           df_temp.iloc[i]['Delayed Transfers of Care (As of Midnight)'][0]),
                                       'No of Total Waiting >24hrs': 0 if pd.isna(
                                           df_temp.iloc[i]['No of Total Waiting >24hrs'][0]) else int(
                                           df_temp.iloc[i]['No of Total Waiting >24hrs'][0]),
                                       'No of >75+yrs Waiting >24hrs': 0 if pd.isna(
                                           df_temp.iloc[i]['No of >75+yrs Waiting >24hrs'][0]) else int(
                                           df_temp.iloc[i]['No of >75+yrs Waiting >24hrs'][0])
                                       }
                                trolleys.append(row)
                region = df_title[('Unnamed: 0_level_0', 'Unnamed: 0_level_1')].iloc[index]
                region_index = df_title[('Unnamed: 0_level_0', 'Unnamed: 0_level_1')].index[index]
            rows.append(trolleys)
            # result = coll.insert_many(trolleys)
            # logger.info(f"Inserted {len(result.inserted_ids)} documents into MongoDB")
            # client.close()
        logger.info(f"Extracted {len(rows)} rows")
        return rows

    except Exception:
        logger.exception("Exception during scraping")
        return []

def store_hse_data_to_mongo(**kwargs):
    logger = logging.getLogger(__name__)
    ti = kwargs['ti']
    rows = ti.xcom_pull(task_ids='scrape_hse_data') or []
    if not rows:
        logger.warning("No rows to insert into MongoDB.")
        return

    client = MongoClient('mongodb://mongodb:27017/')
    db = client['HSE']
    coll = db['trolleys']
    for row in rows:
        result = coll.insert_many(row)
        logger.info(f"Inserted {len(result.inserted_ids)} documents into MongoDB")
    client.close()

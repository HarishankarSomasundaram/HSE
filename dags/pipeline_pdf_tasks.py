import pandas as pd
import camelot
from pymongo import MongoClient
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB configuration
MONGO_URI = "mongodb://localhost:27017"  # Update with your MongoDB URI
MONGO_DB = "health_services"
MONGO_COLLECTION = "employment_reports"


def process_pdf(**kwargs):
    month_config = kwargs['month_config']
    pdf_path = kwargs['pdf_path']
    processed_data = kwargs['processed_data']

    current_month = month_config['currentMonth']
    prev_month = month_config['prevMonth']
    file_name = month_config['fileName']
    page_nos = month_config['pagenos']
    category = month_config['category']

    try:
        # Read PDF
        file_path = os.path.join(pdf_path, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")

        logger.info(f"Processing {file_name} for {current_month}")
        tables = camelot.read_pdf(file_path, flavor='lattice', pages=page_nos)
        logger.info(f"Total tables extracted from {file_name}: {tables.n}")

        # Process tables
        df_list = []
        old_hospital_name = ''
        new_hospital_name = ''

        for i in range(len(tables)):
            name = ' '.join(tables[i].df.loc[0][0].split(' ')[1:]).strip()
            logger.info(f"Extracted hospital name: {name}")
            if name:
                if not old_hospital_name:
                    old_hospital_name = name
                    df_list.append(tables[i].df)
                else:
                    new_hospital_name = name

                if old_hospital_name and new_hospital_name and old_hospital_name != new_hospital_name and df_list:
                    df = concatenate_and_process(df_list)
                    items = process_dataframe(df, category, prev_month, current_month)
                    processed_data.extend(items)
                    old_hospital_name = new_hospital_name
                    df_list = [tables[i].df]
                elif old_hospital_name and new_hospital_name and old_hospital_name == new_hospital_name:
                    df_list.append(tables[i].df)

        # Process remaining tables
        if df_list:
            df = concatenate_and_process(df_list)
            items = process_dataframe(df, category, prev_month, current_month)
            processed_data.extend(items)

        logger.info(f"Completed processing {file_name}")

    except Exception as e:
        logger.error(f"Error processing {file_name}: {str(e)}")
        raise


def concatenate_and_process(df_list):
    df_list2 = []
    for index, df in enumerate(df_list):
        if index == 0:
            df_list2.append(df)
        else:
            df_list2.append(df.iloc[1:])
    return pd.concat(df_list2, ignore_index=True)


def process_dataframe(df, category, prev_month, current_month):
    df = df.fillna('0')
    items = []
    df = df.drop(df.columns[4], axis=1)  # Adjust as needed for specific months
    if len(df.columns) == 5:
        df.columns = [category, 'Dec 2019', 'Dec 2023', f'{prev_month} 2024', f'{current_month} 2024']
        hospital = ''
        titles = [
            'Medical & Dental', 'Nursing & Midwifery', 'Health & Social Care Professionals',
            'Management & Administrative', 'General Support', 'Patient & Client Care'
        ]
        title = titles[0]

        for index, row in df.iterrows():
            if index == 0:
                hospital = ' '.join(row[category].split(' ')[1:]).strip()
            else:
                if hospital:
                    if row[category] == title:
                        titles.pop(0)
                        title = titles[0] if titles else title
                        continue
                    else:
                        if titles:
                            for column in df.columns[1:]:
                                try:
                                    date = datetime.strptime(column, "%b %Y")
                                    value = row[column].replace(',', '') or '0'
                                    items.append({
                                        'name': row[category],
                                        'category': category,
                                        'hospital': hospital,
                                        'title': title,
                                        'date': date,
                                        'value': int(value)
                                    })
                                except ValueError as e:
                                    logger.error(f"Error parsing date {column}: {e}")
                                    continue
    return items


def store_hse_data_to_mongo(**kwargs):
    processed_data = kwargs['processed_data']
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        collection = db[MONGO_COLLECTION]

        if processed_data:
            collection.insert_many(processed_data)
            logger.info(f"Inserted {len(processed_data)} records into MongoDB")
        else:
            logger.warning("No data to insert into MongoDB")

        client.close()
    except Exception as e:
        logger.error(f"Error storing data to MongoDB: {str(e)}")
        raise
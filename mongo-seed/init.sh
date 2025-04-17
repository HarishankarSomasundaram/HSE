#!/bin/bash

set -e

echo "Creating unique index on region,hospital,date in HSE.trolleys..."
mongo --host localhost --port 27017 HSE --eval '
  db.trolleys.createIndex(
    { region: 1, hospital: 1, date: 1 },
    { unique: true, name: "region_1_hospital_1_date_1" }
  );
'

echo "Importing JSON data into MongoDB..."

mongoimport --host localhost --port 27017 --db HSE --collection trolleys --file /docker-entrypoint-initdb.d/trolleys.json --jsonArray

echo "MongoDB JSON import completed."

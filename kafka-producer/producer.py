from kafka import KafkaProducer
import json
import time

producer = KafkaProducer(
    bootstrap_servers='kafka:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

while True:
    message = {'hospital': 'UH Kerry', 'status': 'OK'}
    producer.send('hospital_status', message)
    print(f"Sent: {message}")
    time.sleep(5)

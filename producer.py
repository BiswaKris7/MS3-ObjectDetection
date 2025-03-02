import os
import csv
import json
import glob
from google.cloud import pubsub_v1

#### Set Google Cloud credentials
files=glob.glob("*.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=files[0];

## Constants
project_id = "abstract-stage-451701-j1"
topic_id = "pedestrian-detection"

# Publisher client
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(project_id, topic_id)

def publish_csv_to_topic(csv_file_path):
    with open(csv_file_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            message_data = json.dumps(row).encode("utf-8")  # Serialize dictionary
            try:
                future = publisher.publish(topic_path, message_data)
                print(f"Published message ID: {future.result()}")
            except Exception as e:
                print(f"Failed to publish message: {e}")

if __name__ == "__main__":
    csv_file_path = "Labels.csv"  # label file
    publish_csv_to_topic(csv_file_path)

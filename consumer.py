import os
import json
import glob
from google.cloud import pubsub_v1

### Set Google Cloud credentials
files=glob.glob("*.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=files[0];

## Constants
project_id = "abstract-stage-451701-j1"
subscription_id = "pedestrian-detection-sub"

##Subscriber client
subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(project_id, subscription_id)

def callback(message):
    try:
        # Deserialize the message
        record = json.loads(message.data.decode("utf-8"))
        print(f"Received record: {record}")
        message.ack()  # Acknowledge the message
    except Exception as e:
        print(f"Failed to process message: {e}")
        message.nack()  # Negative acknowledgment if there's an issue

if __name__ == "__main__":
    print("Listening for messages on:", subscription_path)
    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)

    try:
        streaming_pull_future.result()  # Block and wait for messages
    except KeyboardInterrupt:
        streaming_pull_future.cancel()  # Stop the subscription
        print("Subscription cancelled.")

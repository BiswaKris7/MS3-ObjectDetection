import json
import base64
from google.cloud import pubsub_v1

# Replace with your actual project ID and topic name
PROJECT_ID = "abstract-stage-451701-j1"
TOPIC_NAME = "pedestrian-data-topic"

def publish_message():
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, TOPIC_NAME)

    # Message payload (Pedestrian data)
    message_data = {
        "timestamp": 1711900000,
        "car1_location_x": 12.3,
        "car1_location_y": 45.6,
        "car1_location_z": 78.9,
        "car2_location_x": 23.4,
        "car2_location_y": 56.7,
        "car2_location_z": 89.0,
        "occluded_image_view": "occluded1.jpg",
        "occluding_car_view": "occluding1.jpg",
        "ground_truth_view": "groundtruth1.jpg",
        "pedestrian_x_top": 100,
        "pedestrian_y_top": 200,
        "pedestrian_x_bottom": 150,
        "pedestrian_y_bottom": 250
    }

    # Convert JSON to string and Base64 encode it
    json_str = json.dumps(message_data)
    encoded_message = base64.b64encode(json_str.encode("utf-8")).decode("utf-8")

    # Publish message
    future = publisher.publish(topic_path, data=encoded_message.encode("utf-8"))
    print(f"Message published: {future.result()}")

if __name__ == "__main__":
    publish_message()

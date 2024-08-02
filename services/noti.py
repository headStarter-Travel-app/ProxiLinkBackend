from exponent_server_sdk import PushClient
from exponent_server_sdk import PushMessage


def send_expo_push_notification(token, title, message, sound="default", extra=None):
    try:
        client = PushClient()
        response = client.publish(
            PushMessage(
                to=token,
                title=title,
                body=message,
                sound=sound,
                data=extra
            )
        )
        print(f"Push notification sent. Response: {response}")
        return response
    except Exception as e:
        print(f"Error sending push notification: {str(e)}")
        return None


# send_expo_push_notification(
#     "ExponentPushToken[-xVnYmDIY3-N4dUqYItxdh]", "Hello Manas", "I know what type of man you are")

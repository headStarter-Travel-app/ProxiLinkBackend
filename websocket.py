import socketio
from fastapi import FastAPI
import uvicorn

# Create a new FastAPI app
app = FastAPI()

# Create a new Socket.IO server
sio = socketio.AsyncServer(async_mode='asgi')
socket_app = socketio.ASGIApp(sio, app)

# Store connected users
connected_users = {}

@sio.event
async def connect(sid, environ):
    print(f"User connected: {sid}")
    connected_users[sid] = None

@sio.event
async def join(sid, data):
    user_id = data['userId']
    connected_users[sid] = user_id
    sio.enter_room(sid, user_id)
    print(f"User {sid} joined room {user_id}")

@sio.event
async def disconnect(sid):
    print(f"User disconnected: {sid}")
    user_id = connected_users.get(sid)
    if user_id:
        sio.leave_room(sid, user_id)
    connected_users.pop(sid, None)

def send_friend_request_notification(user_id):
    sio.emit('friendRequest', {'message': 'You have a new friend request'}, room=user_id)

# This function can be called whenever a new friend request is made
def new_friend_request(user_id):
    send_friend_request_notification(user_id)

if __name__ == "__main__":
    uvicorn.run(socket_app, host="0.0.0.0", port=8000)
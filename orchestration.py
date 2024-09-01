# Orchestration Server Code
"""
This code implements an orchestration server that manages the execution of machine learning workflows.
It provides health check endpoints, handles websocket connections for remote command execution,
and integrates with a prompt server for executing machine learning tasks.

Key components include:
- Health check and readiness endpoints
- Websocket communication for remote command execution
- Integration with a prompt server for task execution
- Error handling and reconnection logic for websocket communication
"""

import json
import os
import server
import traceback
# Check for availability of workflow checkpointing
import aiohttp
import asyncio
from .workflowcheckpointing import post_prompt_remote

# Static authentication token for secure communication
STATIC_AUTH_TOKEN = os.environ.get("STATIC_AUTH_TOKEN", None)

web = server.web
ps = server.PromptServer.instance

finished_startup = False

# Store the original server start method
original_server_start = ps.start

async def server_start(address, port, verbose=True, call_on_start=None):
    """
    Custom server start method that sets the finished_startup flag when the server is ready.
    """
    original_on_start = call_on_start
    def on_start(*args, **kwargs):
        if original_on_start is not None:
            original_on_start(*args, **kwargs)
        global finished_startup
        finished_startup = True
    return await original_server_start(address, port, verbose, on_start)

# Replace the original server start method with our custom one
ps.start = server_start

@ps.routes.get("/health")
async def heath(request):
    """
    Health check endpoint that returns an empty JSON response.
    """
    # While any of the server endpoints could likely be used
    return web.json_response([])

@ps.routes.get("/startup")
async def startup(request):
    """
    Startup check endpoint that returns 503 if the server hasn't finished starting up.
    """
    if finished_startup:
        return web.json_response([])
    return web.Response(status=503)

@ps.routes.get("/ready")
async def ready(request):
    """
    Readiness check endpoint that returns 503 if there are items in the prompt queue.
    """
    current_queue = ps.prompt_queue.get_current_queue()
    if len(current_queue[0]) == 0 and len(current_queue[1]) == 0:
        return web.json_response(current_queue)
    return web.json_response(current_queue, status=503)

async def websocket_loop():
    """
    Main websocket loop for handling remote commands.
    """
    async with aiohttp.ClientSession() as session:
        if 'ORCHESTRATION_SERVER' not in os.environ:
            while True:
                await asyncio.sleep(60)
        if STATIC_AUTH_TOKEN:
            headers = {"Authorization": f"Bearer {STATIC_AUTH_TOKEN}"}
        else:
            headers = None
        async with session.ws_connect(os.environ["ORCHESTRATION_SERVER"],
                                      headers=headers) as ws:
            print("connected to server")
            async for msg in ws:
                try:
                    print("got command: " + str(msg))
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        js = msg.json()
                        resp = {"message_id": js.get('message_id', 0)}
                        match js['command']:
                            case 'prompt':
                                # Wrap as mock request
                                class MockRequest:
                                    async def json(self):
                                        return js['data']
                                out = await post_prompt_remote(MockRequest())
                                resp['data'] = json.loads(out.body._value)
                            case "queue":
                                resp['data'] = ps.prompt_queue.get_current_queue()
                            case "files":
                                resp['data'] = [f.name for f in os.scandir('fetches') if f.is_file()]
                            case "info":
                                resp['data'] = {}
                                if 'SALAD_MACHINE_ID' in os.environ:
                                    resp['data']['machine_id'] = os.environ['SALAD_MACHINE_ID']
                                else:
                                    resp['data']['machine_id'] = os.environ.get('HOSTNAME', 'local')
                            case "logs":
                                with open('comfyui.log', 'r') as f:
                                    resp['data'] = f.read()
                            case _:
                                resp = {"error": "Unknown command"}
                        print(resp)
                        await ws.send_json(resp)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        await ws.send_json({"error": "Received bad message"})
                except Exception as e:
                    # NOTE: this will reraise if error was socket closing
                    await ws.send_json({"error": str(e)})

async def try_websocket():
    """
    Continuously attempt to maintain a websocket connection, with error handling and reconnection logic.
    """
    while True:
        try:
            await websocket_loop()
        except aiohttp.client_exceptions.ClientConnectorError:
            print("disconnected")
        except:
            print(traceback.format_exc())
        await asyncio.sleep(5)
        print("Attempting re connection")

# Create a task for the websocket connection loop
process_loop = ps.loop.create_task(try_websocket())
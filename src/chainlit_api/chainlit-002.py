import chainlit as cl


@cl.on_message
async def main(message: str):
    # Send a response back to the user
    await cl.Message(
        content=f"Received: {message}",
    ).send()

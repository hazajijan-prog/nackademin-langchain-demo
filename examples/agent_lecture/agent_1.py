# Enkel anget utan verktyg använder endast språkmodellen. 

from langchain.agents import create_agent

from util.models import get_model
from util.streaming_utils import STREAM_MODES, handle_stream
from util.pretty_print import get_user_input

def run():
    # Get predefined attributes
    model = get_model(temperature=0.1, top_p=0.5)
    
    tools = []
    # Create agent
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=(
                "Du är en hjälpsam assistent som svarar på användarens frågor."
                "Svara alltid på svenska och var koncis men informativ."
        ),
    )
    
    while True: 
        # Get user input
        user_input = get_user_input("Ställ din fråga")
        
        if not user_input:
            continue
        if user_input in {"exit","quit"}:
            break
        
        # lägg till användarens meddelande 
        messages = [{"role": "user", "content": user_input}]

        # Call the agent
        process_stream = agent.stream(
        {"messages": messages},
        stream_mode=STREAM_MODES,
    )

     # Stream the process
        handle_stream(process_stream)


if __name__ == "__main__":
    run()

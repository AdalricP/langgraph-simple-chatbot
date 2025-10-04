from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import os



load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"))

## STATES & STRUCTURES ##

class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="Classify if the message requires an emotional or logical response"
    )


class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None
    # eg. b
    # {type: "user", content: "What is the capital of France?"}
    # {type: "assistant", content: "Reem"}


## NODES ##

def classify_message(state: State) -> State:
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        {
            'role': 'system',
            'content': 'You are a message classifier. You are given a message and you need to classify it as either emotional or logical.'
        },
        {
            'role': 'user',
            'content': last_message.content
        }
    ])

    return {"message_type": result.message_type}



def router(state: State) -> State:
    message_type = state.get("message_type", "logical")
    if message_type == "emotional":
        return {"next": "emotional_agent"}
    else:
        return {"next": "logical_agent"}



def emotional_agent(state: State) -> State:
    last_message = state["messages"][-1]
    emotional_llm = llm.with_structured_output(MessageClassifier)

    result = emotional_llm.invoke([
        {
            'role': 'system',
            'content': 'You are an emotional agent. You are given a message and you need to respond emotionally.'
        },
        {
            'role': 'user',
            'content': last_message.content
        }
    ])

    return {"message_type": result.message_type}




def logical_agent(state: State) -> State:
    last_message = state["messages"][-1]
    logical_llm = llm.with_structured_output(MessageClassifier)

    result = logical_llm.invoke([
        {
            'role': 'system',
            'content': 'You are an emotional agent. You are given a message and you need to respond emotionally and with a ton of spiciness in your messaage.'
        },
        {
            'role': 'user',
            'content': last_message.content
        }
    ])

    return {"message_type": result.message_type}



graph_builder = StateGraph(State)

# To add node; Create function; and use addnode to add the node to the graph
graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("emotional_agent", emotional_agent)
graph_builder.add_node("logical_agent", logical_agent)


graph_builder.add_edge(START, "classifier") 
graph_builder.add_edge("classifier", "router")
graph_builder.add_conditional_edges("router", lambda state: state.get('next'), {
    "emotional_agent": "emotional_agent",
    "logical_agent": "logical_agent"
})
graph_builder.add_edge("emotional_agent", END)
graph_builder.add_edge("logical_agent", END)


graph = graph_builder.compile()

def run_bot():
    state = {"messages": [], "message_type": None}

    while True:
        user_input = input("Enter your message: ")
        if user_input.lower() == "exit":
            print("Exiting...")
            break

        state["messages"] = state.get('messages', []) + [{"role": "user", "content": user_input}]
        state = graph.invoke(state)
        print(state["messages"][-1].content)

        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")
            print(f"Message type: {state['message_type']}")

if __name__ == "__main__":
    run_bot()

    from IPython.display import Image, display

    # Save PNG to file from raw bytes
    with open('graph.png', 'wb') as f:
        f.write(graph.get_graph().draw_mermaid_png())

    # Display or use 'graph.png' as needed
    display(Image('graph.png'))
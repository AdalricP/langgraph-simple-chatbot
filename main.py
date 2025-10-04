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




class State(TypedDict):
    messages: Annotated[list, add_messages]
    # eg. b
    # {type: "user", content: "What is the capital of France?"}
    # {type: "assistant", content: "Reem"}



graph_builder = StateGraph(State)


def chatbot(state: State) -> State:
    return {"messages": [llm.invoke(state["messages"])]}

# To add node; Create function; and use addnode to add the node to the graph
graph_builder.add_node("chatbot", chatbot)


graph_builder.add_edge(START, "chatbot") # obv always need a start and end node #this is for start-->chatbot
graph_builder.add_edge("chatbot", END) #this is for chatbot-->end

graph = graph_builder.compile()

user_input = input("Enter your message: ")
state = graph.invoke({"messages": [{"type": "user", "content": user_input}]})

print(state["messages"][-1].content)


from IPython.display import Image, display

# Save PNG to file from raw bytes
with open('graph.png', 'wb') as f:
    f.write(graph.get_graph().draw_mermaid_png())

# Display or use 'graph.png' as needed
display(Image('graph.png'))
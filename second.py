from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage,AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv

#chatbot with memory
load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage,AIMessage]]


llm=ChatGroq(model_name="llama-3.3-70b-versatile")
load_dotenv()

def process(state:AgentState)->AgentState:
    """this node will solve the request you input"""
    response=llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}\n")

    print("CURRENT STATE: ", state["messages"])
    return state


graph=StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)
agent=graph.compile()

conversation_history=[]
user_input=input("User: ")
while user_input!="exit":
    conversation_history.append(HumanMessage(content=user_input))
    result=agent.invoke({"messages":conversation_history})
    conversation_history=result["messages"]
    user_input=input("User: ")


with open("conversation_history.txt", "w") as f:
    f.write("Conversation History:\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            f.write(f"User: {message.content}\n")
        elif isinstance(message, AIMessage):
            f.write(f"AI: {message.content}\n")
    f.write("\nEnd of Conversation\n")


print("Conversation history saved to conversation_history.txt")
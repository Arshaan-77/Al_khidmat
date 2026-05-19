from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

load_dotenv()

class Schema(BaseModel):
    name: Optional[str]
    phone: Optional[str]
    location: Optional[str] = None
    family_members: Optional[int] = Field(..., description="should be the count of family members if in string convert it into integer")
    need: Optional[str] 
    need_cost: Optional[int] = Field(..., description="should be the cost in Rupees if in string or other currency convert it into integer or INR")

parser = PydanticOutputParser(pydantic_object=Schema)

class LangSchema(BaseModel):
    language: Optional[str] = Field(..., description="Detected language of the user conversation")

parser2 = StrOutputParser()

model = ChatGroq(model="llama-3.1-8b-instant")

class BotState(TypedDict):
    messages: Annotated[list, add_messages]
    extracted_dict: dict
    language: str 

graph = StateGraph(BotState)
 

def language_detection_node(state: BotState):
    sys_prompt = """
You are an accurate language detection system.

Your task is to detect the primary language/style used in the user's message.

Possible Outputs:
- English
- Hindi
- Urdu
- Roman English
- Roman Hindi
- Roman Urdu
- Mixed

Definitions:
- English:
  English written in English script with English understanding.

- Hindi:
  Hindi written in Devanagari script.

- Urdu:
  Urdu written in Urdu script.

- Roman English:
  English written in English script with English understanding only.

- Roman Hindi:
  Hindi language written using English alphabets.
  Example:
  "Mera naam Arshaan hai"

- Roman Urdu:
  Urdu language written using English alphabets.
  Example:
  "Mujhe madad chahiye"

- Mixed:
  Multiple languages/scripts significantly mixed together.

Rules:
- Detect based on meaning and writing style, not only script.
- Understand mixed-language sentences carefully.
- Return ONLY one label from the possible outputs.
- Do NOT explain.
- Do NOT add extra text.
- Output must always be in English.
"""
    prompt = ChatPromptTemplate.from_messages([SystemMessage(content=sys_prompt), ("human", "{user_message}")])
    chain = prompt|model|parser2
    print("-"*140)
    print(state["messages"][-1].content)
    print("-"*140)
    response = chain.invoke({"user_message":state["messages"][-1].content})
    return {"language":response}

def input_processing_node(state: BotState):
    user_message = state["messages"][-1].content
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
    You are an extraction system.
    Extract the required fields from the user message.

Required Fields:
- name
- phone
- location
- family members
- need
- need cost

Instructions
- Understand English, Hindi, Urdu, Roman Hindi, Roman Urdu, and Roman English input.
- Convert all extracted values into English before storing.
- Convert Hindi/Urdu/Roman text meaningfully into proper English.
- Do NOT ask questions.
- Do NOT repeat information.
- Keep the response professional and concise.
- Do NOT explain anything
- Do NOT add extra text
- Missing fields should be null

"""),
("human", "User Message: {user_message} \n{format_instructor}")
    ])
    chain = prompt|model|parser
    response = chain.invoke({"user_message":user_message, "format_instructor":parser.get_format_instructions()})

    return{"messages":[AIMessage(content=response.model_dump_json(indent=2))], "extracted_dict":response.model_dump()}

graph.add_node("language_detection_node", language_detection_node)
graph.add_node("input_processing_node", input_processing_node)
# graph.add_node("need_detection_node", need_detection_node)
# graph.add_node("missing_field_detection_node", missing_field_detection_node)
# graph.add_node("dynamic_question_generation_node", dynamic_question_generation_node)
# graph.add_node("validation_node", validation_node)
# graph.add_node("state_update_node", state_update_node)
# graph.add_node("completion_checker_node", completion_checker_node)
# graph.add_node("database_save", database_save)

# graph.add_edge(START, "user_input")
graph.add_edge(START, "language_detection_node")
graph.add_edge("language_detection_node", "input_processing_node")
graph.add_edge("input_processing_node", END)

checkpointer = InMemorySaver()

chatbot = graph.compile(checkpointer=checkpointer)

CONFIG ={"configurable": {
    "thread_id":"thread-1"
}}

initial_state = {"messages":[HumanMessage(content=input("Type Here: "))]}

response = chatbot.invoke(initial_state, config=CONFIG)
print(response["messages"])
print(response["language"])
print(response["messages"][-1].content)
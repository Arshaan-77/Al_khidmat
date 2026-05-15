from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory  import InMemorySaver
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from typing import TypedDict, Annotated, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from database import save_beneficiaries
from dotenv import load_dotenv

load_dotenv()

class Schema(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    need: Optional[str] = None

    location: Optional[str] = None
    family_members: Optional[int] = Field(..., description="should be the count of family members if in string convert it into integer")

parser = PydanticOutputParser(pydantic_object=Schema)
parser2 = StrOutputParser()

llm = ChatGroq(model="llama-3.1-8b-instant")

class KhidmatState(TypedDict):
    messages: Annotated[list, add_messages]
    extracted_data: dict
graph = StateGraph(KhidmatState)


def user_input(state: KhidmatState):
    user_message = state["messages"][-1].content
    prompt = PromptTemplate(
        template="""
You are an extraction system.

Extract the required fields from the user message.

Required Fields:
- name
- phone
- location
- family members
- need

Instructions
- Do NOT ask questions.
- Do NOT repeat information.
- Keep the response professional and concise.
- Do NOT explain anything
- Do NOT add extra text
- Missing fields should be null 

User Message:
{user_message}

{format_instructor}
""",
        input_variables=["user_message"],
        partial_variables={"format_instructor":parser.get_format_instructions()}
    )
    chain = prompt|llm|parser
    response = chain.invoke({"user_message": user_message})
    
    new_data = response.model_dump()

    # Previous stored data
    old_data = state.get("extracted_data", {})

    # Merge old + new
    merged_data = {}

    for key in ["name", "phone", "location", "family_members", "need"]:

        if new_data.get(key) is not None:
            merged_data[key] = new_data.get(key)

        else:
            merged_data[key] = old_data.get(key)

    return {
        "extracted_data": merged_data
    }

def check_fields(state: KhidmatState):

    data = state["extracted_data"]

    mandatory_fields = ["name", "phone", "need"]

    missing_fields = []

    for field in mandatory_fields:
        if not data.get(field):
            missing_fields.append(field)

    if missing_fields:
        return "missing"

    return "complete"

def ask_missing(state: KhidmatState):

    data = state["extracted_data"]

    missing_fields = []

    for field in ["name", "phone", "need"]:
        if not data.get(field):
            missing_fields.append(field)
    prompt = ChatPromptTemplate([
("human", """These fields are missing: {missing_fields}

Ask the user to provide them.

Instructions:
- Ask ONLY for missing fields.
- Do NOT repeat already provided information.
- Do NOT use Markdown formatting.
- Ask naturally like a human assistant.
- Keep the response short and professional.
- If all mandatory fields are present, reply only with:
"All required information has been collected."  
""")])
    
    chain = prompt|llm|parser2
    response = chain.invoke({"missing_fields":missing_fields})
    return {"messages":[AIMessage(content=response)]}


def show_info(state: KhidmatState):
    data = state["extracted_data"]
    response = f"""
# Registration Completed

## Beneficiary Details

- Name: {data.get("name")}
- Phone: {data.get("phone")}
- Location: {data.get("location")}
- Family Members: {data.get("family_members")}
- Need: {data.get("need")}

Please click the Save button to store this information.
"""

    return {
        "messages": [AIMessage(content=response)]
    }


graph.add_node("user_input", user_input)
graph.add_node("ask_missing", ask_missing)
graph.add_node("show_info", show_info)


graph.add_edge(START, "user_input")
graph.add_conditional_edges("user_input", check_fields,{"missing": "ask_missing","complete": "show_info"})
graph.add_edge("ask_missing", END)
graph.add_edge("show_info", END)

checkpointer = InMemorySaver()
workflow = graph.compile(checkpointer=checkpointer)
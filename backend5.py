from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing import TypedDict, Literal, Optional, Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# ==============
# Importing llm
# ==============
llm = ChatGroq(model="llama-3.1-8b-instant")

# ===============================
# importing string output parser
# ===============================
parser2 = StrOutputParser()

# ===================================
# creating pydantic structure output
# ===================================
class Schema(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    need: Optional[str] = None

    location: Optional[str] = None
    family_members: Optional[int] = Field(default=None, description="Count of family members")
    need_cost: Optional[int] = Field(default=None, description="Cost should be INR and numeric value")

parser = PydanticOutputParser(pydantic_object=Schema)

# =============================
# medical schema
# =============================
class MedicalSchema(BaseModel):
    disease: Optional[str] = Field(default=None, description="Name of illness/disease in English")
    hospital: Optional[str] = Field(default=None, description="Hospital name in English")
    urgency: Optional[str] = Field(default=None, description="'high', 'medium', or 'low'")
    need_cost: Optional[int] = Field(default=None, description="Estimated treatment cost in INR (integer only)")

class EducationSchema(BaseModel):
    student_class: Optional[str] = Field(default=None, description="Class/grade e.g. '10th', 'B.Tech 2nd year'")
    institute: Optional[str] = Field(default=None, description="School or college name in English")
    academic_status: Optional[str] = Field(default=None, description="e.g. 'passed', 'failed', 'appearing'")

class FinancialSchema(BaseModel):
    monthly_income: Optional[int] = Field(default=None, description="Monthly household income in INR (integer only)")
    employment_status: Optional[str] = Field(default=None, description="e.g. 'unemployed', 'self-employed', 'daily wage'")
    earning_members: Optional[int] = Field(default=None, description="Number of earning members (integer only)")

medical_parser = PydanticOutputParser(pydantic_object=MedicalSchema)
education_parser = PydanticOutputParser(pydantic_object=EducationSchema)
financial_parser = PydanticOutputParser(pydantic_object=FinancialSchema)    

# ===============
# creating state
# ===============
class BotState(TypedDict):
    messages: Annotated[list, add_messages]
    extracted_dict: dict
    language: str

    need_category: Optional[str] = Field(
        default=None,
        description="One of: 'medical', 'education', 'financial'. Set only if user explicitly names a category."
    )
    medical: Optional[MedicalSchema] = Field(default=None)
    education: Optional[EducationSchema] = Field(default=None)
    financial: Optional[FinancialSchema] = Field(default=None)

# =======================
# Creating Language node
# =======================
def language_detection_node(state: BotState):
    user_message = state["messages"][-1].content
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
You are a language detection assistant.
                      
Detect the language of the user's latest message.

Supported languages:
- english
- hindi
- roman_hindi
- roman_urdu

Rules:
1. If text is written in English script and is proper English:
   -> english

2. If text is written in Devanagari script:
   -> hindi

3. If Hindi is written using English letters:
   Example:
   "mera naam arshaan hai"
   -> roman_hindi

4. If Urdu/Hinglish written using English letters:
   Example:
   "aap kaise ho"
   -> roman_urdu

5. Return ONLY one label.

"""), ("human", "User Message:{user_message}")
    ])
    chain = prompt|llm|parser2
    response = chain.invoke({"user_message":user_message})

    return {"language":response.strip().lower()}

# ===============================
# creating input processing node
# ===============================
def input_processing_node(state: BotState):
    user_message=state["messages"][-1].content
    prompt = PromptTemplate(
        template="""
You are an extraction system.

The user language is:
{language}

Extract the required fields from the user message.

Required Fields:
- name
- phone
- location
- family members
- need
- need cost

Instructions
- Do NOT ask questions.
- Do NOT repeat information.
- Keep the response professional and concise.
- Do NOT explain anything
- Do NOT add extra text
- Missing fields should be null 
- Store information in English


User Message:
{user_message}

{format_instructor}
""",
        input_variables=["user_message", "language"],
        partial_variables={"format_instructor":parser.get_format_instructions()}
    )
    chain = prompt|llm|parser
    response = chain.invoke({"user_message": user_message, "language":state["language"]})
    
    new_data = response.model_dump()

    # Previous stored data
    old_data = state.get("extracted_dict", {})

    # Merge old + new
    merged_data = {}

    for key in ["name", "phone", "location", "family_members", "need", "need_cost"]:

        if new_data.get(key) is not None:
            merged_data[key] = new_data.get(key)

        else:
            merged_data[key] = old_data.get(key)

    return {
        "extracted_dict": merged_data
    }

def need_detection_node(state: BotState):
    user_message = state["messages"][-1].content
    
    prompt = PromptTemplate(
        template="Ask if the need is medical, educational or financial. existing_data:{existing_data}\n use this language:{language}\n user message:{user_message}",
        input_variables=["existing_data", "user_message", "language"]
    
    )
    chain = prompt|llm|parser2
    response = chain.invoke({"existing_data":state["messages"], "user_message":user_message, "language":state["language"]})
    return {"messages":[AIMessage(content=response)]}

def check_need_fields(state: BotState):
    if state["need"] == "medical":
        return "medical"
    if state["need"] == "education":
        return "education"
    if state["need"] == "financial":
        return "financial"
    

    
def check_fields(state: BotState):

    data = state["extracted_dict"]

    mandatory_fields = ["name", "phone", "need", "family_members", "location", "need_cost"]

    missing_fields = []

    for field in mandatory_fields:
        if not data.get(field):
            missing_fields.append(field)

    if missing_fields:
        return "missing"

    return "complete"

def ask_missing(state: BotState):

    data = state["extracted_dict"]

    missing_fields = []

    for field in ["name", "phone", "need", "family_members", "location", "need_cost"]:
        if not data.get(field):
            missing_fields.append(field)
    prompt = ChatPromptTemplate.from_messages([
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

def show_info(state: BotState):
    data = state["extracted_dict"]
    response = f"""
# Registration Completed

## Beneficiary Details

- Name: {data.get("name")}
- Phone: {data.get("phone")}
- Location: {data.get("location")}
- Family Members: {data.get("family_members")}
- Need: {data.get("need")}
- medical: {data.get("medical")}
Please click the Save button to store this information.
"""

    return {
        "messages": [AIMessage(content=response)]
    }

graph = StateGraph(BotState)

# creating nodes
graph.add_node("language_node", language_detection_node)
graph.add_node("input_node", input_processing_node)
graph.add_node("need_node", need_detection_node)
graph.add_node("ask_missing", ask_missing)
graph.add_node("show_info", show_info)

# creating edges
graph.add_edge(START, "language_node")
graph.add_edge("language_node", "input_node")
graph.add_edge("input_node", "need_node")
graph.add_conditional_edges("need_node", check_fields,{"missing": "ask_missing","complete": "show_info"})
graph.add_edge("ask_missing", END)
graph.add_edge("show_info", END)

checkpointer = InMemorySaver()
chatbot = graph.compile(checkpointer=checkpointer)

CONFIG = {"configurable": {"thread_id":"thread-1"}}

while True:
    user_input = input("Type Here: ")

    response = chatbot.invoke({"messages":[HumanMessage(content=user_input)]}, config=CONFIG)

    print("BOT:",response["messages"][-1].content)


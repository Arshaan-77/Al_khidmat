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
    phone: Optional[str] = Field(default=None,description="Beneficiary phone number. Extract only a valid mobile number containing digits. Remove spaces, hyphens, country codes, and extra text.")
    need: Optional[str] = None
 
    location: Optional[str] = None
    family_members: Optional[int] = Field(default=None, description="Count of family members in numbers")
    need_cost: Optional[int] = Field(default=None, description="Cost should be INR and  extract numeric value")

parser = PydanticOutputParser(pydantic_object=Schema)

# =============================
# medical schema
# =============================
class MedicalSchema(BaseModel):
    disease: Optional[str] = Field(default=None, description="Name of illness/disease in English")
    hospital: Optional[str] = Field(default=None, description="Hospital name in English")
    urgency: Optional[Literal["high", "medium", "low"]] = Field(default=None, description="'high', 'medium', or 'low'")


class EducationSchema(BaseModel):
    student_class: Optional[str] = Field(default=None,description="Educational class or academic year of the student. Examples: 'Class 8', '10th', '12th', 'B.Tech 3rd Year', 'B.Com 1st Year', 'MCA Semester 2'. Do not extract classroom names, coaching classes, or institute names.")
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

    medical: dict
    education: dict
    financial: dict

    current_flow: str

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

# ================
# router
# ================
def resume_flow(state: BotState):
    flow = state.get("current_flow", "")

    if flow == "medical":
        medical = state.get("medical", {})
        if not all(medical.get(f) for f in ["disease", "hospital", "urgency"]):
            return "medical_node"
        # ✅ Medical complete — go straight to input/basic check
        return "input_node"

    elif flow == "education":
        education = state.get("education", {})
        if not all(education.get(f) for f in ["student_class", "institute", "academic_status"]):
            return "education_node"
        return "input_node"

    elif flow == "financial":
        financial = state.get("financial", {})
        if not all(financial.get(f) for f in ["monthly_income", "employment_status", "earning_members"]):
            return "financial_node"
        return "input_node"

    return "input_node"

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

    # At the end of input_processing_node, preserve existing flow if new need is empty
    raw_need = merged_data.get("need")
    new_flow = str(raw_need).lower() if raw_need else ""
    return {
        "extracted_dict": merged_data,
        "current_flow": new_flow if new_flow else state.get("current_flow", "")
    }

def check_need_type(state):
    raw_need = state.get("extracted_dict", {}).get("need")
    need = str(raw_need).lower().strip() if raw_need else ""

    # Always fall back to current_flow if need is missing
    if not need:
        need = state.get("current_flow", "").lower().strip()

    if not need:
        return "ask_need"

    if "medical" in need:
        return "medical"
    if "education" in need:
        return "education"
    if "financial" in need:
        return "financial"

    return "ask_need"


def ask_need(state):
    return {"messages":[AIMessage(content=" What type of assistance do you need? Please Write From the Given: Medical, Education or Financial")]}

# ==================
# medical node
# ==================
def medical_node(state: BotState):
    medical = state.get("medical", {})

    if all(
        medical.get(field)
        for field in ["disease", "hospital", "urgency"]
    ):
        return {"medical": medical}

    user_message = state["messages"][-1].content

    prompt = PromptTemplate(
        template="""
You are a medical information extraction system.

Extract ONLY the following fields from the user message:

- disease
- hospital
- urgency

Rules:
- Return data in structured format.
- Extract ONLY disease, hospital, urgency.
- Do NOT infer values.
- Do NOT map city names, locations, addresses, phone numbers,
  names, family members, or costs to medical fields.
- If the message contains location information such as
  "Lucknow", "Bhopal", "Delhi", return null for all fields.
- Hospitals must explicitly contain words such as:
  hospital, clinic, medical college, healthcare center.
- Store values in English.
- If a field is missing, return null.
- urgency should be one of:
  high
  medium
  low

User Message:
{user_message}

{format_instructions}
""",
        input_variables=["user_message"],
        partial_variables={
            "format_instructions": medical_parser.get_format_instructions()
        }
    )

    chain = prompt | llm | medical_parser

    response = chain.invoke(
        {
            "user_message": user_message
        }
    )

    new_data = response.model_dump()

    old_data = state.get("medical", {})

    merged_data = {}

    for key in [
    "disease",
    "hospital",
    "urgency"
]:
        if new_data.get(key) is not None:
            merged_data[key] = new_data.get(key)

        else:
            merged_data[key] = old_data.get(key)

    return {
        "medical": merged_data
    }


def ask_medical_fields(state: BotState):

    medical = state.get("medical", {})
    missing_fields=[]
    required = [
        "disease",
        "hospital",
        "urgency"
    ]

    for field in required:
        if not medical.get(field):
            missing_fields.append(field)
    prompt = ChatPromptTemplate.from_messages([
("human", """This field is missing: {missing_fields}

Ask the user to provide it.

Instructions:
- Ask ONLY for missing field.
- Do NOT repeat already provided information.
- Keep the response short and professional.
- Extract only what is explicitly mentioned in the current message.
- Do not guess or infer values.
- City names, locations, addresses, or states are NOT hospitals.


- urgency should be one of:
  high
  medium
  low
 
 Examples:
"bhopal" → hospital=null
"abm hospital" → hospital="ABM Hospital"
"corona" → disease="Corona"
"high" → urgency="high"
""")])
    
    chain = prompt|llm|parser2
    response = chain.invoke({"missing_fields":missing_fields})
    return {"messages":[AIMessage(content=response)]}

def check_medical_fields(state):
    medical = state.get("medical", {})
    missing_fields=[]
    required = [
        "disease",
        "hospital",
        "urgency"
    ]

    for field in required:
        if not medical.get(field):
            missing_fields.append(field)

    if missing_fields:
        return "missing"

    return "complete"



# ===================
# education node
# ===================
def education_node(state: BotState):

    user_message = state["messages"][-1].content

    prompt = PromptTemplate(
        template="""
You are an education information extraction system.

Extract ONLY the following fields from the user message:

- student_class
- institute
- academic_status

Rules:
- Return data in structured format.
- Store values in English.
- If a field is missing, return null.
- Do not guess values.
- academic_status should be values such as:
  passed
  failed
  appearing
  dropped
  studying.

User Message:
{user_message}

{format_instructions}
""",
        input_variables=["user_message"],
        partial_variables={
            "format_instructions": education_parser.get_format_instructions()
        }
    )

    chain = prompt | llm | education_parser

    response = chain.invoke(
        {
            "user_message": user_message
        }
    )

    new_data = response.model_dump()

    old_data = state.get("education", {})

    merged_data = {}

    for key in [
        "student_class",
        "institute",
        "academic_status"
    ]:

        if new_data.get(key) is not None:
            merged_data[key] = new_data.get(key)

        else:
            merged_data[key] = old_data.get(key)

    return {
        "education": merged_data
    }

def ask_education_fields(state: BotState):

    education = state.get("education", {})
    missing_fields=[]
    required = [
        "student_class",
        "institute",
        "academic_status"
    ]

    for field in required:
        if not education.get(field):
            missing_fields.append(field)
    prompt = ChatPromptTemplate.from_messages([
("human", """This field is missing: {missing_fields}

Ask the user to provide it.

Instructions:
- Ask ONLY for missing field.
- Do NOT repeat already provided information.
- Keep the response short and professional. 
 - academic_status should be values such as:
  passed
  failed
  appearing
  dropped
  studying.
""")])
    
    chain = prompt|llm|parser2
    response = chain.invoke({"missing_fields":missing_fields})
    return {"messages":[AIMessage(content=response)]}

def check_education_fields(state):
    education = state.get("education", {})
    missing_fields=[]
    required = [
        "student_class",
        "institute",
        "academic_status"
    ]

    for field in required:
        if not education.get(field):
            missing_fields.append(field)

    if missing_fields:
        return "missing"

    return "complete"


# ====================
# financial node
# ====================
def financial_node(state: BotState):

    user_message = state["messages"][-1].content

    prompt = PromptTemplate(
        template="""
You are a financial information extraction system.

Extract ONLY the following fields from the user message:

- monthly_income
- employment_status
- earning_members

Rules:
- Return data in structured format.
- Store values in English.
- If a field is missing, return null.
- Do not guess values.
- monthly_income must be an integer INR value.
- earning_members must be an integer.
- employment_status examples:
  unemployed
  employed
  self-employed
  daily wage worker
  retired
  student.

User Message:
{user_message}

{format_instructions}
""",
        input_variables=["user_message"],
        partial_variables={
            "format_instructions": financial_parser.get_format_instructions()
        }
    )

    chain = prompt | llm | financial_parser

    response = chain.invoke(
        {
            "user_message": user_message
        }
    )

    new_data = response.model_dump()

    old_data = state.get("financial", {})

    merged_data = {}

    for key in [
        "monthly_income",
        "employment_status",
        "earning_members"
    ]:

        if new_data.get(key) is not None:
            merged_data[key] = new_data.get(key)

        else:
            merged_data[key] = old_data.get(key)

    return {
        "financial": merged_data
    }

def ask_financial_fields(state: BotState):

    financial = state.get("financial", {})
    missing_fields=[]
    required = [
        "monthly_income",
        "employment_status",
        "earning_members"
    ]

    for field in required:
        if not financial.get(field):
            missing_fields.append(field)
    prompt = ChatPromptTemplate.from_messages([
("human", """This field is missing: {missing_fields}

Ask the user to provide it.

Instructions:
- Ask ONLY for missing field.
- Do NOT repeat already provided information.
- Keep the response short and professional.
 - employment_status examples:
  unemployed
  employed
  self-employed
  daily wage worker
  retired
  student.
""")])
    
    chain = prompt|llm|parser2
    response = chain.invoke({"missing_fields":missing_fields})
    return {"messages":[AIMessage(content=response)]}

def check_financial_fields(state):
    financial = state.get("financial", {})
    missing_fields=[]
    required = [
        "monthly_income",
        "employment_status",
        "earning_members"
    ]

    for field in required:
        if not financial.get(field):
            missing_fields.append(field)

    if missing_fields:
        return "missing"

    return "complete"
    



# =====================
# Other Required fields
# =====================
def check_fields(state: BotState):

    data = state["extracted_dict"]

    mandatory_fields = ["name", "phone", "family_members", "location", "need_cost"]

    missing_fields = []

    for field in mandatory_fields:
        if not data.get(field):
            missing_fields.append(field)

    if missing_fields:
        return "missing"

    return "complete"




def check_basic_node(state):
    return {}

def ask_missing(state: BotState):

    data = state["extracted_dict"]

    missing_fields = []

    for field in ["name", "phone", "family_members", "location", "need_cost"]:
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
    response = chain.invoke({"missing_fields":missing_fields[0]})
    return {"messages":[AIMessage(content=response)]}


# ================
# need cost node
# ================
# def ask_need_cost(state: BotState):
#     data = state.get("extracted_dict", {})

#     if data.get("need_cost"):
#         return {}

#     return {
#         "messages": [
#             AIMessage(
#                 content="Please provide the required assistance amount in INR."
#             )
#         ]
#     }

# def check_need_cost(state: BotState):
#     data = state.get("extracted_dict", {})

#     if data.get("need_cost"):
#         return "complete"

#     return "missing"

# =======================
# Show Information
# =======================
def show_info(state: BotState):

    data = state.get("extracted_dict", {})
    need = str(data.get("need", "")).lower()

    response = f"""
# Registration Completed

## Beneficiary Details

Name: {data.get("name")}\n
Phone: {data.get("phone")}\n
Location: {data.get("location")}\n
Family Members: {data.get("family_members")}\n
Need Type: {data.get("need")}\n
Need Cost: {data.get("need_cost")}\n
"""

    if "medical" in need:

        medical = state.get("medical", {})

        response += f"""

## Medical Details

Disease: {medical.get("disease")}\n
Hospital: {medical.get("hospital")}\n
Urgency: {medical.get("urgency")}
"""

    elif "education" in need:

        education = state.get("education", {})

        response += f"""

## Education Details

Class: {education.get("student_class")}\n
Institute: {education.get("institute")}\n
Academic Status: {education.get("academic_status")}
"""

    elif "financial" in need:

        financial = state.get("financial", {})

        response += f"""

## Financial Details

Monthly Income: {financial.get("monthly_income")}\n
Employment Status: {financial.get("employment_status")}\n
Earning Members: {financial.get("earning_members")}\n
"""

    response += "\nPlease click Save to store the information."

    return {"messages": [AIMessage(content=response)],"current_flow": ""}

graph = StateGraph(BotState)

# creating nodes

# Core nodes
graph.add_node("language_node", language_detection_node)
graph.add_node("input_node", input_processing_node)

# Need selection
graph.add_node("ask_need", ask_need)

# Specialized extraction nodes
graph.add_node("medical_node", medical_node)
graph.add_node("education_node", education_node)
graph.add_node("financial_node", financial_node)

# Common validation node
graph.add_node("check_basic", check_basic_node)

# missing fields check nodes
graph.add_node("ask_missing_medical", ask_medical_fields)
graph.add_node("ask_missing_education", ask_education_fields)
graph.add_node("ask_missing_financial", ask_financial_fields)

# Output nodes
graph.add_node("ask_missing", ask_missing)
graph.add_node("show_info", show_info)


# ===================
# Graph Edges
# ===================

graph.add_edge(START, "language_node")


graph.add_conditional_edges(
    "language_node",
    resume_flow,
    {
        "input_node": "input_node",
        "medical_node": "medical_node",
        "education_node": "education_node",
        "financial_node": "financial_node"
    }
)

graph.add_conditional_edges(
    "input_node",
    check_need_type,
    {
        "medical": "medical_node",
        "education": "education_node",
        "financial": "financial_node",
        "ask_need": "ask_need"
    }
)

graph.add_edge("ask_need", END)

graph.add_conditional_edges(
    "medical_node",
    check_medical_fields,
    {
        "missing": "ask_missing_medical",
        "complete": "check_basic"
    }
)

graph.add_conditional_edges(
    "education_node",
    check_education_fields,
    {
        "missing": "ask_missing_education",
        "complete": "check_basic"
    }
)

graph.add_conditional_edges(
    "financial_node",
    check_financial_fields,
    {
        "missing": "ask_missing_financial",
        "complete": "check_basic"
    }
)

graph.add_conditional_edges(
    "check_basic",
    check_fields,
    {
        "missing": "ask_missing",
        "complete": "show_info"
    }
)

graph.add_edge("ask_missing_medical", END)
graph.add_edge("ask_missing_education", END)
graph.add_edge("ask_missing_financial", END)

graph.add_edge("ask_missing", END)

graph.add_edge("show_info", END)

checkpointer = InMemorySaver()
chatbot = graph.compile(checkpointer=checkpointer)

CONFIG = {"configurable": {"thread_id":"thread-1"}}

# while True:
#     user_input = input("Type Here: ")

#     response = chatbot.invoke({"messages":[HumanMessage(content=user_input)]}, config=CONFIG)

#     print("BOT:",response["messages"][-1].content)
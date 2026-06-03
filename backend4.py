"""
NGO Registration Assistant
==========================
Multilingual (English / Hindi / Roman-Hindi / Roman-Urdu)
LangGraph + LangChain + Groq Llama-3.1-8B + Pydantic
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# LLM
# ──────────────────────────────────────────────────────────────────────────────

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
str_parser = StrOutputParser()

# ──────────────────────────────────────────────────────────────────────────────
# PYDANTIC SCHEMAS
# ──────────────────────────────────────────────────────────────────────────────

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

class ExtractionSchema(BaseModel):
    """Top-level extraction schema. Only fill fields mentioned by the user."""
    name: Optional[str] = Field(default=None, description="Full name of the applicant in English")
    phone: Optional[str] = Field(default=None, description="Phone number digits only")
    location: Optional[str] = Field(default=None, description="City or area in English")
    family_members: Optional[int] = Field(default=None, description="Total family members (integer only)")
    need_category: Optional[str] = Field(
        default=None,
        description="One of: 'medical', 'education', 'financial'. Set only if user explicitly names a category."
    )
    medical: Optional[MedicalSchema] = Field(default=None)
    education: Optional[EducationSchema] = Field(default=None)
    financial: Optional[FinancialSchema] = Field(default=None)

extraction_parser = PydanticOutputParser(pydantic_object=ExtractionSchema)

# ──────────────────────────────────────────────────────────────────────────────
# STATE
# ──────────────────────────────────────────────────────────────────────────────

class BotState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    extracted_dict: dict
    language: str           # Locked after first detection — never changes
    language_locked: bool   # True once language has been confidently detected
    active_category: str
    registration_status: str

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

NEED_FIELDS: dict[str, list[str]] = {
    "medical":   ["disease", "hospital", "urgency", "need_cost"],
    "education": ["student_class", "institute", "academic_status"],
    "financial": ["monthly_income", "employment_status", "earning_members"],
}

BASIC_FIELDS = ["name", "phone", "location", "family_members"]

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "medical": [
        "medical", "cancer", "tumor", "corona", "covid", "surgery", "operation",
        "disease", "illness", "hospital", "treatment", "medicine", "bimari",
        "dawa", "ilaj", "doctor", "emergency", "patient", "bemar", "tibbi",
    ],
    "education": [
        "education", "school", "college", "fees", "scholarship", "admission",
        "university", "class", "study", "padhai", "fee", "exam", "result",
        "taleem",
    ],
    "financial": [
        "financial", "finance", "job", "income", "salary", "unemployed",
        "money", "rent", "loan", "expense", "household", "naukri", "rozi",
        "paisa", "kharcha", "maaliyati", "arthik",
    ],
}

_AMBIGUOUS_PATTERNS = re.compile(
    r"^(yes|no|ok|okay|high|low|medium|medical|education|financial|"
    r"passed|failed|appearing|unemployed|self.employed|daily.wage|\d+[\d,\s]*)$",
    re.IGNORECASE
)

def _is_ambiguous_for_lang_detection(text: str) -> bool:
    stripped = text.strip()
    if len(stripped.split()) <= 2 and _AMBIGUOUS_PATTERNS.match(stripped):
        return True
    if re.match(r"^[\d\s,]+$", stripped):
        return True
    return False

def keyword_detect_category(text: str) -> Optional[str]:
    lower = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return category
    return None

def deep_merge(old: dict, new: dict) -> dict:
    merged = old.copy()
    for key, value in new.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

def get_missing_need_fields(data: dict, category: str) -> list[str]:
    required = NEED_FIELDS.get(category, [])
    cat_data = data.get(category, {})
    return [f for f in required if not cat_data.get(f)]

def get_missing_basic_fields(data: dict) -> list[str]:
    return [f for f in BASIC_FIELDS if not data.get(f)]

_INT_FIELDS = {"need_cost", "monthly_income", "family_members", "earning_members"}

def _extract_bare_value(text: str, field: str):
    text = text.strip()
    if field in _INT_FIELDS:
        digits = re.sub(r"[,\s]", "", text)
        match = re.search(r"\d+", digits)
        return int(match.group()) if match else None
    if len(text) < 100:
        return text.strip()
    return None

def _get_current_field_hint(state: BotState) -> str:
    data = state.get("extracted_dict", {})
    active_category = state.get("active_category", "")
    registration_status = state.get("registration_status", "need_incomplete")

    if registration_status == "basic_incomplete":
        missing = get_missing_basic_fields(data)
        return f"basic.{missing[0]}" if missing else ""

    if active_category:
        missing = get_missing_need_fields(data, active_category)
        return f"{active_category}.{missing[0]}" if missing else ""

    return ""

# ──────────────────────────────────────────────────────────────────────────────
# NODE 1 – LANGUAGE DETECTION  (deterministic, no LLM)
# ──────────────────────────────────────────────────────────────────────────────
#
# Strategy: use Unicode script ranges + a small wordlist.
# Never rely on an LLM for this — it misclassifies names like "Arshaan" as
# Urdu and locks the wrong language for the entire session.
#
# Rules (in priority order):
#   1. Any Devanagari character present  → hindi
#   2. Any Arabic/Urdu-script character  → urdu (not used here, failsafe)
#   3. Known Roman-Urdu marker words     → roman_urdu
#   4. Known Roman-Hindi marker words    → roman_hindi
#   5. Everything else                   → english
# ──────────────────────────────────────────────────────────────────────────────

# Devanagari Unicode block: U+0900–U+097F
_DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')

# Words that only appear in Roman Urdu (not Hindi or English)
_ROMAN_URDU_MARKERS = {
    "aap", "apka", "apki", "apke", "hai", "hain", "mujhe", "humein",
    "kya", "kaise", "kahaan", "kyun", "nahi", "nahin", "zaruri",
    "zaroorat", "madad", "bhai", "jan", "sahab", "theek", "bilkul",
    "shukriya", "meherbani", "khuda", "allah", "inshallah", "mashallah",
    "ghar", "shehar", "paisa", "rupay", "taleem", "tibbi", "maaliyati",
    "daakhil", "ilaj", "dawa", "mareez", "berozgar", "aamdani",
}

# Words that appear in Roman Hindi (but not common English)
_ROMAN_HINDI_MARKERS = {
    "mujhe", "humein", "aapko", "kaunsi", "chahiye", "bataye",
    "kitne", "kamai", "mahine", "rupaye", "padhai", "bimari",
    "hospital", "naukri", "rozi", "paisa", "kharcha", "abhi",
    "kaun", "kahan", "kya", "nahi", "nahin", "hain", "hai",
    "mere", "mera", "meri", "unko", "unka", "uska", "iski",
    "bahut", "thoda", "zyada", "kam", "poora", "poori",
}


def _detect_language(text: str) -> str:
    """
    Deterministic language detector. Returns one of:
    english | hindi | roman_hindi | roman_urdu
    """
    # 1. Devanagari script → Hindi
    if _DEVANAGARI_RE.search(text):
        return "hindi"

    # 2. Tokenise to lowercase words for wordlist checks
    words = set(re.findall(r'[a-z]+', text.lower()))

    # 3. Roman Urdu markers (checked before Hindi — overlapping words exist)
    urdu_hits = words & _ROMAN_URDU_MARKERS
    hindi_hits = words & _ROMAN_HINDI_MARKERS

    if urdu_hits and not hindi_hits:
        return "roman_urdu"
    if hindi_hits or urdu_hits:
        # Both hit → treat as roman_hindi (more common for this NGO context)
        return "roman_hindi"

    # 4. Default → English
    return "english"


def language_node(state: BotState) -> dict:
    # Once locked, never re-detect
    if state.get("language_locked", False):
        return {}

    user_message = state["messages"][-1].content

    # Always run detection — it's instant and deterministic, no LLM needed
    lang = _detect_language(user_message)

    # Lock only when the message is rich enough to be confident:
    # more than 3 words, or contains Devanagari, or hits a marker word
    words = user_message.strip().split()
    has_markers = bool(
        _DEVANAGARI_RE.search(user_message)
        or (set(re.findall(r'[a-z]+', user_message.lower()))
            & (_ROMAN_URDU_MARKERS | _ROMAN_HINDI_MARKERS))
    )
    should_lock = len(words) > 3 or has_markers

    return {
        "language": lang,
        "language_locked": should_lock,
    }




# ──────────────────────────────────────────────────────────────────────────────
# NODE 2 – EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

_EXTRACT_TEMPLATE = """\
You are a multilingual data extraction engine for an NGO registration system.

USER LANGUAGE: {language}
ACTIVE NEED CATEGORY: {active_category}
FIELD CURRENTLY BEING ANSWERED: {current_field}

Extract structured information from the USER MESSAGE below.
Convert ALL values to English before storing.
Return ONLY valid JSON matching the schema. Do NOT add explanation.

==================================================
CRITICAL CONTEXT RULE
==================================================
If FIELD CURRENTLY BEING ANSWERED is set, the user's reply is almost certainly
answering that specific field. Map the value directly.

Examples:
  current_field="medical.need_cost",  user="250000"       → medical.need_cost=250000
  current_field="medical.disease",    user="cancer"        → medical.disease="cancer"
  current_field="medical.urgency",    user="high"          → medical.urgency="high"
  current_field="medical.hospital",   user="bhp hospital"  → medical.hospital="BHP Hospital"
  current_field="financial.monthly_income", user="15000"   → financial.monthly_income=15000
  current_field="education.student_class",  user="10th"    → education.student_class="10th"
  current_field="basic.name",         user="Arshaan Khan"  → name="Arshaan Khan"
  current_field="basic.phone",        user="9876543210"    → phone="9876543210"
  current_field="basic.location",     user="Bhopal"        → location="Bhopal"
  current_field="basic.family_members", user="5"           → family_members=5

==================================================
GENERAL TRANSLATION EXAMPLES
==================================================
  "mere uncle ko cancer hai"   → medical.disease="cancer"
  "hamidia hospital"           → medical.hospital="Hamidia Hospital"
  "bahut emergency hai"        → medical.urgency="high"
  "mahine ki income 20 hazar"  → financial.monthly_income=20000
  "8th class"                  → education.student_class="8th"
  "berozgar hain"              → financial.employment_status="unemployed"

==================================================
RULES
==================================================
1. Missing fields → null (never guess).
2. Numeric fields: need_cost, monthly_income, family_members, earning_members → integer or null.
3. need_category: ONLY set if user explicitly says "medical help", "financial support", etc.
   Do NOT set it for disease names, cost amounts, or hospital names.
4. Do NOT copy old data. Extract ONLY what is in the current user message.

USER MESSAGE:
{user_message}

{format_instructions}
"""

_extract_prompt = PromptTemplate(
    template=_EXTRACT_TEMPLATE,
    input_variables=["language", "user_message", "active_category", "current_field"],
    partial_variables={"format_instructions": extraction_parser.get_format_instructions()},
)
_extract_chain = _extract_prompt | llm | extraction_parser

_CATEGORY_LABELS = {"medical", "education", "financial", "medically", "medico", "med"}


def extraction_node(state: BotState) -> dict:
    user_message = state["messages"][-1].content
    language = state.get("language", "english")
    active_category = state.get("active_category", "")
    current_field = _get_current_field_hint(state)

    try:
        extracted: ExtractionSchema = _extract_chain.invoke({
            "user_message": user_message,
            "language": language,
            "active_category": active_category or "unknown",
            "current_field": current_field or "unknown",
        })
        new_data = extracted.model_dump(exclude_none=True)
    except Exception:
        new_data = {}

    # ── Bare-value injection fallback ──────────────────────────────────────
    if current_field and current_field != "unknown":
        parts = current_field.split(".", 1)
        if len(parts) == 2:
            cat, field = parts
            is_category_word = user_message.strip().lower() in NEED_FIELDS
            if cat != "basic":
                cat_data = new_data.get(cat, {})
                if not cat_data.get(field) and not is_category_word:
                    injected = _extract_bare_value(user_message, field)
                    if injected is not None:
                        new_data.setdefault(cat, {})[field] = injected
            else:
                if not new_data.get(field):
                    injected = _extract_bare_value(user_message, field)
                    if injected is not None:
                        new_data[field] = injected

    # ── Contamination guard ────────────────────────────────────────────────
    for cat in ("medical", "education", "financial"):
        cat_data = new_data.get(cat, {})
        for field in list(cat_data.keys()):
            val = cat_data.get(field)
            if isinstance(val, str) and val.lower() in _CATEGORY_LABELS:
                del new_data[cat][field]

    old_data = state.get("extracted_dict", {})
    merged = deep_merge(old_data, new_data)

    # ── Active category — lock once set, never override ────────────────────
    current_active = active_category
    if not current_active:
        msg_lower = user_message.strip().lower()

        _DIRECT_CATEGORY_MAP = {
            "medical": "medical",    "medial": "medical",   "med": "medical",
            "tibbi": "medical",      "chikitsa": "medical",
            "education": "education","edu": "education",
            "padhai": "education",   "taleem": "education",
            "financial": "financial","finance": "financial",
            "maaliyati": "financial","arthik": "financial",
        }
        if msg_lower in _DIRECT_CATEGORY_MAP:
            current_active = _DIRECT_CATEGORY_MAP[msg_lower]
        elif new_data.get("need_category", "") in NEED_FIELDS:
            current_active = new_data["need_category"]
        else:
            kw_cat = keyword_detect_category(user_message)
            if kw_cat:
                current_active = kw_cat

    merged.pop("need_category", None)

    return {
        "extracted_dict": merged,
        "active_category": current_active,
    }


# ──────────────────────────────────────────────────────────────────────────────
# NODE 3 – EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def evaluation_node(state: BotState) -> dict:
    data = state.get("extracted_dict", {})
    active_category = state.get("active_category", "")

    if not active_category:
        return {"registration_status": "need_incomplete"}

    if get_missing_need_fields(data, active_category):
        return {"registration_status": "need_incomplete"}

    if get_missing_basic_fields(data):
        return {"registration_status": "basic_incomplete"}

    return {"registration_status": "complete"}


# ──────────────────────────────────────────────────────────────────────────────
# ROUTER
# ──────────────────────────────────────────────────────────────────────────────

def registration_router(state: BotState) -> str:
    status = state.get("registration_status", "need_incomplete")
    if status == "need_incomplete":
        return "need_question_node"
    if status == "basic_incomplete":
        return "basic_info_node"
    return "complete"


# ──────────────────────────────────────────────────────────────────────────────
# NODE 4 – NEED QUESTION NODE
# ──────────────────────────────────────────────────────────────────────────────

_NEED_QUESTIONS: dict[str, dict[str, dict[str, str]]] = {
    "medical": {
        "disease": {
            "english":     "What illness or medical condition does the patient have?",
            "hindi":       "मरीज़ को कौन सी बीमारी है?",
            "roman_hindi": "Mareez ko kaunsi bimari hai?",
            "roman_urdu":  "Mareez ko kaunsi bimari hai?",
        },
        "hospital": {
            "english":     "Which hospital is the patient admitted in?",
            "hindi":       "मरीज़ किस अस्पताल में भर्ती है?",
            "roman_hindi": "Mareez kis hospital mein admit hai?",
            "roman_urdu":  "Mareez kis hospital mein daakhil hai?",
        },
        "urgency": {
            "english":     "How urgent is the situation — high, medium, or low?",
            "hindi":       "स्थिति कितनी गंभीर है — बहुत ज़रूरी, सामान्य, या कम?",
            "roman_hindi": "Situation kitni urgent hai — zyada, thodi, ya kam?",
            "roman_urdu":  "Situation kitni zaruri hai — zyada, thodi, ya kam?",
        },
        "need_cost": {
            "english":     "What is the estimated treatment cost in rupees?",
            "hindi":       "इलाज का अनुमानित खर्च कितना है (रुपयों में)?",
            "roman_hindi": "Ilaj mein kitna kharcha aayega (rupaye mein)?",
            "roman_urdu":  "Ilaj mein kitna kharcha hoga (rupay mein)?",
        },
    },
    "education": {
        "student_class": {
            "english":     "Which class or year is the student currently in?",
            "hindi":       "छात्र अभी किस कक्षा में है?",
            "roman_hindi": "Student abhi kaun si class mein hai?",
            "roman_urdu":  "Student abhi kaun si class mein hai?",
        },
        "institute": {
            "english":     "What is the name of the school or college?",
            "hindi":       "स्कूल या कॉलेज का नाम क्या है?",
            "roman_hindi": "School ya college ka naam kya hai?",
            "roman_urdu":  "School ya college ka naam kya hai?",
        },
        "academic_status": {
            "english":     "What is the student's current academic status — passed, failed, or appearing?",
            "hindi":       "छात्र की पढ़ाई की स्थिति क्या है — पास, फेल, या परीक्षा देने वाला?",
            "roman_hindi": "Student ki padhai ki situation kya hai — passed, failed, ya exam de raha hai?",
            "roman_urdu":  "Student ka kya haal hai — passed, failed, ya imtehaan dene wala?",
        },
    },
    "financial": {
        "monthly_income": {
            "english":     "What is the household's total monthly income in rupees?",
            "hindi":       "परिवार की कुल मासिक आय कितनी है (रुपयों में)?",
            "roman_hindi": "Ghar ki poori mahine ki kamai kitni hai (rupaye mein)?",
            "roman_urdu":  "Ghar ki mahina wari aamdani kitni hai (rupay mein)?",
        },
        "employment_status": {
            "english":     "What is the current employment situation — unemployed, self-employed, or daily wage?",
            "hindi":       "रोज़गार की स्थिति क्या है — बेरोज़गार, स्वरोज़गार, या दैनिक मज़दूर?",
            "roman_hindi": "Rozgaar ki kya situation hai — berozgar, apna kaam, ya roz ka kaam?",
            "roman_urdu":  "Rozgaar ka kya haal hai — berozgar, apna kaam, ya roz ka kaam?",
        },
        "earning_members": {
            "english":     "How many members in your family are currently earning?",
            "hindi":       "परिवार में कितने सदस्य अभी कमाई कर रहे हैं?",
            "roman_hindi": "Ghar mein kitne log abhi kamaate hain?",
            "roman_urdu":  "Ghar mein kitne log abhi kamaate hain?",
        },
    },
}

_ASK_NEED_CATEGORY: dict[str, str] = {
    "english":     "What kind of help do you need — medical, education, or financial?",
    "hindi":       "आपको किस प्रकार की सहायता चाहिए — चिकित्सा, शिक्षा, या आर्थिक?",
    "roman_hindi": "Aapko kaunsi madad chahiye — medical, education, ya financial?",
    "roman_urdu":  "Aapko kaunsi madad chahiye — tibbi, taleem, ya maaliyati?",
}


def need_question_node(state: BotState) -> dict:
    language = state.get("language", "english")
    if language not in ("english", "hindi", "roman_hindi", "roman_urdu"):
        language = "english"

    data = state.get("extracted_dict", {})
    active_category = state.get("active_category", "")

    if not active_category:
        q = _ASK_NEED_CATEGORY.get(language, _ASK_NEED_CATEGORY["english"])
        return {"messages": [AIMessage(content=q)]}

    missing = get_missing_need_fields(data, active_category)
    if not missing:
        return {"messages": [AIMessage(content="NEED_COMPLETE")]}

    field_qs = _NEED_QUESTIONS.get(active_category, {}).get(missing[0], {})
    q = field_qs.get(language, field_qs.get("english", "Please provide more details."))
    return {"messages": [AIMessage(content=q)]}


# ──────────────────────────────────────────────────────────────────────────────
# NODE 5 – BASIC INFO NODE
# ──────────────────────────────────────────────────────────────────────────────

_BASIC_QUESTIONS: dict[str, dict[str, str]] = {
    "name": {
        "english":     "Please share your full name.",
        "hindi":       "कृपया अपना पूरा नाम बताइए।",
        "roman_hindi": "Apna poora naam bataye.",
        "roman_urdu":  "Apna poora naam bataye.",
    },
    "phone": {
        "english":     "What is your phone number?",
        "hindi":       "आपका फोन नंबर क्या है?",
        "roman_hindi": "Apna phone number bataye.",
        "roman_urdu":  "Apna phone number bataye.",
    },
    "location": {
        "english":     "Which city or area are you from?",
        "hindi":       "आप किस शहर या इलाके से हैं?",
        "roman_hindi": "Aap kis city ya ilake se hain?",
        "roman_urdu":  "Aap kis shehar ya ilaqe se hain?",
    },
    "family_members": {
        "english":     "How many members are there in your family?",
        "hindi":       "आपके परिवार में कितने सदस्य हैं?",
        "roman_hindi": "Aapke ghar mein kitne log hain?",
        "roman_urdu":  "Aapke ghar mein kitne log hain?",
    },
}


def basic_info_node(state: BotState) -> dict:
    language = state.get("language", "english")
    if language not in ("english", "hindi", "roman_hindi", "roman_urdu"):
        language = "english"

    data = state.get("extracted_dict", {})
    missing = get_missing_basic_fields(data)

    if not missing:
        return {"messages": [AIMessage(content="REGISTRATION_COMPLETE")]}

    field_qs = _BASIC_QUESTIONS.get(missing[0], {})
    q = field_qs.get(language, field_qs.get("english", "Please provide more details."))
    return {"messages": [AIMessage(content=q)]}


# ──────────────────────────────────────────────────────────────────────────────
# GRAPH ASSEMBLY
# ──────────────────────────────────────────────────────────────────────────────

graph = StateGraph(BotState)

graph.add_node("language_node",      language_node)
graph.add_node("extraction_node",    extraction_node)
graph.add_node("evaluation_node",    evaluation_node)
graph.add_node("need_question_node", need_question_node)
graph.add_node("basic_info_node",    basic_info_node)

graph.add_edge(START,             "language_node")
graph.add_edge("language_node",   "extraction_node")
graph.add_edge("extraction_node", "evaluation_node")

graph.add_conditional_edges(
    "evaluation_node",
    registration_router,
    {
        "need_question_node": "need_question_node",
        "basic_info_node":    "basic_info_node",
        "complete":           END,
    },
)

graph.add_edge("need_question_node", END)
graph.add_edge("basic_info_node",    END)

# ──────────────────────────────────────────────────────────────────────────────
# COMPILE  — exported for use by streamlit_app.py
# ──────────────────────────────────────────────────────────────────────────────

checkpointer = InMemorySaver()
chatbot = graph.compile(checkpointer=checkpointer)
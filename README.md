# 🤝 AI-Khidmat — AI Beneficiary Registration System

**AI-Khidmat** is an AI-powered beneficiary registration system that simplifies the process of collecting and verifying beneficiary information through a conversational interface.

Instead of requiring users to fill out a traditional form, the system uses a **Large Language Model (LLM)** to understand natural-language responses, extract structured information, identify the type of assistance required, ask for missing information, and store the completed registration in a PostgreSQL database.

The system also includes **face validation, face embeddings, and duplicate face detection** to help prevent duplicate beneficiary registrations.

---

## ✨ Key Features

* 💬 Conversational beneficiary registration
* 🧠 LLM-based information extraction
* 🔄 Stateful conversation using LangGraph
* 🏥 Medical assistance registration
* 🎓 Education assistance registration
* 💰 Financial assistance registration
* ❓ Automatic detection of missing information
* ✅ Structured output using Pydantic
* 🗄️ PostgreSQL database storage
* 📸 Beneficiary photo upload
* 📷 Camera capture through Streamlit
* 👤 Face validation
* 🧬 Face embedding generation using FaceNet512
* 🔍 Duplicate face detection using cosine similarity
* 🔐 Environment-variable based configuration

---

# 🏗️ System Architecture

```text
                         ┌─────────────────────┐
                         │      Beneficiary    │
                         │        User         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     Streamlit UI    │
                         │  03_streamlit.py    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Conversational AI  │
                         │     backend5.py     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Input Processing  │
                         │  & Need Detection   │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
             ┌───────────┐   ┌────────────┐   ┌────────────┐
             │  Medical  │   │ Education  │   │ Financial  │
             │   Flow    │   │    Flow    │   │    Flow    │
             └─────┬─────┘   └──────┬─────┘   └──────┬─────┘
                   │                │                │
                   └────────────────┼────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Required Fields    │
                         │     Validation      │
                         └──────────┬──────────┘
                                    │
                             ┌──────┴──────┐
                             │             │
                             ▼             ▼
                          Missing       Complete
                             │             │
                             ▼             ▼
                       Ask Missing     Show Info
                                           │
                                           ▼
                                    PostgreSQL Save

                                    +

                         ┌─────────────────────┐
                         │   Face Processing   │
                         │   Face Validation   │
                         │   FaceNet512        │
                         │ Duplicate Detection │
                         └─────────────────────┘
```

---

# 🧠 How the System Works

## 1. Conversational Registration

The beneficiary interacts with the system through a Streamlit chat interface.

For example:

```text
User:
My name is Arshaan. I need financial assistance.
```

The AI extracts the available information instead of requiring the user to enter every field separately.

The system maintains the information collected during the conversation and asks for information that is still missing.

---

# 2. Beneficiary Information Extraction

The system extracts the following basic information:

| Field          | Description                       |
| -------------- | --------------------------------- |
| Name           | Beneficiary's name                |
| Phone          | Beneficiary's mobile number       |
| Location       | Beneficiary's location            |
| Family Members | Number of family members          |
| Need           | Type of assistance                |
| Need Cost      | Required assistance amount in INR |

The information is represented using a **Pydantic schema**.

Example:

```text
Name: Arshaan
Phone: 9876543210
Location: Bhopal
Family Members: 5
Need: Financial
Need Cost: 25000
```

---

# 3. Assistance Type Detection

The system identifies which type of assistance the beneficiary requires.

Currently supported:

### 🏥 Medical

Required information:

```text
Disease
Hospital
Urgency
```

Example:

```text
Disease: Cancer
Hospital: AIIMS Bhopal
Urgency: High
```

### 🎓 Education

Required information:

```text
Student Class
Institute
Academic Status
```

Example:

```text
Student Class: Class 10
Institute: ABC Public School
Academic Status: Studying
```

### 💰 Financial

Required information:

```text
Monthly Income
Employment Status
Earning Members
```

Example:

```text
Monthly Income: ₹12,000
Employment Status: Daily Wage Worker
Earning Members: 1
```

---

# 4. Missing Information Detection

The system checks whether all mandatory fields have been collected.

For example:

```text
User:
I need education assistance for my daughter.
She is studying in class 10.
```

The system can identify that information such as the institute and academic status is still missing.

It then asks the beneficiary for the missing information.

```text
AI:
What is the name of her institute?
```

This process continues until the required information is collected.

---

# 5. LangGraph Workflow

The conversational workflow is implemented using **LangGraph**.

The graph contains separate nodes for different tasks.

```text
START
  │
  ▼
Input Processing
  │
  ▼
Need Detection
  │
  ├───────────────┐
  │               │
  ▼               ▼
Medical        Education
  │               │
  └───────┬───────┘
          │
          ▼
       Financial
          │
          ▼
   Check Required Fields
          │
     ┌────┴────┐
     │         │
     ▼         ▼
   Missing   Complete
     │         │
     ▼         ▼
Ask Missing  Show Info
               │
               ▼
              END
```

LangGraph allows the application to maintain a structured workflow instead of relying entirely on a single LLM prompt.

---

# 6. Structured Output with Pydantic

The project uses Pydantic models to structure the information extracted by the LLM.

### Basic beneficiary schema

```python
class Schema(BaseModel):
    name: Optional[str]
    phone: Optional[str]
    need: Optional[str]
    location: Optional[str]
    family_members: Optional[int]
    need_cost: Optional[int]
```

Separate schemas are used for:

* Medical information
* Education information
* Financial information

This helps convert natural-language responses into structured Python data.

---

# 📸 Face Verification

After completing the conversational registration, the beneficiary must provide a photograph.

The system supports:

* Uploading an image
* Capturing an image using the device camera

Supported formats:

```text
.jpg
.jpeg
.png
```

---

## 👤 Face Validation

Before saving the beneficiary, the system verifies the uploaded image.

The image must contain **exactly one face**.

The system rejects:

```text
No face
```

and:

```text
Multiple faces
```

This prevents invalid images from being used for beneficiary verification.

Face detection is performed using **DeepFace** with the OpenCV detector backend.

---

# 🧬 Face Embeddings

After successful face validation, the system generates a numerical representation of the face.

The project uses:

```text
FaceNet512
```

through DeepFace.

The generated embedding is stored in PostgreSQL as JSON data.

Conceptually:

```text
Beneficiary Photo
       │
       ▼
Face Detection
       │
       ▼
FaceNet512
       │
       ▼
Face Embedding
       │
       ▼
PostgreSQL
```

---

# 🔍 Duplicate Face Detection

Before creating a new beneficiary record, the system checks whether the face already exists in the database.

The process is:

```text
New Beneficiary Photo
        │
        ▼
Generate Embedding
        │
        ▼
Load Existing Embeddings
        │
        ▼
Calculate Cosine Similarity
        │
        ▼
Compare With Threshold
        │
   ┌────┴────┐
   │         │
Duplicate   New Face
   │         │
   ▼         ▼
Reject     Continue
             │
             ▼
           Save
```

The current similarity threshold is:

```text
0.80
```

If the best similarity is greater than or equal to the threshold, the system reports a duplicate.

Example:

```text
Duplicate Face Detected

Existing Beneficiary ID: 15
Similarity: 0.8732
```

---

# 🗄️ Database

The project uses **PostgreSQL**.

Two main tables are created automatically.

## `beneficiary`

Stores the main beneficiary information.

```text
id
name
phone
location
family_members
need
need_cost
need_details
created_at
```

The `need_details` column uses PostgreSQL's `JSONB` format to store category-specific information.

For example:

```json
{
    "disease": "Cancer",
    "hospital": "AIIMS Bhopal",
    "urgency": "high"
}
```

---

## `beneficiary_face`

Stores face-related information.

```text
id
beneficiary_id
image_path
embedding
created_at
```

The `beneficiary_id` is linked to the main beneficiary table through a foreign key.

---

# 📁 Project Structure

```text
AI-Khidmat/
│
├── backend5.py
│   └── LangGraph conversational AI workflow
│
├── 03_streamlit.py
│   └── Streamlit user interface
│
├── database.py
│   └── PostgreSQL database operations
│
├── face_embeddings.py
│   └── Face validation, embeddings and duplicate detection
│
├── README.md
│   └── Project documentation
│
├── requirements.txt
│   └── Python dependencies
│
└── .gitignore
```

---

# 🛠️ Technologies Used

| Technology           | Purpose                     |
| -------------------- | --------------------------- |
| Python               | Main programming language   |
| Streamlit            | User interface              |
| LangGraph            | Conversational workflow     |
| LangChain            | LLM integration and prompts |
| Groq                 | LLM inference               |
| Llama 3.1 8B Instant | Language model              |
| Pydantic             | Structured data validation  |
| DeepFace             | Face processing             |
| FaceNet512           | Face embeddings             |
| NumPy                | Numerical operations        |
| PostgreSQL           | Data storage                |
| psycopg2             | PostgreSQL connection       |
| python-dotenv        | Environment configuration   |
| Git/GitHub           | Version control             |

---

# 📄 File Description

## `backend5.py`

Contains the main AI and LangGraph workflow.

Responsibilities include:

* Natural-language processing
* Beneficiary information extraction
* Need classification
* Medical information extraction
* Education information extraction
* Financial information extraction
* Missing-field detection
* Conversation state management
* Registration completion

---

## `03_streamlit.py`

Contains the Streamlit application.

Responsibilities include:

* Chat interface
* Conversation history
* Session management
* Beneficiary photo upload
* Camera capture
* Face validation
* Duplicate checking
* Saving beneficiary information
* Saving face embeddings

Run with:

```bash
streamlit run 03_streamlit.py
```

---

## `database.py`

Handles PostgreSQL operations.

Responsibilities include:

* Creating tables
* Connecting to PostgreSQL
* Saving beneficiaries
* Retrieving beneficiaries
* Saving face embeddings
* Loading existing embeddings
* Checking duplicate faces

---

## `face_embeddings.py`

Handles face-related functionality.

Responsibilities include:

* Face validation
* Face detection
* Face embedding generation
* Cosine similarity calculation
* Duplicate face detection

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Khidmat.git
```

Then:

```bash
cd AI-Khidmat
```

---

## 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Configuration

Create a `.env` file in the project directory.

Example:

```env
GROQ_API_KEY=your_groq_api_key

DB_HOST=localhost
DB_NAME=your_database_name
DB_USER=your_database_user
DB_PASSWORD=your_database_password
DB_PORT=5432
```

**Never upload the `.env` file to GitHub.**

Add the following to `.gitignore`:

```gitignore
.env
venv/
__pycache__/
*.pyc
uploads/
```

---

# 🗃️ PostgreSQL Setup

Install PostgreSQL and create a database.

For example:

```sql
CREATE DATABASE ai_khidmat;
```

Configure the database credentials in `.env`.

The application automatically creates the required tables through:

```python
create_tables()
```

---

# ▶️ Running the Application

After installing the dependencies and configuring PostgreSQL and Groq:

```bash
streamlit run 03_streamlit.py
```

The application will open in your browser.

---

# 🔄 Complete Registration Flow

```text
1. User opens AI-Khidmat
          ↓
2. User starts conversation
          ↓
3. Beneficiary information is extracted
          ↓
4. Assistance type is detected
          ↓
5. Category-specific information is collected
          ↓
6. Missing information is requested
          ↓
7. Registration information is displayed
          ↓
8. Beneficiary uploads/captures photo
          ↓
9. Face is validated
          ↓
10. Face embedding is generated
          ↓
11. Existing faces are checked
          ↓
12. Duplicate?
       ↙       ↘
     YES        NO
      ↓          ↓
    Reject     Save
                 ↓
        Beneficiary + Face Embedding
                 ↓
              PostgreSQL
```

---

# 🔒 Security & Privacy

This project processes potentially sensitive beneficiary and biometric information.

For development and GitHub publication:

* Do not upload real beneficiary information.
* Do not upload real beneficiary photographs.
* Do not commit `.env`.
* Do not expose Groq API keys.
* Do not expose PostgreSQL passwords.
* Do not commit database dumps containing real users.
* Keep uploaded images outside the public repository.
* Use dummy/test data when demonstrating the project.

Biometric information such as face embeddings should be handled with appropriate security and privacy controls in any production deployment.

---

# 🚧 Current Limitations

This project is currently under development.

Some limitations include:

* Face embeddings are currently stored as JSONB rather than a specialized vector database.
* Duplicate detection depends on the selected similarity threshold.
* The application currently uses in-memory LangGraph checkpointing.
* Authentication and authorization are not implemented.
* Production-level encryption and access controls are not included.
* The current project is intended primarily as an AI/ML prototype.

---

# 🔮 Future Improvements

Possible future improvements include:

* [ ] PostgreSQL `pgvector` integration
* [ ] More robust face verification
* [ ] User authentication
* [ ] Admin dashboard
* [ ] Beneficiary search and management
* [ ] Document/OCR processing
* [ ] Government scheme recommendation
* [ ] Eligibility assessment
* [ ] Production deployment
* [ ] Secure cloud storage for images
* [ ] Role-based access control
* [ ] Advanced analytics and reporting

---

# 🎯 Project Objective

The goal of AI-Khidmat is to combine **Generative AI, conversational workflows, structured data extraction, database systems, and face verification** to create a more efficient beneficiary registration process.

Instead of a conventional form-based workflow:

```text
Traditional System

User → Form → Manual Entry → Database
```

AI-Khidmat provides:

```text
AI-Khidmat

User
  ↓
Conversation
  ↓
AI Information Extraction
  ↓
Missing Information Detection
  ↓
Structured Registration
  ↓
Face Verification
  ↓
Duplicate Detection
  ↓
PostgreSQL
```

---

# 👨‍💻 Author

**Syed Arshaan Hussain**

B.Tech Computer Science Engineering
AI/ML & Generative AI

GitHub: **Arshaan-77**

---

## ⭐ Project Status

🚧 **Under Development**

AI-Khidmat is an AI/ML project focused on building an intelligent and conversational beneficiary registration and verification system.

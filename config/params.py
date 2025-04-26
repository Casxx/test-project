from dotenv import load_dotenv
from llama_index import Settings
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure Azure OpenAI
Settings.llm = OpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_base=AZURE_OPENAI_ENDPOINT,
    api_type="azure",
    api_version="2023-05-15"
)
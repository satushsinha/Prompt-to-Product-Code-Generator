from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()


llm = ChatGroq(model="openai/gpt-oss-120b")

#resp = llm.invoke("What is the today's date")

from pydantic import BaseModel


user_prompt = "Create a simple calculator web application"

prompt = f"""
You are the PLANNER agent. Convert the user prompt into a COMPLETE engineering project plan

User request: {user_prompt}
"""

class Plan(BaseModel):
  pass

resp = llm.with_structured_output(Plan).invoke(prompt)
print(resp)
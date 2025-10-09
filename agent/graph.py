from langchain_groq import ChatGroq

from dotenv import load_dotenv
from prompts import *
from states import *
load_dotenv()

from langgraph.graph import StateGraph
from langgraph.constants import END
from langchain.globals import set_verbose, set_debug


set_debug(True)
set_verbose(True)

llm = ChatGroq(model="openai/gpt-oss-120b")

#resp = llm.invoke("What is the today's date")

def planner_agent(state: dict) -> dict:
    """Converts user prompt into a structured Plan."""
    user_prompt = state["user_prompt"]
    resp = llm.with_structured_output(Plan).invoke(
        planner_prompt(user_prompt)
    )
    if resp is None:
        raise ValueError("Planner did not return a valid response.")
    return {"plan": resp}


def architect_agent(state: dict) -> dict:
    """Creates TaskPlan from Plan."""
    plan: Plan = state["plan"]
    resp = llm.with_structured_output(TaskPlan).invoke(
        architect_prompt(plan=plan.model_dump_json())
    )
    if resp is None:
        raise ValueError("Planner did not return a valid response.")

    resp.plan = plan
    print(resp.model_dump_json())
    return {"task_plan": resp}

graph = StateGraph(dict)
graph.add_node("planner", planner_agent)
graph.add_node("architect", architect_agent)
graph.add_edge("planner", "architect")
graph.set_entry_point("planner")

agent = graph.compile()

if __name__ == "__main__":
  user_prompt = "Create a simple calculator web application"
  result = agent.invoke({"user_prompt": user_prompt})
  print(result)
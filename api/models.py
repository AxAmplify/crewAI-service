from pydantic import BaseModel
from typing import List, Optional

class AgentConfig(BaseModel):
    role: str
    goal: str
    backstory: str
    verbose: bool = True

class TaskConfig(BaseModel):
    description: str
    expected_output: str
    agent_role: str

class CrewRequest(BaseModel):
    agents: List[AgentConfig]
    tasks: List[TaskConfig]
    verbose: bool = True

class CrewResponse(BaseModel):
    result: str
    status: str
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from crewai import Agent, Task, Crew
from api.models import CrewRequest, CrewResponse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AxAmplify CrewAI Service",
    description="API wrapper for CrewAI agent orchestration",
    version="0.1.0"
)

# Configure CORS for your Next.js app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your Next.js app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "service": "AxAmplify CrewAI Service",
        "status": "running",
        "version": "0.1.0"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/crew/run", response_model=CrewResponse)
async def run_crew(request: CrewRequest):
    """
    Run a crew of AI agents with specified tasks
    """
    try:
        logger.info(f"Received crew request with {len(request.agents)} agents and {len(request.tasks)} tasks")
        
        # Create agents from request
        agents = []
        agent_map = {}
        
        for agent_config in request.agents:
            agent = Agent(
                role=agent_config.role,
                goal=agent_config.goal,
                backstory=agent_config.backstory,
                verbose=agent_config.verbose
            )
            agents.append(agent)
            agent_map[agent_config.role] = agent
        
        # Create tasks from request
        tasks = []
        for task_config in request.tasks:
            if task_config.agent_role not in agent_map:
                raise HTTPException(
                    status_code=400,
                    detail=f"Agent role '{task_config.agent_role}' not found"
                )
            
            task = Task(
                description=task_config.description,
                expected_output=task_config.expected_output,
                agent=agent_map[task_config.agent_role]
            )
            tasks.append(task)
        
        # Create and run the crew
        crew = Crew(
            agents=agents,
            tasks=tasks,
            verbose=request.verbose
        )
        
        logger.info("Starting crew execution...")
        result = crew.kickoff()
        logger.info("Crew execution completed")
        
        return CrewResponse(
            result=str(result),
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Error running crew: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Example endpoint for a simple single-agent task
@app.post("/api/agent/simple")
async def run_simple_agent(
    role: str,
    goal: str,
    task_description: str
):
    """
    Run a simple single-agent task
    """
    try:
        agent = Agent(
            role=role,
            goal=goal,
            backstory=f"An AI assistant specialized in {role}",
            verbose=True
        )
        
        task = Task(
            description=task_description,
            expected_output="A detailed response to the task",
            agent=agent
        )
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=True
        )
        result = crew.kickoff()
        
        return {
            "result": str(result),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
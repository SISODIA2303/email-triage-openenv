from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from enum import Enum

class EmailCategory(str,Enum):
    BILLING="billing"
    SUPPORT="support"
    SPAM="spam"
    SALES="sales"
    HR="hr"
    GENERAL="general"

class Priority(str,Enum):
    P1="P1" #Crticial
    P2="P2" #High
    P3="P3" #Medium
    P4="P4" #Low

class RoutingTeam(str,Enum):
    BILLING_TEAM="billing_team"
    SUPPORT_TEAM="support_team"
    SALES_TEAM="sales_team"
    HR_TEAM="hr_team"
    SPAM_FILTER="spam_filter"
    GENERAL_TEAM="general_team"

class Email(BaseModel):
    email_id:str
    sender:str
    subject:str
    body:str
    timestamp:str

class Observation(BaseModel):
    task_id:int=Field(..., description="Current task: 1,2 or 3")
    current_email:Email=Field(..., description="The email to process")
    emails_remaining:int=Field(..., description="How many emails left in queue")
    current_score:float=Field(0.0, description="Score so far this episode")
    step_number:int=Field(0, description="Current step in episode")
    instruction:str=Field(..., description="What the agent must do")

class Action(BaseModel):
    category:EmailCategory=Field(..., description="Email category classification")
    priority:Optional[Priority]=Field(None, description="Required for task 2 and 3")
    routing_team:Optional[RoutingTeam]=Field(None, description="Required for task 2 and 3")
    reply_draft:Optional[str]=Field(None, description="Required for task 3 only")

class Reward(BaseModel):
    value:float=Field(..., description="Reward for this step, range -1.0 to 1.0")
    breakdown:dict=Field(default_factory=dict, description="Per-component scores")
    feedback:str=Field("", description="Human-readable feedback on the action")
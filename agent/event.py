from dataclasses import dataclass,field
from enum import Enum
from typing import Any

@dataclass
class AgentEventType(str,Enum):
    AGENT_START="agent_start"
    AGENT_END="agnet_end"
    AGENT_ERROR="agent_error"

    TEXT_DELTE='text_delta'
    TEXT_COMPLETE='text_complete'

    

class AgentEvent:
    type:AgentEventType
    data: dict[str,Any] = field(default=dict)


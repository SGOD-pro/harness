import os
from dataclasses import dataclass, field

@dataclass
class Config:
    cwd: str = field(default_factory=os.getcwd)
    developer_instructions: str | None = None
    user_instructions: str | None = None

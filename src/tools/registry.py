from tools.base import Tool,ToolResult,ToolInvocation
import logging
from typing import Any
from pathlib import Path
from tools.builtin import ReadFileTool,get_all_builtin_tools

logger = logging.getLogger(__name__)

class ToolRegistry:
    def __init__ (self):
        self._tools:dict[str,Tool]={}

    def register(self,tool:Tool) -> None:

        if tool.name in self._tools:
            logger.warning(f"Over wrting existing tool: {tool.name}")

        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def get(self,name:str) -> Tool | None:
        if name in self._tools:
            return self._tools[name]

        return None
    def unregister(self,name:str)-> bool:
        if name in self._tools:
            del self._tools[name]

            return True
        
        return False

    def get_tools(self) -> list[Tool]:

        tools: list[Tool]=[]
        for tool in self._tools.values():
            tools.append(tool)

        return tools

    def get_schemas(self) -> list[dict[str,Any]]:
        return [tool.to_openai_schema() for tool in self.get_tools()]

    async def invoke(self,name:str,params:dict[str,Any],cwd: Path ) -> ToolResult:
        tool=self.get(name)
        if tool is None:
            return ToolResult.error_result(
                output=f"Unknown tool: {name}",
                metadata={'tool_name':name}
            )
        validation_error=tool.validate_params(params)
        if validation_error:
            return ToolResult.error_result(
                output=f"Invalid parameteres: {'.'.join(validation_error)}",
                metadata={'tool_name':name,'validation errors':validation_error}
            )

        invocation=ToolInvocation(params=params,cwd=cwd)
        try:
            result = await tool.execute(invocation)
        except Exception as e:
            logger.exception(f"Tool {name} raised unexpected error")
            result =ToolResult.error_result(
                f"Intenal error: {str(e)}",
                metadata={
                    "tool_name":name
                }
            )
        
        return result



def create_default_registry()->ToolRegistry:
    registry = ToolRegistry()
    for tool_class in get_all_builtin_tools():
        registry.register(tool_class())

    return registry



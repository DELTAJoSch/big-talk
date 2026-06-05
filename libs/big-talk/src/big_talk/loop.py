import asyncio
from collections import defaultdict
from typing import Sequence

from .tool import Tool
from .tool_execution import ToolExecutionContext, ToolExecutionHandler
from .exceptions import BatchSuspendedException, SuspensionError
from .message import OutputMessage, ToolUse, Message, ToolResult


def extract_tool_uses(message: OutputMessage) -> list[tuple[str, ToolUse]]:
    parent_id = message['id']

    tool_uses_by_parent: list[tuple[str, ToolUse]] = []
    tool_uses = [b for b in message['content'] if b['type'] == 'tool_use']
    for tool_use in tool_uses:
        tool_uses_by_parent.append((parent_id, tool_use))

    return tool_uses_by_parent


async def use_tools(tool_uses_by_parent: list[tuple[str, ToolUse]], messages: Sequence[Message], tools: Sequence[Tool],
                    iteration: int, tool_execution_handler: ToolExecutionHandler) -> dict[str, list[ToolResult]]:
    tool_uses = [tu for _, tu in tool_uses_by_parent]

    tool_execution_ctx = ToolExecutionContext(
        tool_uses=tool_uses,
        tools=tools,
        messages=messages,
        iteration=iteration
    )

    tool_tasks = await tool_execution_handler(tool_execution_ctx)

    # Only SuspensionError should ever pop up here
    raw_tool_results = await asyncio.gather(*tool_tasks, return_exceptions=True)

    suspension_results: dict[str, list[SuspensionError]] = defaultdict(list)
    results_by_parent: dict[str, list[ToolResult]] = defaultdict(list)

    for (parent_id, _), result in zip(tool_uses_by_parent, raw_tool_results):
        if isinstance(result, SuspensionError):
            suspension_results[parent_id].append(result)
        else:
            results_by_parent[parent_id].append(result)

    if len(suspension_results) > 0:
        raise BatchSuspendedException(suspensions=suspension_results, partial_results=results_by_parent)

    return dict(results_by_parent)

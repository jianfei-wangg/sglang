import json
import logging
import re
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.ebnf_composer import EBNFComposer
from sglang.srt.function_call.utils import _is_complete_json

logger = logging.getLogger(__name__)


class Dots2Detector(BaseFormatDetector):
    """
    Detector for DOTS-2 model function call format.

    The DOTS-2 format uses XML-style tags to delimit function calls
    with JSON objects for function name and arguments.

    Format Structure:
    ```
    <dots_function_call>
    {"name": <function-name>, "arguments": <args-json-object>}
    </dots_function_call>
    ```

    Examples:
    ```
    <dots_function_call>
    {"name": "get_current_weather", "arguments": {"location": "Tokyo"}}
    </dots_function_call>
    ```

    Key Components:
    - Function Call Section: Wrapped between `<dots_function_call>` and `</dots_function_call>`
    - JSON Object: Contains "name" and "arguments" fields
    - Supports multiple sequential tool calls
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<dots_function_call>"
        self.eot_token = "</dots_function_call>"
        self.func_call_regex = r"<dots_function_call>\s*(.*?)\s*</dots_function_call>"
        self._last_arguments = ""
        self.current_tool_id = -1

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a DOTS-2 format tool call."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        # Find the start of the first tool call
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text

        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        # Find all complete tool calls
        match_results = re.findall(self.func_call_regex, text, re.DOTALL)
        calls = []

        try:
            for match_result in match_results:
                # Parse the JSON content
                func_call_json = json.loads(match_result.strip())
                func_name = func_call_json.get("name")
                func_args = func_call_json.get("arguments", {})

                # Construct match_result for parse_base_json
                match_result_obj = {"name": func_name, "parameters": func_args}
                calls.extend(self.parse_base_json(match_result_obj, tools))

            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # Return the normal text if parsing fails
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing tool calls for DOTS-2 format.
        """
        self._buffer += new_text
        current_text = self._buffer

        # Check if we have a tool call
        has_tool_call = self.bot_token in current_text

        if not has_tool_call:
            self._buffer = ""
            # Remove any end tokens that might appear in normal text
            for e_token in [self.eot_token]:
                if e_token in new_text:
                    new_text = new_text.replace(e_token, "")
            return StreamingParseResult(normal_text=new_text)

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        calls: list[ToolCallItem] = []

        try:
            # Look for the pattern: <dots_function_call>...JSON...</dots_function_call>
            start_idx = current_text.find(self.bot_token)
            if start_idx == -1:
                return StreamingParseResult(normal_text="", calls=calls)

            # Find the end token
            end_idx = current_text.find(self.eot_token, start_idx)

            if end_idx != -1:
                # Complete tool call found
                json_content = current_text[
                    start_idx + len(self.bot_token) : end_idx
                ].strip()

                try:
                    func_call_json = json.loads(json_content)
                    func_name = func_call_json.get("name")
                    func_args = func_call_json.get("arguments", {})

                    # Initialize state if this is the first tool call
                    if self.current_tool_id == -1:
                        self.current_tool_id = 0
                        self.prev_tool_call_arr = []
                        self.streamed_args_for_tool = [""]

                    # Ensure we have enough entries in our tracking arrays
                    while len(self.prev_tool_call_arr) <= self.current_tool_id:
                        self.prev_tool_call_arr.append({})
                    while len(self.streamed_args_for_tool) <= self.current_tool_id:
                        self.streamed_args_for_tool.append("")

                    # Send the complete tool call
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters=json.dumps(func_args) if func_args else "",
                        )
                    )

                    # Store the tool call info
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": func_name,
                        "arguments": func_args,
                    }

                    # Remove the completed tool call from buffer
                    self._buffer = current_text[end_idx + len(self.eot_token) :]

                    # Reset for next tool call
                    self.current_tool_id += 1
                    self._last_arguments = ""
                    self.current_tool_name_sent = False

                    return StreamingParseResult(normal_text="", calls=calls)

                except json.JSONDecodeError:
                    # JSON is not complete yet, continue buffering
                    pass
            else:
                # Still waiting for the end token
                # Try to parse partial JSON content for streaming
                json_start = start_idx + len(self.bot_token)
                partial_json = current_text[json_start:].strip()

                if partial_json and _is_complete_json(partial_json):
                    try:
                        func_call_json = json.loads(partial_json)
                        func_name = func_call_json.get("name")
                        func_args = func_call_json.get("arguments", {})

                        # Initialize state if this is the first tool call
                        if self.current_tool_id == -1:
                            self.current_tool_id = 0
                            self.prev_tool_call_arr = []
                            self.streamed_args_for_tool = [""]
                            self.current_tool_name_sent = False

                        # Ensure we have enough entries in our tracking arrays
                        while len(self.prev_tool_call_arr) <= self.current_tool_id:
                            self.prev_tool_call_arr.append({})
                        while len(self.streamed_args_for_tool) <= self.current_tool_id:
                            self.streamed_args_for_tool.append("")

                        if not self.current_tool_name_sent:
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name=func_name,
                                    parameters="",
                                )
                            )
                            self.current_tool_name_sent = True
                            self.prev_tool_call_arr[self.current_tool_id] = {
                                "name": func_name,
                                "arguments": {},
                            }

                        # Stream arguments incrementally
                        args_str = json.dumps(func_args) if func_args else ""
                        argument_diff = (
                            args_str[len(self._last_arguments) :]
                            if args_str.startswith(self._last_arguments)
                            else args_str
                        )

                        if argument_diff:
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name=None,
                                    parameters=argument_diff,
                                )
                            )
                            self._last_arguments += argument_diff
                            self.streamed_args_for_tool[
                                self.current_tool_id
                            ] += argument_diff

                            # Update stored arguments
                            self.prev_tool_call_arr[self.current_tool_id][
                                "arguments"
                            ] = func_args

                    except json.JSONDecodeError:
                        pass

            return StreamingParseResult(normal_text="", calls=calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult(normal_text=current_text)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='{"name": "' + name + '", "arguments": ',
            end="}",
            trigger='{"name": "' + name + '", "arguments": ',
        )

    def build_ebnf(self, tools: List[Tool]):
        return EBNFComposer.build_ebnf(
            tools,
            sequence_start_token=self.bot_token,
            sequence_end_token=self.eot_token,
            tool_call_separator="",
            call_rule_fmt='"{\\"name\\": \\"{name}\\", \\"arguments\\": "{arguments_rule}"}"',
            function_format="json",
        )

import pytest
from unittest.mock import MagicMock, patch

from big_talk import SystemMessage, UserMessage, ToolMessage
from big_talk.llm.anthropic import AnthropicProvider
from big_talk.message import Message, ToolResult, AssistantMessageDelta


# Mock the Anthropic stream events
class MockStream:
    def __init__(self, events):
        self.events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.events:
            raise StopAsyncIteration
        return self.events.pop(0)


@pytest.fixture
def anthropic_provider():
    # Patch the import inside the class or mock the client directly
    provider = AnthropicProvider()
    provider._client = MagicMock()
    return provider


@pytest.mark.asyncio
async def test_anthropic_message_conversion(anthropic_provider):
    """Test that System Tool Results are moved to User role."""
    messages = [
        ToolMessage(role="tool", content=[
            ToolResult(type="tool_result", tool_use_id="123", result="Success", is_error=False, parent_id="u1")
        ], metadata=None),
        UserMessage(role="user", content="Hello", id="u1")
    ]

    system, converted, _ = anthropic_provider._convert_messages(messages)

    # Verify System message became a User message with tool_result
    assert converted[0]["role"] == "user"
    assert converted[0]["content"][0]["type"] == "tool_result"
    assert converted[0]["content"][0]["tool_use_id"] == "123"


@pytest.mark.asyncio
async def test_anthropic_streaming(anthropic_provider):
    """Test deltas vs aggregate messages."""

    # Mock events coming from Anthropic
    mock_events = [
        # 1. Text Delta
        MagicMock(type='content_block_stop', content_block=MagicMock(type='text', text="Hello")),
        # 2. Tool Use Delta
        MagicMock(type='content_block_stop',
                  content_block=MagicMock(type='tool_use', id='t1', name='search', input={})),
        # 3. Message Stop (End of Stream)
        MagicMock(type='message_stop')
    ]

    anthropic_provider._client.messages.stream.return_value = MockStream(mock_events)

    stream = anthropic_provider.stream("claude-3", [UserMessage(role="user", content="Hi", id="u1")], tools=[])

    results = [msg async for msg in stream]

    # Expect: 2 Deltas + 1 Aggregate
    assert len(results) == 3

    # Check Delta 1
    assert results[0]["is_aggregate"] is False
    assert results[0]["content"][0]["text"] == "Hello"

    # Check Delta 2
    assert results[1]["is_aggregate"] is False
    assert results[1]["content"][0]["type"] == "tool_use"

    # Check Aggregate
    assert results[2]["is_aggregate"] is True
    assert len(results[2]["content"]) == 2  # Should contain both text and tool


@pytest.mark.asyncio
async def test_anthropic_stream_content_block_delta(anthropic_provider):
    """content_block_delta events produce AssistantMessageDelta objects."""
    text_delta = MagicMock()
    text_delta.type = 'text_delta'
    text_delta.text = 'Hello'

    mock_events = [
        MagicMock(type='content_block_delta', delta=text_delta),
        MagicMock(type='content_block_stop', content_block=MagicMock(type='text', text="Hello")),
        MagicMock(type='message_stop'),
    ]
    anthropic_provider._client.messages.stream.return_value = MockStream(mock_events)

    stream = anthropic_provider.stream("claude-3", [UserMessage(role="user", content="Hi", id="u1")], tools=[])
    results = [msg async for msg in stream]

    delta_msg = results[0]
    assert delta_msg['role'] == 'assistant'
    assert delta_msg['is_aggregate'] is False
    assert delta_msg['type'] == 'text'
    assert delta_msg['delta'] == 'Hello'


def test_anthropic_to_delta_text():
    delta = MagicMock()
    delta.type = 'text_delta'
    delta.text = 'hello'
    result = AnthropicProvider._to_delta(delta, 'msg-1', 'parent-1')
    assert result['type'] == 'text'
    assert result['delta'] == 'hello'
    assert result['id'] == 'msg-1'
    assert result['parent_id'] == 'parent-1'
    assert result['is_aggregate'] is False


def test_anthropic_to_delta_tool_params():
    delta = MagicMock()
    delta.type = 'input_json_delta'
    delta.partial_json = '{"x":'
    result = AnthropicProvider._to_delta(delta, 'msg-1', 'parent-1')
    assert result['type'] == 'tool_use_params'
    assert result['delta'] == '{"x":'


def test_anthropic_to_delta_thinking():
    delta = MagicMock()
    delta.type = 'thinking_delta'
    delta.thinking = 'let me think...'
    result = AnthropicProvider._to_delta(delta, 'msg-1', 'parent-1')
    assert result['type'] == 'thinking'
    assert result['delta'] == 'let me think...'


def test_anthropic_to_delta_signature():
    delta = MagicMock()
    delta.type = 'signature_delta'
    delta.signature = 'abc123'
    result = AnthropicProvider._to_delta(delta, 'msg-1', 'parent-1')
    assert result['type'] == 'signature'
    assert result['delta'] == 'abc123'


def test_anthropic_to_delta_unknown_returns_none():
    delta = MagicMock()
    delta.type = 'unknown_delta_type'
    result = AnthropicProvider._to_delta(delta, 'msg-1', 'parent-1')
    assert result is None


def test_anthropic_init_kwargs():
    """Test that kwargs are passed to the underlying AsyncAnthropic client."""

    # We patch the class where it is IMPORTED, not where it is defined.
    # Since AnthropicProvider imports it inside __init__, we need to patch
    # 'anthropic.AsyncAnthropic' globally or mock the module.

    with patch("anthropic.AsyncAnthropic") as MockClient:
        # Initialize with custom args
        provider = AnthropicProvider(
            api_key="my-secret-key",
            max_retries=5,
            timeout=20.0
        )

        # Verify the client was initialized with these args
        MockClient.assert_called_once_with(
            api_key="my-secret-key",
            max_retries=5,
            timeout=20.0
        )

        # Verify the instance is stored
        assert provider._client == MockClient.return_value

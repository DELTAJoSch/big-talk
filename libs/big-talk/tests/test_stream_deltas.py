import pytest

from big_talk import AssistantMessage, AssistantMessageDelta, Text, ToolUse, UserMessage


@pytest.mark.asyncio
async def test_stream_deltas_param_forwarded_to_provider(bigtalk, create_provider, simple_message):
    provider = create_provider()
    bigtalk.add_provider("test", lambda: provider)

    async for _ in bigtalk.stream("test/m", [simple_message], stream_deltas=True):
        pass

    assert provider.stream_calls[0]["kwargs"]["stream_deltas"] is True


@pytest.mark.asyncio
async def test_stream_deltas_default_false(bigtalk, create_provider, simple_message):
    provider = create_provider()
    bigtalk.add_provider("test", lambda: provider)

    async for _ in bigtalk.stream("test/m", [simple_message]):
        pass

    assert provider.stream_calls[0]["kwargs"]["stream_deltas"] is False


@pytest.mark.asyncio
async def test_delta_messages_are_yielded(bigtalk, create_provider, simple_message):
    provider = create_provider(responses=["hello", "world"])
    bigtalk.add_provider("test", lambda: provider)

    results = [m async for m in bigtalk.stream("test/m", [simple_message], stream_deltas=True)]

    deltas = [m for m in results if 'delta' in m]
    aggregates = [m for m in results if m.get('is_aggregate')]

    assert len(deltas) == 2
    assert len(aggregates) == 2
    assert deltas[0]['delta'] == "hello"
    assert deltas[0]['type'] == 'text'
    assert deltas[0]['is_aggregate'] is False
    assert deltas[0]['role'] == 'assistant'


@pytest.mark.asyncio
async def test_aggregate_still_yielded_with_deltas(bigtalk, create_provider, simple_message):
    provider = create_provider(responses=["hello"])
    bigtalk.add_provider("test", lambda: provider)

    results = [m async for m in bigtalk.stream("test/m", [simple_message], stream_deltas=True)]

    assert any(m.get('is_aggregate') for m in results)
    assert any('delta' in m for m in results)


@pytest.mark.asyncio
async def test_delta_messages_not_added_to_history(bigtalk, create_provider, simple_message):
    """Deltas (is_aggregate=False) must not appear in the history passed to the next iteration."""
    tool_use_msg = AssistantMessage(
        role="assistant",
        content=[ToolUse(type="tool_use", id="call_1", name="my_tool", params={}, metadata=None)],
        id="msg_tool",
        parent_id="parent-id",
        is_aggregate=True,
    )
    final_msg = AssistantMessage(
        role="assistant",
        content=[Text(type="text", text="done")],
        id="msg_final",
        parent_id="parent-id",
        is_aggregate=True,
    )

    provider = create_provider()
    captured_message_lists = []
    call_count = [0]

    async def mock_stream(model, messages, **kwargs):
        captured_message_lists.append(list(messages))
        if call_count[0] == 0:
            call_count[0] += 1
            yield AssistantMessageDelta(
                type='text', id='msg_tool', role='assistant',
                delta='about to use tool', parent_id='parent-id', is_aggregate=False,
            )
            yield tool_use_msg
        else:
            yield final_msg

    provider.stream = mock_stream
    bigtalk.add_provider("test", lambda: provider)

    async def my_tool():
        return "result"

    [m async for m in bigtalk.stream("test/m", [simple_message], tools=[my_tool], stream_deltas=True)]

    # Iteration 1 history: [user_msg, tool_use_msg, tool_result_msg] — no delta
    assert len(captured_message_lists) == 2
    assert len(captured_message_lists[1]) == 3
    assert all('delta' not in m for m in captured_message_lists[1])


@pytest.mark.asyncio
async def test_stream_deltas_accessible_on_stream_context(bigtalk, create_provider, simple_message):
    provider = create_provider()
    bigtalk.add_provider("test", lambda: provider)

    captured = []

    async def inspect_middleware(handler, ctx, **kwargs):
        captured.append(ctx.stream_deltas)
        async for msg in handler(ctx, **kwargs):
            yield msg

    bigtalk.streaming.use(inspect_middleware)

    async for _ in bigtalk.stream("test/m", [simple_message], stream_deltas=True):
        pass

    assert captured == [True]


@pytest.mark.asyncio
async def test_stream_deltas_accessible_on_iteration_context(bigtalk, create_provider, simple_message):
    provider = create_provider()
    bigtalk.add_provider("test", lambda: provider)

    captured = []

    async def inspect_iteration_middleware(handler, ctx, **kwargs):
        captured.append(ctx.stream_deltas)
        async for msg in handler(ctx, **kwargs):
            yield msg

    bigtalk.stream_iteration.use(inspect_iteration_middleware)

    async for _ in bigtalk.stream("test/m", [simple_message], stream_deltas=False):
        pass

    assert captured == [False]

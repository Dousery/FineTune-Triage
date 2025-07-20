import asyncio
import asyncio.events
import collections.abc
import enum
import modal._functions
import modal._utils.async_utils
import modal.client
import modal.functions
import modal.retries
import modal_proto.api_pb2
import typing
import typing_extensions

class _SynchronizedQueue:
    async def init(self): ...
    async def put(self, item): ...
    async def get(self): ...

SUPERSELF = typing.TypeVar("SUPERSELF", covariant=True)

class SynchronizedQueue:
    def __init__(self, /, *args, **kwargs): ...

    class __init_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, /): ...
        async def aio(self, /): ...

    init: __init_spec[typing_extensions.Self]

    class __put_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, /, item): ...
        async def aio(self, /, item): ...

    put: __put_spec[typing_extensions.Self]

    class __get_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, /): ...
        async def aio(self, /): ...

    get: __get_spec[typing_extensions.Self]

class _OutputValue:
    value: typing.Any

    def __init__(self, value: typing.Any) -> None: ...
    def __repr__(self): ...
    def __eq__(self, other): ...

def _map_invocation(
    function: modal._functions._Function,
    raw_input_queue: _SynchronizedQueue,
    client: modal.client._Client,
    order_outputs: bool,
    return_exceptions: bool,
    count_update_callback: typing.Optional[collections.abc.Callable[[int, int], None]],
    function_call_invocation_type: int,
): ...
def _map_helper(
    self: modal.functions.Function,
    async_input_gen: typing.AsyncGenerator[typing.Any, None],
    kwargs={},
    order_outputs: bool = True,
    return_exceptions: bool = False,
) -> typing.AsyncGenerator[typing.Any, None]: ...
def _map_async(
    self: modal.functions.Function,
    *input_iterators: typing.Union[typing.Iterable[typing.Any], typing.AsyncIterable[typing.Any]],
    kwargs={},
    order_outputs: bool = True,
    return_exceptions: bool = False,
) -> typing.AsyncGenerator[typing.Any, None]: ...
def _starmap_async(
    self,
    input_iterator: typing.Union[
        typing.Iterable[typing.Sequence[typing.Any]], typing.AsyncIterable[typing.Sequence[typing.Any]]
    ],
    *,
    kwargs={},
    order_outputs: bool = True,
    return_exceptions: bool = False,
) -> typing.AsyncIterable[typing.Any]: ...
async def _for_each_async(self, *input_iterators, kwargs={}, ignore_exceptions: bool = False) -> None: ...
def _map_sync(
    self, *input_iterators, kwargs={}, order_outputs: bool = True, return_exceptions: bool = False
) -> modal._utils.async_utils.AsyncOrSyncIterable: ...
async def _spawn_map_async(self, *input_iterators, kwargs={}) -> None: ...
def _spawn_map_sync(self, *input_iterators, kwargs={}) -> None: ...
def _for_each_sync(self, *input_iterators, kwargs={}, ignore_exceptions: bool = False): ...
def _starmap_sync(
    self,
    input_iterator: typing.Iterable[typing.Sequence[typing.Any]],
    *,
    kwargs={},
    order_outputs: bool = True,
    return_exceptions: bool = False,
) -> modal._utils.async_utils.AsyncOrSyncIterable: ...

class _MapItemState(enum.Enum):
    # The input is being sent the server with a PutInputs request, but the response has not been received yet.
    SENDING = 1
    # A call to either PutInputs or FunctionRetry has completed, and we are waiting to receive the output.
    WAITING_FOR_OUTPUT = 2
    # The input is on the retry queue, and waiting for its delay to expire.
    WAITING_TO_RETRY = 3
    # The input is being sent to the server with a FunctionRetry request, but the response has not been received yet.
    RETRYING = 4
    # The output has been received and was either successful, or failed with no more retries remaining.
    COMPLETE = 5

class _OutputType(enum.Enum):
    SUCCESSFUL_COMPLETION = 1
    FAILED_COMPLETION = 2
    RETRYING = 3
    ALREADY_COMPLETE_DUPLICATE = 4
    STALE_RETRY_DUPLICATE = 5
    NO_CONTEXT_DUPLICATE = 6

class _MapItemContext:
    state: _MapItemState
    input: modal_proto.api_pb2.FunctionInput
    retry_manager: modal.retries.RetryManager
    sync_client_retries_enabled: bool
    input_id: asyncio.Future
    input_jwt: asyncio.Future
    previous_input_jwt: typing.Optional[str]
    _event_loop: asyncio.events.AbstractEventLoop

    def __init__(
        self,
        input: modal_proto.api_pb2.FunctionInput,
        retry_manager: modal.retries.RetryManager,
        sync_client_retries_enabled: bool,
    ): ...
    def handle_put_inputs_response(self, item: modal_proto.api_pb2.FunctionPutInputsResponseItem): ...
    async def handle_get_outputs_response(
        self,
        item: modal_proto.api_pb2.FunctionGetOutputsItem,
        now_seconds: int,
        function_call_invocation_type: int,
        retry_queue: modal._utils.async_utils.TimestampPriorityQueue,
    ) -> _OutputType: ...
    async def prepare_item_for_retry(self) -> modal_proto.api_pb2.FunctionRetryInputsItem: ...
    def handle_retry_response(self, input_jwt: str): ...

class _MapItemsManager:
    def __init__(
        self,
        retry_policy: modal_proto.api_pb2.FunctionRetryPolicy,
        function_call_invocation_type: int,
        retry_queue: modal._utils.async_utils.TimestampPriorityQueue,
        sync_client_retries_enabled: bool,
        max_inputs_outstanding: int,
    ): ...
    async def add_items(self, items: list[modal_proto.api_pb2.FunctionPutInputsItem]): ...
    async def prepare_items_for_retry(
        self, retriable_idxs: list[int]
    ) -> list[modal_proto.api_pb2.FunctionRetryInputsItem]: ...
    def get_input_jwts_waiting_for_output(self) -> list[str]: ...
    def _remove_item(self, item_idx: int): ...
    def get_item_context(self, item_idx: int) -> _MapItemContext: ...
    def handle_put_inputs_response(self, items: list[modal_proto.api_pb2.FunctionPutInputsResponseItem]): ...
    def handle_retry_response(self, input_jwts: list[str]): ...
    async def handle_get_outputs_response(
        self, item: modal_proto.api_pb2.FunctionGetOutputsItem, now_seconds: int
    ) -> _OutputType: ...
    def __len__(self): ...

import collections.abc
import modal._partial_function
import modal.functions
import typing
import typing_extensions

class PartialFunction(
    typing.Generic[
        modal._partial_function.P, modal._partial_function.ReturnType, modal._partial_function.OriginalReturnType
    ]
):
    raw_f: typing.Optional[collections.abc.Callable[modal._partial_function.P, modal._partial_function.ReturnType]]
    user_cls: typing.Optional[type]
    flags: modal._partial_function._PartialFunctionFlags
    params: modal._partial_function._PartialFunctionParams
    registered: bool

    def __init__(
        self,
        obj: typing.Union[
            collections.abc.Callable[modal._partial_function.P, modal._partial_function.ReturnType], type
        ],
        flags: modal._partial_function._PartialFunctionFlags,
        params: modal._partial_function._PartialFunctionParams,
    ): ...
    def stack(
        self,
        flags: modal._partial_function._PartialFunctionFlags,
        params: modal._partial_function._PartialFunctionParams,
    ) -> typing_extensions.Self: ...
    def validate_flag_composition(self) -> None: ...
    def validate_obj_compatibility(
        self, decorator_name: str, require_sync: bool = False, require_nullary: bool = False
    ) -> None: ...
    def _get_raw_f(self) -> collections.abc.Callable[modal._partial_function.P, modal._partial_function.ReturnType]: ...
    def _is_web_endpoint(self) -> bool: ...
    def __get__(
        self, obj, objtype=None
    ) -> modal.functions.Function[
        modal._partial_function.P, modal._partial_function.ReturnType, modal._partial_function.OriginalReturnType
    ]: ...
    def __del__(self): ...

def method(
    _warn_parentheses_missing=None, *, is_generator: typing.Optional[bool] = None
) -> modal._partial_function._MethodDecoratorType: ...
def web_endpoint(
    _warn_parentheses_missing=None,
    *,
    method: str = "GET",
    label: typing.Optional[str] = None,
    docs: bool = False,
    custom_domains: typing.Optional[collections.abc.Iterable[str]] = None,
    requires_proxy_auth: bool = False,
) -> collections.abc.Callable[
    [
        typing.Union[
            PartialFunction[
                modal._partial_function.P, modal._partial_function.ReturnType, modal._partial_function.ReturnType
            ],
            collections.abc.Callable[modal._partial_function.P, modal._partial_function.ReturnType],
        ]
    ],
    PartialFunction[modal._partial_function.P, modal._partial_function.ReturnType, modal._partial_function.ReturnType],
]: ...
def fastapi_endpoint(
    _warn_parentheses_missing=None,
    *,
    method: str = "GET",
    label: typing.Optional[str] = None,
    custom_domains: typing.Optional[collections.abc.Iterable[str]] = None,
    docs: bool = False,
    requires_proxy_auth: bool = False,
) -> collections.abc.Callable[
    [
        typing.Union[
            PartialFunction[
                modal._partial_function.P, modal._partial_function.ReturnType, modal._partial_function.ReturnType
            ],
            collections.abc.Callable[modal._partial_function.P, modal._partial_function.ReturnType],
        ]
    ],
    PartialFunction[modal._partial_function.P, modal._partial_function.ReturnType, modal._partial_function.ReturnType],
]: ...
def asgi_app(
    _warn_parentheses_missing=None,
    *,
    label: typing.Optional[str] = None,
    custom_domains: typing.Optional[collections.abc.Iterable[str]] = None,
    requires_proxy_auth: bool = False,
) -> collections.abc.Callable[
    [
        typing.Union[
            PartialFunction,
            collections.abc.Callable[[], typing.Any],
            collections.abc.Callable[[typing.Any], typing.Any],
        ]
    ],
    PartialFunction,
]: ...
def wsgi_app(
    _warn_parentheses_missing=None,
    *,
    label: typing.Optional[str] = None,
    custom_domains: typing.Optional[collections.abc.Iterable[str]] = None,
    requires_proxy_auth: bool = False,
) -> collections.abc.Callable[
    [
        typing.Union[
            PartialFunction,
            collections.abc.Callable[[], typing.Any],
            collections.abc.Callable[[typing.Any], typing.Any],
        ]
    ],
    PartialFunction,
]: ...
def web_server(
    port: int,
    *,
    startup_timeout: float = 5.0,
    label: typing.Optional[str] = None,
    custom_domains: typing.Optional[collections.abc.Iterable[str]] = None,
    requires_proxy_auth: bool = False,
) -> collections.abc.Callable[
    [
        typing.Union[
            PartialFunction,
            collections.abc.Callable[[], typing.Any],
            collections.abc.Callable[[typing.Any], typing.Any],
        ]
    ],
    PartialFunction,
]: ...
def build(
    _warn_parentheses_missing=None, *, force: bool = False, timeout: int = 86400
) -> collections.abc.Callable[
    [typing.Union[PartialFunction, collections.abc.Callable[[typing.Any], typing.Any]]], PartialFunction
]: ...
def enter(
    _warn_parentheses_missing=None, *, snap: bool = False
) -> collections.abc.Callable[
    [typing.Union[PartialFunction, collections.abc.Callable[[typing.Any], typing.Any]]], PartialFunction
]: ...
def exit(
    _warn_parentheses_missing=None,
) -> collections.abc.Callable[[collections.abc.Callable[[typing.Any], typing.Any]], PartialFunction]: ...
def batched(
    _warn_parentheses_missing=None, *, max_batch_size: int, wait_ms: int
) -> collections.abc.Callable[
    [
        typing.Union[
            PartialFunction[
                modal._partial_function.P, modal._partial_function.ReturnType, modal._partial_function.ReturnType
            ],
            collections.abc.Callable[modal._partial_function.P, modal._partial_function.ReturnType],
        ]
    ],
    PartialFunction[modal._partial_function.P, modal._partial_function.ReturnType, modal._partial_function.ReturnType],
]: ...
def concurrent(
    _warn_parentheses_missing=None, *, max_inputs: int, target_inputs: typing.Optional[int] = None
) -> collections.abc.Callable[
    [
        typing.Union[
            PartialFunction[
                modal._partial_function.P, modal._partial_function.ReturnType, modal._partial_function.ReturnType
            ],
            collections.abc.Callable[modal._partial_function.P, modal._partial_function.ReturnType],
        ]
    ],
    PartialFunction[modal._partial_function.P, modal._partial_function.ReturnType, modal._partial_function.ReturnType],
]: ...

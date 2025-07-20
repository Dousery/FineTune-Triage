import collections.abc
import google.protobuf.message
import inspect
import modal._functions
import modal._object
import modal._partial_function
import modal.app
import modal.client
import modal.functions
import modal.gpu
import modal.object
import modal.partial_function
import modal.retries
import modal.secret
import modal.volume
import modal_proto.api_pb2
import os
import typing
import typing_extensions

T = typing.TypeVar("T")

def _use_annotation_parameters(user_cls: type) -> bool: ...
def _get_class_constructor_signature(user_cls: type) -> inspect.Signature: ...

class _ServiceOptions:
    secrets: typing.Collection[modal.secret._Secret]
    validated_volumes: typing.Sequence[tuple[str, modal.volume._Volume]]
    resources: typing.Optional[modal_proto.api_pb2.Resources]
    retry_policy: typing.Optional[modal_proto.api_pb2.FunctionRetryPolicy]
    max_containers: typing.Optional[int]
    buffer_containers: typing.Optional[int]
    scaledown_window: typing.Optional[int]
    timeout_secs: typing.Optional[int]
    max_concurrent_inputs: typing.Optional[int]
    target_concurrent_inputs: typing.Optional[int]
    batch_max_size: typing.Optional[int]
    batch_wait_ms: typing.Optional[int]

    def __init__(
        self,
        secrets: typing.Collection[modal.secret._Secret] = (),
        validated_volumes: typing.Sequence[tuple[str, modal.volume._Volume]] = (),
        resources: typing.Optional[modal_proto.api_pb2.Resources] = None,
        retry_policy: typing.Optional[modal_proto.api_pb2.FunctionRetryPolicy] = None,
        max_containers: typing.Optional[int] = None,
        buffer_containers: typing.Optional[int] = None,
        scaledown_window: typing.Optional[int] = None,
        timeout_secs: typing.Optional[int] = None,
        max_concurrent_inputs: typing.Optional[int] = None,
        target_concurrent_inputs: typing.Optional[int] = None,
        batch_max_size: typing.Optional[int] = None,
        batch_wait_ms: typing.Optional[int] = None,
    ) -> None: ...
    def __repr__(self): ...
    def __eq__(self, other): ...

def _bind_instance_method(cls: _Cls, service_function: modal._functions._Function, method_name: str): ...

class _Obj:
    _cls: _Cls
    _functions: dict[str, modal._functions._Function]
    _has_entered: bool
    _user_cls_instance: typing.Optional[typing.Any]
    _args: tuple[typing.Any, ...]
    _kwargs: dict[str, typing.Any]
    _instance_service_function: typing.Optional[modal._functions._Function]
    _options: typing.Optional[_ServiceOptions]

    def __init__(
        self, cls: _Cls, user_cls: typing.Optional[type], options: typing.Optional[_ServiceOptions], args, kwargs
    ): ...
    def _cached_service_function(self) -> modal._functions._Function: ...
    def _get_parameter_values(self) -> dict[str, typing.Any]: ...
    def _new_user_cls_instance(self): ...
    async def update_autoscaler(
        self,
        *,
        min_containers: typing.Optional[int] = None,
        max_containers: typing.Optional[int] = None,
        scaledown_window: typing.Optional[int] = None,
        buffer_containers: typing.Optional[int] = None,
    ) -> None: ...
    async def keep_warm(self, warm_pool_size: int) -> None: ...
    def _cached_user_cls_instance(self): ...
    def _enter(self): ...
    @property
    def _entered(self) -> bool: ...
    @_entered.setter
    def _entered(self, val: bool): ...
    async def _aenter(self): ...
    def __getattr__(self, k): ...

SUPERSELF = typing.TypeVar("SUPERSELF", covariant=True)

class Obj:
    _cls: Cls
    _functions: dict[str, modal.functions.Function]
    _has_entered: bool
    _user_cls_instance: typing.Optional[typing.Any]
    _args: tuple[typing.Any, ...]
    _kwargs: dict[str, typing.Any]
    _instance_service_function: typing.Optional[modal.functions.Function]
    _options: typing.Optional[_ServiceOptions]

    def __init__(
        self, cls: Cls, user_cls: typing.Optional[type], options: typing.Optional[_ServiceOptions], args, kwargs
    ): ...
    def _cached_service_function(self) -> modal.functions.Function: ...
    def _get_parameter_values(self) -> dict[str, typing.Any]: ...
    def _new_user_cls_instance(self): ...

    class __update_autoscaler_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(
            self,
            /,
            *,
            min_containers: typing.Optional[int] = None,
            max_containers: typing.Optional[int] = None,
            scaledown_window: typing.Optional[int] = None,
            buffer_containers: typing.Optional[int] = None,
        ) -> None: ...
        async def aio(
            self,
            /,
            *,
            min_containers: typing.Optional[int] = None,
            max_containers: typing.Optional[int] = None,
            scaledown_window: typing.Optional[int] = None,
            buffer_containers: typing.Optional[int] = None,
        ) -> None: ...

    update_autoscaler: __update_autoscaler_spec[typing_extensions.Self]

    class __keep_warm_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, /, warm_pool_size: int) -> None: ...
        async def aio(self, /, warm_pool_size: int) -> None: ...

    keep_warm: __keep_warm_spec[typing_extensions.Self]

    def _cached_user_cls_instance(self): ...
    def _enter(self): ...
    @property
    def _entered(self) -> bool: ...
    @_entered.setter
    def _entered(self, val: bool): ...
    async def _aenter(self): ...
    def __getattr__(self, k): ...

class _Cls(modal._object._Object):
    _class_service_function: typing.Optional[modal._functions._Function]
    _options: _ServiceOptions
    _app: typing.Optional[modal.app._App]
    _name: typing.Optional[str]
    _method_metadata: typing.Optional[dict[str, modal_proto.api_pb2.FunctionHandleMetadata]]
    _user_cls: typing.Optional[type]
    _method_partials: typing.Optional[dict[str, modal._partial_function._PartialFunction]]
    _callables: dict[str, collections.abc.Callable[..., typing.Any]]

    def _initialize_from_empty(self): ...
    def _initialize_from_other(self, other: _Cls): ...
    def _get_partial_functions(self) -> dict[str, modal._partial_function._PartialFunction]: ...
    def _get_app(self) -> modal.app._App: ...
    def _get_user_cls(self) -> type: ...
    def _get_name(self) -> str: ...
    def _get_class_service_function(self) -> modal._functions._Function: ...
    def _get_method_names(self) -> collections.abc.Collection[str]: ...
    def _hydrate_metadata(self, metadata: google.protobuf.message.Message): ...
    @staticmethod
    def validate_construction_mechanism(user_cls): ...
    @staticmethod
    def from_local(user_cls, app: modal.app._App, class_service_function: modal._functions._Function) -> _Cls: ...
    @classmethod
    def from_name(
        cls: type[_Cls], app_name: str, name: str, *, namespace=1, environment_name: typing.Optional[str] = None
    ) -> _Cls: ...
    def with_options(
        self: _Cls,
        *,
        cpu: typing.Union[float, tuple[float, float], None] = None,
        memory: typing.Union[int, tuple[int, int], None] = None,
        gpu: typing.Union[None, str, modal.gpu._GPUConfig] = None,
        secrets: collections.abc.Collection[modal.secret._Secret] = (),
        volumes: dict[typing.Union[str, os.PathLike], modal.volume._Volume] = {},
        retries: typing.Union[int, modal.retries.Retries, None] = None,
        max_containers: typing.Optional[int] = None,
        buffer_containers: typing.Optional[int] = None,
        scaledown_window: typing.Optional[int] = None,
        timeout: typing.Optional[int] = None,
        concurrency_limit: typing.Optional[int] = None,
        container_idle_timeout: typing.Optional[int] = None,
        allow_concurrent_inputs: typing.Optional[int] = None,
    ) -> _Cls: ...
    def with_concurrency(self: _Cls, *, max_inputs: int, target_inputs: typing.Optional[int] = None) -> _Cls: ...
    def with_batching(self: _Cls, *, max_batch_size: int, wait_ms: int) -> _Cls: ...
    @staticmethod
    async def lookup(
        app_name: str,
        name: str,
        namespace=1,
        client: typing.Optional[modal.client._Client] = None,
        environment_name: typing.Optional[str] = None,
    ) -> _Cls: ...
    def __call__(self, *args, **kwargs) -> _Obj: ...
    def __getattr__(self, k): ...
    def _is_local(self) -> bool: ...

class Cls(modal.object.Object):
    _class_service_function: typing.Optional[modal.functions.Function]
    _options: _ServiceOptions
    _app: typing.Optional[modal.app.App]
    _name: typing.Optional[str]
    _method_metadata: typing.Optional[dict[str, modal_proto.api_pb2.FunctionHandleMetadata]]
    _user_cls: typing.Optional[type]
    _method_partials: typing.Optional[dict[str, modal.partial_function.PartialFunction]]
    _callables: dict[str, collections.abc.Callable[..., typing.Any]]

    def __init__(self, *args, **kwargs): ...
    def _initialize_from_empty(self): ...
    def _initialize_from_other(self, other: Cls): ...
    def _get_partial_functions(self) -> dict[str, modal.partial_function.PartialFunction]: ...
    def _get_app(self) -> modal.app.App: ...
    def _get_user_cls(self) -> type: ...
    def _get_name(self) -> str: ...
    def _get_class_service_function(self) -> modal.functions.Function: ...
    def _get_method_names(self) -> collections.abc.Collection[str]: ...
    def _hydrate_metadata(self, metadata: google.protobuf.message.Message): ...
    @staticmethod
    def validate_construction_mechanism(user_cls): ...
    @staticmethod
    def from_local(user_cls, app: modal.app.App, class_service_function: modal.functions.Function) -> Cls: ...
    @classmethod
    def from_name(
        cls: type[Cls], app_name: str, name: str, *, namespace=1, environment_name: typing.Optional[str] = None
    ) -> Cls: ...
    def with_options(
        self: Cls,
        *,
        cpu: typing.Union[float, tuple[float, float], None] = None,
        memory: typing.Union[int, tuple[int, int], None] = None,
        gpu: typing.Union[None, str, modal.gpu._GPUConfig] = None,
        secrets: collections.abc.Collection[modal.secret.Secret] = (),
        volumes: dict[typing.Union[str, os.PathLike], modal.volume.Volume] = {},
        retries: typing.Union[int, modal.retries.Retries, None] = None,
        max_containers: typing.Optional[int] = None,
        buffer_containers: typing.Optional[int] = None,
        scaledown_window: typing.Optional[int] = None,
        timeout: typing.Optional[int] = None,
        concurrency_limit: typing.Optional[int] = None,
        container_idle_timeout: typing.Optional[int] = None,
        allow_concurrent_inputs: typing.Optional[int] = None,
    ) -> Cls: ...
    def with_concurrency(self: Cls, *, max_inputs: int, target_inputs: typing.Optional[int] = None) -> Cls: ...
    def with_batching(self: Cls, *, max_batch_size: int, wait_ms: int) -> Cls: ...

    class __lookup_spec(typing_extensions.Protocol):
        def __call__(
            self,
            /,
            app_name: str,
            name: str,
            namespace=1,
            client: typing.Optional[modal.client.Client] = None,
            environment_name: typing.Optional[str] = None,
        ) -> Cls: ...
        async def aio(
            self,
            /,
            app_name: str,
            name: str,
            namespace=1,
            client: typing.Optional[modal.client.Client] = None,
            environment_name: typing.Optional[str] = None,
        ) -> Cls: ...

    lookup: __lookup_spec

    def __call__(self, *args, **kwargs) -> Obj: ...
    def __getattr__(self, k): ...
    def _is_local(self) -> bool: ...

class ___get_constructor_args_spec(typing_extensions.Protocol):
    def __call__(self, /, cls: Cls) -> typing.Sequence[modal_proto.api_pb2.ClassParameterSpec]: ...
    async def aio(self, /, cls: Cls) -> typing.Sequence[modal_proto.api_pb2.ClassParameterSpec]: ...

_get_constructor_args: ___get_constructor_args_spec

class ___get_method_schemas_spec(typing_extensions.Protocol):
    def __call__(self, /, cls: Cls) -> dict[str, modal_proto.api_pb2.FunctionSchema]: ...
    async def aio(self, /, cls: Cls) -> dict[str, modal_proto.api_pb2.FunctionSchema]: ...

_get_method_schemas: ___get_method_schemas_spec

class _NO_DEFAULT:
    def __repr__(self): ...

_no_default: _NO_DEFAULT

class _Parameter:
    default: typing.Any
    init: bool

    def __init__(self, default: typing.Any, init: bool): ...
    def __get__(self, obj, obj_type=None) -> typing.Any: ...

def is_parameter(p: typing.Any) -> bool: ...
def parameter(*, default: typing.Any = modal.cls._NO_DEFAULT(), init: bool = True) -> typing.Any: ...

import _io
import asyncio.locks
import collections.abc
import enum
import google.protobuf.message
import modal._object
import modal._utils.blob_utils
import modal.client
import modal.object
import modal_proto.api_pb2
import pathlib
import synchronicity.combined_types
import typing
import typing_extensions

class FileEntryType(enum.IntEnum):
    """Type of a file entry listed from a Modal volume."""

    UNSPECIFIED = 0
    FILE = 1
    DIRECTORY = 2
    SYMLINK = 3

class FileEntry:
    path: str
    type: FileEntryType
    mtime: int
    size: int

    @classmethod
    def _from_proto(cls, proto: modal_proto.api_pb2.FileEntry) -> FileEntry: ...
    def __init__(self, path: str, type: FileEntryType, mtime: int, size: int) -> None: ...
    def __repr__(self): ...
    def __eq__(self, other): ...
    def __setattr__(self, name, value): ...
    def __delattr__(self, name): ...
    def __hash__(self): ...

class _Volume(modal._object._Object):
    _lock: typing.Optional[asyncio.locks.Lock]
    _metadata: typing.Optional[modal_proto.api_pb2.VolumeMetadata]

    async def _get_lock(self): ...
    @staticmethod
    def from_name(
        name: str,
        *,
        namespace=1,
        environment_name: typing.Optional[str] = None,
        create_if_missing: bool = False,
        version: typing.Optional[int] = None,
    ) -> _Volume: ...
    def _hydrate_metadata(self, metadata: typing.Optional[google.protobuf.message.Message]): ...
    def _get_metadata(self) -> typing.Optional[google.protobuf.message.Message]: ...
    @property
    def _is_v1(self) -> bool: ...
    @classmethod
    def ephemeral(
        cls: type[_Volume],
        client: typing.Optional[modal.client._Client] = None,
        environment_name: typing.Optional[str] = None,
        version: typing.Optional[int] = None,
        _heartbeat_sleep: float = 300,
    ) -> typing.AsyncContextManager[_Volume]: ...
    @staticmethod
    async def lookup(
        name: str,
        namespace=1,
        client: typing.Optional[modal.client._Client] = None,
        environment_name: typing.Optional[str] = None,
        create_if_missing: bool = False,
        version: typing.Optional[int] = None,
    ) -> _Volume: ...
    @staticmethod
    async def create_deployed(
        deployment_name: str,
        namespace=1,
        client: typing.Optional[modal.client._Client] = None,
        environment_name: typing.Optional[str] = None,
        version: typing.Optional[int] = None,
    ) -> str: ...
    async def _do_reload(self, lock=True): ...
    async def commit(self): ...
    async def reload(self): ...
    def iterdir(self, path: str, *, recursive: bool = True) -> collections.abc.AsyncIterator[FileEntry]: ...
    async def listdir(self, path: str, *, recursive: bool = False) -> list[FileEntry]: ...
    def read_file(self, path: str) -> collections.abc.AsyncIterator[bytes]: ...
    def _read_file1(self, path: str) -> collections.abc.AsyncIterator[bytes]: ...
    def _read_file2(self, path: str) -> collections.abc.AsyncIterator[bytes]: ...
    async def read_file_into_fileobj(
        self,
        path: str,
        fileobj: typing.IO[bytes],
        progress_cb: typing.Optional[collections.abc.Callable[..., typing.Any]] = None,
    ) -> int: ...
    async def _read_file_into_fileobj1(
        self, path: str, fileobj: typing.IO[bytes], progress_cb: collections.abc.Callable[..., typing.Any]
    ) -> int: ...
    async def _read_file_into_fileobj2(
        self, path: str, fileobj: typing.IO[bytes], progress_cb: collections.abc.Callable[..., typing.Any]
    ) -> int: ...
    async def remove_file(self, path: str, recursive: bool = False) -> None: ...
    async def copy_files(self, src_paths: collections.abc.Sequence[str], dst_path: str) -> None: ...
    async def batch_upload(self, force: bool = False) -> _AbstractVolumeUploadContextManager: ...
    async def _instance_delete(self): ...
    @staticmethod
    async def delete(
        name: str, client: typing.Optional[modal.client._Client] = None, environment_name: typing.Optional[str] = None
    ): ...
    @staticmethod
    async def rename(
        old_name: str,
        new_name: str,
        *,
        client: typing.Optional[modal.client._Client] = None,
        environment_name: typing.Optional[str] = None,
    ): ...

SUPERSELF = typing.TypeVar("SUPERSELF", covariant=True)

class Volume(modal.object.Object):
    _lock: typing.Optional[asyncio.locks.Lock]
    _metadata: typing.Optional[modal_proto.api_pb2.VolumeMetadata]

    def __init__(self, *args, **kwargs): ...

    class ___get_lock_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, /): ...
        async def aio(self, /): ...

    _get_lock: ___get_lock_spec[typing_extensions.Self]

    @staticmethod
    def from_name(
        name: str,
        *,
        namespace=1,
        environment_name: typing.Optional[str] = None,
        create_if_missing: bool = False,
        version: typing.Optional[int] = None,
    ) -> Volume: ...
    def _hydrate_metadata(self, metadata: typing.Optional[google.protobuf.message.Message]): ...
    def _get_metadata(self) -> typing.Optional[google.protobuf.message.Message]: ...
    @property
    def _is_v1(self) -> bool: ...
    @classmethod
    def ephemeral(
        cls: type[Volume],
        client: typing.Optional[modal.client.Client] = None,
        environment_name: typing.Optional[str] = None,
        version: typing.Optional[int] = None,
        _heartbeat_sleep: float = 300,
    ) -> synchronicity.combined_types.AsyncAndBlockingContextManager[Volume]: ...

    class __lookup_spec(typing_extensions.Protocol):
        def __call__(
            self,
            /,
            name: str,
            namespace=1,
            client: typing.Optional[modal.client.Client] = None,
            environment_name: typing.Optional[str] = None,
            create_if_missing: bool = False,
            version: typing.Optional[int] = None,
        ) -> Volume: ...
        async def aio(
            self,
            /,
            name: str,
            namespace=1,
            client: typing.Optional[modal.client.Client] = None,
            environment_name: typing.Optional[str] = None,
            create_if_missing: bool = False,
            version: typing.Optional[int] = None,
        ) -> Volume: ...

    lookup: __lookup_spec

    class __create_deployed_spec(typing_extensions.Protocol):
        def __call__(
            self,
            /,
            deployment_name: str,
            namespace=1,
            client: typing.Optional[modal.client.Client] = None,
            environment_name: typing.Optional[str] = None,
            version: typing.Optional[int] = None,
        ) -> str: ...
        async def aio(
            self,
            /,
            deployment_name: str,
            namespace=1,
            client: typing.Optional[modal.client.Client] = None,
            environment_name: typing.Optional[str] = None,
            version: typing.Optional[int] = None,
        ) -> str: ...

    create_deployed: __create_deployed_spec

    class ___do_reload_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, /, lock=True): ...
        async def aio(self, /, lock=True): ...

    _do_reload: ___do_reload_spec[typing_extensions.Self]

    class __commit_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, /): ...
        async def aio(self, /): ...

    commit: __commit_spec[typing_extensions.Self]

    class __reload_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, /): ...
        async def aio(self, /): ...

    reload: __reload_spec[typing_extensions.Self]

    class __iterdir_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, /, path: str, *, recursive: bool = True) -> typing.Iterator[FileEntry]: ...
        def aio(self, /, path: str, *, recursive: bool = True) -> collections.abc.AsyncIterator[FileEntry]: ...

    iterdir: __iterdir_spec[typing_extensions.Self]

    class __listdir_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, /, path: str, *, recursive: bool = False) -> list[FileEntry]: ...
        async def aio(self, /, path: str, *, recursive: bool = False) -> list[FileEntry]: ...

    listdir: __listdir_spec[typing_extensions.Self]

    class __read_file_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, /, path: str) -> typing.Iterator[bytes]: ...
        def aio(self, /, path: str) -> collections.abc.AsyncIterator[bytes]: ...

    read_file: __read_file_spec[typing_extensions.Self]

    class ___read_file1_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, /, path: str) -> typing.Iterator[bytes]: ...
        def aio(self, /, path: str) -> collections.abc.AsyncIterator[bytes]: ...

    _read_file1: ___read_file1_spec[typing_extensions.Self]

    class ___read_file2_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, /, path: str) -> typing.Iterator[bytes]: ...
        def aio(self, /, path: str) -> collections.abc.AsyncIterator[bytes]: ...

    _read_file2: ___read_file2_spec[typing_extensions.Self]

    class __read_file_into_fileobj_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(
            self,
            /,
            path: str,
            fileobj: typing.IO[bytes],
            progress_cb: typing.Optional[collections.abc.Callable[..., typing.Any]] = None,
        ) -> int: ...
        async def aio(
            self,
            /,
            path: str,
            fileobj: typing.IO[bytes],
            progress_cb: typing.Optional[collections.abc.Callable[..., typing.Any]] = None,
        ) -> int: ...

    read_file_into_fileobj: __read_file_into_fileobj_spec[typing_extensions.Self]

    class ___read_file_into_fileobj1_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(
            self, /, path: str, fileobj: typing.IO[bytes], progress_cb: collections.abc.Callable[..., typing.Any]
        ) -> int: ...
        async def aio(
            self, /, path: str, fileobj: typing.IO[bytes], progress_cb: collections.abc.Callable[..., typing.Any]
        ) -> int: ...

    _read_file_into_fileobj1: ___read_file_into_fileobj1_spec[typing_extensions.Self]

    class ___read_file_into_fileobj2_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(
            self, /, path: str, fileobj: typing.IO[bytes], progress_cb: collections.abc.Callable[..., typing.Any]
        ) -> int: ...
        async def aio(
            self, /, path: str, fileobj: typing.IO[bytes], progress_cb: collections.abc.Callable[..., typing.Any]
        ) -> int: ...

    _read_file_into_fileobj2: ___read_file_into_fileobj2_spec[typing_extensions.Self]

    class __remove_file_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, /, path: str, recursive: bool = False) -> None: ...
        async def aio(self, /, path: str, recursive: bool = False) -> None: ...

    remove_file: __remove_file_spec[typing_extensions.Self]

    class __copy_files_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, /, src_paths: collections.abc.Sequence[str], dst_path: str) -> None: ...
        async def aio(self, /, src_paths: collections.abc.Sequence[str], dst_path: str) -> None: ...

    copy_files: __copy_files_spec[typing_extensions.Self]

    class __batch_upload_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, /, force: bool = False) -> AbstractVolumeUploadContextManager: ...
        async def aio(self, /, force: bool = False) -> AbstractVolumeUploadContextManager: ...

    batch_upload: __batch_upload_spec[typing_extensions.Self]

    class ___instance_delete_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, /): ...
        async def aio(self, /): ...

    _instance_delete: ___instance_delete_spec[typing_extensions.Self]

    class __delete_spec(typing_extensions.Protocol):
        def __call__(
            self,
            /,
            name: str,
            client: typing.Optional[modal.client.Client] = None,
            environment_name: typing.Optional[str] = None,
        ): ...
        async def aio(
            self,
            /,
            name: str,
            client: typing.Optional[modal.client.Client] = None,
            environment_name: typing.Optional[str] = None,
        ): ...

    delete: __delete_spec

    class __rename_spec(typing_extensions.Protocol):
        def __call__(
            self,
            /,
            old_name: str,
            new_name: str,
            *,
            client: typing.Optional[modal.client.Client] = None,
            environment_name: typing.Optional[str] = None,
        ): ...
        async def aio(
            self,
            /,
            old_name: str,
            new_name: str,
            *,
            client: typing.Optional[modal.client.Client] = None,
            environment_name: typing.Optional[str] = None,
        ): ...

    rename: __rename_spec

class _AbstractVolumeUploadContextManager:
    async def __aenter__(self): ...
    async def __aexit__(self, exc_type, exc_val, exc_tb): ...
    def put_file(
        self,
        local_file: typing.Union[pathlib.Path, str, typing.BinaryIO, _io.BytesIO],
        remote_path: typing.Union[pathlib.PurePosixPath, str],
        mode: typing.Optional[int] = None,
    ): ...
    def put_directory(
        self,
        local_path: typing.Union[pathlib.Path, str],
        remote_path: typing.Union[pathlib.PurePosixPath, str],
        recursive: bool = True,
    ): ...
    @staticmethod
    def resolve(
        version: int,
        object_id: str,
        client,
        progress_cb: typing.Optional[collections.abc.Callable[..., typing.Any]] = None,
        force: bool = False,
    ) -> _AbstractVolumeUploadContextManager: ...

class AbstractVolumeUploadContextManager:
    def __init__(self, /, *args, **kwargs): ...
    def __enter__(self): ...
    async def __aenter__(self): ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...
    async def __aexit__(self, exc_type, exc_val, exc_tb): ...
    def put_file(
        self,
        local_file: typing.Union[pathlib.Path, str, typing.BinaryIO, _io.BytesIO],
        remote_path: typing.Union[pathlib.PurePosixPath, str],
        mode: typing.Optional[int] = None,
    ): ...
    def put_directory(
        self,
        local_path: typing.Union[pathlib.Path, str],
        remote_path: typing.Union[pathlib.PurePosixPath, str],
        recursive: bool = True,
    ): ...
    @staticmethod
    def resolve(
        version: int,
        object_id: str,
        client,
        progress_cb: typing.Optional[collections.abc.Callable[..., typing.Any]] = None,
        force: bool = False,
    ) -> AbstractVolumeUploadContextManager: ...

class _VolumeUploadContextManager(_AbstractVolumeUploadContextManager):
    _volume_id: str
    _client: modal.client._Client
    _force: bool
    progress_cb: collections.abc.Callable[..., typing.Any]
    _upload_generators: list[
        collections.abc.Generator[collections.abc.Callable[[], modal._utils.blob_utils.FileUploadSpec], None, None]
    ]

    def __init__(
        self,
        volume_id: str,
        client: modal.client._Client,
        progress_cb: typing.Optional[collections.abc.Callable[..., typing.Any]] = None,
        force: bool = False,
    ): ...
    async def __aenter__(self): ...
    async def __aexit__(self, exc_type, exc_val, exc_tb): ...
    def put_file(
        self,
        local_file: typing.Union[pathlib.Path, str, typing.BinaryIO, _io.BytesIO],
        remote_path: typing.Union[pathlib.PurePosixPath, str],
        mode: typing.Optional[int] = None,
    ): ...
    def put_directory(
        self,
        local_path: typing.Union[pathlib.Path, str],
        remote_path: typing.Union[pathlib.PurePosixPath, str],
        recursive: bool = True,
    ): ...
    async def _upload_file(
        self, file_spec: modal._utils.blob_utils.FileUploadSpec
    ) -> modal_proto.api_pb2.MountFile: ...

class VolumeUploadContextManager(AbstractVolumeUploadContextManager):
    _volume_id: str
    _client: modal.client.Client
    _force: bool
    progress_cb: collections.abc.Callable[..., typing.Any]
    _upload_generators: list[
        collections.abc.Generator[collections.abc.Callable[[], modal._utils.blob_utils.FileUploadSpec], None, None]
    ]

    def __init__(
        self,
        volume_id: str,
        client: modal.client.Client,
        progress_cb: typing.Optional[collections.abc.Callable[..., typing.Any]] = None,
        force: bool = False,
    ): ...
    def __enter__(self): ...
    async def __aenter__(self): ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...
    async def __aexit__(self, exc_type, exc_val, exc_tb): ...
    def put_file(
        self,
        local_file: typing.Union[pathlib.Path, str, typing.BinaryIO, _io.BytesIO],
        remote_path: typing.Union[pathlib.PurePosixPath, str],
        mode: typing.Optional[int] = None,
    ): ...
    def put_directory(
        self,
        local_path: typing.Union[pathlib.Path, str],
        remote_path: typing.Union[pathlib.PurePosixPath, str],
        recursive: bool = True,
    ): ...

    class ___upload_file_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, /, file_spec: modal._utils.blob_utils.FileUploadSpec) -> modal_proto.api_pb2.MountFile: ...
        async def aio(self, /, file_spec: modal._utils.blob_utils.FileUploadSpec) -> modal_proto.api_pb2.MountFile: ...

    _upload_file: ___upload_file_spec[typing_extensions.Self]

class _VolumeUploadContextManager2(_AbstractVolumeUploadContextManager):
    _volume_id: str
    _client: modal.client._Client
    _progress_cb: collections.abc.Callable[..., typing.Any]
    _force: bool
    _hash_concurrency: int
    _put_concurrency: int
    _uploader_generators: list[
        collections.abc.Generator[
            collections.abc.Callable[
                [asyncio.locks.Semaphore], typing.Awaitable[modal._utils.blob_utils.FileUploadSpec2]
            ]
        ]
    ]

    def __init__(
        self,
        volume_id: str,
        client: modal.client._Client,
        progress_cb: typing.Optional[collections.abc.Callable[..., typing.Any]] = None,
        force: bool = False,
        hash_concurrency: int = 4,
        put_concurrency: int = 128,
    ): ...
    async def __aenter__(self): ...
    async def __aexit__(self, exc_type, exc_val, exc_tb): ...
    def put_file(
        self,
        local_file: typing.Union[pathlib.Path, str, typing.BinaryIO, _io.BytesIO],
        remote_path: typing.Union[pathlib.PurePosixPath, str],
        mode: typing.Optional[int] = None,
    ): ...
    def put_directory(
        self,
        local_path: typing.Union[pathlib.Path, str],
        remote_path: typing.Union[pathlib.PurePosixPath, str],
        recursive: bool = True,
    ): ...
    async def _put_file_specs(self, file_specs: list[modal._utils.blob_utils.FileUploadSpec2]): ...

class VolumeUploadContextManager2(AbstractVolumeUploadContextManager):
    _volume_id: str
    _client: modal.client.Client
    _progress_cb: collections.abc.Callable[..., typing.Any]
    _force: bool
    _hash_concurrency: int
    _put_concurrency: int
    _uploader_generators: list[
        collections.abc.Generator[
            collections.abc.Callable[
                [asyncio.locks.Semaphore], typing.Awaitable[modal._utils.blob_utils.FileUploadSpec2]
            ]
        ]
    ]

    def __init__(
        self,
        volume_id: str,
        client: modal.client.Client,
        progress_cb: typing.Optional[collections.abc.Callable[..., typing.Any]] = None,
        force: bool = False,
        hash_concurrency: int = 4,
        put_concurrency: int = 128,
    ): ...
    def __enter__(self): ...
    async def __aenter__(self): ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...
    async def __aexit__(self, exc_type, exc_val, exc_tb): ...
    def put_file(
        self,
        local_file: typing.Union[pathlib.Path, str, typing.BinaryIO, _io.BytesIO],
        remote_path: typing.Union[pathlib.PurePosixPath, str],
        mode: typing.Optional[int] = None,
    ): ...
    def put_directory(
        self,
        local_path: typing.Union[pathlib.Path, str],
        remote_path: typing.Union[pathlib.PurePosixPath, str],
        recursive: bool = True,
    ): ...

    class ___put_file_specs_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, /, file_specs: list[modal._utils.blob_utils.FileUploadSpec2]): ...
        async def aio(self, /, file_specs: list[modal._utils.blob_utils.FileUploadSpec2]): ...

    _put_file_specs: ___put_file_specs_spec[typing_extensions.Self]

async def _put_missing_blocks(
    file_specs: list[modal._utils.blob_utils.FileUploadSpec2],
    missing_blocks: list,
    put_responses: dict[bytes, bytes],
    put_concurrency: int,
    progress_cb: collections.abc.Callable[..., typing.Any],
): ...
def _open_files_error_annotation(mount_path: str) -> typing.Optional[str]: ...

import asyncio

from datetime import datetime, UTC
from typing import Dict, Optional, List

from syl.common.datastores import Datastore, DatastoreRegistration, Status


REGISTERED_DATASTORES: Dict[str, DatastoreRegistration] = {}
reg_lock = asyncio.Lock()


async def get_registered_datastores() -> Dict[str, DatastoreRegistration]:
    """Returns a shallow copy of all registered datastores"""
    async with reg_lock:
        return REGISTERED_DATASTORES.copy()


async def add_registered_datastore(datastore: Datastore, status: Status, file_watch_dirs: Optional[List[str]] = None):
    now = datetime.now(UTC)

    async with reg_lock:
        REGISTERED_DATASTORES[datastore.name] = DatastoreRegistration(
            datastore=datastore,
            status=status,
            file_watch_dirs=file_watch_dirs,
            first_seen=now,
            registered_at=now if status == Status.registered else None,
        )


async def update_registered_datastore_status(name: str, status: Status):
    now = datetime.now(UTC)

    async with reg_lock:
        REGISTERED_DATASTORES[name].status = status
        REGISTERED_DATASTORES[name].registered_at = now if status == Status.registered else None


async def remove_registered_datastore(name: str):
    async with reg_lock:
        del REGISTERED_DATASTORES[name]

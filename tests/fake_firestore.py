"""In-memory stand-in for the subset of the google-cloud-firestore client that
FirestoreStorage uses: collection/document references, get, set, create,
delete and stream. Real-project behavior is checked by offline/check_firestore.py.
"""

from __future__ import annotations

import copy

from google.api_core.exceptions import AlreadyExists


class Snapshot:
    def __init__(self, doc_id: str, data: dict | None):
        self.id = doc_id
        self._data = data

    @property
    def exists(self) -> bool:
        return self._data is not None

    def to_dict(self) -> dict | None:
        return copy.deepcopy(self._data)


class DocumentRef:
    def __init__(self, store: dict, path: tuple[str, ...]):
        self._store = store
        self._path = path
        self.id = path[-1]

    def collection(self, name: str) -> CollectionRef:
        return CollectionRef(self._store, (*self._path, name))

    def get(self) -> Snapshot:
        return Snapshot(self.id, self._store.get(self._path))

    def set(self, data: dict) -> None:
        self._store[self._path] = copy.deepcopy(data)

    def create(self, data: dict) -> None:
        if self._path in self._store:
            raise AlreadyExists(f"document {'/'.join(self._path)} already exists")
        self.set(data)

    def delete(self) -> None:
        self._store.pop(self._path, None)


class CollectionRef:
    def __init__(self, store: dict, path: tuple[str, ...]):
        self._store = store
        self._path = path

    def document(self, doc_id: str) -> DocumentRef:
        return DocumentRef(self._store, (*self._path, doc_id))

    def stream(self):
        n = len(self._path)
        for path, data in list(self._store.items()):
            if len(path) == n + 1 and path[:n] == self._path:
                yield Snapshot(path[-1], copy.deepcopy(data))


class FakeFirestore:
    def __init__(self):
        self._store: dict[tuple[str, ...], dict] = {}

    def collection(self, name: str) -> CollectionRef:
        return CollectionRef(self._store, (name,))

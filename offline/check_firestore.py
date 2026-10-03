"""Check FirestoreStorage against a real Firebase project, then clean up.

Usage (with the service-account JSON downloaded from the Firebase console):
    GOOGLE_APPLICATION_CREDENTIALS=path/to/key.json python offline/check_firestore.py

Creates a throwaway user, saves answers, undoes one, logs a recommendation list
and feedback, checks every read, and deletes all documents it created.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from recsys.session import Answer  # noqa: E402
from recsys.storage import FirestoreStorage, RecItem  # noqa: E402


def main() -> None:
    store = FirestoreStorage()
    db = store.db
    user = store.create_user("adaptive")
    created = {"list": None, "feedback": []}
    print(f"created test user {user.handle}")
    try:
        answers = [Answer.seed(1), Answer(2, "rated", 4), Answer(3, "interested")]
        for pos, a in enumerate(answers):
            store.add_answer(user.user_id, a, pos, response_ms=1000)
        assert store.get_user_by_handle(user.handle) == user, "lookup by handle failed"
        assert store.load_session(user).answers == answers, "session round trip failed"
        assert store.delete_last_answer(user.user_id) == answers[-1], "undo returned the wrong answer"
        assert store.load_session(user).answers == answers[:2], "undo did not delete"

        created["list"] = store.log_rec_list(user.user_id, "voluntary", 0.0, 2, [RecItem(1, 50, 0.4, "test")])
        created["feedback"].append(store.log_rec_feedback(created["list"], user.user_id, 50, "rated", 5))
        tables = store.export()
        assert user.handle in set(tables["users"]["handle"]), "export is missing the user"
        assert created["list"] in set(tables["rec_items"]["list_id"]), "export is missing the list"
        print("all Firestore checks passed")
    finally:
        for doc in store._answers(user.user_id).stream():
            store._answers(user.user_id).document(doc.id).delete()
        db.collection("users").document(user.user_id).delete()
        db.collection("handles").document(user.handle).delete()
        if created["list"]:
            db.collection("rec_lists").document(created["list"]).delete()
        for fid in created["feedback"]:
            db.collection("rec_feedback").document(fid).delete()
        print("cleaned up test documents")


if __name__ == "__main__":
    main()

import os
import firebase_admin
from firebase_admin import credentials, firestore, auth
from datetime import datetime

def init_firebase():
    cred_path = os.getenv("FIREBASE_KEY_PATH", "firebase_key.json")
    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
    return firestore.client()

def save_history(record: dict) -> str:
    """
    Save a record to the 'resume_history' collection and return the doc id.
    """
    db = init_firebase()
    payload = dict(record)
    payload.setdefault("timestamp", datetime.utcnow().isoformat())
    # Firestore Python SDK returns a tuple in some versions; normalize safely
    result = db.collection("resume_history").add(payload)
    try:
        # newer: (write_result, doc_ref)
        doc_id = result[1].id
    except Exception:
        try:
            # some environments: (doc_ref,)
            doc_id = result[0].id
        except Exception:
            doc_id = ""
    return doc_id

def get_recent_history(limit: int = 10) -> list[dict]:
    """
    Fetch the most recent records from 'resume_history' ordered by timestamp desc.
    """
    db = init_firebase()
    docs = (
        db.collection("resume_history")
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .limit(limit)
        .stream()
    )
    out = []
    for d in docs:
        try:
            rec = d.to_dict()
            rec["_id"] = d.id
            out.append(rec)
        except Exception:
            continue
    return out

# components/firebase_listener.py
import firebase_admin
from firebase_admin import credentials, firestore
import logging
import threading
from typing import Callable

from config import (
    FIREBASE_CREDENTIALS_PATH, 
    FIREBASE_COLLECTION, 
    FIREBASE_DOCUMENT, 
    FIREBASE_QUESTION_FIELD,
    FIREBASE_ANSWER_FIELD
)

logger = logging.getLogger(__name__)

class FirebaseListener:
    """
    Listens for changes in a specific Firebase document and triggers a callback.
    """
    def __init__(self, on_question_callback: Callable[[str], None]):
        # only initialize the default app once
        if not firebase_admin._apps:
            cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
            firebase_admin.initialize_app(cred)

        self.db = firestore.client()
        self.doc_ref = self.db.collection(FIREBASE_COLLECTION).document(FIREBASE_DOCUMENT)
        self.on_question_callback = on_question_callback
        self.watch = None
        self.stop_event = threading.Event()
        logger.debug("Firebase initialized successfully.")

    def start(self):
        """Starts listening for document changes in a background thread."""
        self.stop_event.clear()
        thread = threading.Thread(target=self._listen, daemon=True)
        thread.start()
        logger.debug(f"Started listening to Firebase document: {FIREBASE_COLLECTION}/{FIREBASE_DOCUMENT}")

    def stop(self):
        """Stops the listener."""
        self.stop_event.set()
        if self.watch:
            self.watch.unsubscribe()
        logger.debug("Firebase listener stopped.")

    def _listen(self):
        """The internal listening loop."""
        # Callback to run when the document changes
        def on_snapshot(doc_snapshot, changes, read_time):
            if self.stop_event.is_set():
                return

            for doc in doc_snapshot:
                if doc.exists:
                    data = doc.to_dict()
                    question = data.get(FIREBASE_QUESTION_FIELD)
                    if question:
                        logger.debug(f"New question received from Firebase: '{question}'")
                        self.on_question_callback(question)
                        # Clear the question field after processing to avoid re-triggering
                        self.doc_ref.update({FIREBASE_QUESTION_FIELD: None})
                else:
                    logger.debug("Firebase document does not exist.")

        self.watch = self.doc_ref.on_snapshot(on_snapshot)
        
        # Keep the thread alive until stop is called
        self.stop_event.wait()

    def update_answer(self, answer: str):
        """Updates the answer field in the Firebase document."""
        self.doc_ref.update({FIREBASE_ANSWER_FIELD: answer})
        logger.debug(f"Updated Firebase with answer: '{answer}'")

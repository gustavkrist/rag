import logging
import os

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setLevel(os.getenv("LOGLEVEL", "INFO").upper())
logger.addHandler(handler)
logger.setLevel(os.getenv("LOGLEVEL", "INFO").upper())

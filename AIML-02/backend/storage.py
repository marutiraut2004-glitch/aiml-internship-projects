"""Study storage. DICOM files go to Orthanc when it is reachable; everything also lives on local disk."""
from __future__ import annotations

import logging
import os

import requests

log = logging.getLogger(__name__)


class Orthanc:
    def __init__(self, url: str, user: str | None = None, password: str | None = None, timeout: float = 5.0):
        self.url = url.rstrip("/")
        self.auth = (user, password) if user else None
        self.timeout = timeout

    def available(self) -> bool:
        try:
            return requests.get(f"{self.url}/system", auth=self.auth, timeout=self.timeout).ok
        except requests.RequestException:
            return False

    def upload(self, dicom: bytes) -> str | None:
        """POST /instances. Returns the Orthanc instance id, or None on failure."""
        try:
            r = requests.post(f"{self.url}/instances", data=dicom, auth=self.auth, timeout=self.timeout,
                              headers={"Content-Type": "application/dicom"})
            r.raise_for_status()
            return r.json().get("ID")
        except (requests.RequestException, ValueError) as e:
            log.warning("Orthanc upload failed: %s", e)
            return None

    def fetch(self, instance_id: str) -> bytes | None:
        try:
            r = requests.get(f"{self.url}/instances/{instance_id}/file", auth=self.auth, timeout=self.timeout)
            return r.content if r.ok else None
        except requests.RequestException:
            return None


def orthanc_from_env() -> Orthanc | None:
    url = os.environ.get("ORTHANC_URL")
    if not url:
        return None
    return Orthanc(url, os.environ.get("ORTHANC_USER"), os.environ.get("ORTHANC_PASSWORD"))

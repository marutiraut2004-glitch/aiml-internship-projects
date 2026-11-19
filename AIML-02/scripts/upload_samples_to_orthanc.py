"""Push the demo DICOM files to an Orthanc server.   python scripts/upload_samples_to_orthanc.py http://localhost:8042"""
import sys
from pathlib import Path

from backend.storage import Orthanc

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8042"
o = Orthanc(url, "orthanc", "orthanc")
if not o.available():
    sys.exit(f"Orthanc not reachable at {url}")
for p in sorted(Path("data/samples").glob("*.dcm")):
    print(p.name, "->", o.upload(p.read_bytes()))

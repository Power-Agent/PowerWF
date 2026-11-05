# Reviewer failure memory

`verified_seed.jsonl` contains 150 controlled and programmatically verified failures in schema version 2.0, derived only from the development and memory-seed splits. It includes semantic and evidence channels and excludes the test split and numeric ground truth. The current runtime loads this file.

`verified_online.jsonl`, when created at runtime, accepts only program- or expert-confirmed Reflexion records. Test-derived failures are write-locked until evaluation is explicitly closed.

Rebuild the seed without simulations or API calls:

```powershell
.\.venv\Scripts\python.exe experiments\build_reviewer_memory.py
```

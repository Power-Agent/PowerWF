"""Private UI storage, independent of the immutable experiment configuration."""
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from b2g_agent.observability import LLMSettings
from b2g_agent.observability.journal import durable_write


class PersonalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    llm_mode: Literal["offline", "live", "replay"] = "offline"
    llm: LLMSettings = Field(default_factory=LLMSettings)
    maximum_agent_revisions: int = Field(default=4, ge=0, le=8)
    online_memory_enabled: bool = True


@dataclass(frozen=True)
class PersonalWorkspace:
    root: Path
    settings: PersonalSettings

    @classmethod
    def open(cls, repo_root: Path):
        repo_root = repo_root.resolve()
        configured = Path(os.getenv("B2G_PERSONAL_ROOT", "../Personal-workspace"))
        root = (configured if configured.is_absolute() else repo_root / configured).resolve()
        # In-repository private storage may only use the explicitly ignored .personal.
        if (repo_root.is_relative_to(root)
                or (root.is_relative_to(repo_root) and not root.is_relative_to(repo_root / ".personal"))):
            raise ValueError("Personal storage must be outside the repository or under .personal")
        from b2g_agent.case_io import CaseBundle
        data_root = CaseBundle.default(repo_root).local_data_root.resolve()
        if root.is_relative_to(data_root) or data_root.is_relative_to(root):
            raise ValueError("Personal storage cannot overlap the frozen data root")
        root.mkdir(parents=True, exist_ok=True)
        settings_path = root / "settings.json"
        settings = PersonalSettings.model_validate_json(settings_path.read_text(encoding="utf-8")) if settings_path.is_file() else PersonalSettings()
        if not settings_path.exists():
            durable_write(settings_path, settings.model_dump_json(indent=2) + "\n")
        for name in ("sessions", "lr", "interactions", "memory"):
            (root / name).mkdir(exist_ok=True)
        if not (root / ".gitignore").exists():
            durable_write(root / ".gitignore", "*\n")
        if not (root / "README.md").exists():
            durable_write(root / "README.md", "# Private interactive workspace\n\n"
                "All files here are personal and excluded from the formal 600-attempt evaluation.\n"
                "`settings.json` controls this UI only. The initial mode is `offline`.\n"
                "After deciding to use the API, set `llm_mode` to `live` and restart the local UI.\n"
                "From B2G-Agent, run `.venv/Scripts/python.exe -B start_personal.py --host 127.0.0.1 --port 8000`.\n"
                "Open `http://127.0.0.1:8000`. No formal experiment is started by the UI.\n\n"
                "`sessions/`: conversations, results and per-turn physics.\n"
                "`lr/`: original API requests, responses and parsed replies.\n"
                "`interactions/`: durable per-turn stage journals.\n"
                "`memory/`: personal confirmed failures, never loaded by formal evaluation.\n\n"
                "Do not publish this folder. Frozen case assets and knowledge are read-only shared inputs;\n"
                "personal outputs, learned memory and LLM settings never update them.\n")
        return cls(root, settings)

    def pipeline(self, repo_root):
        from b2g_agent.agents.llm import StructuredLLMClient
        from b2g_agent.orchestration import B2GPipeline
        from b2g_agent.orchestration.task_physics import TaskPhysicsExecutor
        llm = StructuredLLMClient.from_env(repo_root=repo_root, mode=self.settings.llm_mode,
            settings=self.settings.llm, record_root=self.root / "lr")
        pipeline = B2GPipeline.default(repo_root=repo_root, llm_client=llm, runtime_root=self.root)
        # Opt in only for private interactive tasks; formal factories keep their
        # original executor, command line and numerical implementation identity.
        pipeline.physics_executor = TaskPhysicsExecutor(
            repo_root, maximum_replans=pipeline.physics_executor.maximum_replans,
            use_interactive_inputs=True,
        )
        pipeline.maximum_agent_revisions = self.settings.maximum_agent_revisions
        pipeline.reviewer.allow_memory_writes = self.settings.online_memory_enabled
        pipeline.reviewer.online_memory_path = (self.root / "memory" / "verified_online.jsonl"
                                               if self.settings.online_memory_enabled else None)
        return pipeline

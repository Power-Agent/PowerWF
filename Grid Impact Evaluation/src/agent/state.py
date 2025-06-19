from __future__ import annotations

from typing import TypedDict, Optional, Dict, Any
from langgraph.graph import add_messages
from typing_extensions import Annotated


class GridImpactStudyState(TypedDict):
    """State for Grid Impact Study workflow."""
    messages: Annotated[list, add_messages]
    
    # User inputs
    powerworld_file_path: Optional[str]
    bus_number: Optional[str]
    load_demand_mw: Optional[float]
    
    # Analysis results
    baseline_results: Optional[Dict[str, Any]]
    modified_results: Optional[Dict[str, Any]]
    contingency_results: Optional[Dict[str, Any]]
    
    # Workflow control
    inputs_collected: bool
    baseline_complete: bool
    modification_complete: bool
    contingency_complete: bool
    
    # Analysis summary
    impact_summary: Optional[Dict[str, Any]]


# Keep the old state for backward compatibility
class ChatState(TypedDict):
    """Simple state for a basic chat agent."""
    messages: Annotated[list, add_messages]

"""
LangGraph Tools and Schemas for Pandapower Analysis

This module contains direct LangGraph tool implementations for pandapower 
power system analysis, replacing the previous MCP-based approach.
"""

import platform
import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import pandapower as pp
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check platform
IS_WINDOWS = platform.system().lower() == 'windows'

# Import Windows-specific modules if on Windows
if IS_WINDOWS:
    try:
        import win32com.client
        import pythoncom
        HAS_WIN32 = True
    except ImportError:
        logger.warning("Windows COM libraries not available. Some features may be limited.")
        HAS_WIN32 = False
else:
    HAS_WIN32 = False
    logger.info("Running on non-Windows platform. Windows-specific features will be disabled.")

# Global variable to store the current network
_current_net = None

def _get_network() -> pp.pandapowerNet:
    """Get the current pandapower network instance.
    
    Returns:
        pp.pandapowerNet: The current network or raises error if none loaded
    """
    global _current_net
    if _current_net is None:
        raise RuntimeError("No pandapower network is currently loaded. Please create or load a network first.")
    return _current_net

# Pydantic schemas for tool inputs
class CreateEmptyNetworkInput(BaseModel):
    """Schema for creating an empty network."""
    pass

class LoadNetworkInput(BaseModel):
    """Schema for loading a network from file."""
    file_path: str = Field(description="Path to the network file (.json, .p)")

class RunPowerFlowInput(BaseModel):
    """Schema for running power flow analysis."""
    algorithm: str = Field(default='nr', description="Power flow algorithm ('nr' for Newton-Raphson, 'bfsw' for backward/forward sweep)")
    calculate_voltage_angles: bool = Field(default=True, description="Consider voltage angles in calculation")
    max_iteration: int = Field(default=10, description="Maximum number of iterations")
    tolerance_mva: float = Field(default=1e-8, description="Convergence tolerance in MVA")

class RunContingencyAnalysisInput(BaseModel):
    """Schema for running contingency analysis."""
    contingency_type: str = Field(default="N-1", description="Type of contingency analysis ('N-1' or 'N-2')")
    elements: Optional[List[str]] = Field(default=None, description="List of specific elements to analyze (optional)")

class GetNetworkInfoInput(BaseModel):
    """Schema for getting network information."""
    pass

class LoadAndRunPowerFlowInput(BaseModel):
    """Schema for loading a network and running power flow in one step."""
    file_path: str = Field(description="Path to the network file (.json, .p)")
    algorithm: str = Field(default='nr', description="Power flow algorithm ('nr' for Newton-Raphson, 'bfsw' for backward/forward sweep)")
    calculate_voltage_angles: bool = Field(default=True, description="Consider voltage angles in calculation")
    max_iteration: int = Field(default=10, description="Maximum number of iterations")
    tolerance_mva: float = Field(default=1e-8, description="Convergence tolerance in MVA")

# LangGraph Tools
@tool(args_schema=CreateEmptyNetworkInput)
def create_empty_network() -> Dict[str, Any]:
    """Create an empty pandapower network.
    
    Returns:
        Dict containing status and network information
    """
    logger.info("Creating an empty pandapower network")
    global _current_net
    try:
        _current_net = pp.create_empty_network()
        return {
            "status": "success",
            "message": "Empty network created successfully",
            "network_info": {
                "buses": len(_current_net.bus),
                "lines": len(_current_net.line),
                "trafos": len(_current_net.trafo)
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to create empty network: {str(e)}"
        }

@tool(args_schema=LoadNetworkInput)
def load_network(file_path: str) -> Dict[str, Any]:
    """Load a pandapower network from a file.
    
    Args:
        file_path: Path to the network file (.json, .p)
        
    Returns:
        Dict containing status and network information
    """
    logger.info(f"Loading network from file: {file_path}")
    global _current_net
    try:
        if file_path.endswith('.json'):
            _current_net = pp.from_json(file_path)
        elif file_path.endswith('.p'):
            _current_net = pp.from_pickle(file_path)
        else:
            raise ValueError("Unsupported file format. Use .json or .p files.")
            
        return {
            "status": "success",
            "message": f"Network loaded successfully from {file_path}",
            "network_info": {
                "buses": len(_current_net.bus),
                "lines": len(_current_net.line),
                "trafos": len(_current_net.trafo)
            }
        }
    except FileNotFoundError:
        return {
            "status": "error",
            "message": f"File not found: {file_path}"
        }
    except ValueError as ve:
        return {
            "status": "error",
            "message": str(ve)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to load network: {str(e)}"
        }

@tool(args_schema=RunPowerFlowInput)
def run_power_flow(algorithm: str = 'nr', calculate_voltage_angles: bool = True, 
                  max_iteration: int = 10, tolerance_mva: float = 1e-8) -> Dict[str, Any]:
    """Run power flow analysis on the current network.
    
    Args:
        algorithm: Power flow algorithm ('nr' for Newton-Raphson, 'bfsw' for backward/forward sweep)
        calculate_voltage_angles: Consider voltage angles in calculation
        max_iteration: Maximum number of iterations
        tolerance_mva: Convergence tolerance in MVA
        
    Returns:
        Dict containing power flow results
    """
    logger.info("Running power flow analysis")
    try:
        net = _get_network()
        pp.runpp(net, algorithm=algorithm, calculate_voltage_angles=calculate_voltage_angles,
                max_iteration=max_iteration, tolerance_mva=tolerance_mva)
        
        # Extract key results
        results = {
            "bus_results": net.res_bus.to_dict(),
            "line_results": net.res_line.to_dict(),
            "trafo_results": net.res_trafo.to_dict(),
            "converged": net.converged
        }
        
        return {
            "status": "success",
            "message": "Power flow calculation completed successfully" if net.converged else "Power flow did not converge",
            "results": results
        }
    except RuntimeError as re:
        return {
            "status": "error",
            "message": str(re)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Power flow calculation failed: {str(e)}"
        }

@tool(args_schema=RunContingencyAnalysisInput)
def run_contingency_analysis(contingency_type: str = "N-1", 
                           elements: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run contingency analysis on the current network.
    
    Args:
        contingency_type: Type of contingency analysis ("N-1" or "N-2")
        elements: List of specific elements to analyze (optional)
        
    Returns:
        Dict containing contingency analysis results
    """
    logger.info("Running contingency analysis")
    try:
        net = _get_network()
        
        # Store original state
        orig_net = net.deepcopy()
        results = []
        
        # Define elements to analyze
        if elements is None:
            elements = ['line', 'trafo']
            
        # Perform contingency analysis
        for element_type in elements:
            for idx in net[element_type].index:
                # Create contingency by taking element out of service
                contingency_net = orig_net.deepcopy()
                contingency_net[element_type].at[idx, 'in_service'] = False
                
                try:
                    pp.runpp(contingency_net)
                    
                    # Check for violations
                    violations = {
                        'voltage_violations': contingency_net.res_bus[
                            (contingency_net.res_bus.vm_pu < 0.95) | 
                            (contingency_net.res_bus.vm_pu > 1.05)
                        ].index.tolist(),
                        'loading_violations': contingency_net.res_line[
                            contingency_net.res_line.loading_percent > 100
                        ].index.tolist()
                    }
                    
                    results.append({
                        'contingency': f"{element_type}_{idx}",
                        'converged': contingency_net.converged,
                        'violations': violations
                    })
                    
                except Exception as e:
                    results.append({
                        'contingency': f"{element_type}_{idx}",
                        'converged': False,
                        'error': str(e)
                    })
        
        return {
            "status": "success",
            "message": "Contingency analysis completed",
            "results": results
        }
    except RuntimeError as re:
        return {
            "status": "error",
            "message": str(re)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Contingency analysis failed: {str(e)}"
        }

@tool(args_schema=GetNetworkInfoInput)
def get_network_info() -> Dict[str, Any]:
    """Get information about the current network.
    
    Returns:
        Dict containing network statistics and information
    """
    logger.info("Retrieving network information")
    try:
        net = _get_network()
        info = {
            "buses": len(net.bus),
            "lines": len(net.line),
            "trafos": len(net.trafo),
            "generators": len(net.gen),
            "loads": len(net.load),
            "switches": len(net.switch),
            "bus_data": net.bus.to_dict(),
            "line_data": net.line.to_dict(),
            "trafo_data": net.trafo.to_dict()
        }
        
        return {
            "status": "success",
            "message": "Network information retrieved successfully",
            "info": info
        }
    except RuntimeError as re:
        return {
            "status": "error",
            "message": str(re)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to get network information: {str(e)}"
        }

@tool(args_schema=LoadAndRunPowerFlowInput)
def load_and_run_power_flow(file_path: str, algorithm: str = 'nr', 
                           calculate_voltage_angles: bool = True, 
                           max_iteration: int = 10, tolerance_mva: float = 1e-8) -> Dict[str, Any]:
    """Load a network from file and immediately run power flow analysis.
    
    This combined function avoids state management issues by performing
    both operations in a single tool call.
    
    Args:
        file_path: Path to the network file (.json, .p)
        algorithm: Power flow algorithm ('nr' for Newton-Raphson, 'bfsw' for backward/forward sweep)
        calculate_voltage_angles: Consider voltage angles in calculation
        max_iteration: Maximum number of iterations
        tolerance_mva: Convergence tolerance in MVA
        
    Returns:
        Dict containing network info and power flow results
    """
    logger.info(f"Loading network from {file_path} and running power flow analysis")
    global _current_net
    
    try:
        # Step 1: Load the network
        if file_path.endswith('.json'):
            _current_net = pp.from_json(file_path)
        elif file_path.endswith('.p'):
            _current_net = pp.from_pickle(file_path)
        else:
            raise ValueError("Unsupported file format. Use .json or .p files.")
        
        network_info = {
            "buses": len(_current_net.bus),
            "lines": len(_current_net.line),
            "trafos": len(_current_net.trafo)
        }
        
        # Step 2: Run power flow analysis
        pp.runpp(_current_net, algorithm=algorithm, 
                calculate_voltage_angles=calculate_voltage_angles,
                max_iteration=max_iteration, tolerance_mva=tolerance_mva)
        
        # Extract key results
        power_flow_results = {
            "bus_results": _current_net.res_bus.to_dict(),
            "line_results": _current_net.res_line.to_dict(),
            "trafo_results": _current_net.res_trafo.to_dict(),
            "converged": _current_net.converged
        }
        
        return {
            "status": "success",
            "message": f"Network loaded from {file_path} and power flow completed successfully" if _current_net.converged else f"Network loaded from {file_path} but power flow did not converge",
            "network_info": network_info,
            "power_flow_results": power_flow_results
        }
        
    except FileNotFoundError:
        return {
            "status": "error",
            "message": f"File not found: {file_path}"
        }
    except ValueError as ve:
        return {
            "status": "error",
            "message": str(ve)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to load network and run power flow: {str(e)}"
        }

# List of all available tools for easy import
PANDAPOWER_TOOLS = [
    create_empty_network,
    load_network,
    run_power_flow,
    run_contingency_analysis,
    get_network_info,
    load_and_run_power_flow
]

# Focused tool list with only the most comprehensive tool
CHAT_TOOLS = [
    load_and_run_power_flow
]

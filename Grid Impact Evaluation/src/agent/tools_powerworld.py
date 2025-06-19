"""
LangGraph Tools and Schemas for PowerWorld Analysis

This module contains direct LangGraph tool implementations for PowerWorld 
power system analysis, replacing the previous pandapower approach.
"""

import logging
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from esa import SAW, PowerWorldError
import os
import pythoncom

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)

# Global SAW instance for state management
_saw = None

def _reset_saw():
    """Reset the global SAW instance."""
    global _saw
    if _saw is not None:
        try:
            _saw.exit()
        except:
            pass  # Ignore errors when closing
        _saw = None

def _get_saw(case_path: Optional[str] = None) -> SAW:
    """Get or create SAW instance."""
    global _saw
    if _saw is None and case_path is not None:
        try:
            # Initialize COM for the current thread (required for PowerWorld)
            pythoncom.CoInitialize()
            
            # Convert to absolute path to ensure it works from any directory
            abs_case_path = os.path.abspath(case_path)
            logger.info(f"Initializing SAW with absolute path: {abs_case_path}")
            
            # Use absolute path directly, ensuring it runs without a visible UI for stability
            _saw = SAW(abs_case_path, UIVisible=False)
            logger.info("SAW instance created successfully.")
        except PowerWorldError as e:
            logger.error(f"Error initializing SAW: {str(e)}")
            # Reset SAW on error to avoid stuck state
            _reset_saw()
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing SAW: {str(e)}")
            # Reset SAW on error to avoid stuck state
            _reset_saw()
            raise
            
    elif _saw is None:
        raise ValueError("No case is currently open. Please open a case first.")
    return _saw

# Pydantic schemas for tool inputs
class OpenCaseInput(BaseModel):
    """Schema for opening a PowerWorld case."""
    case_path: str = Field(description="Path to the PowerWorld case file (.pwb)")

class RunPowerFlowInput(BaseModel):
    """Schema for running power flow analysis."""
    solution_method: str = Field(
        default='RECTNEWT', 
        description="Power flow solution method: 'RECTNEWT', 'POLARNEWTON', 'GAUSSSEIDEL', 'FDXB', or 'DC'"
    )

class AnalyzeContingenciesInput(BaseModel):
    """Schema for contingency analysis."""
    option: str = Field(default="N-1", description="Type of contingency analysis ('N-1' or 'N-2')")
    validate_contingencies: bool = Field(default=False, description="Whether to validate contingencies")

class GetPowerFlowResultsInput(BaseModel):
    """Schema for getting power flow results."""
    object_type: str = Field(description="Object type: 'bus', 'gen', 'load', 'shunt', or 'branch'")
    additional_fields: Optional[List[str]] = Field(default=None, description="Additional fields to retrieve")

class GetKeyFieldListInput(BaseModel):
    """Schema for getting key field list."""
    object_type: str = Field(description="Object type (e.g., 'bus', 'gen', 'branch')")

class ChangeParametersInput(BaseModel):
    """Schema for changing parameters."""
    object_type: str = Field(description="Object type (e.g., 'bus', 'gen', 'branch')")
    param_list: List[str] = Field(description="List of parameter names (must include key fields)")
    value_list: List[List[Any]] = Field(description="List of value lists for each element")

class ChangeAndConfirmParamsInput(BaseModel):
    """Schema for changing and confirming parameters."""
    object_type: str = Field(description="Object type (e.g., 'bus', 'gen', 'branch')")
    command_df: Dict[str, List[Any]] = Field(description="Dictionary representing DataFrame with parameters")

class GetYbusInput(BaseModel):
    """Schema for getting Ybus matrix."""
    full: bool = Field(default=False, description="Return full matrix if True, sparse if False")

class ToGraphInput(BaseModel):
    """Schema for converting to graph."""
    node: str = Field(default='bus', description="Node type: 'bus' or 'substation'")
    geographic: bool = Field(default=False, description="Include geographic coordinates")
    directed: bool = Field(default=False, description="Create directed graph based on power flow")

class GetJacobianInput(BaseModel):
    """Schema for getting Jacobian matrix."""
    full: bool = Field(default=False, description="Return full matrix if True, sparse if False")

class GetLodfMatrixInput(BaseModel):
    """Schema for getting LODF matrix."""
    precision: int = Field(default=3, description="Number of decimal places")
    ignore_open_branch: bool = Field(default=True, description="Ignore open branches")
    method: str = Field(default='DC', description="Power flow method")

class DetermineShortestPathInput(BaseModel):
    """Schema for finding shortest path."""
    start: str = Field(description="Starting bus number")
    end: str = Field(description="Ending bus number")
    branch_distance_measure: str = Field(default="X", description="Distance measure: 'X', 'Z', etc.")
    branch_filter: str = Field(default="ALL", description="Branch filter")

class RunRobustnessAnalysisInput(BaseModel):
    """Schema for robustness analysis."""
    pass

class GetPtdfMatrixInput(BaseModel):
    """Schema for getting PTDF matrix."""
    pass

class OpenCaseAndRunPowerFlowInput(BaseModel):
    """Schema for combined open case and run power flow."""
    case_path: str = Field(description="Path to the PowerWorld case file (.pwb)")
    solution_method: str = Field(
        default='RECTNEWT', 
        description="Power flow solution method: 'RECTNEWT', 'POLARNEWTON', 'GAUSSSEIDEL', 'FDXB', or 'DC'"
    )

# LangGraph Tools
@tool(args_schema=OpenCaseInput)
def open_case(case_path: str) -> Dict[str, Any]:
    """
    Open a PowerWorld case file.
    
    Args:
        case_path: Path to the PowerWorld case file
    
    Returns:
        Dict with status and case information
    """
    logger.info(f"Opening PowerWorld case: {case_path}")
    try:
        # Ensure COM is initialized for this thread
        pythoncom.CoInitialize()
        
        # Initialize SAW with the case
        saw = _get_saw(case_path)
        
        # Get basic case information
        bus_data = saw.get_power_flow_results('bus')
        branch_data = saw.get_power_flow_results('branch')
        gen_data = saw.get_power_flow_results('gen')
        
        return {
            'status': 'success',
            'case_info': {
                'path': case_path,
                'num_buses': len(bus_data) if bus_data is not None else 0,
                'num_branches': len(branch_data) if branch_data is not None else 0,
                'num_generators': len(gen_data) if gen_data is not None else 0
            }
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }

@tool(args_schema=RunPowerFlowInput)
def run_powerflow(solution_method: str = 'RECTNEWT') -> Dict[str, Any]:
    """
    Run power flow analysis on the currently open case.
    
    Args:
        solution_method: Power flow solution method
            
    Returns:
        Dictionary containing power flow results
    """
    logger.info(f"Running power flow with method: {solution_method}")
    try:
        saw = _get_saw()
        
        # Solve power flow
        saw.SolvePowerFlow(SolMethod=solution_method)
        
        # Get branch flow information
        branch_data = saw.get_power_flow_results('branch')
        
        # Process results to check for overflows
        overflows = []
        if branch_data is not None:
            for _, branch in branch_data.iterrows():
                try:
                    from_bus = branch['BusNum']
                    to_bus = branch['BusNum:1']
                    circuit = branch['LineCircuit']
                    mw = branch['LineMW']
                    mvar = branch['LineMVR']
                    
                    # Calculate apparent power and loading
                    mva = (mw**2 + mvar**2)**0.5
                    loading_percent = mva / branch['LineRateA'] * 100 if branch['LineRateA'] > 0 else 0
                    
                    if loading_percent > 100:
                        overflows.append({
                            'line': f"Line {from_bus}-{to_bus} Circuit {circuit}",
                            'loading_percent': loading_percent,
                            'mw': mw,
                            'mvar': mvar,
                            'mva': mva
                        })
                except Exception as e:
                    continue
        
        # Get voltage violations
        bus_data = saw.get_power_flow_results('bus')
        voltage_violations = []
        if bus_data is not None:
            for _, bus in bus_data.iterrows():
                voltage = bus['BusPUVolt']
                if voltage < 0.95 or voltage > 1.05:
                    voltage_violations.append({
                        'bus': int(bus['BusNum']),
                        'voltage': voltage,
                        'angle': bus['BusAngle']
                    })
        
        return {
            'status': 'success',
            'results': {
                'solution_method': solution_method,
                'converged': True,
                'overflows': overflows,
                'voltage_violations': voltage_violations,
                'total_branches': len(branch_data) if branch_data is not None else 0,
                'total_buses': len(bus_data) if bus_data is not None else 0
            }
        }
        
    except PowerWorldError as e:
        return {
            'status': 'error',
            'message': f"PowerWorld Error: {str(e)}"
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Unexpected Error: {str(e)}"
        }

@tool(args_schema=AnalyzeContingenciesInput)
def analyze_contingencies(option: str = "N-1", validate_contingencies: bool = False) -> Dict[str, Any]:
    """
    Run contingency analysis on the currently open case.
    
    Args:
        option: Type of contingency analysis ("N-1" or "N-2")
        validate_contingencies: Whether to validate contingencies
        
    Returns:
        Dictionary containing contingency analysis results
    """
    logger.info(f"Running {option} contingency analysis")
    try:
        saw = _get_saw()
        
        # Run contingency analysis
        if option == "N-1":
            # Get all branches for N-1 analysis
            branch_data = saw.get_power_flow_results('branch')
            violations = []
            
            # Save initial state
            saw.SaveState()
            
            # Test each line outage
            for _, branch in branch_data.iterrows():
                try:
                    # Open the line
                    from_bus = int(branch['BusNum'])
                    to_bus = int(branch['BusNum:1'])
                    circuit = branch['LineCircuit']
                    
                    # Use DataFrame format for parameter changes
                    import pandas as pd
                    command_df = pd.DataFrame([{
                        'BusNum': from_bus,
                        'BusNum:1': to_bus,
                        'LineCircuit': circuit,
                        'LineStatus': 'OPEN'
                    }])
                    saw.change_parameters_multiple_element_df('branch', command_df)
                    
                    # Solve power flow
                    saw.SolvePowerFlow()
                    
                    # Check for violations
                    result = run_powerflow()
                    if result['status'] == 'success' and (
                        len(result['results']['overflows']) > 0 or 
                        len(result['results']['voltage_violations']) > 0
                    ):
                        violations.append({
                            'contingency': f"Line {from_bus}-{to_bus} Circuit {circuit}",
                            'overflows': result['results']['overflows'],
                            'voltage_violations': result['results']['voltage_violations']
                        })
                    
                    # Restore state
                    saw.LoadState()
                    
                except Exception as e:
                    logger.warning(f"Error analyzing contingency: {str(e)}")
                    continue
            
            return {
                'status': 'success',
                'results': {
                    'option': option,
                    'violations': violations,
                    'total_contingencies': len(branch_data),
                    'total_violations': len(violations)
                }
            }
        else:
            return {
                'status': 'error',
                'message': f"Unsupported contingency option: {option}"
            }
            
    except PowerWorldError as e:
        return {
            'status': 'error',
            'message': f"PowerWorld Error: {str(e)}"
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Unexpected Error: {str(e)}"
        }

@tool(args_schema=GetPowerFlowResultsInput)
def get_power_flow_results(object_type: str, additional_fields: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Get power flow results for specified object type with optional additional fields.
    
    Args:
        object_type: Type of power system object ('bus', 'gen', 'load', 'shunt', 'branch')
        additional_fields: Optional list of additional fields to retrieve
            
    Returns:
        Dictionary containing power flow results
    """
    logger.info(f"Getting power flow results for: {object_type}")
    try:
        saw = _get_saw()
        
        # Get results using SAW's get_power_flow_results
        results = saw.get_power_flow_results(object_type, additional_fields)
        
        if results is None:
            return {
                'status': 'error',
                'message': f'No results found for object type: {object_type}'
            }
            
        return {
            'status': 'success',
            'results': results.to_dict('records')  # Convert DataFrame to list of dicts
        }
        
    except PowerWorldError as e:
        return {
            'status': 'error',
            'message': f"PowerWorld Error: {str(e)}"
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Unexpected Error: {str(e)}"
        }

@tool(args_schema=OpenCaseAndRunPowerFlowInput)
def open_case_and_run_powerflow(case_path: str, solution_method: str = 'RECTNEWT') -> Dict[str, Any]:
    """
    Open a PowerWorld case file and immediately run power flow analysis.
    
    This combined function performs both operations in a single tool call for efficiency.
    
    Args:
        case_path: Path to the PowerWorld case file (.pwb)
        solution_method: Power flow solution method
        
    Returns:
        Dictionary containing case info and power flow results
    """
    logger.info(f"Opening case {case_path} and running power flow with {solution_method}")
    
    try:
        # Ensure COM is initialized for this thread
        pythoncom.CoInitialize()
        
        # Reset any existing SAW instance to avoid conflicts
        global _saw
        if _saw is not None:
            logger.info("Resetting existing SAW instance to avoid conflicts")
            _reset_saw()
        
        # Step 1: Open the case
        case_result = open_case(case_path)
        if case_result['status'] != 'success':
            return case_result
        
        # Step 2: Run power flow analysis
        pf_result = run_powerflow(solution_method)
        
        # Combine results
        return {
            'status': 'success',
            'message': f"Case opened and power flow completed with {solution_method}",
            'case_info': case_result['case_info'],
            'power_flow_results': pf_result['results'] if pf_result['status'] == 'success' else None,
            'power_flow_status': pf_result['status']
        }
        
    except Exception as e:
        # Reset SAW on any error to prevent stuck state
        _reset_saw()
        return {
            'status': 'error',
            'message': f"Failed to open case and run power flow: {str(e)}"
        }

@tool(args_schema=ChangeParametersInput)
def change_parameters(object_type: str, param_list: List[str], value_list: List[List[Any]]) -> Dict[str, Any]:
    """
    Change parameters for specified power system objects.
    
    Args:
        object_type: Type of power system object ('bus', 'gen', 'load', 'shunt', 'branch')
        param_list: List of parameter names (must include key fields)
        value_list: List of value lists for each element to modify
        
    Returns:
        Dictionary containing operation status and results
    """
    logger.info(f"Changing parameters for {object_type}")
    try:
        saw = _get_saw()
        
        # Convert to DataFrame format as expected by ESA
        import pandas as pd
        command_df = pd.DataFrame(value_list, columns=param_list)
        
        # Use SAW's change_parameters_multiple_element_df method which expects a DataFrame
        saw.change_parameters_multiple_element_df(object_type, command_df)
        
        return {
            'status': 'success',
            'message': f'Successfully changed parameters for {len(value_list)} {object_type} objects',
            'results': "Parameters changed successfully"
        }
        
    except PowerWorldError as e:
        return {
            'status': 'error',
            'message': f"PowerWorld Error: {str(e)}"
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Unexpected Error: {str(e)}"
        }

# Additional advanced tools
@tool(args_schema=GetYbusInput)
def get_ybus(full: bool = False) -> Dict[str, Any]:
    """Get the bus admittance matrix (Ybus) of the system."""
    logger.info("Getting Ybus matrix")
    try:
        saw = _get_saw()
        ybus = saw.get_ybus(full=full)
        
        # Get bus numbers for reference
        bus_data = saw.get_power_flow_results('bus')
        bus_numbers = bus_data['BusNum'].tolist() if bus_data is not None else []
        
        # Convert to appropriate format for return
        if full:
            matrix_data = ybus.tolist()
        else:
            matrix_data = {
                'data': ybus.data.tolist(),
                'indices': ybus.indices.tolist(),
                'indptr': ybus.indptr.tolist(),
                'shape': ybus.shape
            }
            
        return {
            'status': 'success',
            'results': {
                'matrix': matrix_data,
                'bus_numbers': bus_numbers
            }
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Error getting Ybus: {str(e)}"
        }

@tool(args_schema=ToGraphInput)
def to_graph(node: str = 'bus', geographic: bool = False, directed: bool = False) -> Dict[str, Any]:
    """Convert the power system case to a NetworkX graph representation."""
    logger.info("Converting case to graph representation")
    try:
        saw = _get_saw()
        graph = saw.to_graph(node=node, geographic=geographic, directed=directed)
        
        # Convert NetworkX graph to serializable format
        graph_data = {
            'nodes': [{'id': n, **graph.nodes[n]} for n in graph.nodes()],
            'edges': [{'source': u, 'target': v, **graph.edges[u, v, k]} 
                     for u, v, k in graph.edges(keys=True)]
        }
        
        return {
            'status': 'success',
            'results': graph_data
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Error converting to graph: {str(e)}"
        }

# Add these composite tools at the end of the file, before the tool lists

@tool(args_schema=OpenCaseAndRunPowerFlowInput)
def baseline_analysis_tool(case_path: str, solution_method: str = 'RECTNEWT') -> Dict[str, Any]:
    """
    Complete baseline analysis tool that opens a case and runs power flow analysis.
    This is the only tool needed for baseline analysis phase.
    
    Args:
        case_path: Path to the PowerWorld case file (.pwb)
        solution_method: Power flow solution method
        
    Returns:
        Dictionary containing complete baseline analysis results
    """
    logger.info(f"Running complete baseline analysis for: {case_path}")
    
    try:
        # Ensure COM is initialized for this thread
        pythoncom.CoInitialize()
        
        # Reset any existing SAW instance to avoid conflicts
        global _saw
        if _saw is not None:
            logger.info("Resetting existing SAW instance for baseline analysis")
            _reset_saw()
        
        # Step 1: Open the case
        case_result = open_case.invoke({'case_path': case_path})
        if case_result['status'] != 'success':
            return case_result
        
        # Step 2: Run power flow analysis
        pf_result = run_powerflow.invoke({'solution_method': solution_method})
        
        # Combine results for baseline analysis
        return {
            'status': 'success',
            'phase': 'baseline_analysis',
            'message': f"Baseline analysis completed successfully with {solution_method}",
            'case_info': case_result['case_info'],
            'power_flow_results': pf_result['results'] if pf_result['status'] == 'success' else None,
            'power_flow_status': pf_result['status'],
            'summary': {
                'buses': case_result['case_info']['num_buses'],
                'branches': case_result['case_info']['num_branches'],
                'generators': case_result['case_info']['num_generators'],
                'converged': pf_result['results']['converged'] if pf_result['status'] == 'success' else False,
                'violations': {
                    'overflows': len(pf_result['results']['overflows']) if pf_result['status'] == 'success' else 0,
                    'voltage_violations': len(pf_result['results']['voltage_violations']) if pf_result['status'] == 'success' else 0
                }
            }
        }
        
    except Exception as e:
        # Reset SAW on any error to prevent stuck state
        _reset_saw()
        return {
            'status': 'error',
            'phase': 'baseline_analysis',
            'message': f"Baseline analysis failed: {str(e)}"
        }


class LoadModificationInput(BaseModel):
    """Schema for load modification analysis."""
    bus_number: int = Field(description="Bus number where load will be added")
    load_demand_mw: float = Field(description="Load demand in MW to add")
    solution_method: str = Field(default='RECTNEWT', description="Power flow solution method")

@tool(args_schema=LoadModificationInput)
def load_modification_tool(bus_number: int, load_demand_mw: float, solution_method: str = 'RECTNEWT') -> Dict[str, Any]:
    """
    Complete load modification analysis tool that adds load and runs power flow analysis.
    This is the only tool needed for load modification phase.
    
    Args:
        bus_number: Bus number where load will be added
        load_demand_mw: Load demand in MW to add
        solution_method: Power flow solution method
        
    Returns:
        Dictionary containing complete load modification analysis results
    """
    logger.info(f"Running complete load modification analysis: {load_demand_mw} MW at bus {bus_number}")
    
    try:
        saw = _get_saw()
        
        # Step 1: Add the new load
        # PowerWorld expects load parameters including key fields
        import pandas as pd
        command_df = pd.DataFrame([{
            'BusNum': bus_number,
            'LoadID': '1',
            'LoadMW': load_demand_mw,
            'LoadMVR': 0.0,  # Assume power factor = 1 (no reactive power)
            'LoadStatus': 'Closed'
        }])
        
        # Use SAW's change_parameters_multiple_element_df method directly
        saw.change_parameters_multiple_element_df('load', command_df)
        change_result = {'status': 'success', 'message': 'Load added successfully'}
        if change_result['status'] != 'success':
            return {
                'status': 'error',
                'phase': 'load_modification',
                'message': f"Failed to add load: {change_result['message']}"
            }
        
        # Step 2: Run power flow with the modified system
        pf_result = run_powerflow.invoke({'solution_method': solution_method})
        
        # Step 3: Get updated system results
        bus_results = get_power_flow_results.invoke({'object_type': 'bus'})
        
        return {
            'status': 'success',
            'phase': 'load_modification',
            'message': f"Load modification completed: {load_demand_mw} MW added to bus {bus_number}",
            'modification_details': {
                'bus_number': bus_number,
                'load_added_mw': load_demand_mw,
                'load_added_mvar': 0.0
            },
            'power_flow_results': pf_result['results'] if pf_result['status'] == 'success' else None,
            'power_flow_status': pf_result['status'],
            'bus_results': bus_results['results'] if bus_results['status'] == 'success' else None,
            'summary': {
                'converged': pf_result['results']['converged'] if pf_result['status'] == 'success' else False,
                'violations': {
                    'overflows': len(pf_result['results']['overflows']) if pf_result['status'] == 'success' else 0,
                    'voltage_violations': len(pf_result['results']['voltage_violations']) if pf_result['status'] == 'success' else 0
                }
            }
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'phase': 'load_modification',
            'message': f"Load modification failed: {str(e)}"
        }


class LoadModificationAndContingencyInput(BaseModel):
    """Schema for combined load modification and contingency analysis."""
    bus_number: int = Field(description="Bus number where load will be added")
    load_demand_mw: float = Field(description="Load demand in MW to add")
    solution_method: str = Field(default='RECTNEWT', description="Power flow solution method")
    contingency_option: str = Field(default="N-1", description="Type of contingency analysis ('N-1' or 'N-2')")

@tool(args_schema=LoadModificationAndContingencyInput)
def load_modification_and_contingency_tool(bus_number: int, load_demand_mw: float, solution_method: str = 'RECTNEWT', contingency_option: str = "N-1") -> Dict[str, Any]:
    """
    Complete tool that adds load and runs contingency analysis on the modified system.
    This combines both load modification and contingency analysis in one operation.
    
    Args:
        bus_number: Bus number where load will be added
        load_demand_mw: Load demand in MW to add
        solution_method: Power flow solution method
        contingency_option: Type of contingency analysis ("N-1" or "N-2")
        
    Returns:
        Dictionary containing complete load modification and contingency analysis results
    """
    logger.info(f"Running load modification and {contingency_option} contingency analysis: {load_demand_mw} MW at bus {bus_number}")
    
    try:
        saw = _get_saw()
        
        # Step 1: Add the new load
        import pandas as pd
        command_df = pd.DataFrame([{
            'BusNum': bus_number,
            'LoadID': '1',
            'LoadMW': load_demand_mw,
            'LoadMVR': 0.0,  # Assume power factor = 1 (no reactive power)
            'LoadStatus': 'Closed'
        }])
        
        # Use SAW's change_parameters_multiple_element_df method directly
        saw.change_parameters_multiple_element_df('load', command_df)
        
        # Step 2: Run power flow with the modified system
        pf_result = run_powerflow.invoke({'solution_method': solution_method})
        
        # Step 3: Perform contingency analysis on the modified system
        if contingency_option == "N-1":
            # Get all branches for N-1 analysis
            branch_data = saw.get_power_flow_results('branch')
            violations = []
            
            # Save initial state (with new load)
            saw.SaveState()
            
            # Test each line outage
            for _, branch in branch_data.iterrows():
                try:
                    # Open the line
                    from_bus = int(branch['BusNum'])
                    to_bus = int(branch['BusNum:1'])
                    circuit = branch['LineCircuit']
                    
                    # Use DataFrame format for parameter changes
                    line_command_df = pd.DataFrame([{
                        'BusNum': from_bus,
                        'BusNum:1': to_bus,
                        'LineCircuit': circuit,
                        'LineStatus': 'OPEN'
                    }])
                    saw.change_parameters_multiple_element_df('branch', line_command_df)
                    
                    # Solve power flow
                    saw.SolvePowerFlow()
                    
                    # Check for violations
                    result = run_powerflow.invoke({'solution_method': solution_method})
                    if result['status'] == 'success' and (
                        len(result['results']['overflows']) > 0 or 
                        len(result['results']['voltage_violations']) > 0
                    ):
                        violations.append({
                            'contingency': f"Line {from_bus}-{to_bus} Circuit {circuit}",
                            'overflows': result['results']['overflows'],
                            'voltage_violations': result['results']['voltage_violations']
                        })
                    
                    # Restore state (with new load still in place)
                    saw.LoadState()
                    
                except Exception as e:
                    logger.warning(f"Error analyzing contingency: {str(e)}")
                    continue
            
            contingency_results = {
                'option': contingency_option,
                'violations': violations,
                'total_contingencies': len(branch_data),
                'total_violations': len(violations)
            }
        else:
            return {
                'status': 'error',
                'phase': 'load_modification_and_contingency',
                'message': f"Unsupported contingency option: {contingency_option}"
            }
        
        # Step 4: Get updated system results
        bus_results = get_power_flow_results.invoke({'object_type': 'bus'})
        
        return {
            'status': 'success',
            'phase': 'load_modification_and_contingency',
            'message': f"Load modification and {contingency_option} contingency analysis completed: {load_demand_mw} MW added to bus {bus_number}",
            'modification_details': {
                'bus_number': bus_number,
                'load_added_mw': load_demand_mw,
                'load_added_mvar': 0.0
            },
            'power_flow_results': pf_result['results'] if pf_result['status'] == 'success' else None,
            'power_flow_status': pf_result['status'],
            'contingency_results': contingency_results,
            'bus_results': bus_results['results'] if bus_results['status'] == 'success' else None,
            'summary': {
                'converged': pf_result['results']['converged'] if pf_result['status'] == 'success' else False,
                'violations_modified_system': {
                    'overflows': len(pf_result['results']['overflows']) if pf_result['status'] == 'success' else 0,
                    'voltage_violations': len(pf_result['results']['voltage_violations']) if pf_result['status'] == 'success' else 0
                },
                'contingency_summary': {
                    'total_contingencies': contingency_results['total_contingencies'],
                    'total_violations': contingency_results['total_violations'],
                    'violation_rate': (contingency_results['total_violations'] / contingency_results['total_contingencies'] * 100) if contingency_results['total_contingencies'] > 0 else 0
                }
            }
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'phase': 'load_modification_and_contingency',
            'message': f"Load modification and contingency analysis failed: {str(e)}"
        }

class BaselineContingencyAnalysisInput(BaseModel):
    """Schema for baseline contingency analysis."""
    option: str = Field(default="N-1", description="Type of contingency analysis ('N-1' or 'N-2')")

@tool(args_schema=BaselineContingencyAnalysisInput)
def baseline_contingency_analysis_tool(option: str = "N-1") -> Dict[str, Any]:
    """
    Complete baseline contingency analysis tool that performs N-1 or N-2 analysis on the original system.
    This analyzes the system reliability before any load modifications.
    
    Args:
        option: Type of contingency analysis ("N-1" or "N-2")
        
    Returns:
        Dictionary containing complete baseline contingency analysis results
    """
    logger.info(f"Running baseline {option} contingency analysis")
    
    try:
        saw = _get_saw()
        
        # Run the contingency analysis on baseline system
        if option == "N-1":
            # Get all branches for N-1 analysis
            branch_data = saw.get_power_flow_results('branch')
            violations = []
            
            # Save initial baseline state
            saw.SaveState()
            
            # Test each line outage
            for _, branch in branch_data.iterrows():
                try:
                    # Open the line
                    from_bus = int(branch['BusNum'])
                    to_bus = int(branch['BusNum:1'])
                    circuit = branch['LineCircuit']
                    
                    # Use DataFrame format for parameter changes
                    import pandas as pd
                    line_command_df = pd.DataFrame([{
                        'BusNum': from_bus,
                        'BusNum:1': to_bus,
                        'LineCircuit': circuit,
                        'LineStatus': 'OPEN'
                    }])
                    saw.change_parameters_multiple_element_df('branch', line_command_df)
                    
                    # Solve power flow
                    saw.SolvePowerFlow()
                    
                    # Check for violations
                    result = run_powerflow.invoke({'solution_method': 'RECTNEWT'})
                    if result['status'] == 'success' and (
                        len(result['results']['overflows']) > 0 or 
                        len(result['results']['voltage_violations']) > 0
                    ):
                        violations.append({
                            'contingency': f"Line {from_bus}-{to_bus} Circuit {circuit}",
                            'overflows': result['results']['overflows'],
                            'voltage_violations': result['results']['voltage_violations']
                        })
                    
                    # Restore baseline state
                    saw.LoadState()
                    
                except Exception as e:
                    logger.warning(f"Error analyzing baseline contingency: {str(e)}")
                    continue
            
            contingency_results = {
                'option': option,
                'violations': violations,
                'total_contingencies': len(branch_data),
                'total_violations': len(violations)
            }
        else:
            return {
                'status': 'error',
                'phase': 'baseline_contingency_analysis',
                'message': f"Unsupported contingency option: {option}"
            }
        
        return {
            'status': 'success',
            'phase': 'baseline_contingency_analysis',
            'message': f"Baseline {option} contingency analysis completed successfully",
            'contingency_results': contingency_results,
            'summary': {
                'total_contingencies': contingency_results['total_contingencies'],
                'total_violations': contingency_results['total_violations'],
                'violation_rate': (contingency_results['total_violations'] / contingency_results['total_contingencies'] * 100) if contingency_results['total_contingencies'] > 0 else 0,
                'critical_contingencies': len([v for v in contingency_results['violations'] if len(v['overflows']) > 0 or len(v['voltage_violations']) > 0]),
                'system_reliability': 'Good' if contingency_results['total_violations'] == 0 else 'Needs Attention' if contingency_results['total_violations'] < 5 else 'Poor'
            }
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'phase': 'baseline_contingency_analysis',
            'message': f"Baseline contingency analysis failed: {str(e)}"
        }

class SimpleLoadModificationInput(BaseModel):
    """Schema for simple load modification analysis."""
    bus_number: int = Field(description="Bus number where load will be added")
    load_demand_mw: float = Field(description="Load demand in MW to add")
    solution_method: str = Field(default='RECTNEWT', description="Power flow solution method")

@tool(args_schema=SimpleLoadModificationInput)
def simple_load_modification_tool(bus_number: int, load_demand_mw: float, solution_method: str = 'RECTNEWT') -> Dict[str, Any]:
    """
    Simple load modification tool that adds load and runs power flow analysis only.
    This analyzes the immediate impact of load addition without contingency analysis.
    
    Args:
        bus_number: Bus number where load will be added
        load_demand_mw: Load demand in MW to add
        solution_method: Power flow solution method
        
    Returns:
        Dictionary containing load modification and power flow results
    """
    logger.info(f"Running simple load modification analysis: {load_demand_mw} MW at bus {bus_number}")
    
    try:
        saw = _get_saw()
        
        # Step 1: Add the new load
        import pandas as pd
        command_df = pd.DataFrame([{
            'BusNum': bus_number,
            'LoadID': '1',
            'LoadMW': load_demand_mw,
            'LoadMVR': 0.0,  # Assume power factor = 1 (no reactive power)
            'LoadStatus': 'Closed'
        }])
        
        # Use SAW's change_parameters_multiple_element_df method directly
        saw.change_parameters_multiple_element_df('load', command_df)
        
        # Step 2: Run power flow with the modified system
        pf_result = run_powerflow.invoke({'solution_method': solution_method})
        
        # Step 3: Get updated system results for comparison
        bus_results = get_power_flow_results.invoke({'object_type': 'bus'})
        branch_results = get_power_flow_results.invoke({'object_type': 'branch'})
        
        return {
            'status': 'success',
            'phase': 'simple_load_modification',
            'message': f"Simple load modification completed: {load_demand_mw} MW added to bus {bus_number}",
            'modification_details': {
                'bus_number': bus_number,
                'load_added_mw': load_demand_mw,
                'load_added_mvar': 0.0
            },
            'power_flow_results': pf_result['results'] if pf_result['status'] == 'success' else None,
            'power_flow_status': pf_result['status'],
            'bus_results': bus_results['results'] if bus_results['status'] == 'success' else None,
            'branch_results': branch_results['results'] if branch_results['status'] == 'success' else None,
            'summary': {
                'converged': pf_result['results']['converged'] if pf_result['status'] == 'success' else False,
                'new_violations': {
                    'overflows': len(pf_result['results']['overflows']) if pf_result['status'] == 'success' else 0,
                    'voltage_violations': len(pf_result['results']['voltage_violations']) if pf_result['status'] == 'success' else 0
                },
                'impact_assessment': 'Minimal' if (pf_result['status'] == 'success' and len(pf_result['results']['overflows']) == 0 and len(pf_result['results']['voltage_violations']) == 0) else 'Moderate' if (pf_result['status'] == 'success' and (len(pf_result['results']['overflows']) + len(pf_result['results']['voltage_violations'])) < 5) else 'Significant'
            }
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'phase': 'simple_load_modification',
            'message': f"Simple load modification failed: {str(e)}"
        }

class ContingencyAnalysisInput(BaseModel):
    """Schema for contingency analysis on modified system."""
    option: str = Field(default="N-1", description="Type of contingency analysis ('N-1' or 'N-2')")

@tool(args_schema=ContingencyAnalysisInput)
def modified_contingency_analysis_tool(option: str = "N-1") -> Dict[str, Any]:
    """
    Complete contingency analysis tool that performs N-1 or N-2 analysis on the modified system.
    This analyzes system reliability after load modifications have been made.
    
    Args:
        option: Type of contingency analysis ("N-1" or "N-2")
        
    Returns:
        Dictionary containing complete contingency analysis results
    """
    logger.info(f"Running modified system {option} contingency analysis")
    
    try:
        saw = _get_saw()
        
        # Run the contingency analysis on modified system
        if option == "N-1":
            # Get all branches for N-1 analysis
            branch_data = saw.get_power_flow_results('branch')
            violations = []
            
            # Save initial state (with load modifications)
            saw.SaveState()
            
            # Test each line outage
            for _, branch in branch_data.iterrows():
                try:
                    # Open the line
                    from_bus = int(branch['BusNum'])
                    to_bus = int(branch['BusNum:1'])
                    circuit = branch['LineCircuit']
                    
                    # Use DataFrame format for parameter changes
                    import pandas as pd
                    line_command_df = pd.DataFrame([{
                        'BusNum': from_bus,
                        'BusNum:1': to_bus,
                        'LineCircuit': circuit,
                        'LineStatus': 'OPEN'
                    }])
                    saw.change_parameters_multiple_element_df('branch', line_command_df)
                    
                    # Solve power flow
                    saw.SolvePowerFlow()
                    
                    # Check for violations
                    result = run_powerflow.invoke({'solution_method': 'RECTNEWT'})
                    if result['status'] == 'success' and (
                        len(result['results']['overflows']) > 0 or 
                        len(result['results']['voltage_violations']) > 0
                    ):
                        violations.append({
                            'contingency': f"Line {from_bus}-{to_bus} Circuit {circuit}",
                            'overflows': result['results']['overflows'],
                            'voltage_violations': result['results']['voltage_violations']
                        })
                    
                    # Restore state (with load modifications)
                    saw.LoadState()
                    
                except Exception as e:
                    logger.warning(f"Error analyzing modified system contingency: {str(e)}")
                    continue
            
            contingency_results = {
                'option': option,
                'violations': violations,
                'total_contingencies': len(branch_data),
                'total_violations': len(violations)
            }
        else:
            return {
                'status': 'error',
                'phase': 'modified_contingency_analysis',
                'message': f"Unsupported contingency option: {option}"
            }
        
        return {
            'status': 'success',
            'phase': 'modified_contingency_analysis',
            'message': f"Modified system {option} contingency analysis completed successfully",
            'contingency_results': contingency_results,
            'summary': {
                'total_contingencies': contingency_results['total_contingencies'],
                'total_violations': contingency_results['total_violations'],
                'violation_rate': (contingency_results['total_violations'] / contingency_results['total_contingencies'] * 100) if contingency_results['total_contingencies'] > 0 else 0,
                'critical_contingencies': len([v for v in contingency_results['violations'] if len(v['overflows']) > 0 or len(v['voltage_violations']) > 0]),
                'system_reliability_after_modification': 'Good' if contingency_results['total_violations'] == 0 else 'Needs Attention' if contingency_results['total_violations'] < 5 else 'Poor'
            }
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'phase': 'modified_contingency_analysis',
            'message': f"Modified system contingency analysis failed: {str(e)}"
        }

# Focused tool lists for each analysis phase
BASELINE_TOOLS = [baseline_analysis_tool]
BASELINE_CONTINGENCY_TOOLS = [baseline_contingency_analysis_tool]
SIMPLE_LOAD_MODIFICATION_TOOLS = [simple_load_modification_tool]
MODIFIED_CONTINGENCY_TOOLS = [modified_contingency_analysis_tool]

# Legacy tools (kept for backward compatibility)
LOAD_MODIFICATION_TOOLS = [load_modification_tool]
CONTINGENCY_TOOLS = [modified_contingency_analysis_tool]
COMBINED_MODIFICATION_CONTINGENCY_TOOLS = [load_modification_and_contingency_tool]

# Enhanced tool list for Grid Impact Study (all phases)
GRID_IMPACT_TOOLS = [
    baseline_analysis_tool,
    baseline_contingency_analysis_tool,
    simple_load_modification_tool,
    modified_contingency_analysis_tool
]

# Focused tool list with the most comprehensive tool for chat
CHAT_TOOLS = [
    open_case_and_run_powerflow
]

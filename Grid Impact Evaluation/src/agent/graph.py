import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from agent.state import GridImpactStudyState
from agent.configuration import Configuration
from agent.prompts import get_current_date, chat_instructions
from agent.tools_powerworld import BASELINE_TOOLS, BASELINE_CONTINGENCY_TOOLS, SIMPLE_LOAD_MODIFICATION_TOOLS, MODIFIED_CONTINGENCY_TOOLS
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)

# Configure LangSmith for agent visualization
def setup_langsmith():
    """Setup LangSmith tracing for agent visualization."""
    # Set default values if not provided
    if os.getenv("LANGCHAIN_TRACING_V2") is None:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
    
    if os.getenv("LANGCHAIN_PROJECT") is None:
        os.environ["LANGCHAIN_PROJECT"] = "grid-impact-study-agent"
    
    # Print LangSmith configuration status
    if os.getenv("LANGCHAIN_API_KEY"):
        print(f"✅ LangSmith tracing enabled for project: {os.getenv('LANGCHAIN_PROJECT')}")
        print(f"🔗 View traces at: https://smith.langchain.com/o/default/projects/p/{os.getenv('LANGCHAIN_PROJECT').replace('-', '_')}")
    else:
        print("⚠️  LangSmith API key not found. Set LANGCHAIN_API_KEY to enable tracing.")
        print("   Get your API key at: https://smith.langchain.com/")

setup_langsmith()

if os.getenv("OPENAI_API_KEY") is None:
    raise ValueError("OPENAI_API_KEY is not set")


def input_collection_node(state: GridImpactStudyState, config: RunnableConfig) -> GridImpactStudyState:
    """
    Collect user inputs for the Grid Impact Study.
    
    This node prompts the user to provide:
    1. PowerWorld file path (.pwb)
    2. Bus number for load interconnection
    3. Load demand in MW

    Args:
        state: Current grid impact study state
        config: Configuration for the runnable

    Returns:
        Updated state with user inputs
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Initialize the chat model
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=configurable.temperature,
        max_retries=2,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    # Check if inputs are already collected
    if state.get("inputs_collected", False):
        return state
    
    # Create input collection prompt
    input_prompt = """
Welcome to the Grid Impact Study Agent!

I need three pieces of information to perform your grid impact analysis:

1. **PowerWorld File Path**: Please provide the full path to your PowerWorld case file (.pwb)
2. **Bus Number**: Which bus number do you want to connect the new load to?
3. **Load Demand**: How much load demand (in MW) do you want to interconnect?

Please provide these details, and I'll perform a comprehensive grid impact study including:
- Baseline power flow analysis
- Modified system analysis with your new load
- N-1 contingency analysis to assess system reliability

Example format:
- File path: C:/path/to/your/case.pwb
- Bus number: 1001
- Load demand: 50.0 MW

What are your study parameters?
"""
    
    # Add the prompt to messages if this is the first time
    if not state.get("messages"):
        return {
            **state,
            "messages": [AIMessage(content=input_prompt)]
        }
    
    # Try to extract inputs from the latest user message
    if state["messages"]:
        latest_message = state["messages"][-1]
        
        # Only try to parse if the last message is from the user
        if isinstance(latest_message, HumanMessage) and hasattr(latest_message, 'content'):
            content = latest_message.content.lower()
            
            # Simple parsing logic
            file_path = None
            bus_number = None
            load_demand = None
            
            lines = content.split('\n')
            for line in lines:
                if 'file' in line and ('.pwb' in line or '.pwd' in line):
                    parts = line.split(':')
                    if len(parts) > 1:
                        file_path = ':'.join(parts[1:]).strip()
                elif 'bus' in line and any(char.isdigit() for char in line):
                    import re
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        bus_number = int(numbers[0])
                elif ('load' in line or 'demand' in line or 'mw' in line) and any(char.isdigit() for char in line):
                    import re
                    numbers = re.findall(r'\d+\.?\d*', line)
                    if numbers:
                        load_demand = float(numbers[0])
            
            if file_path and bus_number and load_demand:
                return {
                    **state,
                    "powerworld_file_path": file_path,
                    "bus_number": bus_number,
                    "load_demand_mw": load_demand,
                    "inputs_collected": True,
                    "messages": state["messages"] + [AIMessage(content=f"""
Perfect! I've collected your study parameters:
- PowerWorld File: {file_path}
- Bus Number: {bus_number}
- Load Demand: {load_demand} MW

Now I'll proceed with the Grid Impact Study. Let's begin!
""")]
                }

    # If inputs not complete after a user message, ask for clarification.
    if state["messages"] and isinstance(state["messages"][-1], HumanMessage):
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content="""
I need all three parameters to proceed. Please provide the PowerWorld file path, bus number, and load demand in MW.
""")]
        }

    return state


def baseline_analysis_node(state: GridImpactStudyState, config: RunnableConfig) -> GridImpactStudyState:
    """Perform baseline power flow analysis on the original system."""
    configurable = Configuration.from_runnable_config(config)
    llm = ChatOpenAI(model="gpt-4o", temperature=configurable.temperature, max_retries=2, api_key=os.getenv("OPENAI_API_KEY")).bind_tools(BASELINE_TOOLS)
    
    # Check if we're returning from tools with results
    messages = state.get("messages", [])
    if messages:
        # Look for ToolMessage in recent messages
        for i in range(len(messages)-1, max(0, len(messages)-5), -1):
            msg = messages[i]
            if hasattr(msg, '__class__') and 'ToolMessage' in msg.__class__.__name__:
                # We have tool results! Process them
                tool_content = getattr(msg, 'content', '')
                
                # Now ask LLM to interpret the results
                interpretation_prompt = f"""
Based on the PowerWorld tool results:
{tool_content}

Please analyze these baseline power flow results and provide a summary of:
1. Whether the power flow converged successfully
2. Any violations found (voltage or line overloads)
3. Overall system health assessment

Be factual and base your response only on the data provided.
"""
                response = llm.invoke([HumanMessage(content=interpretation_prompt)])
                
                return {
                    **state,
                    "baseline_results": tool_content,
                    "baseline_complete": True,
                    "messages": state["messages"] + [response]
                }
    
    # First time through - request baseline analysis
    baseline_prompt = f"""
You are performing a baseline power flow analysis.
PowerWorld file: {state['powerworld_file_path']}

IMPORTANT: Use the `baseline_analysis_tool` with the exact file path provided.
This tool will open the case and run power flow analysis in one operation.
Do not generate any error messages unless the tool actually returns an error.
"""
    response = llm.invoke([HumanMessage(content=baseline_prompt)])
    
    return {
        **state,
        "messages": state["messages"] + [response]
    }


def baseline_contingency_analysis_node(state: GridImpactStudyState, config: RunnableConfig) -> GridImpactStudyState:
    """Perform contingency analysis on the baseline system."""
    configurable = Configuration.from_runnable_config(config)
    llm = ChatOpenAI(model="gpt-4o", temperature=configurable.temperature, max_retries=2, api_key=os.getenv("OPENAI_API_KEY")).bind_tools(BASELINE_CONTINGENCY_TOOLS)
    
    # Check if we're returning from tools with results
    messages = state.get("messages", [])
    if messages:
        # Look for ToolMessage in recent messages
        for i in range(len(messages)-1, max(0, len(messages)-5), -1):
            msg = messages[i]
            if hasattr(msg, '__class__') and 'ToolMessage' in msg.__class__.__name__:
                # We have tool results! Process them
                tool_content = getattr(msg, 'content', '')
                
                # Now ask LLM to interpret the results
                interpretation_prompt = f"""
Based on the PowerWorld baseline contingency analysis results:
{tool_content}

Please analyze these baseline contingency results and provide a summary of:
1. System reliability under normal operating conditions
2. Critical contingencies that cause violations
3. Overall system vulnerability assessment
4. Baseline system strengths and weaknesses

Be factual and base your response only on the data provided.
"""
                response = llm.invoke([HumanMessage(content=interpretation_prompt)])
                
                return {
                    **state,
                    "baseline_contingency_results": tool_content,
                    "baseline_contingency_complete": True,
                    "messages": state["messages"] + [response]
                }
    
    # First time through - request baseline contingency analysis
    contingency_prompt = """
Baseline power flow analysis complete. Now perform contingency analysis on the baseline system.

Use the `baseline_contingency_analysis_tool` to:
1. Perform N-1 contingency analysis on the original system
2. Identify critical outages that cause violations
3. Assess baseline system reliability

This analysis establishes the system's inherent vulnerability before load modifications.
"""
    response = llm.invoke([HumanMessage(content=contingency_prompt)])
    
    return {
        **state,
        "messages": state["messages"] + [response]
    }


def simple_load_modification_node(state: GridImpactStudyState, config: RunnableConfig) -> GridImpactStudyState:
    """Add the specified load and run simple power flow analysis."""
    configurable = Configuration.from_runnable_config(config)
    llm = ChatOpenAI(model="gpt-4o", temperature=configurable.temperature, max_retries=2, api_key=os.getenv("OPENAI_API_KEY")).bind_tools(SIMPLE_LOAD_MODIFICATION_TOOLS)
    
    # Check if we're returning from tools with results
    messages = state.get("messages", [])
    if messages:
        # Look for ToolMessage in recent messages
        for i in range(len(messages)-1, max(0, len(messages)-5), -1):
            msg = messages[i]
            if hasattr(msg, '__class__') and 'ToolMessage' in msg.__class__.__name__:
                # We have tool results! Process them
                tool_content = getattr(msg, 'content', '')
                
                # Now ask LLM to interpret the results
                interpretation_prompt = f"""
Based on the PowerWorld load modification results:
{tool_content}

Please analyze these load modification results and provide a summary of:
1. Whether the load was successfully added
2. Power flow convergence after load addition
3. Any new violations caused by the load addition
4. Impact assessment of the load addition on system performance

Be factual and base your response only on the data provided.
"""
                response = llm.invoke([HumanMessage(content=interpretation_prompt)])
                
                return {
                    **state,
                    "modified_results": tool_content,
                    "modification_complete": True,
                    "messages": state["messages"] + [response]
                }
    
    # First time through - request load modification
    modification_prompt = f"""
Baseline contingency analysis complete. Now add the new load and analyze the immediate impact.
Target bus: {state['bus_number']}
Load demand: {state['load_demand_mw']} MW

Use the `simple_load_modification_tool` to:
1. Add the new load at the specified bus
2. Run power flow analysis on the modified system
3. Assess immediate impact without contingency analysis

This step evaluates the direct effect of load addition on system performance.
"""
    response = llm.invoke([HumanMessage(content=modification_prompt)])
    
    return {
        **state,
        "messages": state["messages"] + [response]
    }


def modified_contingency_analysis_node(state: GridImpactStudyState, config: RunnableConfig) -> GridImpactStudyState:
    """Perform contingency analysis on the modified system."""
    configurable = Configuration.from_runnable_config(config)
    llm = ChatOpenAI(model="gpt-4o", temperature=configurable.temperature, max_retries=2, api_key=os.getenv("OPENAI_API_KEY")).bind_tools(MODIFIED_CONTINGENCY_TOOLS)
    
    # Check if we're returning from tools with results
    messages = state.get("messages", [])
    if messages:
        # Look for ToolMessage in recent messages
        for i in range(len(messages)-1, max(0, len(messages)-5), -1):
            msg = messages[i]
            if hasattr(msg, '__class__') and 'ToolMessage' in msg.__class__.__name__:
                # We have tool results! Process them
                tool_content = getattr(msg, 'content', '')
                
                # Now ask LLM to interpret the results
                interpretation_prompt = f"""
Based on the PowerWorld modified system contingency analysis results:
{tool_content}

Please analyze these modified system contingency results and provide a summary of:
1. System reliability after load addition
2. New critical contingencies introduced by the load
3. Comparison with baseline contingency performance
4. Overall impact of load addition on system vulnerability

Be factual and base your response only on the data provided.
"""
                response = llm.invoke([HumanMessage(content=interpretation_prompt)])
                
                return {
                    **state,
                    "contingency_results": tool_content,
                    "contingency_complete": True,
                    "messages": state["messages"] + [response]
                }
    
    # First time through - request modified system contingency analysis
    contingency_prompt = """
Load modification analysis complete. Now perform contingency analysis on the modified system.

Use the `modified_contingency_analysis_tool` to:
1. Perform N-1 contingency analysis on the system with the new load
2. Identify any new critical outages introduced by the load addition
3. Compare system vulnerability with and without the new load

This analysis determines how the load addition affects system reliability.
"""
    response = llm.invoke([HumanMessage(content=contingency_prompt)])
    
    return {
        **state,
        "messages": state["messages"] + [response]
    }


def summary_node(state: GridImpactStudyState, config: RunnableConfig) -> GridImpactStudyState:
    """Generate a comprehensive final summary of the Grid Impact Study with mitigation suggestions."""
    configurable = Configuration.from_runnable_config(config)
    llm = ChatOpenAI(model="gpt-4o", temperature=configurable.temperature, max_retries=2, api_key=os.getenv("OPENAI_API_KEY"))
    
    # Parse the tool results to check actual status
    baseline_status = "Not performed"
    if state.get('baseline_results'):
        results_str = str(state['baseline_results'])
        if '"status": "success"' in results_str:
            baseline_status = "Completed successfully"
        elif '"status": "error"' in results_str:
            baseline_status = "Failed with errors"
    
    summary_prompt = f"""
The comprehensive Grid Impact Study is complete. Please provide a detailed analysis and mitigation recommendations based on the actual tool results provided.

Study Parameters:
- PowerWorld File: {state.get('powerworld_file_path', 'Not specified')}
- Target Bus: {state.get('bus_number', 'Not specified')}
- Load Demand: {state.get('load_demand_mw', 'Not specified')} MW

Analysis Results:
1. Baseline Power Flow Analysis: {baseline_status}
   Raw results: {state.get('baseline_results', 'Not available')}

2. Baseline Contingency Analysis:
   Raw results: {state.get('baseline_contingency_results', 'Not available')}

3. Load Modification Analysis:
   Raw results: {state.get('modified_results', 'Not available')}

4. Modified System Contingency Analysis:
   Raw results: {state.get('contingency_results', 'Not available')}

Please provide a comprehensive summary including:

## Executive Summary
- Overall grid impact assessment
- Key findings and violations

## Detailed Analysis
- Baseline system performance and reliability
- Impact of load addition on power flows and voltages
- Changes in system vulnerability due to load addition
- Critical contingencies before and after load addition

## Mitigation Recommendations
Based on the analysis results, provide specific engineering recommendations such as:
- Infrastructure upgrades (transformers, lines, etc.)
- Operational procedures (load shedding, switching)
- Protection system modifications
- Generation dispatch adjustments
- Power factor correction measures
- Voltage regulation improvements

## Risk Assessment
- Short-term operational risks
- Long-term planning considerations
- Contingency preparedness

## Implementation Priority
- High priority items requiring immediate attention
- Medium priority items for near-term planning
- Low priority items for long-term consideration

Be factual and base all recommendations on the actual analysis results provided.
"""
    response = llm.invoke([HumanMessage(content=summary_prompt)])
    
    return {
        **state,
        "impact_summary": response.content,
        "messages": state["messages"] + [response]
    }


# Define conditional logic for the graph
def should_collect_inputs(state: GridImpactStudyState) -> str:
    return "baseline_analysis" if state.get("inputs_collected") else "input_collection"

def should_proceed_to_baseline_contingency(state: GridImpactStudyState) -> str:
    """Check if baseline analysis has tool calls that need processing."""
    messages = state.get("messages", [])
    if messages and hasattr(messages[-1], 'tool_calls') and messages[-1].tool_calls:
        return "tools"
    return "baseline_contingency_analysis"

def should_proceed_to_load_modification(state: GridImpactStudyState) -> str:
    """Check if baseline contingency has tool calls that need processing."""
    messages = state.get("messages", [])
    if messages and hasattr(messages[-1], 'tool_calls') and messages[-1].tool_calls:
        return "tools"
    return "simple_load_modification"

def should_proceed_to_modified_contingency(state: GridImpactStudyState) -> str:
    """Check if load modification has tool calls that need processing."""
    messages = state.get("messages", [])
    if messages and hasattr(messages[-1], 'tool_calls') and messages[-1].tool_calls:
        return "tools"
    return "modified_contingency_analysis"

def should_proceed_to_summary(state: GridImpactStudyState) -> str:
    """Check if modified contingency has tool calls that need processing."""
    messages = state.get("messages", [])
    if messages and hasattr(messages[-1], 'tool_calls') and messages[-1].tool_calls:
        return "tools"
    return "summary"

def tools_condition(state: GridImpactStudyState) -> str:
    """Route back to appropriate analysis node after tool execution."""
    # Determine which analysis step we came from and should return to
    if not state.get("baseline_complete"):
        return "baseline_analysis"
    elif not state.get("baseline_contingency_complete"):
        return "baseline_contingency_analysis"
    elif not state.get("modification_complete"):
        return "simple_load_modification"
    elif not state.get("contingency_complete"):
        return "modified_contingency_analysis"
    else:
        return "summary"


# Create the tool node with all focused tools
ALL_TOOLS = BASELINE_TOOLS + BASELINE_CONTINGENCY_TOOLS + SIMPLE_LOAD_MODIFICATION_TOOLS + MODIFIED_CONTINGENCY_TOOLS
tool_node = ToolNode(ALL_TOOLS)

# Set up the workflow
workflow = StateGraph(GridImpactStudyState)
workflow.add_node("input_collection", input_collection_node)
workflow.add_node("baseline_analysis", baseline_analysis_node)
workflow.add_node("baseline_contingency_analysis", baseline_contingency_analysis_node)
workflow.add_node("simple_load_modification", simple_load_modification_node)
workflow.add_node("modified_contingency_analysis", modified_contingency_analysis_node)
workflow.add_node("summary", summary_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("input_collection")

workflow.add_conditional_edges(
    "input_collection",
    should_collect_inputs,
    {"baseline_analysis": "baseline_analysis", "input_collection": END}
)
workflow.add_conditional_edges(
    "baseline_analysis",
    should_proceed_to_baseline_contingency,
    {"baseline_contingency_analysis": "baseline_contingency_analysis", "tools": "tools"}
)
workflow.add_conditional_edges(
    "baseline_contingency_analysis",
    should_proceed_to_load_modification,
    {"simple_load_modification": "simple_load_modification", "tools": "tools"}
)
workflow.add_conditional_edges(
    "simple_load_modification",
    should_proceed_to_modified_contingency,
    {"modified_contingency_analysis": "modified_contingency_analysis", "tools": "tools"}
)
workflow.add_conditional_edges(
    "modified_contingency_analysis",
    should_proceed_to_summary,
    {"summary": "summary", "tools": "tools"}
)
workflow.add_edge("summary", END)
workflow.add_conditional_edges("tools", tools_condition, {
    "baseline_analysis": "baseline_analysis",
    "baseline_contingency_analysis": "baseline_contingency_analysis",
    "simple_load_modification": "simple_load_modification",
    "modified_contingency_analysis": "modified_contingency_analysis",
    "summary": "summary"
})

# Compile the graph
graph = workflow.compile()

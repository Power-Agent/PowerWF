# PowerWorkflow ⚡

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PowerWorkflow is an open-source collection of agentic workflows for power system applications. These workflows enable intelligent automation and coordination of power system operations, facilitating efficient management and control in the energy domain.

## 🌟 What is Agentic Workflow?

Agentic workflows are step-by-step processes that process logic together. They provide a structured way to implement intelligent behavior without explicit agent programming. Key features include:

- Automated execution of complex power system operations
- Integration with various power system software and tools ([PowerMCP](https://github.com/power-agent/PowerMCP))
- Reusable workflow components for different use cases
- Flexible orchestration of multiple tools and services

## 🤝 Our Community Vision

We're building an open-source community focused on advancing workflow automation in the power domain. Our goals are:

- **Collaboration**: Connect power system experts, workflow developers, and automation specialists
- **Innovation**: Create and share powerful workflows for various use cases
- **Education**: Provide resources and examples for implementing agentic workflows
- **Standardization**: Develop best practices for workflow design in the energy sector

## 🚀 Getting Started with PowerWorkflow

### Video Demos

Check out these demos showcasing PowerWorkflow in action:

- [**Grid Impact Evaluation for Large Load Interconnection**](https://www.youtube.com/watch?v=LxF8RqsNf04): This demo showcases how Agentic AI can streamline grid impact evaluation for large loads like data centers. Using domain-adapted AI agents equipped with tools such as PowerWorld, the workflow automates data analysis, contingency studies, and report generation.

![Grid Impact Evaluation](Grid%20Impact%20Evaluation/graph.png)

- [**Electricity Market Expert Demo**](https://www.youtube.com/watch?v=ccY93ssvUwU): An automated workflow that handles electricity market inquiries, generates professional responses using an LLM, logs actions for traceability, and supports human-in-the-loop review.

![Electricity Market Expert Workflow](Electricity%20Market%20Expert/MEWF.png)

### Cross-Engine Transient Simulation

- **[Cross-Engine Transient Simulation Workflow](cross_engine_validation/README.md)**: An agent-facing PowerWF workflow for executing one canonical transient-stability study through two independent simulation MCPs, preserving both outcomes, and delegating deterministic trajectory comparison to PowerMCP. The workflow is engine-agnostic and reports engine/comparator failures explicitly.

### Community Workflow Examples

- [**B2G-Agent: Building-Grid Interaction Toolkit**](B2G-Agent/README.md): A Translator-Actor-Reviewer toolkit and local browser interface for an IEEE 33-bus feeder with ten building energy managers and 15-minute intervals. The complete source snapshot supports the final five-method study (200 tasks repeated three times per method), with [result figures](B2G-Agent/README.md#results), reproduction instructions, and separately hosted data. See [PowerWF setup and snapshot provenance](B2G-Agent/POWERWF.md) for the working-directory layout. B2G-Agent is independently developed and maintained by [Xueyuan Cui](https://github.com/cuixueyuan); ongoing development remains in the canonical [cuixueyuan/B2G-Agent](https://github.com/cuixueyuan/B2G-Agent) repository.

### Useful Agentic Workflow Development Platforms

- [**FlowiseAI**](https://flowiseai.com/): Open-source visual workflow builder
- [**Dify**](https://dify.ai/): AI workflow development platform
- [**Coze**](https://www.coze.com/): Bot development platform
- [**n8n**](https://n8n.io/): No-code workflow builder
- [**Gumloop**](https://gumloop.com/): No-code workflow builder
- [**Relevance AI**](https://relevanceai.com/): No-code Agent and workflow builder
- [**Microsoft Copilot Studio**](https://www.microsoft.com/en-us/microsoft-copilot/microsoft-copilot-studio): Enterprise workflow development

### Representative Agent Orchestration Frameworks

- [**LangGraph**](https://python.langchain.com/docs/langgraph): Python framework for building agent workflows
- [**AutoGen**](https://microsoft.github.io/autogen/): Microsoft's framework for collaborative agents
- [**CrewAI**](https://github.com/joaomdmoura/crewAI): Framework for collaborative AI agents

### Agent to Agent Communication

- [**A2A Protocol**](https://google.github.io/A2A/): An application-level protocol that enables agents to collaborate in their natural modalities

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](https://power-agent.github.io/) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

### Core Team
- Lin Dong, [Qian Zhang](https://www.linkedin.com/in/qian-zhang-75323111b/)

### Special Thanks
- All contributors who help make this project better
- [The Power and AI Initiative (PAI) at Harvard SEAS](https://pai.seas.harvard.edu/)

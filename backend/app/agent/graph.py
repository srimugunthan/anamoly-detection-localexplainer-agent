from langgraph.graph import END, START, StateGraph

from app.agent.nodes.interpret import interpret_node
from app.agent.nodes.lime_tool import lime_node
from app.agent.nodes.pdp_tool import pdp_node
from app.agent.nodes.predict import predict_node
from app.agent.nodes.shap_tool import shap_node
from app.agent.state import ExplainerState


def build_graph():
    """Build and compile the LangGraph StateGraph for anomaly explanation.

    Graph topology:
        START → predict → shap_tool ┐
                        → lime_tool  ├→ interpret → END
                        → pdp_tool  ┘
    """
    graph = StateGraph(ExplainerState)

    graph.add_node("predict", predict_node)
    graph.add_node("shap_tool", shap_node)
    graph.add_node("lime_tool", lime_node)
    graph.add_node("pdp_tool", pdp_node)
    graph.add_node("interpret", interpret_node)

    # Fan-out: predict → each parallel tool node
    graph.add_edge(START, "predict")
    graph.add_edge("predict", "shap_tool")
    graph.add_edge("predict", "lime_tool")
    graph.add_edge("predict", "pdp_tool")

    # Fan-in: all three tool nodes must finish before interpret runs
    graph.add_edge(["shap_tool", "lime_tool", "pdp_tool"], "interpret")

    graph.add_edge("interpret", END)

    return graph.compile()

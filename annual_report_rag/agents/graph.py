"""LangGraph agent for annual report Q&A."""

from __future__ import annotations

import json
import re
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from annual_report_rag.agents.tools.report_tools import TOOL_DEFINITIONS, ReportTools
from annual_report_rag.config import get_settings, load_yaml_config
from annual_report_rag.retrieval import HybridSearch


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    citations: list[dict[str, Any]]
    steps: list[str]


SYSTEM_PROMPT = """你是企业年报分析助手。必须遵守：
1. 先调用 search_annual_report 等工具检索，再回答。
2. 回答中的事实必须来自检索结果，并标注引用 [chunk_id]。
3. 金额、比例等数字优先引用 table 类型切片；不确定时明确说“未在年报检索到”。
4. 不要编造未出现的财务数据。"""


class AnnualReportAgent:
    def __init__(self, search: HybridSearch | None = None) -> None:
        self.settings = get_settings()
        model_cfg = load_yaml_config("models.yaml")["llm"]
        agent_cfg = load_yaml_config("models.yaml")["agent"]
        self.max_steps = agent_cfg.get("max_steps", 6)
        self.enable_critic = agent_cfg.get("enable_critic", True)
        self.tools = ReportTools(search or HybridSearch())
        self.llm = ChatOpenAI(
            model=self.settings.llm_model or model_cfg.get("model", "gpt-4o-mini"),
            temperature=model_cfg.get("temperature", 0.1),
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
        )
        self.llm_with_tools = self.llm.bind_tools(TOOL_DEFINITIONS)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", self._tool_node)
        if self.enable_critic:
            graph.add_node("critic", self._critic_node)
        graph.set_entry_point("agent")
        graph.add_conditional_edges(
            "agent",
            self._route_after_agent,
            {"tools": "tools", "critic": "critic" if self.enable_critic else END, END: END},
        )
        graph.add_edge("tools", "agent")
        if self.enable_critic:
            graph.add_edge("critic", END)
        return graph.compile()

    def _route_after_agent(self, state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        if self.enable_critic:
            return "critic"
        return END

    def _agent_node(self, state: AgentState) -> dict[str, Any]:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = self.llm_with_tools.invoke(messages)
        step = "agent called tools" if response.tool_calls else "agent drafted answer"
        return {"messages": [response], "steps": state.get("steps", []) + [step]}

    def _tool_node(self, state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        tool_messages = []
        citations = list(state.get("citations", []))
        for call in last.tool_calls:
            name = call["name"]
            args = call.get("args", {})
            result = self._dispatch_tool(name, args)
            if name == "search_annual_report":
                try:
                    hits = json.loads(result)
                    for hit in hits[:5]:
                        citations.append(
                            {
                                "chunk_id": hit.get("chunk_id"),
                                "page": hit.get("metadata", {}).get("page_start"),
                                "excerpt": hit.get("content_preview", "")[:200],
                                "citation": hit.get("citation"),
                            }
                        )
                except json.JSONDecodeError:
                    pass
            from langchain_core.messages import ToolMessage

            tool_messages.append(
                ToolMessage(content=result, tool_call_id=call["id"], name=name)
            )
        return {
            "messages": tool_messages,
            "citations": citations,
            "steps": state.get("steps", []) + [f"tools: {[c['name'] for c in last.tool_calls]}"],
        }

    def _critic_node(self, state: AgentState) -> dict[str, Any]:
        last_ai = next(
            (m for m in reversed(state["messages"]) if isinstance(m, AIMessage) and m.content),
            None,
        )
        if not last_ai or not isinstance(last_ai.content, str):
            return {}
        numbers = re.findall(r"\d+\.?\d*", last_ai.content)
        corpus = json.dumps(state.get("citations", []), ensure_ascii=False)
        unsupported = [n for n in numbers if n not in corpus and len(n) > 2]
        if unsupported:
            note = (
                f"\n\n[校验提示] 以下数字可能缺少直接引用，请谨慎采信：{', '.join(unsupported[:5])}"
            )
            return {"messages": [AIMessage(content=last_ai.content + note)]}
        return {}

    def _dispatch_tool(self, name: str, args: dict[str, Any]) -> str:
        fn = getattr(self.tools, name, None)
        if not fn:
            return json.dumps({"error": f"unknown tool {name}"})
        return fn(**args)

    def ask(self, question: str) -> dict[str, Any]:
        state = self.graph.invoke(
            {
                "messages": [HumanMessage(content=question)],
                "citations": [],
                "steps": [],
            }
        )
        answer_msg = next(
            (m for m in reversed(state["messages"]) if isinstance(m, AIMessage) and m.content),
            None,
        )
        answer = answer_msg.content if answer_msg else ""
        citations = state.get("citations", [])
        confidence = "high" if citations else "low"
        return {
            "answer": answer,
            "citations": citations,
            "steps": state.get("steps", []),
            "confidence": confidence,
        }

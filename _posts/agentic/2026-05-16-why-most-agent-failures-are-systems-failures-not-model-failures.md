---
layout: post
title: "Why Most Agent Failures Are Systems Failures, Not Model Failures"
date: 2026-05-16
category: agentic
author:
 - Stephen Oni
 - Opeyemi Bamigbade

excerpt: "In production, agents usually fail less because the model is weak and more because state, tools, observability, and governance are weak."
---

When an agent fails, it is tempting to say: the model is not smart enough yet.

Sometimes that is true. But more often, especially in production, the model is only one part of the failure. The agent lost the right state. It called the wrong tool. It had no safe execution boundary. Nobody could inspect what happened. Or the workflow needed a human checkpoint and did not have one.

That is the useful way to think about agents: not as "LLM plus prompt", but as small runtime systems wrapped around a model.

{% include toc.html %}

![System failures around agents](/images/articles/agent-systems-failure.svg)

## Start With The System

The best public guidance from frontier labs is surprisingly pragmatic here.

OpenAI's practical guide recommends starting with a single agent, adding tools deliberately, setting eval baselines, and introducing human intervention around high-risk actions. {% include sidenote.html ref="https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/" ref_title="OpenAI: A practical guide to building agents" %} Anthropic makes a similar point from a different angle: effective agentic systems are often simple, composable workflows, and tool definitions deserve the same engineering attention as prompts. In its SWE-bench work, Anthropic says it spent more time improving tools than improving the main prompt. {% include sidenote.html ref="https://www.anthropic.com/research/building-effective-agents" ref_title="Anthropic: Building effective agents" %}

That is the signal. The frontier labs are not saying "just wait for the next model." They are saying: build the system around the model carefully.

## Where Agents Actually Fail

### State Is Not Memory

Memory is a loaded word in agent discourse. People use it to mean conversation history, user preferences, retrieved facts, long-term state, task state, and sometimes just "more tokens."

But production agents usually need something more precise: state that can be inspected, resumed, compacted, and updated without blindly stuffing everything back into the prompt.

That is why LangGraph treats persistence and durable execution as core pieces of the framework. Its docs describe saving state at each execution step so long-running workflows can resume after interruptions or human review. {% include sidenote.html ref="https://docs.langchain.com/oss/python/langgraph/durable-execution" ref_title="LangGraph Docs: Durable execution" %} OpenAI's Agents SDK has sessions for conversation state, {% include sidenote.html ref="https://openai.github.io/openai-agents-python/sessions/" ref_title="OpenAI Agents SDK: Sessions" %} and OpenHands documents a condenser that compresses conversation history when it grows beyond the context budget. {% include sidenote.html ref="https://docs.openhands.dev/sdk/arch/condenser" ref_title="OpenHands Docs: Condenser architecture" %}

The lesson is simple: context is not memory, and memory is not automatically useful. The hard part is deciding what state should survive, what should be summarized, and what should be forgotten.

### Tools Break Before Reasoning Does

A lot of "bad reasoning" is just bad tool design wearing a clever disguise.

If a tool schema is vague, the agent guesses. If two tools overlap, the agent may pick the wrong one. If execution is not isolated, a coding agent can touch more than it should. If permissions are too broad, a small planning mistake becomes a business risk.

You can see this in the way open-source agent projects are built. AutoGen provides command-line executors and recommends Docker-based execution for isolation when available. {% include sidenote.html ref="https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/components/command-line-code-executors.html" ref_title="AutoGen Docs: Command line code executors" %} OpenHands recommends Docker sandboxes, labels process mode unsafe, and points production SDK users toward managed workspaces for sandboxing and credential support. {% include sidenote.html ref="https://docs.openhands.dev/openhands/usage/sandboxes/overview" ref_title="OpenHands Docs: Sandbox overview" %}

The real product surface of an agent is often the tool layer. The model may decide, but the tools define what decisions are possible.

### What You Cannot See You Cannot Fix

Once an agent can branch, retry, call tools, hand off work, and recover from failure, a bad result is no longer just a bad output. It is a bad trace.

This is why observability is becoming part of the agent stack itself. OpenAI's Agents SDK traces model generations, tool calls, handoffs, guardrails, and custom events. {% include sidenote.html ref="https://openai.github.io/openai-agents-python/tracing/" ref_title="OpenAI Agents SDK: Tracing" %} AutoGen follows OpenTelemetry conventions for agent and tool tracing. {% include sidenote.html ref="https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tracing.html" ref_title="AutoGen Docs: Tracing and observability" %} CrewAI's tracing exposes agent decisions, task timelines, tool usage, and LLM calls through CrewAI AMP. {% include sidenote.html ref="https://docs.crewai.com/en/observability/tracing" ref_title="CrewAI Docs: Tracing" %}

That convergence matters. It means serious agent builders are treating agents more like distributed systems than like chat prompts. The trace is where you learn whether the failure was planning, retrieval, state, tool arguments, permissions, latency, or final response quality.

### Autonomy Needs Boundaries

There is a common belief that more autonomy means a more advanced agent. I think that is backwards.

In production, good autonomy is bounded autonomy. OpenAI's guide emphasizes guardrails and human intervention for sensitive or irreversible actions. LangGraph's human-in-the-loop patterns are built around interrupting execution, exposing state, and resuming after review. {% include sidenote.html ref="https://docs.langchain.com/oss/python/langgraph/human-in-the-loop" ref_title="LangGraph Docs: Human-in-the-loop" %} OpenAI's Agents SDK also separates input guardrails, output guardrails, and tool-related checks. {% include sidenote.html ref="https://openai.github.io/openai-agents-python/guardrails/" ref_title="OpenAI Agents SDK: Guardrails" %}

The point is not to make the agent timid. The point is to make its freedom legible and governed.

## How the Ecosystem Is Responding

The interesting thing about today's agent ecosystem is that the projects look different on the surface, but they are solving similar operational problems underneath.

- LangGraph leans into persistence, resumability, memory, and human checkpoints.
- AutoGen leans into orchestration patterns, execution environments, and telemetry.
- CrewAI leans into memory plus operational tracing through AMP.
- OpenHands leans into sandboxes, workspace isolation, and long-context management.

Commercial products are moving in the same direction. OpenAI's AgentKit is framed around workflow versioning, connector governance, trace grading, prompt optimization, and embedded agent UI. {% include sidenote.html ref="https://openai.com/index/introducing-agentkit/" ref_title="OpenAI: Introducing AgentKit" %} OpenAI's platform docs also position trace grading and prompt optimization as ways to monitor and improve agents after they are built. {% include sidenote.html ref="https://platform.openai.com/docs/guides/agent-evals" ref_title="OpenAI Platform Docs: Agent evals" %}

That is not just a tooling story. It is a market signal. Agent products are becoming systems products.

## The Business Meaning

For technical teams, the implication is straightforward: better prompts and stronger models still help, but they are no longer enough to make an agent reliable in production.

For business teams, the implication is even more important: agent ROI depends less on whether a model can produce a brilliant demo and more on whether the surrounding system can remember the right things, use the right tools safely, expose failures clearly, and hand control back when it should.

The model may be the brain of the agent. The systems layer is what keeps it employable.

# BDP-v1 Wire Protocol

BDP-v1 is a dense key-value protocol for communication **between** AI agents.
Human language is used only at the entrance and exit.

## 1. Front door — human to spec

The orchestrator compiles a natural-language request into one line:

```text
BDP:v1|INTENT:<intent>|TGT:<target>|REQ:<constraints>|GATE:<verification>
```

Example:

```text
BDP:v1|INTENT:CODE_SYNTH|TGT:ring_buffer|REQ:BOUNDS_SAFE+NULL_GUARD|GATE:ASAN_CLEAN
```

## 2. Internal bus — strategist to verifier

The strategist responds with a reasoning scratchpad and one packet:

```text
<think>[one line of reasoning]</think>
BDP:v1|NODE:STRATEGIST|VOTE:<A|B|C>|ROOT_CAUSE:<short>|VERIFY:<short>
```

The verifier responds with its own packet:

```text
<think>[one line of check]</think>
BDP:v1|NODE:VERIFIER|STAT:<PASS|FAIL>|ISSUE:<short>|CONF:<0-100>
```

## 3. Back door — result to human language

The verified packet is translated back into a human answer. In the benchmark,
the orchestrator emits the final option:

```text
FINAL: B
```

## The Anti-Rush rule

Compression must never remove reasoning. Every agent gets an isolated
`<think>` scratchpad governed by this instruction:

> "Compress the syntax, never the reasoning. Verify bounds, edge cases and
> failure modes inside your `<think>` scratchpad BEFORE emitting the dense
> telemetry vector. Never sacrifice architectural depth for brevity."

The scratchpad stays local. Only the packet crosses the wire.

## Schema checks used in the benchmark

A BDP run counts as schema-compliant when:

- the spec contains `BDP:v1|` and `INTENT`, `TGT`, `REQ`, `GATE`;
- the strategist packet contains `BDP:v1|`, `VOTE:`, and `ROOT_CAUSE`;
- the verifier packet contains `BDP:v1|`, `STAT:`, and `CONF`.

The final answer is parsed from `FINAL: <A|B|C>`.

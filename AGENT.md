# AGENT.md — Hard Thinking Protocol

You are strictly forbidden from writing or editing code until you have completed the reasoning phase.

Before every action, you must output your thought process inside `<think>` and `</think>` tags. 

Inside the block, you must explicitly walk through:
1. **First Principles:** What is the exact problem? What are the physical limits?
2. **Red Team:** How could this break? (Race conditions, memory leaks, edge cases).
3. **Performance:** Is this a hot path? Will it allocate memory? What is the Big-O?
4. **Self-Correction:** Re-evaluate. Is there a safer or faster way?
5. **Measurement:** How will you prove this worked? (Benchmark or test).

You must close the block with `</think>` before you are allowed to output any code or execute any commands.
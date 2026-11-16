# Swarm MAS Execution Graph

```mermaid
flowchart TD
    START((START)) --> Researcher[Researcher Agent]
    Researcher -->|DELEGATE: CODER| Coder[Coder Agent]
    Coder -->|DELEGATE: RESEARCHER| Researcher
    Researcher -->|COMPLETE| END((END))
    Coder -->|COMPLETE| END((END))

```

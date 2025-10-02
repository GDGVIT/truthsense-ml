# System Architecture

**Flow:** Video → (Audio, Frames) → Audio & Video feature extractors → Feature fusion → LLM → JSON → Frontend UI.

``` mermaid
flowchart LR
    A[Video Upload] --> B[Audio Extract]
    A --> C[Frame Extract]
    B --> D[Audio Features/Models]
    C --> E[Video Landmarks/Features]
    D --> F[Fusion]
    E --> F
    F --> G[LLM: Report JSON]
    G --> H[Rendered Feedback]
```
# Frame Finder Workflow Diagrams

## Current Workflow

```mermaid
graph TD
    A[User uploads files] --> B[Process videos]
    B --> C[Extract frames at interval]
    C --> D[Compare frames with CLIP]
    D --> E{Confidence >= threshold?}
    E -->|Yes| F[Store match]
    E -->|No| G[Discard result]
    F --> H[Apply temporal clustering]
    H --> I[Return results]
    G --> H
    I --> J[Display results]
```

## Proposed Workflow

```mermaid
graph TD
    A[User uploads files] --> B[Process videos]
    B --> C[Extract frames at interval]
    C --> D[Compare frames with CLIP]
    D --> E[Store all matches]
    E --> F[Apply temporal clustering]
    F --> G[Return all results with metadata]
    G --> H[Display results]
    H --> I[Dynamic filtering slider]
    I --> J[Show/hide results based on threshold]
```

## Settings Persistence Flow

```mermaid
graph TD
    A[User visits main page] --> B{Settings in localStorage?}
    B -->|Yes| C[Restore settings]
    B -->|No| D[Use defaults]
    C --> E[Display form]
    D --> E
    E --> F[User adjusts settings]
    F --> G[User submits form]
    G --> H[Save settings to localStorage]
    H --> I[Process videos]
```

## Time Display Flow

```mermaid
graph TD
    A[Backend returns timestamp in seconds] --> B[Template receives data]
    B --> C[JavaScript formats timestamp]
    C --> D[Display in hh:mm:ss format]
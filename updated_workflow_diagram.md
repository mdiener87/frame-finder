# Frame Finder Updated Workflow Diagrams

## Main Page Workflow with Collapsible Panels

```mermaid
graph TD
    A[User visits main page] --> B{Settings in localStorage?}
    B -->|Yes| C[Restore settings]
    B -->|No| D[Use defaults]
    C --> E[Display collapsible panels]
    D --> E
    E --> F[Settings panel expanded]
    E --> G[Progress panel collapsed]
    
    G --> H[User adjusts settings]
    H --> I[User clicks Analyze Videos]
    I --> J[Save settings to localStorage]
    J --> K[Disable settings panel]
    K --> L[Collapse settings panel]
    L --> M[Expand progress panel]
    M --> N[Start processing in background]
    
    N --> O{Task completed or cancelled?}
    O -->|Completed| P[Redirect to results page]
    O -->|Cancelled| Q[Enable settings panel]
    Q --> R[Expand settings panel]
    R --> S[Collapse progress panel]
```

## Analysis Process with Cancellation Support

```mermaid
graph TD
    A[Background processing starts] --> B[Extract frames at interval]
    B --> C[Compare frames with CLIP]
    C --> D[Store all matches]
    D --> E[Apply temporal clustering]
    E --> F{Cancellation requested?}
    F -->|Yes| G[Stop processing]
    F -->|No| H{More videos to process?}
    H -->|Yes| B
    H -->|No| I[Return all results with metadata]
    I --> J[Update task status to completed]
    
    G --> K[Update task status to cancelled]
```

## Results Page with Multi-Video Filtering

```mermaid
graph TD
    A[Display results for all videos] --> B[Show dynamic filter sliders]
    B --> C[User adjusts slider for Video 1]
    C --> D[Filter results for Video 1]
    D --> E[Update visible count for Video 1]
    
    B --> F[User adjusts slider for Video 2]
    F --> G[Filter results for Video 2]
    G --> H[Update visible count for Video 2]
    
    B --> I[User adjusts slider for Video N]
    I --> J[Filter results for Video N]
    J --> K[Update visible count for Video N]
```

## Cancel Confirmation Flow

```mermaid
graph TD
    A[User clicks Cancel button] --> B[Show confirmation dialog]
    B --> C{User confirms cancellation?}
    C -->|Yes| D[Send cancel request to backend]
    D --> E[Update UI to cancelled state]
    E --> F[Enable settings panel]
    F --> G[Expand settings panel]
    G --> H[Collapse progress panel]
    
    C -->|No| I[Close confirmation dialog]
    I --> J[Continue processing]
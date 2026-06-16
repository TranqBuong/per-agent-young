# Multi-Agent Quality Engineer — Workflow Diagram

## 1. High-Level System Overview

```mermaid
flowchart TD
    USER(["👤 User"])

    subgraph FE["Frontend (index.html)"]
        INPUT["Requirement Input\n(text / file upload / URL)"]
        TABS["Result Tabs\nOverview · Scenarios · Test Cases\nCode Files · RTM"]
    end

    subgraph API["Backend API (FastAPI)"]
        EP1["/preview-requirements"]
        EP2["/analyze-requirements"]
        EP3["/generate-test-cases"]
        EP4["/generate-automation-code"]
        EP5["/run-full-pipeline"]
        CACHE["File-Based Cache\n(SHA256 hash key, TTL 2 days)"]
    end

    subgraph AGENTS["AI Agents (Groq · llama-3.1-8b / llama-3.3-70b)"]
        A1["Agent 1\nRequirementAnalyzer"]
        A2["Agent 2\nTestCasePayloadGenerator"]
        A3["Agent 3\nAutomationCodeWriter"]
    end

    STORE[("Output Store\nbackend/app/output/\n*.json · *.md")]

    USER --> INPUT
    INPUT --> EP1
    INPUT --> EP2
    EP2 --> EP3
    EP3 --> EP4
    INPUT --> EP5

    EP1 --> A1
    EP2 --> A1
    EP3 --> A2
    EP4 --> A3
    EP5 --> A1
    EP5 --> A2
    EP5 --> A3

    A1 & A2 & A3 --> CACHE
    EP5 --> STORE
    CACHE --> TABS
    TABS --> USER
```

---

## 2. Step-by-Step Data Flow

```mermaid
flowchart LR
    REQ["📄 Requirement Text\n(≤ 8000 chars)"]

    subgraph STEP1["Step 1 — Requirement Analysis"]
        direction TB
        P1["preview-requirements\n(fast scan)"]
        A1_CALLS["LLM Call 1: quality_checks\nLLM Call 2: scenarios + summary"]
        A1_OUT["Output:\n• overview (summary, features,\n  endpoints, business_rules)\n• suggestions[]\n• quality_score (overall 0-100)\n• requirements_summary[]\n• scenarios[]\n• missing_information[]"]
    end

    subgraph STEP2["Step 2 — Test Case Generation"]
        direction TB
        P2["generate-test-cases"]
        A2_CALLS["LLM Call 1: technique selection\nLLM Call 2: test case generation\n(batched per technique)"]
        A2_VALIDATE["Grounding Validation\n• drop hallucinated scenario_ids\n• add fallback TC for uncovered scenarios\n• rebuild TC IDs: TC-TECH-NNN"]
        A2_OUT["Output:\n• system_type (api/web/mobile)\n• applied_techniques[]\n• test_cases[]\n• payload_templates{}\n• test_data_matrix{}"]
    end

    subgraph STEP3["Step 3 — Code Generation"]
        direction TB
        P3["generate-automation-code"]
        A3_CALLS["LLM Call: full test suite\n(separator format)"]
        A3_PARSE["Parse ===FILE:path===...===END===\nblocks → GeneratedCodeFile[]"]
        A3_OUT["Output:\n• framework (pytest/playwright/…)\n• generated_files[]\n  - file_name\n  - code\n  - explanation"]
    end

    ARTIFACTS[("💾 Artifacts\nfull_pipeline_output.json\nfull_pipeline_output.md")]

    REQ --> STEP1
    A1_OUT --> STEP2
    A2_OUT --> STEP3
    A3_OUT --> ARTIFACTS
```

---

## 3. Agent 1 — RequirementAnalyzer (Detail)

```mermaid
flowchart TD
    IN["requirement_text (truncated to 1500 chars)"]

    subgraph CALL1["LLM Call 1 — Quality Checks"]
        QC["quality_checks JSON:\n• completeness (6 boolean flags)\n• testability (5 boolean flags)\n• clarity_positive (4 boolean flags)\n• clarity_issues (vague words, undefined terms…)"]
    end

    subgraph SCORE["Deterministic Scoring (no LLM arithmetic)"]
        COMP["Completeness\n6 flags × ~17pt each → 0–100"]
        TEST["Testability\n5 flags × 20pt each → 0–100"]
        AMB["Clarity/Ambiguity\nbase 50 + bonuses − deductions → 0–100"]
        OVR["Overall = C×0.40 + T×0.35 + A×0.25\nRisk: ≥70 Low · ≥40 Medium · else High"]
    end

    subgraph CALL2["LLM Call 2 — Scenarios + Summary"]
        SCN["scenarios[]:\n• scenario_id (SCN-NNN)\n• title, description\n• given / when / then\n• priority (high/medium/low)\n• type (positive/negative/boundary…)\n• related_endpoint"]
        SUM["requirements_summary[]:\n• id (REQ-NNN)\n• text\n• category"]
        MISS["missing_information[]"]
    end

    IN --> CALL1 --> SCORE
    IN --> CALL2
    SCORE & CALL2 --> OUT["RequirementAnalysisResult"]
```

---

## 4. Agent 2 — TestCasePayloadGenerator (Detail)

```mermaid
flowchart TD
    IN2["scenarios[] + requirement_text\n+ overview"]

    subgraph CALL1_2["LLM Call 1 — Technique Selection"]
        TECH["applied_techniques[]:\n• technique (EP/BVA/DT/ST/UC/PT/EG…)\n• rationale\n• applicable_scenarios[]"]
        STYPE["system_type: api | web | mobile | mixed"]
    end

    subgraph CALL2_2["LLM Call 2 — Test Case Generation\n(slim scenario view, per-technique batch)"]
        TC["test_cases[]:\n• test_case_id (rebuilt post-gen)\n• name (≤50 chars)\n• scenario_id\n• technique\n• priority\n• type\n• preconditions\n• steps[]\n• expected_result\n• test_data{}\n• tags[]"]
        TMPL["payload_templates{}"]
        MATRIX["test_data_matrix{}"]
    end

    subgraph VALIDATE["Post-Processing"]
        DROP["Drop TCs with invalid scenario_id"]
        FILL["Add fallback TC for uncovered scenarios"]
        REBUILD["Rebuild IDs: TC-{TECH}-{NNN}"]
    end

    IN2 --> CALL1_2
    CALL1_2 --> CALL2_2
    CALL2_2 --> VALIDATE
    VALIDATE --> OUT2["TestCasePayloadResult"]
```

---

## 5. Agent 3 — AutomationCodeWriter (Detail)

```mermaid
flowchart TD
    IN3["test_cases[] + framework\n(pytest/playwright/k6/selenium/postman)\n+ requirement_text\n+ payload_templates + test_data_matrix"]

    subgraph CALL_3["LLM Call — Full Test Suite"]
        SEP["Separator format output:\n===FILE:path/to/test.py===\n...code...\n===END==="]
    end

    subgraph PARSE_3["Parser"]
        REGEX["Regex split on FILE/END markers"]
        FB["Fallback: JSON parse if no markers found"]
    end

    IN3 --> CALL_3 --> PARSE_3
    PARSE_3 --> OUT3["AutomationCodeResult:\n• framework\n• generated_files[]\n  - file_name\n  - code\n  - explanation"]
```

---

## 6. API Endpoint Map

```mermaid
flowchart LR
    subgraph ENDPOINTS["REST API"]
        H["GET /health"]
        PF["POST /parse-file\n(txt, pdf, docx, md, yaml, json)"]
        FU["GET /fetch-url"]
        PR["POST /preview-requirements\n→ Agent 1 preview only"]
        AR["POST /analyze-requirements\n→ Agent 1 full"]
        GT["POST /generate-test-cases\n→ Agent 2"]
        GC["POST /generate-automation-code\n→ Agent 3"]
        FP["POST /run-full-pipeline\n→ Agent 1 + 2 + 3 sequential"]
    end

    subgraph CACHE_MAP["Cache Layer"]
        C1["cache key = SHA256(text)[:16]\nhit → skip LLM, return cached\nTTL 2 days, similarity fallback"]
    end

    PR & AR & GT & GC & FP --> C1
```

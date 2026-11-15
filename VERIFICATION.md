# Prototype verification

- Generated 600 unique users, 498 leads, 605 offers, 540 shown offers, 181 booked users and 378 traces.
- Eight pipeline assertions passed: session identity, booking join cardinality, exposure references, replay removal, malformed JSON quarantine, assignment uniqueness, user grain and explanation additivity.
- Independently recalculated the two-sided Fisher exact p-value using integer combinations: 0.016433653418818646, matching the dashboard's 0.0164.
- Browser checks covered every dashboard tab, workflow-to-conversion navigation, campaign Search → All (64 → 181 converted users), credit-bracket filtering, lead selection and an information-button disclosure.
- Desktop and 390px mobile renders inspected; mobile document width stayed 390px. Wide evidence tables scroll inside their panels.
- No browser page errors occurred during the interaction run.
- Final build passed authored-content verification and is marked complete.

No live model training, LLM judging, production orchestration or external publishing was performed. Synthetic labels and fixed model coefficients are intentionally identified in the dashboard and assessment brief.

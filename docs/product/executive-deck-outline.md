<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# Executive Presentation Deck Outline (5 Slides)

> Purpose: an initial presentation to customer decision-makers (Chairperson/CEO/CIO), with a recommended speaking time of 10–15 minutes.
> Each slide includes a title, key message (one conclusion per slide), suggested visual, and speaker notes.
> Companion materials: the two-page Executive Summary and the full Product Value Whitepaper.
>
> **Narrative arc:** resonate with the pain points (Slides 1–2) → present the solution and value (Slides 3–4) → recommend action (Slide 5). Avoid feature lists and speak the language of business throughout.

---

## Slide 1 | We Have Run Plenty of AI Pilots—Why Are They Still Not in Production?

**Key conclusion: What organizations lack is not models, but a compliant model production line.**

- Left half: icons for three familiar customer scenarios—production-line quality inspection, camera surveillance, and drone inspection—with the line: "The AI that creates real value runs in the field."
- Right half: three barriers (large text and icons, with no technical jargon):
  - **Data cannot leave the premises** (a compliance red line that rules out cloud services)
  - **Experts are hard to hire and retain** (a manual workshop whose expertise resides with individuals)
  - **Models cannot reach the field** (excellent in the lab, but unable to run on field devices)
- Emphasize in one line at the bottom: **"Every project feels like the first—delivery takes quarters, costs stay high, and success cannot be replicated."**

> Speaker notes: Do not rush into the product. First earn agreement from the executives: "Yes, that is exactly our situation." Use one of the customer's own previous pilots as an example.

---

## Slide 2 | The Three Traditional Approaches All Have Economics That Do Not Add Up

**Key conclusion: For customers with sensitive data and a need to deploy at scale, cloud platforms, in-house assembly, and outsourcing are either unworkable or uneconomical.**

- Three comparison cards (use red sparingly to highlight disadvantages):

| Approach | Critical Problem |
|---|---|
| Buy a public-cloud AI platform | Data must leave the premises and **cannot pass compliance review**; ongoing fees accumulate while data assets remain on someone else's platform |
| Assemble open-source components in-house | Requires a highly skilled platform team, **at least six months and millions in investment**, while still leaving expertise dependent on individuals |
| Outsource project by project | **Costs increase with every project and no in-house capability develops**; model ownership is easily disputed and responses are slow |

- Bottom-line message: "What customers really need is a **fourth path**: private deployment, turnkey operation, and full asset ownership."

> Speaker notes: This slide helps executives rule out distractions and answers common questions such as "Can our IT department build this itself?" and "Why not use free open-source software?"

---

## Slide 3 | NereidForge: A Model Factory in the Customer's Data Center

**Key conclusion: Turn an expert-dependent manual workshop into a standardized production line that keeps running even when disconnected.**

- Main visual: a **large production-line illustration** (seven stations from left to right, all described in business language):

```
Business data ingestion → machine-assisted annotation → standardized training → automated quality inspection → multi-hardware compression → unified registration → field deployment
   (Raw-material warehouse)     (Machines do 80%)          (Production line)       (Quality gate)          (Multi-format packaging)   (Finished-goods warehouse)   (Delivery)
```

- Three badges in language executives can relate to:
  - 🔒 **Entirely within the private network and fully operational while disconnected**
  - 👥 **Ordinary engineers can operate the system after two weeks of training, eliminating dependence on scarce experts**
  - 🔁 **Trained models immediately return to annotate new data—more data makes the factory faster** (small flywheel graphic)

> Speaker notes: Focus on the "machines do 80% first" flywheel, which is central to the cost story. One line: "Add business use cases continuously without expanding the annotation team."

---

## Slide 4 | Five Business Cases: How to Evaluate This Investment

**Key conclusion: The cases for compliance, efficiency, workforce, assets, and technological independence all add up, and a pilot can validate them.**

- Upper half: an icon wall for the five business cases (one sentence each; see Section 4 of the whitepaper):
  - **Compliance**: Data never leaves the premises, so the system can pass inspections
  - **Efficiency**: Delivery cycles shrink from quarters to weeks
  - **Workforce**: Machine preannotation prevents labor requirements from growing linearly with data volume
  - **Assets**: Data and models are centrally registered and traceable, so assets remain when people leave
  - **Independence**: Adaptation to multiple hardware types, including domestic platforms, avoids lock-in to chip vendors, cloud providers, or suppliers
- Lower half: a set of **verifiable figures** (clearly labeled "internal test results" or "pilot targets," with no unsupported promises):
  - Internal test: across 128 images, **the machine automatically produced 269 bounding boxes on 75 images**, so nearly 60% of images required no annotation from scratch
  - Pilot targets: **at least 50% fewer annotation hours, delivery time reduced to one-third, and independent operation after two weeks of training**
  - Include a mini ROI equation: *annual annotation volume × per-image cost × savings percentage (the whitepaper includes a calculation table where you can enter your company's data)*

> Speaker notes: Proactively state, "These figures must be validated in a pilot using your company's own data." This candor builds trust and leads naturally to Slide 5.

---

## Slide 5 | Recommendation: Prove the Value in One Use Case with a Four-to-Eight-Week Pilot

**Key conclusion: Start with low risk, let real data demonstrate the value, and replicate horizontally after success.**

- Left half: how to select the pilot (three conditions):
  - A solid data foundation (an existing body of images or video)
  - Clear business value (quantifiable labor savings or loss reduction)
  - Strong field cooperation (participation by a designated business expert)
- Right half: pilot roadmap (timeline):

```
Week 1                 Weeks 2–4                         Weeks 5–7                                      Week 8
Deploy environment     Data ingestion + machine-assisted annotation     Model training/quality inspection/trial field deployment     Joint review
Measure baselines      (Record baseline work hours)                     (Record cycle-time and outcome metrics)                    Produce ROI report → decide on replication
```

- Bottom call to action (give executives three clear options):
  1. **Approve the joint pilot** (recommended)
  2. Arrange an in-depth technical session or on-site demonstration first
  3. Review the whitepaper first and reconvene later
- Footer: **Shanghai Yiqi Software Co., Ltd. | NereidForge Privately Deployed Model Factory**

> Speaker notes: Always position a pilot, not a purchase, as the next step. Its low commitment and low risk make it easier for executives to approve on the spot.

---

## Appendix: Presentation Guidelines

1. **Avoid technical jargon**: Use factory metaphors such as "production line," "quality inspection," "warehouse," and "packaging" throughout; the accompanying engineers can handle technical questions during Q&A.
2. **Be honest about the numbers**: Strictly distinguish internal test results from pilot targets; confirm every assumption in the whitepaper's ROI table with the customer on site.
3. **Prepare answers to three questions**:
   - "How does this compare with a cloud or major-vendor solution?" → Return to Slide 2: their data must leave the premises and their costs continue indefinitely.
   - "Why not build it ourselves with open source?" → Explain the three costs: time (at least six months), talent (a platform team), and maintenance (a long-term burden).
   - "How is data security guaranteed?" → Fully disconnected deployment, customer ownership of data in standard formats, and integration with the customer's own identity system.
4. **Follow up after the meeting**: Send the two-page Executive Summary and full whitepaper within 24 hours. Complete the ROI table in Section 6 of the whitepaper with data supplied by the customer on site, then send the updated version.

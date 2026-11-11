<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# NereidForge Privately Deployed Model Factory

## Scaling AI Across the Organization

**Product Value Whitepaper | 2026 Edition**

**Published by Shanghai Yiqi Software Co., Ltd.**

---

> Intended readers: organizational decision-makers, IT leaders, and business-unit executives.
> The full text avoids technical jargon and takes approximately 15 minutes to read.

## Executive Summary

Artificial intelligence is moving from experimental pilots to infrastructure for production and operations. Yet traditional-industry organizations with sensitive data, field-based use cases, and lean teams consistently encounter three barriers to AI adoption: **data cannot leave the premises, talent is unaffordable, and models cannot reach field devices.**

NereidForge is a **model factory platform** deployed in the customer's own data center. It transforms the previously expert-driven manual workflow of data preparation → data annotation → model training → quality inspection → model compression → field deployment into a standardized, reusable production line that operates continuously.

Customers gain not merely a tool, but an **organizational capability**: a small team of three to five people can continuously turn business data into intelligent models that run in real time on field devices. Data and model assets remain entirely under the customer's control and become more valuable with use.

---

## 1. Background: AI's Main Battleground Is Shifting to Traditional Industries

Foundation models have shown society what AI can do, but most AI that creates real business value is far less glamorous:

- Production-line cameras detect product defects;
- Drones identify equipment defects at electrical substations;
- Traffic cameras track vehicles involved in violations;
- Campus systems identify people, vehicles, and abnormal events.

These scenarios share common characteristics: **the data consists of images and video, the intelligence must run in the field, and decisions must be made in real time.**

As policies advance and competition intensifies, every type of organization faces the same question:

> **We have run plenty of pilots, so why has AI still not become part of daily operations like our other information systems?**

## 2. Four Challenges Facing Customers

### Challenge 1: Data Compliance Is a Red Line That Rules Out Cloud Services

Industrial, government, financial, and security data is subject to laws, regulations, and internal policies and **cannot leave the organization's own environment**. Most AI platforms on the market are public-cloud services; regardless of their capabilities, they cannot pass the security review.

### Challenge 2: AI Projects Are Manual Workshops That Cannot Be Replicated

Taking an AI project from data to production often requires more than a dozen steps—data cleansing, sourcing annotators, building a training environment, repeated parameter tuning, and hardware adaptation—using five or six disconnected tools, with a few experts manually connecting the entire process.

The result: **every project feels like the first, delivery takes quarters, costs cannot be amortized, and success cannot be replicated.**

### Challenge 3: Expertise Resides with Individuals, Not the Organization

AI talent is expensive and scarce. Critical project knowledge—how data was processed, how models were tuned, and how well they performed—often exists only in the minds and computers of core engineers.

**When those people leave, the project stalls; models are scattered across personal computers, and no one can account for their versions or performance.** For management, this is a serious asset risk.

### Challenge 4: Laboratory Models Cannot Reach the Shop Floor

Algorithms that perform well in a data center fail in the field. Field devices have limited computing power and use a wide variety of chips, including domestic chips, while requiring millisecond-level real-time responses. Without dedicated compression and adaptation, models simply will not run.

**A gap separates research from delivery**, leaving many projects stuck at the stage where "the demo looks great, but the system is unusable."

## 3. NereidForge's Answer: Build a Model Production Line in the Customer's Data Center

If an AI model is treated as a product, the traditional approach is to hire craftspeople to build each one by hand. NereidForge instead provides a **modern factory**:

| Factory Stage | NereidForge Equivalent |
|---|---|
| Raw-material warehouse | Centralized ingestion of business data, with every version traceable |
| Raw-material preprocessing | Data annotation, with machines completing 80% and people the remaining 20% |
| Automated production line | Model training through standard templates that ordinary engineers can operate |
| Quality-inspection workshop | Automated evaluation before release, with models that fail quality targets blocked from release |
| Multi-format packaging | Train once and automatically produce models adapted to different field hardware |
| Finished-goods warehouse and shipping records | Central registration of model assets, with full traceability of origin, version, and performance |
| Logistics and delivery | One-click deployment to data-center servers or field edge devices |

The entire production line runs in **the customer's own data center** and continues to operate even when physically isolated from the internet.

## 4. Core Value: Five Business Cases for Decision-Makers

### Case 1: Compliance—Data Never Leaves the Premises, and the System Passes Inspections

- The entire system is deployed in the organization's own environment, and **data never leaves the private network at any stage**;
- The system works out of the box in a completely disconnected environment, with even foundation models preinstalled, eliminating the need for ad hoc internet downloads;
- It meets mandatory data-localization requirements for classified environments, cybersecurity compliance, and industry regulation.

> What this means for management: **it is both a ticket to participate in the AI era and the peace of mind that comes from knowing organizational data is secure.**

### Case 2: Efficiency—Delivery Cycles Shrink from Quarters to Weeks

- Data, annotation, training, evaluation, and deployment are completed on one platform, eliminating repeated handoffs between tools and teams;
- Standardized templates turn expert knowledge into repeatable processes, enabling ordinary engineers to run projects independently after brief training;
- For the most time-consuming data types, such as video, machines can automatically perform frame-by-frame tracking annotation while people only review and correct.

> What this means for management: **the same team can grow from delivering two or three use cases per year to more than ten.**
> (We recommend measuring the exact reduction in delivery time in the first pilot use case; see Section 6.)

### Case 3: Workforce—The Cost Flywheel of Machines Doing 80% First

The system can use the customer's existing models to perform preannotation automatically, leaving annotators to handle only the uncertain cases. As data accumulates, the models continue to improve:

```
More data → more accurate models → less annotation labor → more use cases for the team
```

This is not a linear reduction in labor, but a **self-accelerating flywheel**: the annotation team can remain stable in size while business use cases continue to expand.

Video annotation is usually the largest labor cost in an AI project. Traditionally, people draw boxes frame by frame; on this platform, they annotate only a small number of key frames, and machines automatically fill in object movement between frames. **Video annotation is where this platform's advantages are most pronounced.**

### Case 4: Assets—Data, Models, and Experience Become Organizational Assets

- Every data batch, model version, and evaluation report is centrally registered, tamper-proof, and traceable;
- Every model can answer three questions: **What data was it trained on? What are its performance metrics? Where is it deployed?**
- New projects can continue training from historical models instead of starting from scratch;
- Staff turnover no longer interrupts projects—**expertise is retained in the platform rather than in individuals.**

### Case 5: Technological Independence—No Lock-in to Chip Vendors, Cloud Providers, or Suppliers

- Train once and automatically produce models adapted to multiple chip and hardware types, including NVIDIA, AMD, Intel, and domestic hardware platforms, **preserving freedom of choice in hardware procurement**;
- The platform is based on mature, open technologies and integrates with the customer's existing unified identity system, avoiding a closed, proprietary account system;
- When the platform later needs to join the group's unified technology platform, it can be integrated smoothly, so **today's investment will not become tomorrow's isolated silo.**

## 5. Typical Application Scenarios

| Industry | Typical Scenarios | Work Performed by the Platform |
|---|---|---|
| **Manufacturing** | Product quality inspection, assembly verification, workplace-safety monitoring | Continuous annotation and training on defect images, followed by model compression and deployment to production-line edge devices for real-time detection of defective products |
| **Transportation and Security** | Vehicle tracking, violation recognition, event detection | Automated tracking annotation and model iteration across massive video volumes, supporting continuous object tracking across cameras |
| **Energy and Power** | Substation and power-line inspection, drone-based defect recognition | Inspection-image ingestion, continuous accumulation of small-sample scenarios, and model adaptation for portable and airborne devices |
| **Government and Defense** | Image and text intelligence in classified environments | Deployment in fully disconnected environments, with data never leaving the data center and auditable, traceable model performance |
| **Campus and Property Management** | People and vehicle management, perimeter security, facility inspection | Parallel delivery of multiple use cases by a small team, with historical models reused across scenarios |
| **AI Solution Integrators** | Project delivery for the industries above | Use the platform as a standardized foundation for proprietary solutions, significantly reducing marginal project costs and eliminating dependence on individual experts |

## 6. How to Measure Results: Reference Data and ROI Calculations

We do not advocate using unverified percentages in marketing. The data below falls into two categories: **measured internal platform results** (reproducible) and a **customer ROI calculation model** (using measured pilot data).

### 6.1 Internal Platform Test (September 2026)

In a standard demonstration scenario (128 real imported street-scene images and 8 object-detection classes):

| Metric | Measured Result | Customer Impact |
|---|---|---|
| Images with boxes automatically generated in one batch preannotation run | **75 of 128** (approximately 59%) | On nearly 60% of images, people **do not need to draw boxes from scratch** and only review and correct |
| Total detection boxes generated automatically | **269** | An average of approximately 3.6 manual drawing operations saved per image with generated boxes |
| Preannotation operating mode | One-click batch processing, background execution, pause/resume/stop support, and real-time progress | Annotators can perform other work simultaneously instead of watching the screen and waiting |
| Video annotation | People annotate only key frames, while machines automatically track and complete intermediate frames; supports one-click reruns with a new model and automatic replacement of old machine annotations | Video-project labor shifts from frame-by-frame annotation to sampled review |
| Model validation | Upload one image to any model version for a trial run and view a visualization of where the model is looking, **without deploying it first** | Performance assessment shrinks from hours to minutes, avoiding unnecessary deployments |

> Note: These are internal platform regression-test results. Detection coverage depends heavily on how well the model matches the scenario. Actual coverage in a customer scenario should be determined using pilot data—and because the pilot itself runs on the platform, that data is naturally available.

### 6.2 Customer ROI Calculation Model (Example)

The following is a **calculation method and example**. All figures in the table are explicitly labeled assumptions and should be replaced with measured customer pilot results in formal commercial discussions.

**[Example Assumptions]**

| Parameter | Assumed Value | Notes |
|---|---|---|
| Annual image-annotation volume | 100,000 images | Replace with the customer's business volume |
| Fully manual all-in cost per image | RMB 1.5/image | Includes annotation, quality inspection, rework, and allocated management costs |
| Machine-preannotation coverage | Approximately 60% | Based on the 59% internal test result; mature scenarios are usually higher |
| Efficiency of reviewing one machine-preannotated image | 3× annotation from scratch | In other words, approximately one-third of the original time per image |

**[Calculation]**

- Fully manual annual cost: 100,000 × RMB 1.5 = **RMB 150,000/year**
- Equivalent workload after introducing preannotation: 40% of images annotated from scratch + 60% reviewed at one-third of the original labor
  = 0.4 + 0.6 ÷ 3 = **0.6 (60% of the original)**
- Annual annotation cost: RMB 150,000 × 0.6 = **RMB 90,000/year**
- **Direct annotation-cost savings of approximately 40%, or RMB 60,000/year** (excluding the potentially greater savings on video projects, business benefits from shorter project cycles, and cumulative benefits from reusing model assets)

As the flywheel turns (coverage and accuracy improve through model iteration), the savings percentage in the second year is usually higher than in the first.

### 6.3 Recommended Pilot Validation Metrics

We recommend that both parties jointly record the following baselines in the first pilot use case and let the customer's own data demonstrate the results:

| Metric | Pre-pilot Baseline | Pilot Target |
|---|---|---|
| Annotation hours per image or per minute of video | Measure on site | Reduction of at least 50% (recommended target) |
| Delivery cycle for one use case, from initiation to field trial | Measure in weeks | Reduced to no more than one-third of the original cycle |
| Evaluation pass rate and rework count for model versions | Historical statistics | Traceable evaluation reports and quantifiable rework |
| Training time required for an ordinary engineer to deliver a use case independently | — | Independent operation within two weeks |

## 7. Customer Benefits at a Glance

| Dimension | Traditional Approach | After Introducing NereidForge |
|---|---|---|
| Data security | Difficult to achieve compliance; cloud solutions unusable | End-to-end private-network workflow with fully disconnected operation |
| Single-project delivery cycle | Weeks to months | Days and weeks (measured in the pilot) |
| Annotation labor | Grows linearly with data volume | Machine preannotation + human correction, with labor requirements remaining largely constant |
| Team skill requirements | Dependent on senior algorithm experts | Ordinary engineers can operate the system after training |
| Field hardware adaptation | Separate development for each platform, with long cycles | Train once and adapt automatically to multiple hardware types |
| Asset management | Scattered across personal devices and untraceable | Central registration, versioning, and full-process audit trails |
| Project experience | Lost through staff turnover | Retained as templates and models for reuse across projects |
| Supplier risk | Closed ecosystems and lock-in | Open architecture, technological independence, and integration into a group platform |

## 8. Why Choose NereidForge

Organizations generally have only three paths for building AI capabilities. NereidForge's positioning is clear by comparison:

| Approach | Advantages | Practical Problems |
|---|---|---|
| **Purchase a public-cloud AI platform** | Ready to use upon activation | Data must leave the premises and cannot pass compliance review; long-term usage costs continue indefinitely; capabilities and data remain on someone else's platform |
| **Build in-house with open-source components** | Free software | Requires a highly skilled platform team and costly integration and maintenance; takes at least six months and still leaves expertise dependent on individuals |
| **Outsource project by project** | Low initial burden | No in-house capability develops and costs increase over time; data and model ownership is easily disputed; responses are slow and iteration is difficult |
| **NereidForge** | Turnkey private deployment, integrated end-to-end workflow, and full asset ownership | Requires basic customer operations capabilities or the purchase of complementary operations services |

The fundamental differentiator is that **NereidForge does not simply package several open-source tools. It addresses the central question of how an organization whose data cannot leave its premises can continuously produce usable models by connecting annotation, training, evaluation, compression, and deployment into a closed-loop production line**. In particular, general-purpose platforms commonly lack both its self-reinforcing mechanism—trained models immediately returning to annotate new data—and its complete video-data workflow from annotation to edge deployment.

## 9. Delivery and Engagement Models

- **Private deployment**: The complete system is installed on the customer's own server cluster and supports disconnected data centers;
- **Phased adoption**: We recommend starting with a use case that has a solid data foundation and clear business value, completing the data–model–field feedback loop, and then replicating it horizontally;
- **Ongoing partnership**: We provide platform deployment, team training, template tuning, and operations support to help the customer team gradually become self-sufficient;
- **For integrators**: The platform can serve as the foundation for industry solutions while protecting the integrator's own industry knowledge and delivery assets.

## 10. Investment Protection: Independent Today, Seamlessly Evolvable Tomorrow

NereidForge is designed around a no-lock-in principle:

- **No identity lock-in**: Integrates with the customer's existing unified account system, so employees do not need to remember new credentials;
- **No data lock-in**: Data and models are stored in standard formats and belong to the customer;
- **No architectural lock-in**: When the group builds a unified AI platform in the future, NereidForge can be integrated as a standard component rather than replaced from scratch.

**The customer is purchasing not a one-off project, but a production capability that can operate continuously and increase in value over time.**

## Conclusion

Ultimately, competition in AI is not about models alone. It is about **whether an organization can continuously, economically, and compliantly turn data into field intelligence**.

NereidForge enables organizations with sensitive data, real-world scenarios, and lean teams to avoid choosing between compliance and intelligence. In their own data centers, they can own a model factory that becomes faster with use.

---

**Shanghai Yiqi Software Co., Ltd.**

*NereidForge Privately Deployed Model Factory | Product Value Whitepaper | 2026 Edition*

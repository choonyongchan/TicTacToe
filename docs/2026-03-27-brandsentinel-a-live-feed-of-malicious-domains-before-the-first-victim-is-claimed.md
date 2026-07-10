---
title: "Every Scam Site Leaves One Trace Before It Goes Live. We Built a Tool to Catch It."
subtitle: "Introducing BrandSentinel: how we automated 95% of a Digital Risk Protection analyst's workload — and what we learned doing it"
date: 2026-03-27
category: Cybersecurity
abstract: "In a previous post, I showed how Certificate Transparency logs give us a real-time window into newly registered domains — including scam sites — before they go live. But surfacing suspicious domains is only half the battle. In this post, I introduce BrandSentinel: an open-source pipeline that automatically classifies those domains using an ensemble of heuristics, eliminating the majority of manual analyst work and catching threats before the first victim is claimed."
cover: /assets/images/covers/brandsentinel-cover.svg
comments: true
---

*This post was proofread with the assistance of AI. The source code backing this story is open-sourced and available on [GitHub](https://github.com/choonyongchan/BrandSentinel).*

---

When we pointed CertStream at two influential Singapore organisations, it surfaced over ten thousands of suspicious domains in a single day. No human team could triage that volume before the threats became active. So we built a machine to do it — one that classifies domains in real time, flags the dangerous ones before the first victim ever loads the page, and hands analysts a curated shortlist instead of a firehose. This is the story of how that tool came to be, what it took to build it, and what it taught us about defending brands at machine speed.

---

Before I get into the technical details, I want to take a moment to acknowledge the people who made this work possible.

During my internship, I had the privilege of learning from the Asia-Pacific Digital Risk Protection team at a leading cybersecurity company. They showed me what it really looks like to defend a brand in the wild — the volume, the pace, the judgment calls analysts make under pressure, and the genuine satisfaction of taking down a scam site before it harms someone. This project is built on everything they taught me, and I dedicate it to them.

---

## The Problem: We Found the Domains. Now What?

In [my previous post](./2026-03-19-hunting-scam-domains-before-they-strike-with-certstream.md), I described how Certificate Transparency logs — and specifically CertStream, the real-time WebSocket feed built by Cali Dog Security — give us a powerful early warning system. Every domain that obtains an HTTPS certificate is publicly logged the moment it does, and brand-targeted scam domains are no exception. That means we can see them at the point of certificate issuance, often before the site is even reachable.

The problem we were left with was a different one entirely: **volume**.

CertStream processes millions of certificate events every day. Even after filtering to domains that contain your brand keywords, you're looking at thousands of candidates per day for a single brand. A human analyst cannot manually investigate each one fast enough for the intelligence to be actionable. By the time they've worked through the queue, the scam domains they flagged are either weaponised or gone.

We had solved the sourcing problem. We had created a new one.

## The Implication: Speed Is Not a Nice-to-Have

Let me be direct about the stakes. In threat intelligence, there are two kinds of errors. **False positives** — flagging a benign domain as malicious — waste analyst time and erode trust in the tool. Annoying, but recoverable. **False negatives** — missing a real scam domain — mean a live phishing site with your brand's logo on it is collecting victims' credentials, authorising fraudulent transactions, and destroying customer trust, with nobody at your organisation aware it exists.

False negatives are not a metric. They are harm to real people.

The implication is this: an automated classification layer that works around the clock, surfaces threats with high recall, and hands analysts only what truly requires human judgment is not a convenience — it is a prerequisite for any serious DRP operation. That is what we set out to build.

## BrandSentinel: A Live Feed of Malicious Domains

### Methodology

BrandSentinel is an open-source Digital Risk Protection pipeline. It ingests domains from multiple threat intelligence feeds continuously, classifies them using an ensemble of heuristics, and writes verdicts to per-category output files — all while the analyst monitors a live dashboard in Prefect's UI.

**Ingestion.** Ten source workers run as independent async tasks, each polling a different feed on its own schedule:

| Source | Frequency |
|---|---|
| CertStream | Live (CT log stream) |
| URLhaus | Every 5 minutes |
| PhishTank | Every 1 hour |
| OpenPhish | Every 12 hours |
| CERT Polska, Phishing.Database, Phishing Army, Botvrij.eu, DigitalSide | Configurable |
| Manual import file | Every 30 seconds |

Every domain that any source produces is placed onto a shared, deduplicated `asyncio.Queue`. If a domain has been seen before — across any source — it is silently dropped. The pipeline never processes the same domain twice.

**Filtering.** A keyword-based filter checks each domain against the brand configurations in `config.yaml`. Domains that do not contain a brand keyword are classified as `IRRELEVANT` and logged immediately. This is the first gate, and it is intentionally cheap — regex matching costs nothing compared to what comes next.

**Classification.** For every domain that passes the filter, BrandSentinel concurrently fetches the HTTP response (following redirects, capturing page content), resolves DNS records, and retrieves the TLS certificate — all in parallel. This enriched context is then passed through a pipeline of 15 heuristics.

The heuristics run lazily. If any single heuristic produces a *definitive* verdict — such as matching a known brand favicon hash on a non-canonical domain, or detecting a phishing kit directory structure — the pipeline short-circuits and assigns that verdict immediately, without running the remaining checks. For contributing signals, scores are accumulated and normalised. The final verdict is one of four categories:

- **SCAM** — High confidence. Investigate and initiate takedown.
- **INCONCLUSIVE** — Suspicious. Requires analyst review.
- **BENIGN** — Passed all checks. Continue monitoring passively.
- **IRRELEVANT** — Not related to the monitored brand. Dropped.

A few heuristics are worth calling out explicitly because they do the heaviest lifting:

- **Inactive domain**: An HTTP timeout, connection refusal, or non-200 response is a strong signal — scam domains are frequently pre-registered before they're weaponised, or already retired.
- **Parking detection**: The page content or DNS fingerprint matches a known domain parking service. These are typically benign, but they are watched — parked domains can be activated at any time.
- **Brand lookalike**: Levenshtein distance between the registered domain and the brand's canonical domain detects typosquatting and combosquatting that a keyword filter alone would miss.
- **Favicon hash matching**: A SHA-1 hash of the site's favicon is compared against a curated list of brand favicon hashes. A match on a non-canonical domain is a near-definitive SCAM signal — an attacker has copied the brand's own visual identity.
- **Forms exfiltration**: Login forms or input fields that submit to a different domain than the one being analysed are a reliable indicator of a credential-harvesting phishing page.

**Output.** Verdicts are written continuously to `scam.txt`, `inconclusive.txt`, `benign.txt`, and `irrelevant.txt`. The entire pipeline is orchestrated by Prefect, which provides a live UI for observing flow runs, retrying failed tasks, and inspecting per-domain logs without touching the terminal.

### Results

The version I am describing here is the current, fully instrumented release. But the insight that validated the approach came from something far simpler.

An early prototype ran only two heuristics: the **inactive domain check** and the **parking detection check**. No lookalike scoring, no favicon hashing, no content analysis — just those two. Even with only CertStream as the data source, the results were striking.

Of every hundred domains flagged by the brand keyword filter, roughly **ninety-five were resolved by those two heuristics alone**. The inactive check cleared out domains that had been registered but not yet deployed — or already retired. The parking check cleared out benign infrastructure that had been registered speculatively by domain investors, not attackers. What remained — the five or so domains that were live, active, and neither parked nor obviously benign — was the set that actually warranted human attention.

In operational terms, this meant a DRP analyst who previously faced a hundred manual lookups per day was reduced to reviewing fewer than ten. The pipeline had not replaced the analyst. It had given them back their day.

### Discussion

The two-heuristic result is encouraging, but it is not the ceiling — it is the floor. The more sources you ingest from, and the more precisely your heuristics are calibrated, the tighter the classification becomes.

This brings us back to the two-error asymmetry. **False positives** are an inconvenience: the analyst reviews a benign domain and clears it. The cost is time. **False negatives** are a failure: a scam domain is classified as benign or irrelevant and goes unactioned. The cost is victims. BrandSentinel is deliberately calibrated toward recall over precision — when evidence is ambiguous, the verdict is `INCONCLUSIVE`, not `BENIGN`. The analyst sees it. The domain does not slip through.

As research matures, more can be layered in. New ingestion sources plug in by implementing a single `Source` base class. New heuristics extend `HeuristicBase` and register themselves; the orchestrator handles the rest. Commercial feeds — URLScan.io, OTX, VirusTotal — can join the source pool without touching the classification logic. An ML-based scoring layer can replace or complement the weighted heuristic sum for teams with labelled datasets. The architecture was designed to absorb these improvements without structural change.

## Who Should Use This

**Digital Risk Protection teams** are the primary audience. BrandSentinel is not a replacement for experienced analysts — it is the triage layer that lets them work on what matters. Instead of spending their shift manually checking whether `mybank-secure-login.xyz` resolves to anything, analysts can focus on the pre-scored shortlist that BrandSentinel surfaces: live domains, with suspicious content, with brand-identical favicons, submitting credentials to a foreign server. That is a very different kind of work.

**Small and medium enterprises** are the second audience, and perhaps the more important one. Most SMEs do not have a DRP team. They may not even know how actively their brand is being impersonated online. Running BrandSentinel — even in its simplest configuration, with CertStream and a handful of brand keywords — produces a sobering and actionable picture of digital threat exposure. For an SME trying to decide whether to invest in a commercial DRP solution, that picture is exactly the evidence they need.

## What Comes Next

BrandSentinel is open source. It is a gift to the cybersecurity community, in the same spirit as the team that first gave me this perspective. If you work in threat intelligence, digital risk, or brand protection, I hope you find it useful — and I hope you make it better. Open an issue, submit a heuristic, add a feed.

The threat is automated. The defence should be too.

## What I Took Away

**1. Open-source intelligence is a treasure trove — especially for those who need it most.**

Open-source intelligence will not beat a dedicated proprietary threat intelligence platform on visibility, reliability, or breadth of coverage. The commercial vendors have more sources, better enrichment, and dedicated teams curating signal quality. That is simply the reality.

But here is the counterpoint: for the organisations most exposed to brand-impersonation attacks — small and medium enterprises — a commercial DRP subscription is often out of reach. The same businesses that cannot absorb the cost of a cyberattack are frequently the ones that cannot afford the tooling to prevent one. That asymmetry is where open-source intelligence earns its keep.

The insight BrandSentinel reinforced for me is that the value of OSINT is not in any single source — it is in the fusion. CertStream alone produces noise. URLhaus alone has coverage gaps. PhishTank alone misses the newest campaigns. But a pipeline that ingests all of them, deduplicates across sources, and applies a consistent classification layer produces something genuinely useful: a picture of your brand's threat exposure that would otherwise require a commercial contract to see. For an SME deciding whether to invest in a DRP solution, BrandSentinel running for a week is a more persuasive argument than any sales deck.

**2. Building a real-time, scalable ETL pipeline in Python is harder than it looks — and the right answer is to iterate, not over-engineer.**

I designed BrandSentinel to be event-driven from the start. The first version used Python `async` functions throughout, with Redis Pub/Sub as the message bus between ingestion and classification. For a small proof-of-concept, this is clean and simple — everything lives in one file, the architecture is easy to reason about, and the latency between domain observation and verdict is low.

The problem became obvious under load. Python's threading model means CPU-bound work competes with I/O on a single thread, and the volume of domains BrandSentinel processes is enough to make that a real bottleneck. Worse, several libraries I relied on — DNS resolution in particular — do not expose async interfaces. Every blocking call was a stall. Converting the entire codebase to async is not just tedious; it is a maintenance burden that multiplies with every new heuristic.

My first instinct was to reach for a microservice architecture: split ingestion, classification, and output into separate processes, use a proper message queue, and let each component scale independently. The design is sound on paper. In practice, the overhead of inter-process communication directly conflicts with BrandSentinel's primary goal, which is the speed of scam domain detection. Latency introduced at the architecture level is just as damaging as latency introduced by slow heuristics.

The solution I landed on was Prefect — a workflow orchestration framework that let me run the full pipeline from a single Python file while genuinely parallelising I/O-bound work across tasks. No async contortions. No microservice topology. No operational complexity beyond spinning up the Prefect server. The pipeline became both faster and easier to reason about than either of its predecessors.

The lesson here is not that Prefect is the answer to every pipeline problem. The lesson is to resist the temptation to design for theoretical scale before you understand where the real bottlenecks are — and to search aggressively for existing tools before building your own solutions to solved problems. Iterative solutions are not a sign of weak engineering. They are a sign of engineering focused on the right outcome: in this case, getting a scam domain classified and surfaced to an analyst as fast as possible.

---

*If you're running BrandSentinel against your own brand, I'd genuinely like to hear what you find. What heuristics are generating the most signal? What sources are you missing? Leave a comment below.*

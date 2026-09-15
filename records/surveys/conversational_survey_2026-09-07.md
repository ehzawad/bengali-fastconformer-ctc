# Workstream 4 — Bengali conversational / spontaneous speech survey

**Registered against:** SPEC §8m (council G) Q5 — "conduct a bounded conversational-data survey for the
next registration". **Status: RESEARCH ONLY.** No GPU work, no pipeline file changed, no frozen artifact
touched, no audio downloaded or decoded (Ben-10's PCM geometry was pinned by two 96-byte HTTP range reads
of WAV headers; nothing else). Nothing here is admitted, and nothing here may be admitted without its own
registration, its own Gate-0 pass and its own arm.

**Governing constraint on how to read this file.** Council G's own claims-correction (iii) retracted the
statement that every candidate bucket is read speech, and correction (ii) marked transfer from any new
register to SUBAK.KO or FLEURS as a *hypothesis, not a finding*. Everything below inherits that. A candidate
with conversational hours is a candidate with conversational hours; it is not evidence that D will move
0.87 pp. The one arithmetic anchor the survey is allowed to use is council G Q5's: D is 52.9 % IndicVoices
reference words and 18.9 % SUBAK.KO, so closing D's outstanding 0.87 pp needs ≈1.64 pp on IndicVoices alone
or ≈4.62 pp on SUBAK.KO alone.

**Evidence policy applied throughout: a candidate whose licence I could not verify from a custodian-controlled
source is a NO,** stated as a NO, not as a "promising but unclear". Where a number is an estimate I say so and
give the bias direction; where evidence is missing I say *missing* rather than interpolating.

---

## 1. What we already train on (measured, for the overlap column)

Counted from `data2/train_corpus_ready.json` this session:

| source | rows | hours |
|---|---:|---:|
| indicvoices | 277,948 | 580.47 |
| openslr53 | 166,585 | 163.99 |
| subakko | 35,284 | 110.76 |
| kathbath | 46,240 | 81.98 |
| commonvoice | 24,856 | 39.71 |
| fleurs_train | 2,019 | 7.22 |
| **Phase-1 total** | **552,932** | **984.13** |

Plus the Phase-4 candidates already ingested and frozen: `macro_filtered` 736.8 h, `madasr` 894.2 h,
`vaani` 117.4 h retained (SPEC §1).

**Audit note for the owner, not a survey finding.** SPEC §2 labels `data2/holdout_corpus.json` as
"speaker-disjoint IndicVoices-R". The file's actual composition is 6,767 rows across `commonvoice` (464),
`indicvoices` (2,493), `kathbath` (626), `openslr53` (2,413) and `subakko` (771) — it contains no
IndicVoices-**R** rows at all. Either the SPEC parenthetical is a labelling error or it means "disjoint from
IndicVoices-R". This does not change any result, but the phrase is load-bearing for the IndicVoices-R
question in §3.2 and should be corrected or clarified in the SPEC.

---

## 2. Ranking

Score = expected **conversational hours actually admissible under our rules** × label quality × licence
cleanliness. "Admissible hours" is post-cleaning, not the headline hours in the paper — that is the whole
point of the exercise, and it reorders the table.

| # | candidate | headline h (Bengali) | est. admissible conversational h | labels | licence | verdict |
|---:|---|---:|---:|---|---|---|
| 1 | **SPRING-INX R1+R2, Bengali-script-only subset** | 647.7 | **≈280** (est.) | human, vendor QC | CC BY 4.0, record archived | **NEEDS OWNER AMENDMENT** to G0.9(a) |
| 2 | **IndicVoices upstream delta** | ≤115.8 unresolved | **0–116** (unresolved) | human, verified | CC-BY-4.0, gate accepted | **MEASURE FIRST** (CPU, free) |
| 3 | **Ben-10 (bengaliAI)** | 78 (73.8 released) | **≈35** | human, linguist-validated | CC0-1.0 | **VIABLE, but likely fails G0.6 as written** |
| 4 | RESPIN semi-noisy / noisy Bengali slabs | ≈200 / ≈140 est. | 0 (read speech) | human, quality-downgraded | CC-BY-4.0 | wrong register for Q5 |
| 5 | RegSpeech12 | ~100 | — | human (claimed) | **unverified** | **NO** |
| 6 | Vaani untranscribed remainder | ~31,000 (all langs) | 0 | **none** | CC-BY-4.0 | **NO** |
| 7 | Shrutilipi | large | 0 | forced alignment | — | **NO, permanently** |
| 8 | IndicVoices-R Bengali | 39.9K rows | 0 | derived from IndicVoices | CC-BY-4.0 | **NO** |
| 9 | Bengali telecom / call-centre on HF | ~7.5K clips | 0 | **synthetic TTS** | CC BY 4.0 | **NO** |
| 10 | Parliament / broadcast (BanSpeech, Bengali-Loop, WorldSpeech) | 6.5 / 158.6 / — | 0 | subtitles + alignment, YouTube | unverifiable | **NO** |
| 11 | OpenSLR SLR37 | small | 0 | read TTS studio | CC BY-SA 4.0 | no conversational value |

---

## 3. Candidate detail

### 3.1 SPRING-INX Bengali R1 + R2 — the largest genuinely conversational pool, and we excluded it for script, not quality

- **Custodian:** SPRING Lab, IIT Madras (NLTM). HF `SPRINGLab/SPRING_INX_Bengali_R1` and `_R2`, **ungated, open**.
- **Hours (Bengali):** R1 374.73 h train, R2 272.98 h train (SPEC §1); dev 70 h and eval 10 h are protected
  one-shot panels and stay protected.
- **Spontaneous vs read:** **mock conversations and monologues** (SPEC §1). Verified by inspection of the
  first rows: R1 utterance ids are conversation-grouped (`KaInDe_utt00000002`) and the text is dialogue —
  *"এইতো বসে আছি। বল তুই কী করছিস?"*. This is the only large Bengali corpus in the survey that is
  dialogue rather than image-description or elicited monologue.
- **Transcript provenance:** manually transcribed, vendor-collected with preliminary QC. Human, not machine-assisted.
- **Licence:** CC BY 4.0, portal text already archived in-repo at `experiments/p4/licences/SPRING_INX_licence_record.md`.
  This is the one candidate whose licence record we already hold. **Verified.**
- **Format / rate:** 16 kHz WAV. Duration statistics from the datasets-server (partial scan): R1 mean 7.58 s,
  median 7.13 s, min 2.00 s, max 24.70 s; R2 mean 9.30 s, median 10.09 s, min 2.00 s, **max 47.21 s**. R2 therefore
  loses a tail to the 23.0 s cap; R1 barely does.
- **Overlap with what we train on:** none by construction — different collection, different custodian, no shared
  prompt pool with MaCro/Common Voice. PCM-hash dedup and the fingerprint audit still apply as registered.
- **Published WER baselines:** none located for Bengali specifically. **Evidence missing.**

**Why it is excluded today and what changes.** SPEC §3 G0.9(a) (council D Q4) measured 48.3 % of R1 rows carrying
Latin-script code-switch words and excluded SPRING-INX whole, with the explicit escape hatch: *"owner may amend
explicitly with a replacement threshold and the stated bias"*. I re-measured through the frozen surface on 500
sampled train rows per release:

| | Latin-bearing rows | retained share of **rows** | retained share of **surfaced characters** |
|---|---:|---:|---:|
| R1 | 46.8 % (n=500) | 53.2 % | **40.3 %** |
| R2 | 46.2 % (n=500) | 53.8 % | **47.3 %** |

My R1 row rate (46.8 %, ±≈4.4 pp at n=500) reproduces the SPEC's full-corpus 48.3 %, which is the sampling check.
R2 is the same corpus by another name — 46.2 % — so there is no "cleaner release" to prefer.

**The number that matters is the character share, not the row share, and it is smaller.** Code-switched rows are
*longer* rows (R1 clean rows average 74 surfaced characters, Latin-bearing rows 124), because speakers reach for
English mid-turn in longer turns. So filtering to Bengali-script-only rows keeps 53 % of the rows but only ~40–47 %
of the *speech*. Estimated admissible hours: R1 374.73 × 0.403 ≈ **151 h**, R2 272.98 × 0.473 ≈ **129 h**,
**≈280 h total** — not the ≈345 h a row-share estimate would have claimed. Characters are a duration proxy, not
duration; the true figure needs the `duration` column and could move by tens of hours in either direction.

**The honest cost of the amendment.** A Bengali-script-only subset is not a neutral cleaning rule. It removes
*exactly the code-switched utterances*, so the retained subset is systematically biased toward the speech of
speakers and topics that do not code-switch, in a corpus where nearly half of all conversation does. Bengali
conversational speech genuinely contains English; training on the half that does not, then claiming
conversational coverage, would overstate what the model saw. This must be disclosed as a *content* filter, not
a *quality* filter, in exactly the language council G used to retract the CV17 "defect filter" claim (§8m
correction iv). It is still, on the numbers, the best conversational option available.

### 3.2 IndicVoices — is there more Bengali than our 580 h, and what about IndicVoices-R

**More Bengali: unresolved, and cheap to resolve.** The HF card records `bengali/train` at **335,367 examples**
today, against our **277,948 ingested rows / 580.47 h**. That is a nominal gap of 57,419 rows. The dataset card's
2025-12-23 note ("We now have 11,200 hours of transcribed data") confirms the corpus grew after our Phase-1
ingest. The upstream per-clip mean duration from the datasets-server statistics endpoint is **7.26 s** (partial
scan, 41,928 rows; median 6.50 s), so the gap is **at most ≈116 h**.

**But it may be zero.** I cannot separate (a) genuinely new upstream rows from (b) rows our Phase-1 cleaning
deleted, without the Phase-1 raw ingest manifest. Both explanations fit the arithmetic. Claiming "+116 h of
IndicVoices is available" would be exactly the kind of optimism this survey is supposed to avoid, so I do not
claim it. **This is the single highest-value unresolved question in the survey**, because IndicVoices is 52.9 %
of D's reference words, is 76 % extempore + 15 % conversational by the corpus's own accounting, already carries
the transcription convention our targets use, is CC-BY-4.0 with the gate already accepted, and needs no new
audit conventions. Per council G Q5, adding IndicVoices hours is *not* the same intervention as upweighting
IndicVoices — the sampling weights stay frozen either way.

Bengali metadata worth noting for the register question: the released Bengali rows are **West Bengal, India**
districts (Paschim Bardhaman, Paschim Medinipur, Purulia, Jhargram, Purba/…), 58.6 % rural, spread evenly over
four age bands. It is Indian Bengali, not Bangladeshi Bengali — which is the opposite side of the register gap
from Ben-10 and SUBAK.KO.

**IndicVoices-R: NO.** `ai4bharat/indicvoices_r`, Bengali train 39.9K rows, CC-BY-4.0, gated. It is *"the largest
multilingual Indian TTS dataset **derived from an ASR dataset**"* — it is IndicVoices audio that has been
enhanced/denoised and re-segmented for TTS, with `task_categories: text-to-speech`. Three independent reasons it
stays out and none of them has weakened: (i) the audio is a **derived artifact** of rows we already train on, so
admitting it trains twice on the same speech through a processing chain we did not perform and cannot audit —
G0.9(b) counts cross-corpus copies and explicitly does not exempt them; (ii) it adds no new *labels*, only new
*renderings*; (iii) SPEC §2 names it in the holdout's provenance line, so touching it risks the protected panel
(see the audit note in §1 — resolve that wording before anyone re-opens this). Phase 1 was right.

### 3.3 Ben-10 (bengaliAI) — the cleanest labels and licence in the survey, and the smallest usable yield

- **Custodian:** Bengali.AI, with IUT and BRAC University (arXiv 2510.23252). HF `bengaliAI/Ben-10`, **open, ungated**.
- **Licence:** **CC0-1.0**, declared in the repo's own README front-matter by the custodian. The paper itself
  states only "publicly available" and names no licence, so the CC0 rests on the repo declaration alone —
  custodian-controlled, therefore **verified**, but it is a one-line README with no LICENSE file.
- **Spontaneous:** yes, unambiguously. 394 speakers, 155 topic stimuli (family, religion, sports, politics),
  *"All conversations were informal to ensure the inclusion of authentic regional dialects"*, ten Bangladeshi
  regions (Rangpur, Kishoreganj, Narail, Chittagong, Narsingdi, Tangail, Barishal, Habiganj, Sylhet, Sandwip).
- **Transcript provenance: the best in this survey.** 34 local transcribers, each resident in the region they
  transcribed, selected by a linguist-scored 100-clip test, trained by a linguist, working 14 months; an expert
  linguist then validated spelling and transcription errors for homogeneity and gave annotators feedback.
  Human, verified, single convention by design. Annotators were paid (BDT 2,000/h of speech).
- **Format / rate: measured, not assumed.** RIFF headers read over HTTP range on one train and one valid file:
  PCM, mono, **16 kHz, 16-bit** — same geometry as our corpus, no resampling needed.
- **Hours and durations: measured from 15,036 LFS blob sizes.** **73.82 h released**, mean 17.67 s, median 16.10 s,
  p95 23.52 s, max 60.0 s. The paper's 78 h / 16,690 clips includes the **1,682-clip test split, which is withheld
  from the HF release** (train 13,342 + valid 1,666 = 15,008 rows in the CSVs, matching the paper's Table 1 exactly).
- **Overlap:** none with anything we train on. The paper's own t-SNE contrasts Ben-10 against OOD-Speech (the
  MaCro/Bengali.AI competition lineage) and shows a clear distribution shift. Same custodian as MaCro, so
  PCM-hash dedup and the fingerprint audit are still mandatory, but shared *prompts* are impossible: there are no
  prompts, the speech is spontaneous.
- **Published WER baselines — and they are alarming.** Averaged over the ten regions (paper Table 3, ratios):
  Google ASR 1.03, **Hishab Conformer 0.87**, Wav2Vec2(SCB) 0.89, Tugstugi/Whisper 0.81; fine-tuned on Ben-10
  itself, Tugstugi drops only to 0.70.

**Three gate problems, stated plainly.**

1. **`<>` markers on 46.4 % of train rows (6,187/13,342) and 50.2 % of valid rows.** These are the only
   non-Bengali characters in the entire corpus. The paper uses `<>` for a *null model prediction* in its Table 5
   and **never defines the marker for the transcripts themselves**. If it is an unintelligible/untranscribed
   span — which is the natural reading, and which matches Vaani's `[unintelligible]` — then SPEC §1's registered
   Vaani policy DROPS those rows (partial labels teach deletion, the Phase-1 lesson), and Ben-10's yield collapses
   from 73.8 h to roughly **35 h** after the duration band is also applied. **This is missing evidence and it is
   decisive: no admission decision on Ben-10 is possible until the custodian confirms what `<>` means.**
2. **5.53 % of rows (832) exceed the registered 23.0 s cap** (0 % below 0.5 s). 73.82 h becomes **64.71 h** inside
   the band, before the `<>` question. Ben-10's clips are an order of magnitude longer than anything currently in
   the corpus; the cap was not written with 30-second VAD segments in mind.
3. **It would probably fail G0.6 as written, for the wrong reason.** The machine-consensus audit flags a clip when
   both recognisers are >30 % WER from the reference but <15 % from each other. On Ben-10 the *published* numbers
   put Hishab at 0.87 WER and every SCB model at 0.8–1.0 — that is, essentially every clip clears the ">30 % from
   the reference" half of the test, on a corpus whose labels are the most carefully human-validated in the survey.
   G0.6 would be measuring dialect distance, not label noise. This is precisely council G's correction (iv) about
   the CV17 filter, arriving again from the other direction, and it is an argument about **our gate**, not about
   Ben-10.

**And a register caveat.** Ben-10 is *regional dialect* Bengali from Bangladesh, with genuinely divergent
morphology and lexicon (আঁই ন যাই for আমি যাই না). D's conversational slice is 774 **IndicVoices** rows —
Indian, standard-colloquial. That Ben-10 would move D's conversational guard is a hypothesis with no supporting
evidence; if anything the paper's distribution-shift analysis argues the two are far apart. Its plausible value
is robustness and OOD coverage, which the registered objective S does not measure.

**One genuinely good property:** **0.00 % Latin-script rows after the frozen surface**, in 15,008 rows. English
loanwords are written in Bengali script throughout (*"ডিপারমেন্ট অব সপটোওয়্যার ইনজিনারিং"*). Ben-10 passes
G0.9(a) with the largest margin of any candidate, and its convention matches FLEURS's. Only 45 stray ZWJ
characters, which the registered audit key strips.

### 3.4 RESPIN — other slabs exist, none of them help Q5

RESPIN-S1.0 categorises by transcription quality into clean (10,416.58 h across nine languages), semi-noisy
(2,288.06 h) and noisy (1,617.47 h). We took Bengali `train_clean` **whole** (894.2 h). Pro-rating the
Bengali share gives roughly **200 h semi-noisy and 140 h noisy** — estimate only, the per-language split was not
located. Two reasons this is not the answer to Q5: (a) it is **read speech** of expert-composed agriculture/finance
prompts, the same register as `madasr`, so it adds zero conversational coverage; (b) the slabs are defined by
*lower transcription quality*, which is the axis the owner's "quality over quantity" rule pushes against. Licence
and access are already settled (CC-BY-4.0 via the archived IISc TAC), so this remains available if the phase ever
wants more read hours — it just does not answer the question asked.

### 3.5 Vaani's untranscribed remainder — NO

The dataset card (updated 2026-09-07) reports ~31,255 h of spontaneous image-prompted speech across 165 districts
and 106 languages, of which **only 2,043 h are transcribed at all**, spread evenly across districts. Our Bengali
transcription part is 118.1 h ingested / 117.4 h retained. **The remainder has no human transcripts**, and the
owner's admission rule (SPEC §0) permits model-filtered human labels but forbids pseudo-labels outright. There is
nothing to admit: transcribing it ourselves would be pseudo-labelling by definition. **NO, and not a close call.**

*Unresolved sub-question, flagged not guessed:* the card was updated **today (2026-09-07)** and I could not read
its README from this session (the MCP identity is not on Vaani's authorised list, though the project's own token
holds the accepted gate). Whether ARTPARK released *additional Bengali transcription* beyond our 118.1 h is
therefore **unverified**. Exact check, to run with the project's HF token, no GPU, no download:

```
hf download ARTPARK-IISc/Vaani --repo-type dataset --include "*.md" --local-dir /tmp/vaani_card
python - <<'PY'   # compare transcribed Bengali row counts against data4/vaani manifests
from huggingface_hub import HfApi
print([f for f in HfApi().list_repo_files("ARTPARK-IISc/Vaani", repo_type="dataset")
       if "engali" in f or "/bn" in f][:50])
PY
```

### 3.6 Shrutilipi — NO, permanently, and for a better reason than the ablation

Phase 1 excluded it after a −24.6 pp ablation and both councils confirmed (SPEC §1: "OUT, 0 rows, 0 GPU-h"). The
survey adds an independent and stronger ground: Shrutilipi's labels come from **forced alignment** of news audio
to separately-published text. That is neither a human transcript of the recording nor a model-filtered human
transcript — it is an alignment artifact, and the −24.6 pp result is what an alignment artifact does to a CTC
model. It fails SPEC §0's admissible-labels rule on provenance alone, so it stays out even if someone produces a
better-aligned release. `ucalyptus/shrutilipi_bengali` is a third-party mirror and changes nothing.

### 3.7 Telephone / call-centre Bengali — NO, nothing real exists publicly

The only Bengali telecom/customer-care speech datasets I could find on the Hub are
`kawshikbuet17/bengali-telecom-customer-care-speech` and `-v2` (7.5K rows). Both are **synthetic TTS**: the card
says "synthetic Bengali speech generated from telecom and customer-care style text prompts" and the schema
carries an `is_synthetic` boolean. SPEC §0 bans TTS audio outright. The associated paper (arXiv 2608.20346) is
titled *"Building and Evaluating a **Synthetic** Bengali Speech Resource for Telecom Customer Care"*, which
settles it. **No genuine Bengali call-centre corpus with human transcripts and a public licence was located** —
this is a real gap in the ecosystem, not an oversight in the search.

### 3.8 Parliamentary / broadcast archives — NO

- **BanSpeech**: 8,085 utterances / **6.52 h** verified, drawn from YouTube across 20 domains including TV news
  and parliament speech. Too small to matter and YouTube-sourced, which the owner ruled out ("useless YouTube …
  should not be there").
- **Bengali-Loop**: 191 recordings / 158.6 h from public YouTube channels, transcripts obtained by **subtitle
  extraction** followed by human-in-the-loop correction. Subtitle-derived provenance plus YouTube audio whose
  underlying rights we cannot clear for a model release. **NO.**
- **WorldSpeech**: 65k h / 76 languages of *aligned* audio-transcript from parliamentary proceedings and
  broadcasts. Same alignment-artifact class as Shrutilipi. Bengali share unknown. **NO.**
- Bangladesh's Jatiya Sangsad publishes no aligned human-transcribed speech corpus that I could locate.

### 3.9 RegSpeech12 — NO, on the licence rule, and this is the one worth re-checking

arXiv 2510.24096: ~100 h of Bengali **spontaneous** speech across twelve Bangladeshi regional dialects. On
subject matter it is the second-best fit in this entire survey after SPRING-INX. But: the paper says only "released
for public use"; it is distributed via **Kaggle** (`mdrezuwanhassan/regspeech12`), not a Hub repo I can inspect;
and the only licence signal I found is that a *derived model* (`Nirban10/mms-1b-bangla-regspeech12`) carries
CC-BY-NC-4.0 — which is a downstream author's choice and establishes nothing about the corpus. **Under the stated
rule, an unverifiable licence is a NO.** If the owner wants it, the resolution is one look at the Kaggle dataset
page's licence field, and if that reads CC0/CC-BY it moves to roughly Ben-10's rank with more hours.

### 3.10 OpenSLR and the rest of the Hub

- **SLR53** (~196K utterances, CC BY-SA 4.0) — already in, 163.99 h.
- **SLR37** — high-quality multi-speaker **TTS studio read** data for bn-BD/bn-IN. Read speech, tiny, no
  conversational value.
- **OOD-Speech** (1,178 h train) — the MaCro/Bengali.AI competition lineage, already in as `macro_filtered`.
- Everything else a Hub search surfaced for "bengali speech" is hate-speech *text* classification, tiny sample
  packs (`psdn-ai/*`, `Speech-data/*`, `shunyalabs/*` — all unsized, undocumented, provenance-free), speech-
  enhancement derivatives of SLR53, or gated personal mirrors. **None passes provenance or licence verification.**
  There is no undiscovered large Bengali conversational corpus sitting on the Hub.

---

## 4. What admitting the top three would require under Gate 0

Preconditions that apply to **all three**, since they are not candidate-specific: G0.2 licence record archived
under `experiments/p4/licences/`; G0.3 text-leakage audit (frozen surface + the three audit keys + fuzzy ≥0.90)
against **every** protected panel with zero unresolved matches; G0.4 decoded-PCM SHA-256 and the ≥12-hit
fingerprint audit against every protected panel, with any matching training row excluded; G0.5 zero `<unk>`
under the frozen tokenizer plus the `L+R ≤ T`, `(L+R)/T ≤ 0.90` feasibility rule with ≤1 % loss; G0.9(b)
incoming-duplicate rate; MAX_PER_TEXT; and — since the SPEC's multiplicity rule is that a slot announced is a
slot counted (§8m, m stayed 4 for a deferred CV17) — **registering any of these raises m and re-tests every
prior admission at the new quantile.** MaCro's m=4 margin is −0.306 at 0.99375; at m=5 (1−0.025/5 = 0.995)
it would have to be re-checked, and its conversational-guard headroom is ~0.08 pp.

**1. SPRING-INX Bengali-script-only (≈280 h est.)**
- **An explicit owner amendment to SPEC §3 G0.9(a)**, in writing, before any ingest: G0.9(a) currently fails the
  bucket at >1 % script violations and SPRING is at ~47 %. The amendment must (i) name the replacement rule
  ("rows whose surfaced target contains a Latin letter are removed; the bucket is judged on the remainder"),
  (ii) state the registered bias verbatim — that this is a *content* filter removing code-switched conversation,
  not a defect filter, and that the resulting corpus under-represents how Bengali conversation is actually
  spoken — and (iii) fix the threshold before the retained rate is known, so it cannot be tuned to the yield.
- **Measure the real hours before deciding.** My ≈280 h is a surfaced-character proxy over 1,000 sampled rows.
  Compute the exact retained hours from the `duration` column, Latin-filtered, at CPU cost only.
- G0.9(c) duration-vs-metadata and the 23.0 s cap on **R2 specifically** (max 47.21 s observed).
- G0.9(e): the orchestrator reads 200 random retained targets. The specific risk is that the retained subset has
  a *different* convention from the discarded one — check that Bengali-script transliterations of English do not
  appear in the retained rows (which would mean two conventions coexist rather than one).
- G0.6 machine-consensus audit on 1,000 retained clips + the 230 board clips. SPRING is standard-colloquial
  dialogue, so unlike Ben-10 it has no structural reason to fail this.
- Its official dev (70 h) and eval (10 h) stay protected one-shot panels, never trained on, hashes recorded.
- Then its own 8,000-update arm against O_ref at the 1−0.025/m quantile, with the conversational guard against
  the frozen P1 baseline (≤22.075 %). **At ≈280 h it does not clear the §8k Q3 ≥500-unique-hour floor, so it can
  never be the union fallback on its own** — it is a union contributor only.

**2. IndicVoices upstream delta (0–116 h, unresolved)**
- **Resolve the delta first — this is free and is the single cheapest thing in this document.** Reconstruct the
  Phase-1 raw ingest count for `bengali/train` and diff it against today's 335,367 rows, separating upstream
  growth from our own cleaning deletions. Until that split exists there is nothing to register, and the honest
  status is "unknown", not "+116 h available".
- If the delta is real: the rows must be **speaker- and sentence-disjoint from panels D, M, K, R and the
  holdout**, which is harder here than anywhere else in the survey precisely because IndicVoices is 52.9 % of D.
  G0.3 + G0.4 must be run against D and the holdout with zero tolerance; any new row sharing a speaker with a D
  row is a selection-integrity problem even when the text differs.
- No new convention work: same custodian, same `verbatim`/`normalized` fields, same frozen surface, gate already
  accepted, CC-BY-4.0 record extends trivially.
- It would still need its own arm and would still raise m — and council G Q5 is explicit that this is *adding
  hours*, a different intervention from *upweighting* IndicVoices, whose sampling weights stay frozen.

**3. Ben-10 (≈35 h after `<>`, ≈65 h if `<>` rows are keepable)**
- **Blocking, before anything else: get the custodian to define `<>`.** 46.4 % of rows carry it. If it marks an
  unintelligible span, SPEC §1's registered Vaani policy drops those rows and the bucket is ~35 h; if it marks
  something benign (a pause, a redaction of a name), it is ~65 h. Registering a bucket without knowing which is
  registering a number we do not have. Ask Bengali.AI, or read the annotation guideline; do **not** infer it from
  the audio, which would require decoding and is not what a survey does.
- **A registered decision on G0.6 before the audit runs, not after.** Published baselines put every SCB
  recogniser at 0.8–1.0 WER on Ben-10, so the consensus-disagreement gate as written will flag near-everything.
  Either (i) exclude Ben-10 on that basis and say so, or (ii) amend G0.6 with a dialect-specific rule fixed in
  advance and disclose that the amendment was made *because* the gate mis-measures dialect distance. What is not
  allowed is running the audit, seeing it fail, and then relaxing the threshold.
- Duration: 832 rows (5.53 %) over the 23.0 s cap; 73.82 h → 64.71 h in-band. Disclose the removal count by
  reason as §1 requires.
- Licence: archive the CC0-1.0 declaration under `experiments/p4/licences/` **with the repo revision hash**,
  since it is a one-line README with no LICENSE file and no licence statement in the paper.
- PCM-hash dedup against MaCro is mandatory (shared custodian) even though shared prompts are impossible.
- Withheld test split: the 1,682 test clips are not in the HF release. If they ever appear, they are a protected
  panel, never training data.
- **At ≈35–65 h it is far below the §8k Q3 500-hour floor.** Ben-10's honest value proposition is *not* the
  registered objective S — it is dialect robustness and OOD coverage, which S does not measure at all. Admitting
  it to chase D's 0.87 pp would be admitting it for a benefit nothing in this survey supports.

---

## 5. The finding that matters most

There is **no large, clean, unencumbered pool of Bengali conversational speech with human transcripts that we
have overlooked.** The ecosystem's conversational Bengali is: SPRING-INX (which we excluded on a script-uniformity
rule, not a quality one), IndicVoices (which we already train on, and may have grown), Vaani (already in, 117 h),
Ben-10 and RegSpeech12 (~75 h and ~100 h, both regional dialect, one with an unverifiable licence), and then a
long tail of YouTube-derived, alignment-derived or outright synthetic material that our admission rules exclude
by design.

So council G Q5's arithmetic — 1.64 pp on IndicVoices or 4.62 pp on SUBAK.KO to close D's 0.87 pp — cannot be
answered by acquiring conversational hours at Phase-4 scale. The largest admissible conversational addition on
the table is SPRING-INX at ≈280 estimated hours, it requires an owner amendment that must be disclosed as a
content filter, and **whether it moves D at all remains a hypothesis with no supporting evidence** — exactly as
council G ruled for CV17. Nothing in this survey changes the Phase-4 execution order in §8m, and nothing in it
should be read as a recommendation to change it.

# Lamina VLM Parser

A vision-language model pipeline for extracting structured electrical data from solar Single Line Diagrams (SLDs). Takes a raw engineering drawing as input and outputs a fully queryable SQLite site configuration database; inverter, DC combiner, string, and module-level specifications, enriched with manufacturer spec sheet data.

Provided as a portfolio piece demonstrating applied LLM engineering, structured data extraction, and ETL pipeline design.

> **Status: Work in Progress**
> Functional end-to-end but currently tuned against a small set of test SLDs. Error rate is approximately 5%, primarily in information-dense diagram regions. 
> The main avenue being explored is further decomposing extraction into narrower, single-concern prompts, particularly for inverter and string-level data where overlapping values increase ambiguity. This constrains VLM attention to a tighter scope, reducing hallucination and making errors easier to isolate.

---

## Before & After

**Input: Raw SLD drawing:**

![SLD Input](../assets/Anon_SLD_1MWp.png)

**Output: Interactive site configuration viewer:**

![HTML Output](../assets/example_SQL_1MWp.png)

The pipeline reads the SLD image and produces an interactive radial visualisation with every inverter, DC combiner box, and string fully populated and queryable. Clicking any node surfaces its full electrical specification, including module-level STC/NOCT data enriched from a component library.

---

## Methodology

1. **SLD Parsing:** The SLD image is passed through a set of modular, per-component VLM extractors, each scoped to a single component type (metadata, inverters, DC combiners, strings, modules). Decomposing extraction this way constrains the model's attention to a specific diagram region, improving structured output reliability.
2. **Schema Validation:** Each extracted component is validated against a Pydantic schema before writing to the database.
3. **Database Seeding:** Validated JSON is seeded into a per-project SQLite database via SQLModel ORM, building a relational hierarchy: Site → Inverters → DC Combiner Boxes → Strings.
4. **BOM Enrichment:** A BOM CSV is parsed and fuzzy-matched against a component library to link full manufacturer spec sheet data to every string row.
5. **Query & Visualisation:** The enriched database is queried to generate an interactive radial HTML viewer for drill-down component inspection.

---

## Folder Structure

```
vlm_parser/
├── prompts/        # Per-component VLM prompt templates
├── schema/         # Pydantic models for each SLD entity   
├── sqldb/          # SQLModel ORM models and seed scripts
└── src/            # Extraction and pipeline scripts
```

---

Note: No proprietary data or credentials are included.

Here’s a **compressed version** of the prompt — short, efficient, and still powerful for financial data extraction:

---

**Prompt:**

Extract only the information a skilled financial analyst would find useful from the given financial text.
Ignore generic commentary or non-financial content.
Return clean, valid **JSON** with dynamic keys reflecting extracted data.
Include only relevant quantitative and qualitative insights.

**Possible keys (add or omit as needed):**
`company_name`, `sector`, `fiscal_year`, `financial_metrics`, `growth_rates`, `assets`, `liabilities`, `debt`, `equity`, `valuation_metrics`, `market_trends`, `risk_factors`, `analyst_sentiment`, `data_source`

**Output example:**

```json
{
  "company_name": "ABC Ltd.",
  "fiscal_year": "2024-25",
  "financial_metrics": {
    "revenue_growth": "15.2%",
    "retained_earnings": "₹36T",
    "long_term_debt": "₹33.3T"
  },
  "valuation_metrics": {"debt_to_equity": "0.9"},
  "market_trends": "Expansion in non-current assets",
  "risk_factors": "Stretched valuations, weak demand"
}
```

Focus only on financially material data and omit null or irrelevant fields.

---

Would you like me to make it even shorter — optimized for embedding inside a script or API call (e.g., a one-paragraph version)?

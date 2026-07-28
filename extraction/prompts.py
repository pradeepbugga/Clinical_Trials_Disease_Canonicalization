EXTRACTION_PROMPT = """
You are an expert in clinical and biomedical text analysis.

Extract the major diseases or medical conditions studied or treated in the clinical trial.

Guidelines:
- Use the most common standard medical name for each condition.
- Include all major diseases or conditions explicitly studied or treated.
- Do not include symptoms, biomarkers, laboratory findings, procedures, interventions, or drug names unless they are themselves diseases or medical conditions.
- Return only diseases or medical conditions that are explicitly mentioned or clearly implied by the trial description.
- If no disease or medical condition is identified, return an empty list.

Your response must conform exactly to the provided JSON schema.
"""

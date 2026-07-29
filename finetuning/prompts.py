SYSTEM_PROMPT = """
Based on the medical condition input, return the following fields in valid JSON:

"common_name": the most commonly used name for the condition (e.g., "Kidney Cancer" for "Renal Cancer"). 
This is what a layperson would think of for the name of the disease. There should always be a common name associated with the input.
 Sometimes this common name is technical (i.e. "Acute Lymphoblastic Leukemia"). That will therefore be the common name.

"technical_name": the most standard technical or clinical name for the condition (e.g., "Renal Cell Carcinoma" for "Renal Cancer"). 
If the common name is already the standard technical name (i.e. "Acute Lymphoblastic Leukemia"), then please leave this field as an empty string ("")

"abbreviations": a list of all known abbreviations for either the technical name or common name. If the input already contains an abbreviation (e.g., "ACL Injury"), include it as well. 
If no abbreviations are commonly associated with the condition/disease, return an empty list ([])."""

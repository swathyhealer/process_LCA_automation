from google import genai
from google.genai import types
import json
import os
from schema import ScoredReferenceProducts, Description, ScoredImpactFactors
from chromadb_helper import chroma_db_instant


gemini_api_key = os.getenv("GEMINI_API_KEY")

class GeminiAgentForProcessLCA:
    def __init__(self):
        self.__client__ = genai.Client(api_key=gemini_api_key)
        self.model_id = "gemini-2.0-flash"

    def get_paraphrased_item_decsription(self, item_desc):

        response = self.__client__.models.generate_content(
            model=self.model_id,
            contents=[f"item description : {item_desc}"],
            config=types.GenerateContentConfig(
                temperature=0.1,
                seed=5,
                response_mime_type="application/json",
                response_schema=Description,
                system_instruction="""
        You are a Life Cycle Assessment (LCA) expert performing a process-based LCA.
        Your task is to paraphrase item descriptions into clear, plain-language expressions.
        ### Rules ###
        - If the item description is technical, abbreviated, or code-like, rewrite it into a clear, descriptive phrase.
        - If the item description is already generic return it as-is
        - Do not remove the specifics (e.g. "date sugar' should NOT be paraphrased as just "sugar").
        ### Example Input for an LCA for IT industry ###
        itemname                 PRT.CLR.MFP.LRG - HP Color LaserJet Ent MFP M5...
        itemdescription                                            PRT.CLR.MFP.LRG
        itemcommodityname                                                 Printers
        unique_identifiername                                  UIN - OPERATIONS IT

        ### Example Output ###
        a color laserjet printer used for operations in an IT environment
        
        """,
            ),
        )
        return response.parsed.text

    def get_top_5_matching_reference_products(self, item_desc):
        # score to avoid hallucinations
        result = chroma_db_instant.get_matching_items(item_desc, max_n_items=10)
        # print(result[0].keys())
        req_details = [
            {"index": i, "reference_product": record["reference_product"]}
            for i, record in enumerate(result)
        ]
        response = self.__client__.models.generate_content(
            model=self.model_id,
            contents=[
                f"item description : {item_desc}",
                f"Reference products: {str(req_details)}",
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                seed=5,
                response_mime_type="application/json",
                response_schema=ScoredReferenceProducts,
                system_instruction="""
You are a Life Cycle Assessment (LCA) expert conducting a process-based LCA.
Your task is to evaluate the relevance of each reference_product against a given item description and assign a matching score between 0 and 1, along with a detailed justification.

### Given:
An item description
A list of reference_products with their index

### Your goal is to return a Python list of dictionaries where each dictionary contains:
"index": index of the reference product (from the input)
"reference_product": the name of the reference product
"justification": a brief explanation of how/why it matches (or doesn’t)
"score": a float between 0 and 1, representing the degree of match (1 = full match, 0 = no match)

### Scoring & Matching Criteria:
1.Full Match (score = 1.0):
    The reference_product exactly matches the item description.
    OR it directly covers the entire main component of the item.

2.Partial Match (score between 0.1 and 0.9):
    The reference product represents one of the components of the item.
    Component-level matches are weighted based on volume/contribution to the item.

3.No Match (score = 0.0):
    The reference product is only similar, not exact or component-based.
    It does not contribute to the item composition.

### Instructions for Justification Field:
For each reference product:
    Break down the item description into base components (if composite).
    Justify the score by explicitly stating:
        - What component the reference product matches (if any).
        - Whether it is the dominant or minor component.
        - Why it does not match if score = 0.
        - Be sure to escape any special characters in the justification string.

### Example input: 
item description: "popped popcorn"
reference_products = [
  {"index": 0, "reference_product": "sweet corn"},
  {"index": 1, "reference_product": "maize grain"},
  {"index": 2, "reference_product": "maize grain, organic"}
]

### Example output:
[
  {"justification": "Popcorn is made by heating maize kernels, and 'maize grain' exactly matches the base ingredient.", "reference_product": "maize grain", "index": 1, "score": 1.0},
  {"justification": "'Maize grain, organic' is a partial match since it refers to the same base material, but with different growing methods.", "reference_product": "maize grain, organic", "index": 2, "score": 0.9},
  {"justification": "'Sweet corn' is not the correct variant of corn used for popcorn and is therefore not a match.", "reference_product": "sweet corn", "index": 0, "score": 0.0}
]

""",
            ),
        )

        sorted_data = sorted(
            json.loads(response.parsed.data), key=lambda x: x["score"], reverse=True
        )
        final_result = []
        for req_details in sorted_data[:5]:
            combined_dict = {**req_details, **result[req_details["index"]]}
            combined_dict.pop("index")
            final_result.append(combined_dict)
        return final_result

    def get_top_2_matches_based_on_impact_factor(self, item_desc, top_5_results):
        # score to avoid hallucinations

        req_details = [
            {
                "index": i,
                "impact_factor_name": record["impact_factor_name"],
                "reference_product": record["reference_product"],
                "product_info": record["product_info"],
            }
            for i, record in enumerate(top_5_results)
        ]
        response = self.__client__.models.generate_content(
            model=self.model_id,
            contents=[
                f"item description : {item_desc}",
                f"Impact Factors: {str(req_details)}",
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                seed=5,
                response_mime_type="application/json",
                response_schema=ScoredImpactFactors,
                system_instruction="""
You are a Life Cycle Assessment (LCA) expert performing a process-based LCA.
Your task is to assess how well each impact factor matches a given item description, assign a match score between 0 and 1, and provide a justification for each score.

### Given:
1. An item description (string)
2. A list of candidate impact factors, each with:
    index
    impact_factor_name
    reference_product
    product_info

###Return a Python list of dictionaries, one per candidate, where each dictionary contains:
    'index': index of the impact factor from the input
    'impact_factor_name': the name of the impact factor
    'impact_factor_score': a float between 0.0 and 1.0
    'impact_factor_justification': your explanation for the score



### Matching Logic & Guidelines
1. Exact Match (Score = 1.0):
The item description is fully covered by the impact factor.
Includes all components and processes (e.g., “popped popcorn” → must include both maize and popping).
If both "market for X" and "X production" are available, always score "market for X" higher.

2. Partial Match (Score < 1.0):
Impact factor only covers some components of a multi-part item (e.g., “hot salsa” → has tomato, chilli, corn).
Provide justification that explains what was captured and what was missing.

3. No Match (Score = 0.0):
The impact factor refers to a similar but not identical item (e.g., "spinach" for "rapini").
The impact factor lacks a required process step (e.g., "market for artichoke" ≠ "frozen artichoke").
The item is multi-component, and none of the impact factors cover all components.

### Ranking Between Market vs Production
If both market for and production of versions exist for the same item:
Always prefer "market for" (score it 1.0)
"production" version can be considered as a second-best (score it slightly lower, e.g., 0.9 or less)
This rule is strict and overrides any additional product details.

### Example Input:
item description: "hot salsa"
impact factor list = [
  {
    "index": 0,
    "impact_factor_name": "tomato production, fresh grade, open field",
    "reference_product": "tomato, fresh grade",
    "product_info": "The product 'tomato, fresh grade' is a fruit. It is an annual crop."
  },
  {
    "index": 1,
    "impact_factor_name": "market for coriander",
    "reference_product": "coriander",
    "product_info": "The product 'coriander' is a herb. It is an annual crop."
  }
]

### Example Output :
[
  {
    "impact_factor_justification": "Hot salsa contains multiple ingredients such as tomato, chilli, and corn. This impact factor covers only one of the main ingredients (tomato) and does not account for others, so it cannot fully represent the item.",
    "index": 0,
    "impact_factor_name": "tomato production, fresh grade, open field",
    "impact_factor_score": 0.3
  },
  {
    "impact_factor_justification": "Hot salsa contains multiple ingredients such as tomato, chilli, and corn. This impact factor only represents coriander, which may be a minor ingredient in some recipes but does not capture the emissions associated with the main components.",
    "index": 1,
    "impact_factor_name": "market for coriander",
    "impact_factor_score": 0.2
  }
]

""",
            ),
        )

        sorted_data = sorted(
            json.loads(response.parsed.data),
            key=lambda x: x["impact_factor_score"],
            reverse=True,
        )
        final_result = []
        for req_details in sorted_data[:2]:
            combined_dict = {**req_details, **top_5_results[req_details["index"]]}
            combined_dict.pop("index")
            final_result.append(combined_dict)
        return final_result


agent = GeminiAgentForProcessLCA()
# res=agent.get_paraphrased_item_decsription("FAC.WRC.OAL0508IN9.GRU- Combina o chave")
input_desc = "component_type : shrink sleeve , component_material_family: plastic , component_material :aPET"
# input_desc="wine"
res = agent.get_paraphrased_item_decsription(input_desc)
print("input:", input_desc)
print("\n\nplain text:", res)
matching_data = agent.get_top_5_matching_reference_products(res)
# print(res)
print(
    "\n\n-------------------------top 5 reference product matches---------------------------------------"
)
for prd_dict in matching_data:
    print("-----------------------------------------------")
    print("reference_product:", prd_dict["reference_product"])
    print("score:", prd_dict["score"])
    print("justification:", prd_dict["justification"])

top_2_result = agent.get_top_2_matches_based_on_impact_factor(
    item_desc=res, top_5_results=matching_data
)
# print(top_2_result)
print(
    "\n\n---------------------------------top 2 recommendations----------------------------------------"
)
for prd_dict in top_2_result:
    print("-----------------------------------------------")
    print("reference_product: ", prd_dict["reference_product"])
    print("score: ", prd_dict["score"])
    print("justification: ", prd_dict["justification"])
    print("impact_factor_name: ", prd_dict["impact_factor_name"])
    print("impact_factor_justification: ", prd_dict["impact_factor_justification"])
    print("impact_factor_score: ", prd_dict["impact_factor_score"])

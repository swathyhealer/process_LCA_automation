# Process-Based LCA Automation Application

This repository contains an interactive application built on top of the research work presented in the paper:  
**"Carbon Assessment with Machine Learning"** by Amazon Science.

📄 **Paper**: [Carbon Assessment with Machine Learning (Amazon Science)](https://www.amazon.science/publications/parakeet-emission-factor-recommendation-for-carbon-footprinting-with-generative-ai)  
💻 **Original Code Repository**: [amazon-science/carbon-assessment-with-ml](https://github.com/amazon-science/carbon-assessment-with-ml)

---

# About Application


Measuring greenhouse gas (GHG) emissions is key for organizations aiming for net-zero goals. Life Cycle Assessment (LCA) helps estimate these emissions across a product’s life—from raw materials to disposal. A major challenge in this process is matching each product or activity with the right emission factor (EF), which is often manual, slow, and complex.

Our app automates this step using AI. It takes a product or service description, searches the ecoinvent EF database, and recommends the top two matching EFs with easy-to-understand explanations. This saves time, reduces errors, and helps companies report their carbon footprint more accurately.

ie, This application automates **Process-Based Life Cycle Assessment (LCA)** by mapping free-form item descriptions to relevant environmental **impact factors** using semantic search and LLMs.

---

##  What It Does

Given a product or item description, the app performs the following:

1. **Embeds the input** using the `thenlper/gte-large` model.
2. **Matches** the description to the most semantically similar reference products from the **Ecoinvent** dataset using **cosine similarity**.
3. **Selects** the top 2 relevant **impact factors** based on embeddings and LLM-based reasoning.
4. **Returns**:
   - `reference_product`: Closest match from Ecoinvent
   - `score`: Embedding similarity score between item description and reference product
   - `justification`: Explanation for this match
   - `impact_factor_name`: Top 2 selected impact factor names
   - `impact_factor_score`: Similarity score between item description and impact factor
   - `impact_factor_justification`: LLM-generated rationale for why the impact factor is relevant

---

##  How It Works

| Component             | Technology Used                   |
|----------------------|-----------------------------------|
| Embedding Model      | [`thenlper/gte-large`](https://huggingface.co/thenlper/gte-large) |
| LLM Model            | `gemini-2.0-flash`                |
| Vector Store         | [ChromaDB](https://www.trychroma.com/) |
| Similarity Metric    | Cosine Similarity                 |
| Dataset              | Ecoinvent                         |

The system stores embeddings for Ecoinvent entries in ChromaDB and leverages an LLM to provide contextual explanations and select the most relevant impact factors.

---

##  Features

-  Natural language input (e.g., *"LED light bulb 10W"*)
-  Embedding-based semantic matching to Ecoinvent
-  LLM reasoning for impact factor selection and justification


---


#  Citation

This application draws inspiration and design principles from the following work:

```bibtex
@Inproceedings{Balaji2024,
  author    = {Bharathan Balaji and Fahimeh Ebrahimi and Nina Domingo and Gargeya Vunnava and Abu-Zaher Faridee and Soma Ramalingam and Shikhar Gupta and Anran Wang and Harsh Gupta and Domenic Belcastro and Kellen Axten and Jeremie Hakian and Jared Kramer},
  title     = {Parakeet: Emission factor recommendation for carbon footprinting with generative AI},
  year      = {2024},
  url       = {https://www.amazon.science/publications/parakeet-emission-factor-recommendation-for-carbon-footprinting-with-generative-ai},
  booktitle = {NeurIPS 2024 Workshop on Tackling Climate Change with Machine Learning},
}

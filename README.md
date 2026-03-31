# CordisMine

This is the repository associated with the paper "Mapping European Funding for New Approach Methodologies (NAMs) and Replacement, Reduction and Refinement (3Rs) Projects from 2007 to 2025" by Matyjasiak and Avîrvarei et al (2026).

Data source: CORDIS (European Commission)
https://cordis.europa.eu/
Reused under EU reuse policy. This is not an official EU app. Data has been cleaned and transformed for this application.

- RELEASE 1. The projectSelection folder contains the scripts used to select projects (A. Project identification and screening):
  - train_classification_NAMs.ipynb contains the code used to train the classification model (Tier 1a). 
  - semantic_search.py contains the code used for semantic retrieval (Tier1b)
  - the prodigy folder contains the configuration file (prodigy.json) and helper functions (recipe.py) to cross-check projects (Tier 2)
  
- RELEASE 2 (IN PROGRESS) The analysis folder contains the raw data files and code for project categorisation and visualization in the form of a Jupyter notebook: Mapping 3Rs and NAM Research.ipynb. It includes the figures in the paper but also other analyses.

- RELEASE 2 (IN PROGRESS) The app folder contains the code for a standalone app that you can run locally to explore NAM-relevant projects and their labels. It also contains an experimental feature to query and summarize projects with a local LLM.

Each folder contains its own requirements file to reproduce analysis.

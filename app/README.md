# EU NAMs / 3Rs Funding Explorer
## A Streamlit Dashboard for Visualising NAM/3RS Related EU Funded Project Categories + Searching CORDIS NAM/3Rs Projects

The app is available via Streamlit at the following link: (LINK TBD)

This project provides a dashboard to explore EU-funded New Approach Methodologies (NAMs) and 3Rs (Replacement, Reduction, Refinement) research projects funded by the EU between 2007 and 2025.
It includes:

1. The full downloadable dataset of EU NAM/3Rs projects via Streamlit web dashboard
2. Search function using BM25 + MiniLM-L6-v2 embeddings
3. A local WebLLM, running entirely inside your browser, which gives a short description of the most relevant (top 8) projects to your query.

### Set-up
1. Clone repository
<pre> git clone https://github.com/DataScienceINT/CordisMine/new/main/app </pre>
<pre> cd CordisMine </pre>

2. Create environment
Windows:
<pre> python -m venv nam_env </pre>
<pre> .\nam_env\Scripts\activate </pre>

Mac:
<pre> python3 -m venv nam_env </pre>
<pre> source nam_env/bin/activate </pre>

3. Install requirements
<pre> pip install -r requirements.txt </pre>

4. Run the app
<pre> streamlit run streamlit_app.py </pre>

### Features
User Query ➡️

Hybrid Search (BM25 + Embeddings) ➡️

Relevant Projects (threshold-based) ➡️

Top 8 rows (Compressed + Sanitized) ➡️

Browser LLM (WebLLM) ➡️

Structured summary + analysis of first 8 projects

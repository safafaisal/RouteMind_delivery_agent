# RouteMind — RL + CrewAI + Groq Delivery Agent

A small educational delivery-routing application combining:

- Q-learning for route/order decisions
- CrewAI for the LLM agent layer
- Groq for LLM inference
- Streamlit for the UI
- GitHub + Streamlit Community Cloud for deployment

## Files

- `app.py` — complete application
- `requirements.txt` — Python dependencies
- `.gitignore` — prevents local secrets from being committed

## Run locally

Use Python 3.12 to match Streamlit Community Cloud's current default.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Set your Groq key locally:

PowerShell:
```powershell
$env:GROQ_API_KEY="YOUR_KEY"
streamlit run app.py
```

The RL simulator works without a Groq key. Only the CrewAI explanation feature needs the key.

## Streamlit Community Cloud

1. Push this folder to GitHub.
2. Create a Streamlit Community Cloud app from the repository.
3. Set the main file to `app.py`.
4. In the app's Secrets settings, add:

```toml
GROQ_API_KEY = "YOUR_KEY"
```

Do NOT put the key in `app.py` or commit it to GitHub.

## Architecture

User → Streamlit → Q-learning simulator → learned route

Optional LLM layer:

User question → CrewAI agents → Groq → explanation of the actual RL result

The LLM does not calculate the RL policy. It only analyzes and explains the numerical result supplied by the simulator.

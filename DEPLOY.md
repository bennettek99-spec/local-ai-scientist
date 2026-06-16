# Deploying to the web (Streamlit Community Cloud)

This puts the app at a permanent URL (e.g. `https://your-app.streamlit.app`),
auto-redeploys on every `git push`, and is **free**. It's protected by a
password and uses **Groq** for the LLM (a deployed app can't reach your local or
cloud Ollama).

## What changes when it's on the web
- **LLM = Groq.** Set via secrets (below). Your local setup is unaffected.
- **Library data is ephemeral.** Free hosting wipes the filesystem on each
  restart/redeploy, so your saved papers + embeddings reset. Rebuild anytime by
  adding papers from the Live / bioRxiv assistant tabs. (Persisting data needs
  external storage — ask if you want that.)
- **Password-gated.** Only people with `APP_PASSWORD` can open it.

---

## One-time setup

### 1. Push this project to GitHub
From the `local-ai-scientist/` folder:

```bash
git init
git add .
git commit -m "Local AI Scientist"
git branch -M main
git remote add origin https://github.com/<your-username>/local-ai-scientist.git
git push -u origin main
```

> Your `.env` and `.streamlit/secrets.toml` are gitignored — **no keys get
> pushed.** Verify with `git status` that neither appears.

### 2. Create the app on Streamlit Community Cloud
1. Go to <https://share.streamlit.io> and sign in **with GitHub**.
2. Click **New app** → pick your `local-ai-scientist` repo, branch `main`.
3. Set **Main file path** to: `ui/streamlit_app.py`
4. Open **Advanced settings → Secrets** and paste (using your real values):

   ```toml
   APP_PASSWORD = "choose-a-strong-password"
   LLM_PROVIDER = "groq"
   OPENAI_BASE_URL = "https://api.groq.com/openai/v1"
   OPENAI_API_KEY = "gsk_your_groq_key_here"
   OPENAI_MODEL = "openai/gpt-oss-120b"
   EMBEDDING_MODEL = "all-MiniLM-L6-v2"
   ```

5. Click **Deploy**. The first build takes a few minutes (it installs PyTorch,
   ChromaDB, etc. and downloads the embedding model on first load).

### 3. (Optional) Lock it down further
In the app's **Settings → Sharing**, set it to **private** and add only your own
email. That's a second layer on top of the password.

---

## Notes & troubleshooting
- **Free tier ≈ 1 GB RAM.** This app is on the heavier side (PyTorch + ChromaDB);
  it runs for a modest library but can be slow to wake from sleep.
- **First load is slow** — it downloads the ~90 MB embedding model once.
- **Updating the app:** just `git push`; Streamlit redeploys automatically.
- **If the build fails on `torch` size**, add these two lines to the **top** of
  `requirements.txt` to force the smaller CPU build, then push again:

  ```
  --extra-index-url https://download.pytorch.org/whl/cpu
  torch
  ```

- **Rotating secrets:** edit them in the Streamlit app's Secrets box — no code
  change or redeploy of the repo needed.

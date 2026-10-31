# VocalKey Streamlit — Browser Voice MFA Prototype

This version turns the Python GUI into a browser-based Streamlit app. It uses real microphone input through Streamlit's browser audio widget when available, with WAV upload as a fallback.

## Run locally

Requirements: Python 3.10+.

```bash
cd voice_mfa_streamlit
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

Install and run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the local URL shown by Streamlit, usually `http://localhost:8501`. Allow microphone permission in the browser. If the microphone widget is unavailable, upgrade Streamlit or use the WAV upload controls.

## Public panel access

For a temporary public HTTPS URL from your own computer, install the Streamlit community tunnel tool:

```bash
python -m pip install streamlit
streamlit run streamlit_app.py
```

Then use a secure tunnel such as Cloudflare Tunnel or deploy the repository to Streamlit Community Cloud. Do not expose sensitive recordings or real user accounts. The app is designed for consented academic demonstration only.

### Streamlit Community Cloud

1. Create a GitHub repository containing `streamlit_app.py` and `requirements.txt`.
2. Go to Streamlit Community Cloud and create a new app.
3. Select the repository and set the main file to `streamlit_app.py`.
4. Deploy and open the generated HTTPS URL from the panel laptop.
5. Test browser microphone permission in advance.

## Panel runbook

1. Explain that the app uses a real browser microphone but only a classical MFCC baseline.
2. Record three enrolment samples from your own voice or a consenting participant.
3. Record a fresh verification phrase and show the score and threshold.
4. Upload or replay a previous WAV and select `Replay attempt`. Explain that this app does not implement a real PAD detector; the scenario is for threat-model teaching.
5. Change the threshold to explain the false-accept/false-reject trade-off.
6. Use the panel notes and audit trail to explain why a real deployment needs PAD, secure template protection, rate limiting and FIDO2/passkeys.

## Limitations

- MFCC statistics are not ECAPA-TDNN or x-vector embeddings.
- No trained deepfake or presentation-attack detector is included.
- The threshold is not validated.
- No FAR, FRR or EER claims should be made from this demo.
- Streamlit processes audio in the running app session. It does not write a database or cloud storage record in this code, but public hosting environments have their own temporary-storage and logging policies.
- Use only consented recordings and delete sensitive test material after assessment.

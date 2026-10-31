"""VocalKey Streamlit browser demo.

Real, consented browser microphone/WAV input with a lightweight MFCC baseline.
Academic demonstration only: no production biometric claims and no server-side
persistence is implemented by this app.
"""
from __future__ import annotations

from io import BytesIO
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import streamlit as st

SAMPLE_RATE = 16_000


def clean_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise ValueError("The audio sample is empty.")
    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio)))
    if peak > 0.001:
        audio = audio / max(peak, 1.0)
    return audio


def read_audio(uploaded_file) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(BytesIO(uploaded_file.getvalue()), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = clean_audio(audio)
    if int(sample_rate) != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=int(sample_rate), target_sr=SAMPLE_RATE)
        sample_rate = SAMPLE_RATE
    return audio, int(sample_rate)


def extract_features(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    if len(audio) < int(sample_rate * 0.5):
        raise ValueError("Please provide at least half a second of speech.")
    emphasised = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    mfcc = librosa.feature.mfcc(
        y=emphasised,
        sr=sample_rate,
        n_mfcc=20,
        n_fft=1024,
        hop_length=256,
        n_mels=40,
    )
    delta = librosa.feature.delta(mfcc)
    vector = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1), delta.mean(axis=1)])
    vector = np.nan_to_num(vector).astype(np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("The audio did not contain a usable feature signal.")
    return vector / norm


def similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def init_state() -> None:
    defaults = {
        "enrol_features": [],
        "centroid": None,
        "enrol_names": [],
        "verification_result": None,
        "event_log": ["Session opened — no audio has been uploaded to a remote service by this app."],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_event(message: str) -> None:
    st.session_state.event_log.insert(0, message)
    st.session_state.event_log = st.session_state.event_log[:8]


def process_enrolment(uploaded_file) -> None:
    audio, sample_rate = read_audio(uploaded_file)
    feature = extract_features(audio, sample_rate)
    st.session_state.enrol_features.append(feature)
    st.session_state.enrol_names.append(uploaded_file.name or "browser microphone sample")
    centroid = np.mean(np.stack(st.session_state.enrol_features), axis=0)
    st.session_state.centroid = centroid / np.linalg.norm(centroid)
    st.session_state.verification_result = None
    add_event(f"Enrolment sample added — {uploaded_file.name or 'microphone capture'}; raw audio retained only in this Streamlit session.")


def process_verification(uploaded_file, threshold: float, scenario: str) -> None:
    if st.session_state.centroid is None:
        raise ValueError("Create at least one enrolment sample first.")
    audio, sample_rate = read_audio(uploaded_file)
    feature = extract_features(audio, sample_rate)
    score = similarity(st.session_state.centroid, feature)
    allowed = score >= threshold
    st.session_state.verification_result = {
        "score": score,
        "allowed": allowed,
        "scenario": scenario,
        "name": uploaded_file.name or "browser microphone sample",
    }
    add_event(f"Verification evaluated — {scenario}; similarity {score:.3f}; {'ALLOW' if allowed else 'DENY'} at threshold {threshold:.2f}.")


def main() -> None:
    st.set_page_config(page_title="VocalKey · Streamlit Voice MFA", page_icon="🎙️", layout="wide")
    init_state()
    st.markdown(
        """
        <style>
        .stApp { background: #071018; color: #e9f2f7; }
        [data-testid="stHeader"] { background: rgba(7,16,24,.9); }
        .block-container { max-width: 1180px; padding-top: 2.5rem; }
        h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.04em; }
        .eyebrow { color: #8ce9ea; font: 600 0.72rem/1.2 monospace; letter-spacing: .15em; text-transform: uppercase; }
        .hero-note { color: #91a8b5; font-size: 1.05rem; line-height: 1.65; max-width: 760px; }
        .guardrail { border: 1px solid rgba(237,176,93,.35); background: rgba(237,176,93,.08); padding: .9rem 1rem; border-radius: .5rem; color: #efd3a2; }
        .metric-card { border: 1px solid rgba(140,233,234,.2); background: rgba(13,32,43,.8); padding: 1rem; border-radius: .55rem; }
        .metric-label { color: #8ca2ae; font: .72rem monospace; text-transform: uppercase; letter-spacing: .08em; }
        .metric-value { color: #8ce9ea; font: 700 2.2rem 'Space Grotesk', sans-serif; }
        .allow { color: #82e7a6; font-weight: 700; }
        .deny { color: #ff9a96; font-weight: 700; }
        .small-note { color: #8ca2ae; font-size: .85rem; line-height: 1.55; }
        </style>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="eyebrow">BSc Cybersecurity / real-voice applied lab</div>', unsafe_allow_html=True)
    st.title("Can a voice be a useful second factor?")
    st.markdown('<p class="hero-note">A browser-accessible, consent-first prototype using real microphone samples and a lightweight MFCC similarity baseline. It demonstrates the workflow; it does not claim production biometric accuracy.</p>', unsafe_allow_html=True)
    st.markdown('<div class="guardrail"><strong>Ethics guardrail:</strong> Use only your own voice or a consenting participant. This app keeps audio in the current Streamlit session and does not implement a production database, biometric template protection, or a real PAD model.</div>', unsafe_allow_html=True)

    st.sidebar.header("Session controls")
    threshold = st.sidebar.slider("Similarity threshold", 0.50, 0.98, 0.80, 0.01)
    st.sidebar.caption("This is a demonstration threshold. Calibrate it only with a documented evaluation set.")
    if st.sidebar.button("Reset session", use_container_width=True):
        for key in ["enrol_features", "centroid", "enrol_names", "verification_result", "event_log"]:
            st.session_state.pop(key, None)
        st.rerun()
    st.sidebar.divider()
    st.sidebar.subheader("Current state")
    st.sidebar.write(f"Enrolment samples: **{len(st.session_state.enrol_features)}**")
    st.sidebar.write(f"Reference available: **{'Yes' if st.session_state.centroid is not None else 'No'}**")
    st.sidebar.caption("For public deployment, use HTTPS so the browser can request microphone permission.")

    enrol_tab, verify_tab, panel_tab = st.tabs(["1 · Enrol voice", "2 · Verify voice", "3 · Panel notes"])

    with enrol_tab:
        st.subheader("Create a local voice reference")
        st.write("Record three short samples for a more stable demonstration centroid. Speak naturally and vary the phrase slightly.")
        if hasattr(st, "audio_input"):
            enrol_audio = st.audio_input("Record an enrolment sample", key="enrol_audio")
        else:
            st.warning("This Streamlit version does not provide browser recording. Upgrade Streamlit or upload WAV files below.")
            enrol_audio = None
        enrol_upload = st.file_uploader("Or upload a WAV enrolment sample", type=["wav"], key="enrol_upload")
        enrol_source = enrol_audio or enrol_upload
        if enrol_source is not None:
            if st.button("Add this enrolment sample", type="primary", key="add_enrol"):
                try:
                    process_enrolment(enrol_source)
                    st.success(f"Enrolment sample added. Total samples: {len(st.session_state.enrol_features)}")
                except Exception as exc:
                    st.error(str(exc))
        if st.session_state.enrol_names:
            st.write("**Local session samples**")
            for index, name in enumerate(st.session_state.enrol_names, start=1):
                st.write(f"{index}. {name}")
        st.info("The feature vector is a classical MFCC baseline. It is not the same as an ECAPA-TDNN speaker embedding and should not be called a secure biometric template.")

    with verify_tab:
        st.subheader("Verify a new voice sample")
        scenario = st.selectbox("Teaching scenario", ["Live human voice", "Replay attempt", "Synthetic / deepfake voice"])
        st.write("For the live scenario, speak a new phrase. For a replay demonstration, play a previous recording through a speaker and capture it again, or upload the WAV file.")
        if hasattr(st, "audio_input"):
            verify_audio = st.audio_input("Record a verification sample", key="verify_audio")
        else:
            verify_audio = None
        verify_upload = st.file_uploader("Or upload a verification WAV", type=["wav"], key="verify_upload")
        verify_source = verify_audio or verify_upload
        if verify_source is not None and st.button("Evaluate voice sample", type="primary", key="evaluate"):
            try:
                process_verification(verify_source, threshold, scenario)
            except Exception as exc:
                st.error(str(exc))

        result = st.session_state.verification_result
        if result is not None:
            st.divider()
            st.subheader("Verification result")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Speaker similarity</div><div class="metric-value">{result["score"]:.3f}</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Policy threshold</div><div class="metric-value">{threshold:.2f}</div></div>', unsafe_allow_html=True)
            with col3:
                decision_class = "allow" if result["allowed"] else "deny"
                st.markdown(f'<div class="metric-card"><div class="metric-label">Decision</div><div class="metric-value {decision_class}">{"ALLOW" if result["allowed"] else "DENY"}</div></div>', unsafe_allow_html=True)
            st.progress(max(0.0, min(1.0, result["score"])))
            if scenario == "Live human voice":
                st.success("The MFCC baseline passed the selected threshold. This is a voice-signal result, not proof of identity.") if result["allowed"] else st.warning("The sample failed the selected threshold. Discuss noise, microphone differences, and threshold trade-offs.")
            else:
                st.warning("This is an attack-teaching scenario. The current app does not implement a real PAD/deepfake detector; explain that replay or synthetic audio may still achieve a high similarity score.")

    with panel_tab:
        st.subheader("Panel runbook")
        st.markdown(
            """
            1. Obtain consent and record three enrolment samples.
            2. Record a fresh sample from the same speaker and show the similarity score.
            3. Record or load a sample from a second consenting speaker and discuss the score.
            4. Play a previous recording through a speaker, capture it again, and select **Replay attempt**. Explain why speaker similarity alone is insufficient.
            5. Change the threshold and explain the FAR/FRR trade-off.
            6. State that a real deployment would add PAD, rate limits, secure template protection, audit logging, and FIDO2/passkey possession proof.
            """
        )
        st.subheader("Session audit trail")
        for event in st.session_state.event_log:
            st.code(event, language="text")
        st.subheader("Suggested viva wording")
        st.info("This prototype demonstrates a real-microphone voice signal and the authentication workflow. It does not claim biometric accuracy because the MFCC baseline has not been evaluated on a validated genuine/impostor dataset and does not contain a trained presentation-attack detector.")
        st.subheader("Data handling")
        st.write("The browser sends the captured file to the running Streamlit process so it can be analysed. This app does not write audio to a database or cloud storage, but the hosting environment may retain temporary request/session data according to its configuration. For a public panel URL, use only consenting voices and avoid sensitive recordings.")

    st.divider()
    st.caption("VocalKey Streamlit · no production authentication · no fabricated accuracy · MFCC baseline only")


if __name__ == "__main__":
    main()

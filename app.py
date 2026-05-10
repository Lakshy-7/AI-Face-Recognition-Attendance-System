"""
╔══════════════════════════════════════════════════════════════╗
║   AI Face Recognition Attendance System — Gradio App         ║
║   Deployment: Hugging Face Spaces                            ║
║   Stack: Gradio · face_recognition · OpenCV · Pandas         ║
╚══════════════════════════════════════════════════════════════╝

FLOW:
  Tab 1 → Register a person (upload photo + enter name)
  Tab 2 → Take attendance  (upload photo → detect → recognise → log)
  Tab 3 → View & download  attendance records
"""

import gradio as gr
import face_recognition
import cv2
import numpy as np
import pandas as pd
import os
import pickle
from datetime import datetime
from PIL import Image

# ─────────────────────────────────────────────────────────────
#  PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────
KNOWN_DIR      = "known_faces"       # registered face photos
ENCODINGS_FILE = "encodings.pkl"     # cached 128-d face fingerprints
ATTENDANCE_CSV = "attendance.csv"    # output log
TOLERANCE      = 0.50                # match threshold (lower = stricter)

os.makedirs(KNOWN_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
#  PERSISTENCE HELPERS
# ─────────────────────────────────────────────────────────────

def load_encodings():
    """Load saved encodings + names from disk. Returns ([], []) if none."""
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        return data["encodings"], data["names"]
    return [], []


def save_encodings(encodings, names):
    """Persist encodings + names to disk."""
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)


def load_attendance():
    """Load attendance CSV or return empty DataFrame."""
    if os.path.exists(ATTENDANCE_CSV):
        return pd.read_csv(ATTENDANCE_CSV)
    return pd.DataFrame(columns=["Name", "Date", "Time", "Status"])


def save_attendance(df):
    df.to_csv(ATTENDANCE_CSV, index=False)


# ─────────────────────────────────────────────────────────────
#  CORE LOGIC
# ─────────────────────────────────────────────────────────────

def pil_to_rgb(pil_img):
    """PIL Image → RGB numpy array (required by face_recognition)."""
    return np.array(pil_img.convert("RGB"))


def encode_image(image_rgb):
    """
    Find the first face in image_rgb and return its 128-d encoding.
    Returns None if no face is found.
    """
    locs = face_recognition.face_locations(image_rgb)
    if not locs:
        return None
    encs = face_recognition.face_encodings(image_rgb, locs)
    return encs[0] if encs else None


def draw_boxes(image_rgb, face_locations, face_names):
    """
    Draw labelled bounding boxes on a copy of the image.
    Green = recognised, Red = unknown.
    Returns annotated RGB numpy array.
    """
    out = image_rgb.copy()
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        color  = (34, 197, 94) if name != "Unknown" else (220, 60, 60)
        # Box
        cv2.rectangle(out, (left, top), (right, bottom), color, 3)
        # Label background
        cv2.rectangle(out, (left, bottom), (right, bottom + 32), color, cv2.FILLED)
        # Name text
        cv2.putText(out, name, (left + 6, bottom + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return out


def mark_attendance(name, df, marked_set):
    """
    Append a record for `name` if not already marked this session.
    Returns (updated_df, updated_marked_set).
    """
    if name == "Unknown" or name in marked_set:
        return df, marked_set

    now = datetime.now()
    row = pd.DataFrame([{
        "Name"  : name,
        "Date"  : now.strftime("%Y-%m-%d"),
        "Time"  : now.strftime("%H:%M:%S"),
        "Status": "Present",
    }])
    df = pd.concat([df, row], ignore_index=True)
    marked_set.add(name)
    return df, marked_set


# ─────────────────────────────────────────────────────────────
#  SESSION STATE  (mutable dict shared across callbacks)
# ─────────────────────────────────────────────────────────────
# Gradio doesn't have built-in session state like Streamlit,
# so we use a simple module-level dict. For multi-user
# deployments on HF Spaces, use gr.State() components instead.

_session = {
    "marked_today" : set(),
    "attendance_df": load_attendance(),
}


# ─────────────────────────────────────────────────────────────
#  TAB 1 — REGISTER
# ─────────────────────────────────────────────────────────────

def register_face(name, pil_img):
    """
    Encode the uploaded photo and save the person's face fingerprint.
    Called when the user clicks 'Register'.
    Returns a status message string.
    """
    # ── Validation ────────────────────────────────────────────
    if not name or not name.strip():
        return "⚠️ Please enter the person's name."
    if pil_img is None:
        return "⚠️ Please upload a photo."

    name = name.strip()

    # ── Encode face ───────────────────────────────────────────
    image_rgb = pil_to_rgb(pil_img)
    encoding  = encode_image(image_rgb)

    if encoding is None:
        return (
            "❌ No face detected in this photo.\n"
            "Please use a clear, front-facing, well-lit image."
        )

    # ── Check for duplicate ───────────────────────────────────
    enc_list, name_list = load_encodings()
    if name in name_list:
        return f"⚠️ '{name}' is already registered. Use a different name or remove them first."

    # ── Save encoding + photo ─────────────────────────────────
    enc_list.append(encoding)
    name_list.append(name)
    save_encodings(enc_list, name_list)

    photo_path = os.path.join(KNOWN_DIR, f"{name}.jpg")
    pil_img.save(photo_path)

    total = len(name_list)
    return (
        f"✅ '{name}' registered successfully!\n"
        f"📊 Total registered people: {total}\n"
        f"👥 All registered: {', '.join(name_list)}"
    )


def get_registered_list():
    """Return a formatted string of all registered people."""
    _, names = load_encodings()
    if not names:
        return "No people registered yet."
    lines = [f"  {i+1}. {n}" for i, n in enumerate(names)]
    return f"Registered ({len(names)} people):\n" + "\n".join(lines)


# ─────────────────────────────────────────────────────────────
#  TAB 2 — TAKE ATTENDANCE
# ─────────────────────────────────────────────────────────────

def take_attendance(pil_img):
    """
    Main attendance pipeline:
      1. Detect all faces in the uploaded image
      2. Encode each detected face
      3. Compare against known encodings
      4. Draw labelled boxes
      5. Mark attendance for recognised faces
      6. Return annotated image + summary text + updated table
    """
    if pil_img is None:
        return None, "⚠️ Please upload an image.", _session["attendance_df"]

    # ── Load known encodings ──────────────────────────────────
    known_enc, known_names = load_encodings()
    if not known_names:
        return (
            None,
            "⚠️ No registered faces found.\nPlease register people in Tab 1 first.",
            _session["attendance_df"],
        )

    # ── Detect faces ──────────────────────────────────────────
    image_rgb      = pil_to_rgb(pil_img)
    face_locations = face_recognition.face_locations(image_rgb)
    face_encodings = face_recognition.face_encodings(image_rgb, face_locations)

    if not face_locations:
        return pil_img, "❌ No faces detected in this image.", _session["attendance_df"]

    # ── Identify each face ────────────────────────────────────
    face_names    = []
    summary_lines = [f"🔍 {len(face_locations)} face(s) detected:\n"]

    for face_enc in face_encodings:
        name       = "Unknown"
        confidence = 0.0

        if known_enc:
            distances = face_recognition.face_distance(known_enc, face_enc)
            best_idx  = int(np.argmin(distances))
            best_dist = float(distances[best_idx])

            if best_dist <= TOLERANCE:
                name       = known_names[best_idx]
                confidence = round((1 - best_dist) * 100, 1)

        face_names.append(name)

        if name != "Unknown":
            summary_lines.append(f"  ✅ {name} — {confidence}% confidence")
        else:
            summary_lines.append("  ❓ Unknown face")

    # ── Draw boxes on image ───────────────────────────────────
    annotated = draw_boxes(image_rgb, face_locations, face_names)
    annotated_pil = Image.fromarray(annotated)

    # ── Mark attendance ───────────────────────────────────────
    df      = _session["attendance_df"]
    marked  = _session["marked_today"]
    newly   = []

    for name in face_names:
        if name != "Unknown" and name not in marked:
            df, marked = mark_attendance(name, df, marked)
            newly.append(name)

    _session["attendance_df"] = df
    _session["marked_today"]  = marked
    save_attendance(df)

    # ── Build summary text ────────────────────────────────────
    summary_lines.append("")
    if newly:
        summary_lines.append(f"📋 Attendance marked for: {', '.join(newly)}")
    else:
        recognised = [n for n in face_names if n != "Unknown"]
        if recognised:
            summary_lines.append("ℹ️ All recognised people already marked this session.")
        else:
            summary_lines.append("⚠️ No registered faces were recognised.")

    summary_lines.append(f"\n📊 Total records today: {len(df)}")

    return annotated_pil, "\n".join(summary_lines), df


# ─────────────────────────────────────────────────────────────
#  TAB 3 — ATTENDANCE LOG
# ─────────────────────────────────────────────────────────────

def refresh_log():
    """Reload attendance from disk and return the DataFrame."""
    df = load_attendance()
    _session["attendance_df"] = df
    return df


def download_csv():
    """
    Save current attendance to CSV and return the file path
    so Gradio can serve it as a downloadable file.
    """
    df = load_attendance()
    if df.empty:
        # Return an empty CSV so the download still works
        path = "attendance_empty.csv"
        pd.DataFrame(columns=["Name", "Date", "Time", "Status"]).to_csv(path, index=False)
        return path
    save_attendance(df)
    return ATTENDANCE_CSV


def clear_attendance():
    """Wipe all attendance records."""
    if os.path.exists(ATTENDANCE_CSV):
        os.remove(ATTENDANCE_CSV)
    _session["attendance_df"] = pd.DataFrame(columns=["Name", "Date", "Time", "Status"])
    _session["marked_today"]  = set()
    return _session["attendance_df"], "✅ All attendance records cleared."


def clear_registrations():
    """Remove all registered face encodings and photos."""
    import shutil
    if os.path.exists(ENCODINGS_FILE):
        os.remove(ENCODINGS_FILE)
    if os.path.exists(KNOWN_DIR):
        shutil.rmtree(KNOWN_DIR)
    os.makedirs(KNOWN_DIR, exist_ok=True)
    return "✅ All registrations cleared."


# ─────────────────────────────────────────────────────────────
#  GRADIO UI
# ─────────────────────────────────────────────────────────────

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
)

CSS = """
/* ── Global ── */
body, .gradio-container { background: #0f1117 !important; }

/* ── Header banner ── */
#header-banner {
    background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 50%, #2563eb 100%);
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 8px;
    border: 1px solid #3b82f6;
}
#header-banner h1 {
    color: #ffffff !important;
    font-size: 1.9rem !important;
    font-weight: 800 !important;
    margin: 0 0 6px 0 !important;
    letter-spacing: -0.02em;
}
#header-banner p {
    color: #bfdbfe !important;
    font-size: 0.95rem !important;
    margin: 0 !important;
}

/* ── Cards ── */
.card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 18px 20px;
    margin: 6px 0;
}

/* ── Status box ── */
#status-box textarea {
    background: #0f172a !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
}

/* ── Image outputs ── */
.image-container { border-radius: 12px !important; overflow: hidden; }

/* ── Buttons ── */
button.primary { border-radius: 8px !important; font-weight: 700 !important; }

/* ── Tab styling ── */
.tab-nav button {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 10px 22px !important;
}
"""

with gr.Blocks(theme=THEME, css=CSS, title="Face Attendance System") as demo:

    # ── HEADER ────────────────────────────────────────────────
    gr.HTML("""
    <div id="header-banner">
        <h1>🎓 AI Face Recognition Attendance System</h1>
        <p>Register faces · Recognise people · Mark attendance automatically · Export CSV</p>
    </div>
    """)

    # ════════════════════════════════════════════════════════
    #  TAB 1 — REGISTER FACES
    # ════════════════════════════════════════════════════════
    with gr.Tab("👤  Register Faces"):

        gr.Markdown("""
        ### Step 1 — Register People
        Upload a **clear, front-facing photo** of each person and type their name.
        The system encodes their face as a unique 128-number fingerprint.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                reg_name  = gr.Textbox(
                    label="Full Name",
                    placeholder="e.g. Lakshy Choudhary",
                    info="This exact name will appear in the attendance log."
                )
                reg_image = gr.Image(
                    label="Upload Photo",
                    type="pil",
                    image_mode="RGB",
                    sources=["upload"],
                )
                reg_btn   = gr.Button("✅  Register Person", variant="primary")

            with gr.Column(scale=1):
                reg_status = gr.Textbox(
                    label="Registration Status",
                    lines=6,
                    interactive=False,
                    elem_id="status-box",
                )
                reg_list_btn = gr.Button("🔄  Refresh Registered List", variant="secondary")
                reg_list     = gr.Textbox(
                    label="Currently Registered",
                    lines=8,
                    interactive=False,
                    value=get_registered_list,   # auto-load on startup
                    elem_id="status-box",
                )

        gr.Markdown("---")

        with gr.Accordion("⚠️ Danger Zone — Clear All Registrations", open=False):
            gr.Markdown("This will permanently delete all registered face encodings and photos.")
            clear_reg_btn    = gr.Button("🗑️  Delete All Registrations", variant="stop")
            clear_reg_status = gr.Textbox(label="Status", interactive=False, lines=1)

        # ── Wiring ────────────────────────────────────────────
        reg_btn.click(
            fn=register_face,
            inputs=[reg_name, reg_image],
            outputs=reg_status,
        )
        reg_list_btn.click(
            fn=get_registered_list,
            inputs=[],
            outputs=reg_list,
        )
        # Also refresh the list automatically after registration
        reg_btn.click(
            fn=get_registered_list,
            inputs=[],
            outputs=reg_list,
        )
        clear_reg_btn.click(
            fn=clear_registrations,
            inputs=[],
            outputs=clear_reg_status,
        )

    # ════════════════════════════════════════════════════════
    #  TAB 2 — TAKE ATTENDANCE
    # ════════════════════════════════════════════════════════
    with gr.Tab("📷  Take Attendance"):

        gr.Markdown("""
        ### Step 2 — Take Attendance
        Upload a photo (individual or group). The system will:
        detect every face → match against registered people → mark attendance.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                att_image = gr.Image(
                    label="Upload Photo",
                    type="pil",
                    image_mode="RGB",
                    sources=["upload"],
                )
                att_btn = gr.Button("🔍  Detect & Mark Attendance", variant="primary")

                gr.Markdown("""
                **Tips for best results:**
                - Use a well-lit, clear photo
                - Faces should be front-facing
                - Works with group photos too
                - Each person is marked only once per session
                """)

            with gr.Column(scale=1):
                att_result_img = gr.Image(
                    label="Recognition Result",
                    type="pil",
                    interactive=False,
                )
                att_status = gr.Textbox(
                    label="Attendance Status",
                    lines=8,
                    interactive=False,
                    elem_id="status-box",
                )

        gr.Markdown("### 📋 Live Attendance Table")
        att_table = gr.Dataframe(
            value=load_attendance,     # auto-load on startup
            label="Attendance Records",
            headers=["Name", "Date", "Time", "Status"],
            interactive=False,
            wrap=True,
        )

        # ── Wiring ────────────────────────────────────────────
        att_btn.click(
            fn=take_attendance,
            inputs=[att_image],
            outputs=[att_result_img, att_status, att_table],
        )

    # ════════════════════════════════════════════════════════
    #  TAB 3 — ATTENDANCE LOG & EXPORT
    # ════════════════════════════════════════════════════════
    with gr.Tab("📋  Attendance Log"):

        gr.Markdown("""
        ### Step 3 — View & Export Attendance
        Browse all records and download the complete CSV file.
        """)

        with gr.Row():
            refresh_btn  = gr.Button("🔄  Refresh Log",          variant="secondary")
            download_btn = gr.Button("⬇️  Prepare CSV Download", variant="primary")

        log_table = gr.Dataframe(
            value=load_attendance,
            label="Full Attendance Log",
            headers=["Name", "Date", "Time", "Status"],
            interactive=False,
            wrap=True,
        )

        csv_file = gr.File(
            label="Download Attendance CSV",
            interactive=False,
            visible=True,
        )

        gr.Markdown("---")
        with gr.Accordion("⚠️ Danger Zone — Clear Attendance Records", open=False):
            gr.Markdown("This will permanently delete all attendance records.")
            clear_att_btn    = gr.Button("🗑️  Clear All Records", variant="stop")
            clear_att_status = gr.Textbox(label="Status", interactive=False, lines=1)

        # ── Wiring ────────────────────────────────────────────
        refresh_btn.click(
            fn=refresh_log,
            inputs=[],
            outputs=log_table,
        )
        download_btn.click(
            fn=download_csv,
            inputs=[],
            outputs=csv_file,
        )
        clear_att_btn.click(
            fn=clear_attendance,
            inputs=[],
            outputs=[log_table, clear_att_status],
        )

    # ── FOOTER ────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center; margin-top:20px; padding:14px;
                color:#64748b; font-size:0.82rem; border-top:1px solid #1e293b;">
        Built with &nbsp;
        <a href="https://gradio.app" target="_blank" style="color:#3b82f6">Gradio</a> ·
        <a href="https://github.com/ageitgey/face_recognition" target="_blank" style="color:#3b82f6">face_recognition</a> ·
        OpenCV · Pandas
    </div>
    """)


# ─────────────────────────────────────────────────────────────
#  LAUNCH
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   # required for Hugging Face Spaces
        server_port=7860,         # default HF Spaces port
        share=False,              # set True for a temporary public link locally
    )

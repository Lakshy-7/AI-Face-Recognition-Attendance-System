"""
╔══════════════════════════════════════════════════════════════╗
║   AI Face Recognition Attendance System — Gradio App         ║
║   Deployment: Hugging Face Spaces                            ║
║   Stack: Gradio · face_recognition · OpenCV · Pandas · SQLite ║
╚══════════════════════════════════════════════════════════════╝

Professional attendance system with student registration, face recognition,
SQLite persistence, analytics, and multi-format export.

DEBUG MODE: All operations log to terminal for troubleshooting.
"""

import gradio as gr
import face_recognition
import cv2
import numpy as np
import pandas as pd
import sqlite3
import os
import pickle
import socket
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from PIL import Image

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly not installed. Analytics charts will be disabled. Run: pip install plotly")

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False
    print("⚠️ fpdf2 not installed. PDF export will be disabled. Run: pip install fpdf2")

BASE_DIR = Path(__file__).resolve().parent
KNOWN_DIR = BASE_DIR / "known_faces"
DB_PATH = BASE_DIR / "attendance.db"
CSV_PATH = BASE_DIR / "attendance.csv"
PDF_PATH = BASE_DIR / "attendance_report.pdf"

TOLERANCE = 0.50
MAX_PROCESS_WIDTH = 480
ENCODING_CACHE_INTERVAL = 5
DATE_FORMAT = "%Y-%m-%d"
TIME_FORMAT = "%H:%M:%S"

os.makedirs(KNOWN_DIR, exist_ok=True)

_encoding_cache = {
    "encodings": None,
    "student_ids": None,
    "names": None,
    "timestamp": 0,
}

# ──────────────────────────────────────────────────────────────
#  LOGGING & DEBUG HELPERS
# ──────────────────────────────────────────────────────────────

def log_debug(msg, level="INFO"):
    """Print debug messages with timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    if level == "ERROR":
        print(f"[{ts}] ❌ ERROR: {msg}")
    elif level == "SUCCESS":
        print(f"[{ts}] ✅ {msg}")
    elif level == "WARNING":
        print(f"[{ts}] ⚠️  {msg}")
    else:
        print(f"[{ts}] ℹ️  {msg}")


def log_exception(exc, context=""):
    """Log exception with full traceback."""
    print(f"\n{'='*60}")
    print(f"❌ EXCEPTION in {context}")
    print(f"{'='*60}")
    traceback.print_exc()
    print(f"{'='*60}\n")


def find_available_port(start_port=7865, max_port=7899):
    """Find the first available port in a range."""
    for port in range(start_port, max_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise OSError(f"No available ports found between {start_port} and {max_port}.")


# ──────────────────────────────────────────────────────────────
#  DATABASE OPERATIONS
# ──────────────────────────────────────────────────────────────

def get_db_connection():
    """Get SQLite connection with row factory."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    """Create database tables if they don't exist."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                photo_path TEXT NOT NULL,
                encoding BLOB NOT NULL,
                registered_at TEXT NOT NULL
            )
            """
        )
        
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY,
                student_id INTEGER,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(student_id) REFERENCES students(id)
            )
            """
        )
        
        cursor.execute("CREATE INDEX IF NOT EXISTS ix_attendance_date ON attendance(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS ix_attendance_student ON attendance(student_id)")
        conn.commit()
        conn.close()
        log_debug("Database initialized successfully.", "SUCCESS")
    except Exception as e:
        log_exception(e, "initialize_database")
        raise


def cache_student_encodings(force_reload=False):
    """Load and cache student face encodings from database."""
    try:
        current_time = time.time()
        
        if not force_reload and _encoding_cache["encodings"] is not None:
            if current_time - _encoding_cache["timestamp"] < ENCODING_CACHE_INTERVAL:
                log_debug(f"Using cached encodings ({len(_encoding_cache['encodings'])} students)")
                return _encoding_cache["encodings"], _encoding_cache["student_ids"], _encoding_cache["names"]

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, encoding FROM students ORDER BY name")
        rows = cursor.fetchall()
        conn.close()

        encodings = []
        student_ids = []
        names = []
        for row in rows:
            try:
                enc = pickle.loads(row["encoding"])
                encodings.append(enc)
                student_ids.append(row["id"])
                names.append(row["name"])
            except Exception as e:
                log_debug(f"Failed to load encoding for student {row['name']}: {str(e)}", "WARNING")
                continue

        _encoding_cache.update({
            "encodings": encodings,
            "student_ids": student_ids,
            "names": names,
            "timestamp": current_time,
        })
        log_debug(f"Reloaded {len(encodings)} student encodings from database.", "SUCCESS")
        return encodings, student_ids, names
    except Exception as e:
        log_exception(e, "cache_student_encodings")
        return [], [], []


# ──────────────────────────────────────────────────────────────
#  IMAGE PROCESSING
# ──────────────────────────────────────────────────────────────

def resize_image(pil_img, max_width=MAX_PROCESS_WIDTH):
    """Resize PIL image while maintaining aspect ratio."""
    try:
        if pil_img.width <= max_width:
            return pil_img, 1.0
        scale = max_width / pil_img.width
        new_height = int(pil_img.height * scale)
        resized = pil_img.resize((max_width, new_height), Image.Resampling.LANCZOS)
        return resized, scale
    except Exception as e:
        log_exception(e, "resize_image")
        return pil_img, 1.0


def pil_to_rgb(pil_img):
    """Convert PIL image to RGB numpy array."""
    try:
        return np.array(pil_img.convert("RGB"))
    except Exception as e:
        log_exception(e, "pil_to_rgb")
        return None


def encode_face(image_rgb):
    """Extract and encode first face from RGB image."""
    try:
        face_locations = face_recognition.face_locations(image_rgb, model="hog")
        if not face_locations:
            log_debug("No faces detected in image.", "WARNING")
            return None
        face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
        if face_encodings:
            log_debug(f"Successfully encoded face.", "SUCCESS")
            return face_encodings[0]
        return None
    except Exception as e:
        log_exception(e, "encode_face")
        return None


def draw_faces(image_rgb, face_locations, face_names, face_confidence=None):
    """Draw bounding boxes and labels on detected faces."""
    try:
        image = image_rgb.copy()
        for idx, ((top, right, bottom, left), name) in enumerate(zip(face_locations, face_names)):
            color = (34, 197, 94) if name != "Unknown" else (235, 73, 86)
            cv2.rectangle(image, (left, top), (right, bottom), color, 2)
            label = name
            if face_confidence is not None and idx < len(face_confidence) and name != "Unknown":
                label = f"{name} — {face_confidence[idx]:.1f}%"
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (left, bottom - 22), (left + text_size[0] + 10, bottom), color, cv2.FILLED)
            cv2.putText(image, label, (left + 5, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return image
    except Exception as e:
        log_exception(e, "draw_faces")
        return image_rgb


# ──────────────────────────────────────────────────────────────
#  REGISTRATION
# ──────────────────────────────────────────────────────────────

def register_student(name, pil_img):
    """Register a new student with face encoding."""
    try:
        name = name.strip()
        if not name:
            msg = "⚠️ Please enter a valid name."
            log_debug(msg, "WARNING")
            return msg
        
        if pil_img is None:
            msg = "⚠️ Please upload a clear registration photo."
            log_debug(msg, "WARNING")
            return msg

        log_debug(f"Registering student: {name}")
        
        resized, _ = resize_image(pil_img)
        image_rgb = pil_to_rgb(resized)
        if image_rgb is None:
            msg = "❌ Failed to process image."
            log_debug(msg, "ERROR")
            return msg
            
        encoding = encode_face(image_rgb)
        if encoding is None:
            msg = "❌ No face detected. Please use a clear front-facing photo."
            log_debug(msg, "ERROR")
            return msg

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM students WHERE LOWER(name)=LOWER(?)", (name,))
        if cursor.fetchone():
            conn.close()
            msg = f"⚠️ '{name}' is already registered. Use a different name."
            log_debug(msg, "WARNING")
            return msg

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{name.replace(' ', '_')}_{timestamp}.jpg"
        photo_path = KNOWN_DIR / filename
        pil_img.save(photo_path)
        log_debug(f"Saved photo to {photo_path}")

        cursor.execute(
            "INSERT INTO students (name, photo_path, encoding, registered_at) VALUES (?, ?, ?, ?)",
            (name, str(photo_path), pickle.dumps(encoding), datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()
        
        cache_student_encodings(force_reload=True)
        msg = f"✅ {name} registered successfully!"
        log_debug(msg, "SUCCESS")
        return msg
    except Exception as e:
        log_exception(e, "register_student")
        return f"❌ Registration failed: {str(e)}"


def get_registered_students():
    """Get list of all registered students."""
    try:
        conn = get_db_connection()
        df = pd.read_sql_query("SELECT name FROM students ORDER BY name", conn)
        conn.close()
        
        if df.empty:
            msg = "No students registered yet."
            log_debug(msg, "WARNING")
            return msg
        
        result = "\n".join([f"{idx+1}. {row['name']}" for idx, row in df.iterrows()])
        log_debug(f"Retrieved {len(df)} registered students.")
        return result
    except Exception as e:
        log_exception(e, "get_registered_students")
        return "❌ Failed to retrieve students."


def load_student_options():
    """Get list of student names for dropdown."""
    try:
        conn = get_db_connection()
        df = pd.read_sql_query("SELECT name FROM students ORDER BY name", conn)
        conn.close()
        students = [row['name'] for _, row in df.iterrows()]
        log_debug(f"Loaded {len(students)} students for dropdown.")
        return students
    except Exception as e:
        log_exception(e, "load_student_options")
        return []


# ──────────────────────────────────────────────────────────────
#  ATTENDANCE OPERATIONS
# ──────────────────────────────────────────────────────────────

def attendance_exists(student_id, date_str):
    """Check if student already marked present on a given date."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM attendance WHERE student_id=? AND date=? LIMIT 1", (student_id, date_str))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists
    except Exception as e:
        log_exception(e, "attendance_exists")
        return False


def save_attendance_record(student_id, name, confidence, status="Present"):
    """Record attendance for a student."""
    try:
        now = datetime.now()
        date_str = now.strftime(DATE_FORMAT)
        time_str = now.strftime(TIME_FORMAT)
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO attendance (student_id, name, status, confidence, date, time, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (student_id, name, status, float(confidence), date_str, time_str, now.isoformat()),
        )
        conn.commit()
        conn.close()
        log_debug(f"✅ Attendance recorded: {name} ({confidence:.1f}% match) on {date_str} at {time_str}")
    except Exception as e:
        log_exception(e, "save_attendance_record")


def load_attendance_frame():
    """Load all attendance records as DataFrame."""
    try:
        conn = get_db_connection()
        df = pd.read_sql_query(
            "SELECT name AS Name, date AS Date, time AS Time, status AS Status, confidence AS Confidence FROM attendance ORDER BY date DESC, time DESC",
            conn,
        )
        conn.close()
        log_debug(f"Loaded {len(df)} attendance records.")
        return df
    except Exception as e:
        log_exception(e, "load_attendance_frame")
        return pd.DataFrame(columns=["Name", "Date", "Time", "Status", "Confidence"])


def filter_attendance(name_filter, date_filter):
    """Filter attendance records by name and/or date."""
    try:
        query = "SELECT name AS Name, date AS Date, time AS Time, status AS Status, confidence AS Confidence FROM attendance"
        conditions = []
        values = []
        
        if name_filter:
            conditions.append("LOWER(name) LIKE ?")
            values.append(f"%{name_filter.lower()}%")
        if date_filter:
            conditions.append("date = ?")
            values.append(date_filter)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY date DESC, time DESC"
        
        conn = get_db_connection()
        df = pd.read_sql_query(query, conn, params=values)
        conn.close()
        
        log_debug(f"Filtered attendance: {len(df)} records (name='{name_filter}', date='{date_filter}')")
        return df
    except Exception as e:
        log_exception(e, "filter_attendance")
        return pd.DataFrame(columns=["Name", "Date", "Time", "Status", "Confidence"])


def reset_filters():
    """Reset attendance filters."""
    try:
        log_debug("Resetting attendance filters.")
        return load_attendance_frame(), "", ""
    except Exception as e:
        log_exception(e, "reset_filters")
        return pd.DataFrame(columns=["Name", "Date", "Time", "Status", "Confidence"]), "", ""


# ──────────────────────────────────────────────────────────────
#  FACE RECOGNITION & ATTENDANCE
# ──────────────────────────────────────────────────────────────

def recognize_attendance(pil_img):
    """Recognize faces in image and mark attendance."""
    try:
        if pil_img is None:
            msg = "⚠️ Upload an attendance image first."
            log_debug(msg, "WARNING")
            return None, msg, load_attendance_frame()

        log_debug("Starting face recognition...")
        known_encodings, student_ids, names = cache_student_encodings()
        
        if not known_encodings:
            msg = "⚠️ No registered students. Register first."
            log_debug(msg, "WARNING")
            return None, msg, load_attendance_frame()

        resized_pil, scale = resize_image(pil_img)
        image_rgb = pil_to_rgb(resized_pil)
        if image_rgb is None:
            msg = "❌ Failed to process image."
            log_debug(msg, "ERROR")
            return pil_img, msg, load_attendance_frame()
        
        face_locations = face_recognition.face_locations(image_rgb, model="hog")
        face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
        
        if not face_locations:
            msg = "❌ No faces detected."
            log_debug(msg, "WARNING")
            return pil_img, msg, load_attendance_frame()

        log_debug(f"Detected {len(face_locations)} face(s).")
        
        original_image = pil_to_rgb(pil_img)
        original_locations = [
            (
                int(top / scale),
                int(right / scale),
                int(bottom / scale),
                int(left / scale),
            ) for (top, right, bottom, left) in face_locations
        ]

        face_names = []
        face_confidences = []
        summary_lines = []
        unknown_count = 0
        date_str = datetime.now().strftime(DATE_FORMAT)

        for face_enc in face_encodings:
            distances = face_recognition.face_distance(known_encodings, face_enc)
            best_idx = int(np.argmin(distances))
            best_dist = float(distances[best_idx])
            match_name = "Unknown"
            confidence = 0.0
            student_id = None

            if best_dist <= TOLERANCE:
                match_name = names[best_idx]
                student_id = student_ids[best_idx]
                confidence = round((1 - best_dist) * 100, 1)

            face_names.append(match_name)
            face_confidences.append(confidence)

            if match_name == "Unknown":
                unknown_count += 1
                summary_lines.append("❓ Unknown face detected")
                log_debug(f"Unknown face (distance: {best_dist:.3f})")
            else:
                if attendance_exists(student_id, date_str):
                    summary_lines.append(f"✅ {match_name} already marked today ({confidence}% match)")
                    log_debug(f"{match_name} already marked today.")
                else:
                    save_attendance_record(student_id, match_name, confidence)
                    summary_lines.append(f"✅ {match_name} marked present ({confidence}% match)")

        annotated = draw_faces(original_image, original_locations, face_names, face_confidence=face_confidences)
        annotated_pil = Image.fromarray(annotated)

        if unknown_count:
            summary_lines.append(f"⚠️ {unknown_count} unknown face(s) were not recognised.")

        result = "\n".join(summary_lines)
        log_debug(result, "SUCCESS")
        return annotated_pil, result, load_attendance_frame()
    except Exception as e:
        log_exception(e, "recognize_attendance")
        return pil_img, f"❌ Recognition failed: {str(e)}", load_attendance_frame()


# ──────────────────────────────────────────────────────────────
#  STUDENT PROFILES
# ──────────────────────────────────────────────────────────────

def get_student_profile(name):
    """Get student profile with attendance history."""
    try:
        if not name:
            return None, "Select a student to view details.", pd.DataFrame(columns=["Date", "Time", "Status", "Confidence"]), "0%"
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, photo_path, registered_at FROM students WHERE name=?", (name,))
        student = cursor.fetchone()
        
        if not student:
            conn.close()
            log_debug(f"Student '{name}' not found.", "WARNING")
            return None, "Student not found.", pd.DataFrame(columns=["Date", "Time", "Status", "Confidence"]), "0%"

        df = pd.read_sql_query(
            "SELECT date AS Date, time AS Time, status AS Status, confidence AS Confidence FROM attendance WHERE name=? ORDER BY date DESC, time DESC",
            conn,
            params=(name,),
        )
        conn.close()
        
        total_days = df['Date'].nunique() if not df.empty else 0
        attended_days = df[df['Status'] == 'Present']['Date'].nunique() if not df.empty else 0
        attendance_pct = (attended_days / total_days * 100) if total_days else 0.0

        profile_text = (
            f"### {student['name']}\n"
            f"**Registered on:** {student['registered_at'][:10]}\n\n"
            f"**Attendance records:** {len(df)}\n"
            f"**Days attended:** {attended_days}\n"
            f"**Attendance percentage:** {attendance_pct:.1f}%"
        )
        
        photo_path = student['photo_path'] if os.path.exists(student['photo_path']) else None
        log_debug(f"Loaded profile for {name}: {attendance_pct:.1f}% attendance")
        return photo_path, profile_text, df, f"{attendance_pct:.1f}%"
    except Exception as e:
        log_exception(e, "get_student_profile")
        return None, f"❌ Failed to load profile: {str(e)}", pd.DataFrame(columns=["Date", "Time", "Status", "Confidence"]), "0%"


def update_student_profile(name):
    """Update student profile display."""
    try:
        return get_student_profile(name)
    except Exception as e:
        log_exception(e, "update_student_profile")
        return None, f"❌ Error: {str(e)}", pd.DataFrame(columns=["Date", "Time", "Status", "Confidence"]), "0%"


# ──────────────────────────────────────────────────────────────
#  ANALYTICS
# ──────────────────────────────────────────────────────────────

def make_stats():
    """Calculate basic statistics."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM students")
        total_students = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM attendance")
        total_records = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT date) FROM attendance")
        total_days = cursor.fetchone()[0]
        
        conn.close()
        
        attendance_pct = (total_records / (total_students * total_days) * 100) if total_students and total_days else 0.0
        attendance_pct = min(attendance_pct, 100.0)
        
        log_debug(f"Stats: {total_students} students, {total_records} records, {total_days} days, {attendance_pct:.1f}% average")
        return total_students, total_records, attendance_pct
    except Exception as e:
        log_exception(e, "make_stats")
        return 0, 0, 0.0


def make_dashboard_figures():
    """Generate Plotly charts for analytics."""
    try:
        if not PLOTLY_AVAILABLE:
            log_debug("Plotly not available. Skipping charts.", "WARNING")
            return None, None
        
        conn = get_db_connection()
        
        daily_df = pd.read_sql_query(
            "SELECT date AS Date, COUNT(*) AS Count FROM attendance GROUP BY date ORDER BY date",
            conn,
        )
        
        monthly_df = pd.read_sql_query(
            "SELECT SUBSTR(date, 1, 7) AS Month, COUNT(*) AS Count FROM attendance GROUP BY Month ORDER BY Month",
            conn,
        )
        
        conn.close()
        
        if daily_df.empty and monthly_df.empty:
            log_debug("No attendance data for charts.", "WARNING")
            return None, None
        
        daily_fig = None
        monthly_fig = None
        
        if not daily_df.empty:
            daily_fig = px.bar(daily_df, x="Date", y="Count", title="Daily Attendance", labels={"Count": "Records"}, template="plotly_dark")
            log_debug(f"Generated daily chart with {len(daily_df)} days.")
        
        if not monthly_df.empty:
            monthly_fig = px.line(monthly_df, x="Month", y="Count", title="Monthly Attendance", markers=True, template="plotly_dark")
            log_debug(f"Generated monthly chart with {len(monthly_df)} months.")
        
        return daily_fig, monthly_fig
    except Exception as e:
        log_exception(e, "make_dashboard_figures")
        return None, None


def refresh_dashboard():
    """Refresh analytics dashboard."""
    try:
        registered, records, attendance_pct = make_stats()
        daily_fig, monthly_fig = make_dashboard_figures()
        
        return (
            f"**Registered Students:** {registered}",
            f"**Attendance Records:** {records}",
            f"**Attendance Rate:** {attendance_pct:.1f}%",
            daily_fig,
            monthly_fig,
        )
    except Exception as e:
        log_exception(e, "refresh_dashboard")
        return "❌ Error loading stats", "❌ Error loading stats", "0%", None, None


# ──────────────────────────────────────────────────────────────
#  EXPORT & DOWNLOAD
# ──────────────────────────────────────────────────────────────

def export_csv():
    """Export attendance records to CSV."""
    try:
        df = load_attendance_frame()
        if df.empty:
            log_debug("No attendance records to export.", "WARNING")
            return None

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        csv_path = Path(tempfile.gettempdir()) / f"attendance_export_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        log_debug(f"CSV exported to {csv_path}")
        return str(csv_path)
    except Exception as e:
        log_exception(e, "export_csv")
        return None


def export_pdf_report():
    """Export attendance records to PDF."""
    try:
        if not FPDF_AVAILABLE:
            log_debug("fpdf2 not available. PDF export disabled.", "WARNING")
            return None

        df = load_attendance_frame()
        if df.empty:
            log_debug("No attendance records to export.", "WARNING")
            return None

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        pdf_path = Path(tempfile.gettempdir()) / f"attendance_report_{timestamp}.pdf"

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(255, 255, 255)
        pdf.set_fill_color(15, 23, 42)
        pdf.rect(0, 0, 210, 297, style="F")
        pdf.set_xy(10, 10)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 10, "AI Face Recognition Attendance Report", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(5)

        if df.empty:
            pdf.cell(0, 8, "No attendance records available.")
        else:
            pdf.set_font("Helvetica", "B", 10)
            header = ["Name", "Date", "Time", "Status", "Confidence"]
            widths = [50, 30, 25, 30, 30]

            for i, item in enumerate(header):
                pdf.cell(widths[i], 8, item, border=1, align="C")
            pdf.ln()

            pdf.set_font("Helvetica", "", 10)
            for _, row in df.iterrows():
                pdf.cell(widths[0], 8, str(row['Name'])[:22], border=1)
                pdf.cell(widths[1], 8, str(row['Date']), border=1)
                pdf.cell(widths[2], 8, str(row['Time']), border=1)
                pdf.cell(widths[3], 8, str(row['Status']), border=1)
                confidence_text = f"{row['Confidence']:.1f}%" if pd.notna(row['Confidence']) else "N/A"
                pdf.cell(widths[4], 8, confidence_text, border=1)
                pdf.ln()

        pdf.output(str(pdf_path))
        log_debug(f"PDF exported to {pdf_path}")
        return str(pdf_path)
    except Exception as e:
        log_exception(e, "export_pdf_report")
        return None


def export_pdf_or_csv():
    """Export as PDF or fallback to CSV."""
    try:
        pdf_path = export_pdf_report()
        if pdf_path:
            return pdf_path
        return export_csv()
    except Exception as e:
        log_exception(e, "export_pdf_or_csv")
        return None


def clear_database():
    """Clear all data from database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM attendance")
        cursor.execute("DELETE FROM students")
        conn.commit()
        conn.close()
        
        cache_student_encodings(force_reload=True)
        
        for file in KNOWN_DIR.glob("*.jpg"):
            try:
                file.unlink()
            except Exception:
                pass
        
        msg = "✅ Database cleared. All registrations and records removed."
        log_debug(msg, "SUCCESS")
        return msg
    except Exception as e:
        log_exception(e, "clear_database")
        return f"❌ Failed to clear database: {str(e)}"


# ──────────────────────────────────────────────────────────────
#  GRADIO UI
# ──────────────────────────────────────────────────────────────

def build_app():
    """Build Gradio UI with all tabs and callbacks."""
    try:
        log_debug("Building Gradio UI...")
        initialize_database()
        
        theme = gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
        )
        
        css = """
        body, .gradio-container { background: #0b1220 !important; color: #e2e8f0 !important; }
        #header-banner { background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%); border-radius: 18px; padding: 28px 32px; margin-bottom: 16px; box-shadow: 0 12px 30px rgba(0,0,0,0.35); }
        #header-banner h1 { color: #ffffff !important; font-size: 2rem !important; }
        #header-banner p { color: #dbeafe !important; }
        .card { background: #111827 !important; border: 1px solid #2e3a59 !important; border-radius: 16px !important; padding: 20px !important; margin-bottom: 16px !important; }
        .gr-button.primary { background: #2563eb !important; color: white !important; }
        .gr-button.secondary { background: #334155 !important; color: white !important; }
        .gr-button.stop { background: #dc2626 !important; color: white !important; }
        .tab-nav button { border-radius: 12px !important; }
        textarea, .gr-textbox textarea { background: #0f172a !important; color: #e2e8f0 !important; }
        .gr-dataframe { background: #0f172a !important; color: #e2e8f0 !important; }
        .gr-file { background: #0f172a !important; }
        """

        with gr.Blocks() as demo:
            gr.HTML(
                """
                <div id='header-banner'>
                    <h1>🎓 AI Face Recognition Attendance System</h1>
                    <p>Professional attendance tracking with registration, image-based recognition, analytics, and export.</p>
                </div>
                """
            )

            # ════════════════════════════════════════════════════════════
            # TAB 1: REGISTER STUDENTS
            # ════════════════════════════════════════════════════════════
            with gr.Tab("👤 Register Students"):
                with gr.Row():
                    with gr.Column(scale=1):
                        reg_name = gr.Textbox(label="Student Name", placeholder="e.g. Lakshy Choudhary")
                        reg_image = gr.Image(label="Upload Student Photo", type="pil", image_mode="RGB", sources=["upload"])
                        reg_btn = gr.Button("Register Student", variant="primary")
                    with gr.Column(scale=1):
                        reg_status = gr.Textbox(label="Registration Status", lines=6, interactive=False)
                        reg_list = gr.Textbox(label="Registered Students", lines=10, interactive=False)
                
                with gr.Accordion("Danger Zone — Delete All Registrations and Records", open=False):
                    gr.Markdown("This will clear all students and attendance history from the database.")
                    clear_all_btn = gr.Button("Clear Database", variant="stop")
                    clear_all_status = gr.Textbox(label="Status", interactive=False, lines=1)

                # CALLBACKS for Registration Tab
                def on_register_click(name, img):
                    log_debug("on_register_click triggered")
                    result = register_student(name, img)
                    students_list = get_registered_students()
                    return result, students_list
                
                def on_clear_database():
                    log_debug("on_clear_database triggered")
                    return clear_database()

                reg_btn.click(
                    fn=on_register_click,
                    inputs=[reg_name, reg_image],
                    outputs=[reg_status, reg_list]
                )
                
                clear_all_btn.click(
                    fn=on_clear_database,
                    inputs=[],
                    outputs=[clear_all_status]
                )

            # ════════════════════════════════════════════════════════════
            # TAB 2: TAKE ATTENDANCE
            # ════════════════════════════════════════════════════════════
            with gr.Tab("📷 Take Attendance"):
                with gr.Row():
                    with gr.Column(scale=1):
                        att_image = gr.Image(
                            label="Upload or Capture Attendance Photo",
                            type="pil",
                            image_mode="RGB",
                            sources=["upload", "webcam"],
                        )
                        att_btn = gr.Button("Detect & Mark Attendance", variant="primary")
                    with gr.Column(scale=1):
                        att_result = gr.Image(label="Recognition Result", type="pil", interactive=False)
                        att_status = gr.Textbox(label="Attendance Status", lines=8, interactive=False)
                
                att_table = gr.Dataframe(
                    value=load_attendance_frame(),
                    label="Attendance Records",
                    headers=["Name", "Date", "Time", "Status", "Confidence"],
                    interactive=False,
                    wrap=True
                )
                
                # CALLBACK for Attendance Recognition
                def on_attendance_click(img):
                    log_debug("on_attendance_click triggered")
                    return recognize_attendance(img)

                att_btn.click(
                    fn=on_attendance_click,
                    inputs=[att_image],
                    outputs=[att_result, att_status, att_table]
                )

            # ════════════════════════════════════════════════════════════
            # TAB 3: ATTENDANCE LOG
            # ════════════════════════════════════════════════════════════
            with gr.Tab("📋 Attendance Log"):
                with gr.Row():
                    name_filter = gr.Textbox(label="Search by Name", placeholder="Search student name...")
                    date_filter = gr.Textbox(label="Filter by Date (YYYY-MM-DD)", placeholder="2026-05-10")
                    filter_btn = gr.Button("Search / Filter", variant="primary")
                    reset_btn = gr.Button("Reset", variant="secondary")
                
                log_table = gr.Dataframe(
                    value=load_attendance_frame(),
                    label="Attendance Records",
                    headers=["Name", "Date", "Time", "Status", "Confidence"],
                    interactive=False,
                    wrap=True
                )
                
                with gr.Row():
                    csv_btn = gr.Button("Export CSV", variant="primary")
                    pdf_btn = gr.Button("Export PDF", variant="secondary")
                
                csv_file = gr.File(label="Download CSV", interactive=False)
                pdf_file = gr.File(label="Download PDF", interactive=False)
                
                # CALLBACKS for Attendance Log
                def on_filter_click(name, date):
                    log_debug("on_filter_click triggered")
                    return filter_attendance(name, date)

                def on_reset_click():
                    log_debug("on_reset_click triggered")
                    return reset_filters()

                def on_export_csv_click():
                    log_debug("on_export_csv_click triggered")
                    return export_csv()

                def on_export_pdf_click():
                    log_debug("on_export_pdf_click triggered")
                    return export_pdf_or_csv()

                filter_btn.click(
                    fn=on_filter_click,
                    inputs=[name_filter, date_filter],
                    outputs=[log_table]
                )
                
                reset_btn.click(
                    fn=on_reset_click,
                    inputs=[],
                    outputs=[log_table, name_filter, date_filter]
                )
                
                csv_btn.click(
                    fn=on_export_csv_click,
                    inputs=[],
                    outputs=[csv_file]
                )
                
                pdf_btn.click(
                    fn=on_export_pdf_click,
                    inputs=[],
                    outputs=[pdf_file]
                )

            # ════════════════════════════════════════════════════════════
            # TAB 4: STUDENT PROFILES
            # ════════════════════════════════════════════════════════════
            with gr.Tab("👥 Student Profiles"):
                student_dropdown = gr.Dropdown(
                    label="Select Student",
                    choices=load_student_options(),
                    interactive=True,
                    value=None
                )
                profile_image = gr.Image(label="Profile Photo", interactive=False, value=None)
                profile_summary = gr.Markdown("Select a student to view details.")
                profile_history = gr.Dataframe(
                    value=pd.DataFrame(columns=["Date", "Time", "Status", "Confidence"]),
                    label="Attendance History",
                    interactive=False
                )
                attendance_pct = gr.Textbox(label="Attendance Percentage", interactive=False, value="0%")
                
                # CALLBACK for Profile Selection
                def on_student_change(name):
                    log_debug("on_student_change triggered")
                    return update_student_profile(name)

                student_dropdown.change(
                    fn=on_student_change,
                    inputs=[student_dropdown],
                    outputs=[profile_image, profile_summary, profile_history, attendance_pct]
                )

            # ════════════════════════════════════════════════════════════
            # TAB 5: ANALYTICS
            # ════════════════════════════════════════════════════════════
            with gr.Tab("📊 Analytics"):
                dashboard_total_text, dashboard_records_text, dashboard_percent_text, dashboard_daily_fig, dashboard_monthly_fig = refresh_dashboard()
                dashboard_total = gr.Markdown(dashboard_total_text)
                dashboard_records = gr.Markdown(dashboard_records_text)
                dashboard_percent = gr.Markdown(dashboard_percent_text)
                daily_plot = gr.Plot(label="Daily Attendance")
                monthly_plot = gr.Plot(label="Monthly Attendance")
                refresh_btn = gr.Button("Refresh Dashboard", variant="primary")
                
                # CALLBACK for Analytics Refresh
                def on_refresh_click():
                    log_debug("on_refresh_click triggered")
                    return refresh_dashboard()

                refresh_btn.click(
                    fn=on_refresh_click,
                    inputs=[],
                    outputs=[dashboard_total, dashboard_records, dashboard_percent, daily_plot, monthly_plot]
                )

            gr.HTML(
                """
                <div style='text-align:center; margin-top:20px; color:#94a3b8; font-size:0.82rem;'>
                    Built with Gradio · face_recognition · OpenCV · SQLite · Plotly · Pandas
                </div>
                """
            )

        demo.queue()
        log_debug("Gradio UI built successfully.", "SUCCESS")
        return demo, theme, css
    except Exception as e:
        log_exception(e, "build_app")
        raise


# ══════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════

log_debug("=" * 60)
log_debug("AI FACE RECOGNITION ATTENDANCE SYSTEM", "INFO")
log_debug("=" * 60)

demo, theme, css = build_app()

if __name__ == "__main__":
    # Hugging Face Spaces exposes the port via PORT; fall back to GRADIO_SERVER_PORT or 7865 for local runs.
    port_env = os.environ.get("PORT") or os.environ.get("GRADIO_SERVER_PORT")
    base_port = int(port_env) if port_env and port_env.isdigit() else 7865
    port = base_port

    if "PORT" not in os.environ:
        try:
            port = find_available_port(base_port, base_port + 20)
        except Exception as e:
            log_exception(e, "find_available_port")
            port = base_port

    log_debug(f"Launching Gradio app on port {port}...")
    if port != base_port:
        log_debug(f"Port {base_port} was busy. Using port {port} instead.", "WARNING")

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        theme=theme,
        css=css,
    )

from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
import hashlib
import re
import sqlite3

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

try:
    import av
    from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
except ImportError:
    av = None
    VideoProcessorBase = object
    webrtc_streamer = None

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "Q1" / "runs" / "detect" / "vest_helmet_final" / "weights" / "best.pt"
PERSON_MODEL_PATH = PROJECT_DIR / "yolov8n.pt"
DASHBOARD_DIR = PROJECT_DIR / "dashboard"
EVENTS_CSV = DASHBOARD_DIR / "events.csv"
DB_PATH = DASHBOARD_DIR / "guardian360.db"
SCREENSHOT_DIR = DASHBOARD_DIR / "screenshots"
PHOTO_DIR = DASHBOARD_DIR / "employee_photos"
BACKUP_DIR = DASHBOARD_DIR / "backups"
REMOTE_CAMERA_URL = "http://10.130.108.143/live"
REMOTE_CAMERA_CANDIDATES = [
    "http://10.130.108.143/live",
    "http://10.130.108.143/live.mjpg",
    "http://10.130.108.143/stream",
    "http://10.130.108.143/stream.mjpg",
    "http://10.130.108.143/video",
    "http://10.130.108.143/video.mjpg",
    "http://10.130.108.143/",
    "http://10.130.108.143:81/live",
    "http://10.130.108.143:81/stream",
    "http://10.130.108.143:8080/live",
    "http://10.130.108.143:8080/stream",
]
CLASS_NAMES = {0: "No vest", 1: "Helmet", 2: "Vest"}
CLASS_COLORS = {0: (40, 40, 220), 1: (220, 190, 30), 2: (30, 190, 70)}
EVENT_COLUMNS = ["timestamp", "event_type", "confidence", "worker_id", "ppe_status", "source_camera", "screenshot_path", "details"]
LOGINS = {
    "admin": (hashlib.sha256(b"admin123").hexdigest(), "admin"),
    "safety": (hashlib.sha256(b"safety123").hexdigest(), "safety"),
}

st.set_page_config(page_title="Guardian 360", page_icon=":material/health_and_safety:", layout="wide")
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #f4f7f9 0%, #e7eef2 58%, #fff7e8 100%); }
[data-testid="stSidebar"] { background: #172b3a; }
[data-testid="stSidebar"] * { color: #eef5f7; }
[data-testid="stSidebar"] input { color: #172b3a; }
.brand { color:#102332; font-size:2rem; font-weight:800; letter-spacing:-.03em; }
.eyebrow { color:#d9822b; font-size:.75rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }
.status { border-left:5px solid var(--status); background:rgba(255,255,255,.82); padding:.8rem 1rem; border-radius:4px; box-shadow:0 5px 18px rgba(23,43,58,.08); }
.status strong { color:var(--status); font-size:1.35rem; }
</style>
""", unsafe_allow_html=True)


def safe_filename(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._") or "file"


def init_database():
    for path in (DASHBOARD_DIR, PHOTO_DIR, SCREENSHOT_DIR, BACKUP_DIR):
        path.mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS workers (
            worker_id TEXT PRIMARY KEY, full_name TEXT NOT NULL, role TEXT DEFAULT '',
            department TEXT DEFAULT '', phone TEXT DEFAULT '', emergency_contact TEXT DEFAULT '',
            active INTEGER DEFAULT 1, created_at TEXT NOT NULL)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS worker_photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT, worker_id TEXT NOT NULL, view_label TEXT,
            photo_path TEXT NOT NULL, uploaded_at TEXT NOT NULL,
            FOREIGN KEY(worker_id) REFERENCES workers(worker_id) ON DELETE CASCADE)""")


@st.cache_resource(show_spinner=False)
def load_model(path):
    return YOLO(str(path))


@st.cache_data(ttl=5)
def load_workers():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM workers ORDER BY worker_id", conn)


@st.cache_data(ttl=5)
def load_photos(worker_id):
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM worker_photos WHERE worker_id = ? ORDER BY id", conn, params=(worker_id,))


def face_photo_signature(workers):
    signature = []
    for worker_id in workers["worker_id"].astype(str) if not workers.empty else []:
        for row in load_photos(worker_id).itertuples(index=False):
            path = Path(row.photo_path)
            if path.exists():
                signature.append((worker_id, str(path), path.stat().st_mtime_ns))
    return tuple(signature)


@st.cache_resource(show_spinner=False)
def build_face_recognizer(signature):
    if not signature or not hasattr(cv2, "face"):
        return None, {}, None
    cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    if cascade.empty():
        return None, {}, None
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels, worker_labels = [], [], {}
    for label, (worker_id, path, _) in enumerate(signature):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        found = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        for x, y, width, height in found:
            faces.append(image[y:y + height, x:x + width])
            labels.append(label)
            worker_labels[label] = worker_id
    if not faces:
        return None, {}, cascade
    recognizer.train(faces, np.array(labels, dtype=np.int32))
    return recognizer, worker_labels, cascade


def recognize_worker(frame, recognizer, worker_labels, cascade):
    if recognizer is None or cascade is None:
        return None, None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if not len(found):
        return None, None
    x, y, width, height = max(found, key=lambda item: item[2] * item[3])
    label, distance = recognizer.predict(gray[y:y + height, x:x + width])
    worker_id = worker_labels.get(label) if distance <= 75 else None
    return worker_id, float(distance)


def save_worker(worker):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""INSERT INTO workers
            (worker_id, full_name, role, department, phone, emergency_contact, active, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(worker_id) DO UPDATE SET full_name=excluded.full_name, role=excluded.role,
            department=excluded.department, phone=excluded.phone, emergency_contact=excluded.emergency_contact,
            active=excluded.active""", (*worker, datetime.now().isoformat(timespec="seconds")))
    load_workers.clear()


def save_worker_photos(worker_id, files, view_label):
    if not files:
        return 0
    worker_dir = PHOTO_DIR / safe_filename(worker_id)
    worker_dir.mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        for index, file in enumerate(files, 1):
            suffix = Path(file.name).suffix.lower() or ".jpg"
            path = worker_dir / f"{safe_filename(worker_id)}_{datetime.now():%Y%m%d_%H%M%S_%f}_{index}{suffix}"
            path.write_bytes(file.getvalue())
            conn.execute("INSERT INTO worker_photos (worker_id, view_label, photo_path, uploaded_at) VALUES (?, ?, ?, ?)",
                         (worker_id, view_label, str(path), datetime.now().isoformat(timespec="seconds")))
    load_photos.clear()
    return len(files)


def load_events():
    if not EVENTS_CSV.exists():
        return pd.DataFrame(columns=EVENT_COLUMNS)
    events = pd.read_csv(EVENTS_CSV)
    if "timestamp" in events:
        events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce")
    return events


def append_event(event):
    if EVENTS_CSV.exists():
        backup = BACKUP_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
        backup.mkdir(exist_ok=True)
        backup.joinpath("events.csv").write_bytes(EVENTS_CSV.read_bytes())
        old = pd.read_csv(EVENTS_CSV)
    else:
        old = pd.DataFrame(columns=EVENT_COLUMNS)
    pd.concat([old, pd.DataFrame([event])], ignore_index=True).to_csv(EVENTS_CSV, index=False)


def detect_frame(model, frame, confidence, person_model):
    result = model(frame, conf=confidence, verbose=False)[0]
    detections = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        if class_id in CLASS_NAMES:
            detections.append({"class_id": class_id, "label": CLASS_NAMES[class_id], "confidence": float(box.conf.item()), "box": tuple(int(v) for v in box.xyxy[0].tolist())})
    if not detections:
        result = person_model(frame, classes=[0], conf=max(.25, confidence * .6), verbose=False)[0]
        for box in result.boxes:
            detections.append({"class_id": 0, "label": "No vest", "confidence": float(box.conf.item()), "box": tuple(int(v) for v in box.xyxy[0].tolist())})
    return detections


def status_for(detections):
    found = {item["class_id"] for item in detections}
    if 1 in found and 2 in found:
        return "SAFE"
    if 1 in found or 2 in found:
        return "PARTIAL"
    if 0 in found:
        return "UNSAFE"
    return "NO DETECTIONS"


def annotate(frame, detections, status):
    output = frame.copy()
    for item in detections:
        x1, y1, x2, y2 = item["box"]
        color = CLASS_COLORS[item["class_id"]]
        label = f'{item["label"]} {item["confidence"]:.2f}'
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
        top = max(0, y1 - 32)
        width = max(190, cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .6, 2)[0][0] + 12)
        cv2.rectangle(output, (x1, top), (min(output.shape[1], x1 + width), y1), color, -1)
        cv2.putText(output, label, (x1 + 6, max(18, y1 - 9)), cv2.FONT_HERSHEY_SIMPLEX, .6, (15, 20, 20), 2)
    cv2.rectangle(output, (0, 0), (300, 48), (24, 31, 35), -1)
    cv2.putText(output, f"STATUS: {status}", (14, 32), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
    return output


def log_check(worker_id, worker_name, source, annotated_frame, detections, status):
    stamp = datetime.now()
    screenshot = SCREENSHOT_DIR / f"{safe_filename(source)}_{stamp:%Y%m%d_%H%M%S}.jpg"
    cv2.imwrite(str(screenshot), annotated_frame)
    counts = {label: sum(item["label"] == label for item in detections) for label in CLASS_NAMES.values()}
    missing = [label for label, count in (("helmet", counts["Helmet"]), ("vest", counts["Vest"])) if not count]
    event_type = "ppe_violation" if missing else "ppe_checked"
    detail = f"{worker_id} {worker_name} is not wearing {', '.join(missing)}." if missing else f"{worker_id} {worker_name} was checked. No explicit PPE violation detected."
    append_event({"timestamp": stamp.isoformat(timespec="seconds"), "event_type": event_type, "confidence": round(max((item["confidence"] for item in detections), default=0), 3), "worker_id": worker_id, "ppe_status": status, "source_camera": source, "screenshot_path": str(screenshot), "details": detail})
    return detail


def read_remote_frame(url):
    if not url:
        return None
    urls = []
    trimmed = url.strip()
    if trimmed:
        urls.append(trimmed)
    if trimmed and not trimmed.endswith("/"):
        urls.append(trimmed + "/")
    for candidate in REMOTE_CAMERA_CANDIDATES:
        if candidate not in urls:
            urls.append(candidate)
    for candidate in urls:
        try:
            capture = cv2.VideoCapture(candidate)
            if not capture.isOpened():
                continue
            ok, frame = capture.read()
            capture.release()
            if ok and frame is not None:
                return frame, candidate
        except Exception:
            continue
    return None, None


def render_login():
    st.title("Guardian 360")
    st.caption("Construction-site worker database and PPE monitoring.")
    admin_col, safety_col = st.columns(2)
    for column, role, title, button in [(admin_col, "admin", "Admin login", "Login as admin"), (safety_col, "safety", "Safety dashboard login", "Open dashboard")]:
        with column, st.container(border=True):
            st.subheader(title)
            with st.form(f"{role}_login"):
                username = st.text_input("Username", value=role)
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button(button, type="primary")
            if submitted:
                account = LOGINS.get(username.strip().lower())
                if account and account[1] == role and hashlib.sha256(password.encode()).hexdigest() == account[0]:
                    st.session_state.authenticated = True
                    st.session_state.role = role
                    st.session_state.username = username.strip()
                    st.rerun()
                st.error("Invalid login details.")
    st.info("Demo credentials: admin/admin123 and safety/safety123.")


def render_photos(worker_id):
    photos = load_photos(worker_id)
    if photos.empty:
        st.caption("No employee photos saved yet.")
        return
    for row in photos.itertuples(index=False):
        path = Path(row.photo_path)
        if path.exists():
            st.image(str(path), caption=row.view_label or "Employee view")


def render_events():
    events = load_events()
    if events.empty:
        st.info("No events logged yet.")
        return
    with st.sidebar:
        st.header("Event filters")
        days = st.date_input("Date range", (date.today() - timedelta(days=30), date.today()))
        types = sorted(events["event_type"].dropna().unique())
        selected_types = st.multiselect("Event type", types, default=types)
    start, end = days if isinstance(days, tuple) and len(days) == 2 else (days, days)
    filtered = events[events["timestamp"].dt.date.between(start, end) & events["event_type"].isin(selected_types)].sort_values("timestamp", ascending=False)
    with st.container(horizontal=True):
        st.metric("Total alerts", len(filtered), border=True)
        st.metric("PPE events", int(filtered["event_type"].str.contains("ppe", na=False).sum()), border=True)
        st.metric("Employees", filtered["worker_id"].nunique(), border=True)
        st.download_button("Download event report", filtered.to_csv(index=False), "guardian360_events.csv", "text/csv", icon=":material/download:")
    chart_col, image_col = st.columns(2)
    with chart_col, st.container(border=True):
        st.subheader("Alert mix")
        chart = filtered.groupby("event_type").size().rename("alerts") if not filtered.empty else pd.Series(dtype=int)
        if not chart.empty:
            st.bar_chart(chart)
    with image_col, st.container(border=True):
        st.subheader("Latest processed frame")
        if not filtered.empty and Path(filtered.iloc[0]["screenshot_path"]).exists():
            st.image(filtered.iloc[0]["screenshot_path"])
        else:
            st.caption("No screenshot found.")
    with st.container(border=True):
        st.subheader("Recent events")
        st.dataframe(filtered.head(200), hide_index=True, width="stretch")


def render_admin():
    st.title("Admin worker database")
    workers = load_workers()
    with st.container(horizontal=True):
        st.metric("Workers", len(workers), border=True)
        st.metric("Active workers", int(workers["active"].sum()) if not workers.empty else 0, border=True)
        st.metric("Saved events", len(load_events()), border=True)
    form_col, list_col = st.columns([.9, 1.1])
    with form_col, st.container(border=True):
        st.subheader("Add or update employee")
        with st.form("worker_form"):
            worker_id = st.text_input("Employee ID", placeholder="101")
            full_name = st.text_input("Full name")
            role = st.text_input("Work role")
            department = st.text_input("Department / site area")
            phone = st.text_input("Phone")
            emergency = st.text_input("Emergency contact")
            active = st.toggle("Active employee", True)
            label = st.text_input("Photo view label", placeholder="front view")
            photos = st.file_uploader("Employee photos", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
            submitted = st.form_submit_button("Save employee", type="primary", icon=":material/save:")
        if submitted:
            if not worker_id.strip() or not full_name.strip():
                st.error("Employee ID and full name are required.")
            else:
                save_worker((worker_id.strip(), full_name.strip(), role.strip(), department.strip(), phone.strip(), emergency.strip(), int(active)))
                count = save_worker_photos(worker_id.strip(), photos, label.strip())
                st.success(f"Saved {worker_id.strip()} and {count} photo(s).")
                st.rerun()
    with list_col, st.container(border=True):
        st.subheader("Employee records")
        st.dataframe(workers, hide_index=True, width="stretch")
    if not workers.empty:
        selected_id = st.selectbox("View employee profile", workers["worker_id"].tolist())
        selected = workers[workers["worker_id"] == selected_id].iloc[0]
        profile_col, photo_col = st.columns([.8, 1.2])
        with profile_col, st.container(border=True):
            st.subheader(f"{selected.worker_id} - {selected.full_name}")
            st.write(f"Role: {selected.role or 'Not set'}")
            st.write(f"Department: {selected.department or 'Not set'}")
            st.write(f"Phone: {selected.phone or 'Not set'}")
            st.write(f"Emergency contact: {selected.emergency_contact or 'Not set'}")
        with photo_col, st.container(border=True):
            st.subheader("Employee photos")
            render_photos(str(selected_id))
    st.subheader("Safety event dashboard")
    render_events()


def render_safety():
    st.title("Safety PPE dashboard")
    st.caption("The employee ID is recognized from the face photos saved in Admin.")
    workers = load_workers()
    recognizer, worker_labels, face_cascade = build_face_recognizer(face_photo_signature(workers))
    with st.sidebar:
        confidence = st.slider("Detection confidence", .05, .95, .25, .05)
        source = st.selectbox("Input source", ["Upload image", "Camera feed", "Live camera"])
        remote_url = st.text_input("Camera URL", value=REMOTE_CAMERA_URL) if source == "Camera feed" else ""
    if recognizer is None:
        st.warning("No usable employee face photos found. Add a clear front-facing photo for each employee in Admin.")
    if source == "Live camera":
        if webrtc_streamer is None:
            st.error("Install live camera support with: pip install streamlit-webrtc av")
        else:
            model = load_model(MODEL_PATH)
            person_model = load_model(PERSON_MODEL_PATH)
            st.info("Allow browser camera access to start live detection.")
            webrtc_streamer(key="guardian-live-camera", video_processor_factory=lambda: PPEProcessor(model, person_model, confidence, recognizer, worker_labels, face_cascade), media_stream_constraints={"video": True, "audio": False})
        return
    upload = st.file_uploader("Upload employee/site photo", type=["jpg", "jpeg", "png", "webp"]) if source == "Upload image" else None
    if source == "Camera feed":
        image, working_url = read_remote_frame(remote_url)
        if image is None:
            st.error(
                "This camera module is not exposing a direct MJPEG stream yet. "
                "Use the ESP32-CAM endpoint like http://IP_ADDRESS/live or http://IP_ADDRESS/live.mjpg."
            )
            st.code(f"Tried: {', '.join(REMOTE_CAMERA_CANDIDATES[:6])}")
            render_events()
            return
        selected_image = None
        frame = image
        source_label = working_url or remote_url
    else:
        selected_image = upload
        if selected_image is None:
            st.info("Upload an image to check PPE.")
            render_events()
            return
        image = np.array(Image.open(BytesIO(selected_image.getvalue())).convert("RGB"))
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        source_label = upload.name
    model = load_model(MODEL_PATH)
    person_model = load_model(PERSON_MODEL_PATH)
    with st.spinner("Running PPE detection..."):
        detections = detect_frame(model, frame, confidence, person_model)
        recognized_id, face_distance = recognize_worker(frame, recognizer, worker_labels, face_cascade)
    selected = workers[workers["worker_id"].astype(str) == recognized_id].iloc[0] if recognized_id and not workers.empty and any(workers["worker_id"].astype(str) == recognized_id) else None
    worker_id = str(selected["worker_id"]) if selected is not None else "UNKNOWN"
    worker_name = str(selected["full_name"]) if selected is not None else "Unregistered person"
    status = status_for(detections)
    result = annotate(frame, detections, status)
    detail = log_check(worker_id, worker_name, source_label, result, detections, status)
    st.image(result, channels="BGR", caption="YOLO detection result", width="stretch")
    if selected is not None:
        st.success(f"Recognized employee: {worker_id} - {worker_name} (face distance {face_distance:.1f})")
    else:
        st.error("Unregistered person: no saved employee face matched this image.")
    color = {"SAFE": "#2fbf71", "PARTIAL": "#e6a23c", "UNSAFE": "#e05252", "NO DETECTIONS": "#84909f"}[status]
    st.markdown(f'<div class="status" style="--status:{color}"><strong>{status}</strong><br>{detail}</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([{"Detection": d["label"], "Confidence": round(d["confidence"], 3)} for d in detections]), hide_index=True, width="stretch")
    st.subheader("Safety event dashboard")
    render_events()


class PPEProcessor(VideoProcessorBase):
    def __init__(self, model, person_model, confidence, recognizer, worker_labels, face_cascade):
        self.model, self.person_model, self.confidence = model, person_model, confidence
        self.recognizer, self.worker_labels, self.face_cascade = recognizer, worker_labels, face_cascade

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        detections = detect_frame(self.model, image, self.confidence, self.person_model)
        worker_id, _ = recognize_worker(image, self.recognizer, self.worker_labels, self.face_cascade)
        if worker_id:
            cv2.putText(image, f"EMPLOYEE: {worker_id}", (14, 78), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
        else:
            cv2.putText(image, "EMPLOYEE: UNKNOWN", (14, 78), cv2.FONT_HERSHEY_SIMPLEX, .7, (40, 40, 220), 2)
        return av.VideoFrame.from_ndarray(annotate(image, detections, status_for(detections)), format="bgr24")


init_database()
st.session_state.setdefault("authenticated", False)
st.session_state.setdefault("role", "")
st.markdown('<div class="eyebrow">Construction site intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="brand">Guardian 360</div>', unsafe_allow_html=True)
if not st.session_state.authenticated:
    render_login()
else:
    with st.sidebar:
        st.caption(f"Logged in as {st.session_state.get('username')} ({st.session_state.role})")
        if st.button("Logout", icon=":material/logout:"):
            st.session_state.authenticated = False
            st.session_state.role = ""
            st.rerun()
    if st.session_state.role == "admin":
        render_admin()
    else:
        render_safety()

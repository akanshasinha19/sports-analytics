# app.py

# === Imports ===
import streamlit as st
import tempfile
import os
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import sys
sys.path.append("/opt/anaconda3/lib/python3.12/site-packages")

# === App Config ===
st.set_page_config(page_title="Soccer Player Analyzer", layout="wide")
st.title("⚽ Upload a Soccer Video to Analyze Player Behavior")

from moviepy.editor import VideoFileClip

# === Sidebar Controls ===
with st.sidebar:
    video_file = st.file_uploader("Upload a soccer video", type=["mp4", "mov"])
    video_url = st.text_input("Or paste a video URL")
    model_choice = st.selectbox("Choose YOLOv8 Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0)
    frame_limit = st.slider("Number of Frames to Analyze", min_value=30, max_value=200, value=50, step=10)

# === Video Path Resolution ===
video_path = None
if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name
elif video_url:
    try:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        headers = {"User-Agent": "Mozilla/5.0"}
        import requests
        with requests.get(video_url, stream=True, headers=headers, timeout=15) as r:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    tfile.write(chunk)
        video_path = tfile.name
    except Exception as e:
        st.error(f"Failed to load video from URL: {e}")

# === Audio + Whisper Transcription ===
import whisper
import spacy
import re

left_team_goals = 0
right_team_goals = 0

if video_path:
    audio_path = video_path + ".mp3"
    try:
        video_clip = VideoFileClip(video_path)
        video_clip.audio.write_audiofile(audio_path)
        st.audio(audio_path)

        try:
            st.info("🗣️ Transcribing commentary using Whisper...")
            whisper_model = whisper.load_model("base")
            transcript = whisper_model.transcribe(audio_path)

            jersey_mentions = re.findall(r"number\s\d+", transcript["text"], re.IGNORECASE)
            jersey_mentions = list(set(jersey_mentions))

            nlp = spacy.load("en_core_web_sm")
            doc = nlp(transcript["text"])
            player_names = list(set([ent.text for ent in doc.ents if ent.label_ == "PERSON"]))

            all_mentions = sorted(set(jersey_mentions + player_names))

            if all_mentions:
                st.markdown("### 🧢 Mentioned Players from Commentary")
                cols = st.columns(3)
                for i, mention in enumerate(all_mentions):
                    cols[i % 3].markdown(
                        f"<span style='background:#f0f0f0; padding:6px 10px; border-radius:20px; display:inline-block'>{mention}</span>",
                        unsafe_allow_html=True)

            st.markdown(f"""
            <div style='background:#f9f9f9; padding:1.2rem; border-left:5px solid #4CAF50; border-radius:8px; margin-top:1rem;'>
                <h4 style='margin-top:0;'>🎯 Key Commentary Insights</h4>
                <ul style='margin-bottom:0;'>
                    <li><b>Standout Players:</b> {', '.join(all_mentions[:5])}</li>
                    <li><b>Mentions:</b> {', '.join(jersey_mentions[:3])}</li>
                    <li><b>Summary:</b> Commentary captured important moments like transitions, key player actions, and close shots</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.warning(f"Transcription failed: {e}")
    except Exception as e:
        st.warning(f"Audio extraction failed: {e}")

    st.video(video_path)

    # === Goal Detection Helpers ===
    def estimate_field_position(cx, frame_width):
        ratio = cx / frame_width
        if ratio < 0.33:
            return "Defender"
        elif ratio < 0.66:
            return "Midfielder"
        else:
            return "Forward"

    def check_goal(ball_x, ball_y, frame_width, frame_height):
        goal_margin = int(frame_width * 0.15)
        goal_top = int(frame_height * 0.25)
        goal_bottom = int(frame_height * 0.75)
        if ball_x < goal_margin and goal_top <= ball_y <= goal_bottom:
            return "Right Team Goal"
        elif ball_x > frame_width - goal_margin and goal_top <= ball_y <= goal_bottom:
            return "Left Team Goal"
        return None

    # === YOLOv8 + DeepSort Detection and Analysis ===
    model = YOLO(model_choice)
    tracker = DeepSort(max_age=10)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    player_stats = {}
    ball_positions = []
    player_centers = {}

    with st.spinner(f"Analyzing first {frame_limit} frames..."):
        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= frame_limit:
                break

            if frame_count % 2 != 0:
                frame_count += 1
                continue

            frame = cv2.resize(frame, (640, 360))
            frame_height, frame_width = frame.shape[:2]
            results = model.predict(frame, imgsz=416, conf=0.4, iou=0.5)[0]
            detections = []

            goal_margin = int(frame_width * 0.15)
            goal_top = int(frame_height * 0.25)
            goal_bottom = int(frame_height * 0.75)
            cv2.rectangle(frame, (0, goal_top), (goal_margin, goal_bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (frame_width - goal_margin, goal_top), (frame_width, goal_bottom), (255, 0, 0), 2)

            for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
                label = model.names[int(cls)]
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                height = y2 - y1
                if height < 50:
                    continue
                if label == "sports ball":
                    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                    goal_result = check_goal(cx, cy, frame_width, frame_height)
                    if goal_result == "Left Team Goal":
                        left_team_goals += 1
                    elif goal_result == "Right Team Goal":
                        right_team_goals += 1
                    ball_positions.append((frame_count, cx, cy))
                if label == "person":
                    detections.append(([x1, y1, x2, y2], 0.9, 'player'))

            tracks = tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                l, t, r, b = map(int, track.to_ltrb())
                cx, cy = (l + r) // 2, (t + b) // 2
                position = estimate_field_position(cx, frame_width)

                if track_id not in player_stats:
                    player_stats[track_id] = {
                        "Position": position,
                        "NumberOfTouches": 0,
                        "GameAwareness": 0,
                        "Teamwork": 0,
                        "Speed": 0.0,
                        "FrameCount": 0,
                        "Last": (cx, cy)
                    }
                    player_centers[track_id] = []

                last_cx, last_cy = player_stats[track_id]["Last"]
                dist = ((cx - last_cx)**2 + (cy - last_cy)**2)**0.5
                player_stats[track_id]["Speed"] += dist
                player_stats[track_id]["Last"] = (cx, cy)
                player_centers[track_id].append((frame_count, cx, cy))

                for _, bx, by in ball_positions[-5:]:
                    if abs(cx - bx) < 60 and abs(cy - by) < 60:
                        if "touched" not in player_stats[track_id] or player_stats[track_id]["touched"]:
                            player_stats[track_id]["NumberOfTouches"] += 1
                            player_stats[track_id]["touched"] = False
                        break
                else:
                    player_stats[track_id]["touched"] = True
                player_stats[track_id]["FrameCount"] += 1

            frame_count += 1

        cap.release()

    for pid, centers in player_centers.items():
        contacts = set()
        for f1, x1, y1 in centers:
            for qid, qcenters in player_centers.items():
                if pid == qid:
                    continue
                for f2, x2, y2 in qcenters:
                    if abs(f1 - f2) <= 2:
                        dist = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
                        if dist < 80:
                            contacts.add(qid)
                            break
        player_stats[pid]["Teamwork"] = len(contacts)

    valid_players = {pid: stats for pid, stats in player_stats.items() if stats["FrameCount"] >= 1}

    st.markdown(f"### 📊 Player Stats Summary ({len(valid_players)} players detected)")
    st.markdown(f"### 🥅 Goals Scored")
    st.markdown(f"- ⚽ Left Team Goals: **{left_team_goals}**")
    st.markdown(f"- ⚽ Right Team Goals: **{right_team_goals}**")

    grouped_players = {'Defender': [], 'Midfielder': [], 'Forward': []}
    for pid, stats in valid_players.items():
        grouped_players[stats['Position']].append((pid, stats))

    for role in ['Defender', 'Midfielder', 'Forward']:
        st.markdown(f"## 🛡️ {role}s")
        role_players = grouped_players[role]
        if not role_players:
            continue
        cols = st.columns(3)
        for i, (pid, stats) in enumerate(sorted(role_players, key=lambda x: x[1]['NumberOfTouches'], reverse=True)):
            avg_speed = stats["Speed"] / max(stats["FrameCount"], 1)
            awareness = round(min(1.0, stats["FrameCount"] / frame_count), 2)
            teamwork = stats["Teamwork"]
            fit_score = round(0.5 * avg_speed + 0.3 * teamwork + 0.2 * awareness, 2)
            score_color = '#4CAF50' if fit_score >= 0.6 else ('#FF9800' if fit_score >= 0.4 else '#F44336')
            with cols[i % 3]:
                st.markdown(f"""
                    <div style='border:1px solid #ddd; border-radius:10px; padding:15px; margin-bottom:10px'>
                        <h4 style='color:{score_color};'>Player {pid}</h4>
                        <p><b>Position:</b> {stats['Position']}</p>
                        <p><b>Speed:</b> {avg_speed:.2f} px/frame</p>
                        <p><b>Touches:</b> {stats['NumberOfTouches']}</p>
                        <p><b>Game Awareness:</b> {awareness}</p>
                        <p><b>Teamwork:</b> {teamwork}</p>
                        <p><b>Overall Fit Score:</b> <span style='color:{score_color}; font-weight:bold;'>{fit_score}</span></p>
                    </div>
                """, unsafe_allow_html=True)

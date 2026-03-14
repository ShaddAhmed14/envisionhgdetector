# ============================================================================
# STRUCTURED FEATURE EXTRACTION v3.2 (Data Creation Only)
# ============================================================================
# Creates THREE structured datasets:
# 1. Basic:    41 features (29 body + 6 visibility + 6 move-distinguishing)
# 2. Extended: 61 features (29 body + 20 hand + 6 visibility + 6 move-distinguishing)
# 3. World:    92 features (23 landmarks × 4: x, y, z, visibility)
#
# Four categories: NoGesture, Gesture, Move, Objman
# Full metadata for speaker-independent splits
# Timestamps preserved per video
#
# For plotting, use: plot_dataset_statistics.py

from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any
import os
import json
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm
from enum import auto, Enum
from multiprocessing import Pool, cpu_count
from collections import defaultdict

mp_holistic = mp.solutions.holistic


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    # Output paths
    output_dir: str = "../TrainingDataProcessed"
    
    # Dataset filenames
    npz_basic: str = "landmarks_basic_41_structured_v3.npz"
    npz_extended: str = "landmarks_extended_61_structured_v3.npz"
    npz_world: str = "landmarks_world_92_structured_v3.npz"
    metadata_filename: str = "dataset_metadata_v3.json"
    
    # Feature counts
    num_basic_features: int = 41     # 29 + 6 visibility + 6 move-distinguishing
    num_extended_features: int = 61  # 29 + 20 hand + 6 visibility + 6 move-distinguishing
    num_world_features: int = 92     # 23 × 4 (x, y, z, visibility)
    
    # Processing
    target_fps: int = 25
    max_frames_per_video: int = 800
    min_detection_percentage: float = 0.1
    
    # Categories (4 categories)
    category_labels: Tuple[str, ...] = ("NoGesture", "Gesture", "Move", "Objman")


CONFIG = Config()


# ============================================================================
# LANDMARK DEFINITIONS
# ============================================================================

BODY_LANDMARKS = [
    'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_OUTER', 
    'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 
    'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX',
    'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP',
    'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL',
    'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
]

UPPER_BODY_LANDMARKS = [
    'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 
    'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 
    'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX',
    'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB'
]
UPPER_BODY_INDICES = list(range(23))


# ============================================================================
# FEATURE ENUMS
# ============================================================================

class FeatureBasicV3(Enum):
    """Basic 41 features: 29 body + 6 visibility + 6 move-distinguishing"""
    rot_x = auto()
    rot_y = auto()
    rot_z = auto()
    nose_x = auto()
    nose_y = auto()
    nose_z = auto()
    norm_dist = auto()
    left_brow_left_eye_norm_dist = auto()
    right_brow_right_eye_norm_dist = auto()
    mouth_corners_norm_dist = auto()
    mouth_apperture_norm_dist = auto()
    left_right_wrist_norm_dist = auto()
    left_right_elbow_norm_dist = auto()
    left_elbow_midpoint_shoulder_norm_dist = auto()
    right_elbow_midpoint_shoulder_norm_dist = auto()
    left_wrist_midpoint_shoulder_norm_dist = auto()
    right_wrist_midpoint_shoulder_norm_dist = auto()
    left_shoulder_left_ear_norm_dist = auto()
    right_shoulder_right_ear_norm_dist = auto()
    left_thumb_left_index_norm_dist = auto()
    right_thumb_right_index_norm_dist = auto()
    left_thumb_left_pinky_norm_dist = auto()
    right_thumb_right_pinky_norm_dist = auto()
    x_left_wrist_x_left_elbow_norm_dist = auto()
    x_right_wrist_x_right_elbow_norm_dist = auto()
    y_left_wrist_y_left_elbow_norm_dist = auto()
    y_right_wrist_y_right_elbow_norm_dist = auto()
    left_index_finger_nose_norm_dist = auto()
    right_index_finger_nose_norm_dist = auto()
    # Visibility features (6)
    left_wrist_visibility = auto()
    right_wrist_visibility = auto()
    left_elbow_visibility = auto()
    right_elbow_visibility = auto()
    left_hand_detected = auto()
    right_hand_detected = auto()
    # Move-distinguishing features (6)
    min_hand_face_dist = auto()
    hand_face_contact = auto()
    min_hand_shoulder_y_diff = auto()
    left_wrist_above_shoulder = auto()
    right_wrist_above_shoulder = auto()
    hand_position_symmetry = auto()


class FeatureExtendedV3(Enum):
    """Extended 61 features: 29 body + 20 hand + 6 visibility + 6 move-distinguishing"""
    rot_x = auto()
    rot_y = auto()
    rot_z = auto()
    nose_x = auto()
    nose_y = auto()
    nose_z = auto()
    norm_dist = auto()
    left_brow_left_eye_norm_dist = auto()
    right_brow_right_eye_norm_dist = auto()
    mouth_corners_norm_dist = auto()
    mouth_apperture_norm_dist = auto()
    left_right_wrist_norm_dist = auto()
    left_right_elbow_norm_dist = auto()
    left_elbow_midpoint_shoulder_norm_dist = auto()
    right_elbow_midpoint_shoulder_norm_dist = auto()
    left_wrist_midpoint_shoulder_norm_dist = auto()
    right_wrist_midpoint_shoulder_norm_dist = auto()
    left_shoulder_left_ear_norm_dist = auto()
    right_shoulder_right_ear_norm_dist = auto()
    left_thumb_left_index_norm_dist = auto()
    right_thumb_right_index_norm_dist = auto()
    left_thumb_left_pinky_norm_dist = auto()
    right_thumb_right_pinky_norm_dist = auto()
    x_left_wrist_x_left_elbow_norm_dist = auto()
    x_right_wrist_x_right_elbow_norm_dist = auto()
    y_left_wrist_y_left_elbow_norm_dist = auto()
    y_right_wrist_y_right_elbow_norm_dist = auto()
    left_index_finger_nose_norm_dist = auto()
    right_index_finger_nose_norm_dist = auto()
    # Hand features (20)
    left_wrist_thumb_tip_norm_dist = auto()
    left_wrist_index_tip_norm_dist = auto()
    left_wrist_middle_tip_norm_dist = auto()
    left_wrist_ring_tip_norm_dist = auto()
    left_wrist_pinky_tip_norm_dist = auto()
    left_thumb_index_angle = auto()
    left_index_middle_angle = auto()
    left_middle_ring_angle = auto()
    left_ring_pinky_angle = auto()
    left_hand_openness = auto()
    right_wrist_thumb_tip_norm_dist = auto()
    right_wrist_index_tip_norm_dist = auto()
    right_wrist_middle_tip_norm_dist = auto()
    right_wrist_ring_tip_norm_dist = auto()
    right_wrist_pinky_tip_norm_dist = auto()
    right_thumb_index_angle = auto()
    right_index_middle_angle = auto()
    right_middle_ring_angle = auto()
    right_ring_pinky_angle = auto()
    right_hand_openness = auto()
    # Visibility features (6)
    left_wrist_visibility = auto()
    right_wrist_visibility = auto()
    left_elbow_visibility = auto()
    right_elbow_visibility = auto()
    left_hand_detected = auto()
    right_hand_detected = auto()
    # Move-distinguishing features (6)
    min_hand_face_dist = auto()
    hand_face_contact = auto()
    min_hand_shoulder_y_diff = auto()
    left_wrist_above_shoulder = auto()
    right_wrist_above_shoulder = auto()
    hand_position_symmetry = auto()


# ============================================================================
# METADATA PARSER
# ============================================================================

@dataclass
class VideoMetadata:
    """Metadata extracted from video filename"""
    filepath: str
    filename: str
    corpus: str
    speaker: str
    clip_id: str
    category: str
    subtype: str
    is_mirror: bool
    training_label: str = ""
    speaker_id: str = ""
    
    def __post_init__(self):
        self.speaker_id = f"{self.corpus}_{self.speaker}"
        self.training_label = self._get_training_label()
    
    def _get_training_label(self) -> str:
        """Determine training label based on category and subtype"""
        if self.category == 'nogesture':
            return 'NoGesture'
        elif self.subtype.lower() in ['move', 'adaptor']:
            return 'Move'
        elif self.subtype.lower() == 'objman':
            return 'Objman'
        else:
            return 'Gesture'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'filepath': self.filepath,
            'filename': self.filename,
            'corpus': self.corpus,
            'speaker': self.speaker,
            'clip_id': self.clip_id,
            'category': self.category,
            'subtype': self.subtype,
            'is_mirror': self.is_mirror,
            'training_label': self.training_label,
            'speaker_id': self.speaker_id
        }


def parse_filename(filepath: str) -> Optional[VideoMetadata]:
    """Parse video filename to extract metadata."""
    filename = os.path.basename(filepath)
    
    is_mirror = "_mirror" in filename
    clean_name = filename.replace("_mirror", "").replace(".mp4", "")
    
    parts = clean_name.split("_")
    
    if len(parts) < 4:
        print(f"Warning: Cannot parse filename (too few parts): {filename}")
        return None
    
    corpus = parts[0]
    
    # Find category index
    category_idx = None
    for i, p in enumerate(parts):
        if p.lower() in ['gesture', 'nogesture']:
            category_idx = i
            break
    
    if category_idx is None or category_idx < 2:
        print(f"Warning: Cannot find category in: {filename}")
        return None
    
    clip_id = parts[category_idx - 1]
    
    if category_idx > 2:
        speaker = "_".join(parts[1:category_idx-1])
    else:
        speaker = parts[1]
    
    category = parts[category_idx].lower()
    subtype = parts[category_idx + 1] if category_idx + 1 < len(parts) else "NA"
    
    return VideoMetadata(
        filepath=filepath,
        filename=filename,
        corpus=corpus,
        speaker=speaker,
        clip_id=clip_id,
        category=category,
        subtype=subtype,
        is_mirror=is_mirror
    )


# ============================================================================
# FEATURE EXTRACTION HELPERS
# ============================================================================

def calculate_angle(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    """Calculate angle between three points in degrees."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def extract_hand_features(hand_landmarks, wrist_pos: Tuple[float, float], 
                         norm_dist: float, w: int, h: int) -> List[float]:
    """Extract 10 hand features from MediaPipe hand landmarks."""
    if hand_landmarks is None or norm_dist == 0:
        return [0.0] * 10
    
    lm = hand_landmarks.landmark
    
    wrist = (lm[0].x * w, lm[0].y * h)
    thumb_tip = (lm[4].x * w, lm[4].y * h)
    index_tip = (lm[8].x * w, lm[8].y * h)
    middle_tip = (lm[12].x * w, lm[12].y * h)
    ring_tip = (lm[16].x * w, lm[16].y * h)
    pinky_tip = (lm[20].x * w, lm[20].y * h)
    
    index_mcp = (lm[5].x * w, lm[5].y * h)
    middle_mcp = (lm[9].x * w, lm[9].y * h)
    ring_mcp = (lm[13].x * w, lm[13].y * h)
    pinky_mcp = (lm[17].x * w, lm[17].y * h)
    
    dists = [
        np.linalg.norm(np.array(wrist) - np.array(tip)) / norm_dist
        for tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
    ]
    
    angles = [
        calculate_angle(thumb_tip, wrist, index_tip) / 180.0,
        calculate_angle(index_tip, index_mcp, middle_tip) / 180.0,
        calculate_angle(middle_tip, middle_mcp, ring_tip) / 180.0,
        calculate_angle(ring_tip, ring_mcp, pinky_tip) / 180.0,
    ]
    
    extensions = []
    for tip, mcp in [(index_tip, index_mcp), (middle_tip, middle_mcp), 
                     (ring_tip, ring_mcp), (pinky_tip, pinky_mcp)]:
        tip_dist = np.linalg.norm(np.array(wrist) - np.array(tip))
        mcp_dist = np.linalg.norm(np.array(wrist) - np.array(mcp))
        extensions.append(tip_dist / (mcp_dist + 1e-6))
    openness = np.mean(extensions)
    
    return dists + angles + [openness]


def extract_visibility_features(results) -> List[float]:
    """Extract 6 visibility/confidence features."""
    visibility = [0.0] * 6
    
    if results.pose_landmarks:
        pose_lm = results.pose_landmarks.landmark
        visibility[0] = pose_lm[15].visibility if len(pose_lm) > 15 else 0.0
        visibility[1] = pose_lm[16].visibility if len(pose_lm) > 16 else 0.0
        visibility[2] = pose_lm[13].visibility if len(pose_lm) > 13 else 0.0
        visibility[3] = pose_lm[14].visibility if len(pose_lm) > 14 else 0.0
    
    visibility[4] = 1.0 if results.left_hand_landmarks else 0.0
    visibility[5] = 1.0 if results.right_hand_landmarks else 0.0
    
    return visibility


def extract_move_distinguishing_features(body: Dict, nose_2d: Tuple[float, float], 
                                         norm_dist: float) -> List[float]:
    """Extract 6 move-distinguishing features."""
    features = [0.0] * 6
    
    lw, rw = body.get('LEFT_WRIST'), body.get('RIGHT_WRIST')
    ls, rs = body.get('LEFT_SHOULDER'), body.get('RIGHT_SHOULDER')
    li, ri = body.get('LEFT_INDEX'), body.get('RIGHT_INDEX')

    # 1. Hand-to-face distance
    l_dist = np.linalg.norm(np.array(li) - np.array(nose_2d)) / norm_dist if li else 99
    r_dist = np.linalg.norm(np.array(ri) - np.array(nose_2d)) / norm_dist if ri else 99
    min_dist = min(l_dist, r_dist)
    features[0] = min_dist if min_dist < 99 else 0.0

    # 2. Contact threshold
    features[1] = 1.0 if min_dist < 0.35 else 0.0 

    # 3. Min height relative to shoulders
    if lw and ls and rw and rs:
        features[2] = min((lw[1]-ls[1])/norm_dist, (rw[1]-rs[1])/norm_dist)

    # 4 & 5. Wrists above shoulders
    features[3] = 1.0 if (lw and ls and lw[1] < ls[1]) else 0.0
    features[4] = 1.0 if (rw and rs and rw[1] < rs[1]) else 0.0

    # 6. Symmetry
    if lw and rw and ls and rs:
        center_x = (ls[0] + rs[0]) / 2
        symm = abs(abs(lw[0]-center_x) - abs(rw[0]-center_x)) / norm_dist
        features[5] = max(0, 1.0 - symm)
    
    return features


def extract_world_landmarks_with_visibility(results) -> Optional[List[float]]:
    """Extract upper body world landmarks with visibility (23 × 4 = 92 features)."""
    if not results.pose_world_landmarks:
        return None
    
    features = []
    for idx in UPPER_BODY_INDICES:
        if idx < len(results.pose_world_landmarks.landmark):
            lm = results.pose_world_landmarks.landmark[idx]
            features.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
    
    return features


def extract_basic_features(results, w: int, h: int) -> Tuple[List[float], Dict, Tuple, float]:
    """Extract 29 basic features."""
    pose_landmarks = results.pose_landmarks.landmark
    nose_pose = pose_landmarks[0]
    nose_2d = (nose_pose.x * w, nose_pose.y * h)
    nose_3d = (nose_pose.x * w, nose_pose.y * h, nose_pose.z * 3000)

    face_2d, face_3d = [], []
    chin, top_head = (0, 0), (0, 0)
    left_inner_eye, left_brow = (0, 0), (0, 0)
    right_inner_eye, right_brow = (0, 0), (0, 0)
    left_mouth, right_mouth = (0, 0), (0, 0)
    upper_lip, lower_lip = (0, 0), (0, 0)

    for idx, lm in enumerate(results.face_landmarks.landmark):
        if idx in [33, 263, 1, 61, 291, 199]:
            x, y = int(lm.x * w), int(lm.y * h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])
        
        if idx == 152: chin = (lm.x * w, lm.y * h)
        elif idx == 10: top_head = (lm.x * w, lm.y * h)
        elif idx == 133: left_inner_eye = (lm.x * w, lm.y * h)
        elif idx == 443: left_brow = (lm.x * w, lm.y * h)
        elif idx == 223: right_brow = (lm.x * w, lm.y * h)
        elif idx == 362: right_inner_eye = (lm.x * w, lm.y * h)
        elif idx == 87: left_mouth = (lm.x * w, lm.y * h)
        elif idx == 308: right_mouth = (lm.x * w, lm.y * h)
        elif idx == 13: upper_lip = (lm.x * w, lm.y * h)
        elif idx == 14: lower_lip = (lm.x * w, lm.y * h)

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    focal_length = w
    cam_matrix = np.array([[focal_length, 0, h / 2], [0, focal_length, w / 2], [0, 0, 1]])
    success, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, np.zeros((4, 1)))
    
    if success:
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        xrot, yrot, zrot = angles[0] * w, angles[1] * h, angles[2] * 3000
    else:
        xrot, yrot, zrot = 0.0, 0.0, 0.0

    norm_dist = np.linalg.norm(np.array(chin) - np.array(top_head))
    if norm_dist == 0: norm_dist = 1
    
    body = {}
    for idx, lm in enumerate(pose_landmarks):
        if idx < len(BODY_LANDMARKS):
            body[BODY_LANDMARKS[idx]] = (lm.x * w, lm.y * h)

    def sd(p1, p2): 
        return np.linalg.norm(np.array(p1) - np.array(p2)) / norm_dist if (p1 and p2) else 0.0

    features = [
        xrot, yrot, zrot, nose_3d[0], nose_3d[1], nose_3d[2], norm_dist,
        sd(left_inner_eye, left_brow), sd(right_inner_eye, right_brow),
        sd(left_mouth, right_mouth), sd(upper_lip, lower_lip),
        sd(body.get('LEFT_WRIST'), body.get('RIGHT_WRIST')),
        sd(body.get('LEFT_ELBOW'), body.get('RIGHT_ELBOW')),
        sd(body.get('LEFT_ELBOW'), body.get('LEFT_SHOULDER')),
        sd(body.get('RIGHT_ELBOW'), body.get('RIGHT_SHOULDER')),
        sd(body.get('LEFT_WRIST'), body.get('LEFT_SHOULDER')),
        sd(body.get('RIGHT_WRIST'), body.get('RIGHT_SHOULDER')),
        sd(body.get('LEFT_SHOULDER'), body.get('LEFT_EAR')),
        sd(body.get('RIGHT_SHOULDER'), body.get('RIGHT_EAR')),
        sd(body.get('LEFT_THUMB'), body.get('LEFT_INDEX')),
        sd(body.get('RIGHT_THUMB'), body.get('RIGHT_INDEX')),
        sd(body.get('LEFT_THUMB'), body.get('LEFT_PINKY')),
        sd(body.get('RIGHT_THUMB'), body.get('RIGHT_PINKY')),
        (body.get('LEFT_WRIST')[0] - body.get('LEFT_ELBOW')[0])/norm_dist if (body.get('LEFT_WRIST') and body.get('LEFT_ELBOW')) else 0,
        (body.get('RIGHT_WRIST')[0] - body.get('RIGHT_ELBOW')[0])/norm_dist if (body.get('RIGHT_WRIST') and body.get('RIGHT_ELBOW')) else 0,
        (body.get('LEFT_WRIST')[1] - body.get('LEFT_ELBOW')[1])/norm_dist if (body.get('LEFT_WRIST') and body.get('LEFT_ELBOW')) else 0,
        (body.get('RIGHT_WRIST')[1] - body.get('RIGHT_ELBOW')[1])/norm_dist if (body.get('RIGHT_WRIST') and body.get('RIGHT_ELBOW')) else 0,
        sd(body.get('LEFT_INDEX'), nose_2d),
        sd(body.get('RIGHT_INDEX'), nose_2d),
    ]
    
    return features, body, nose_2d, norm_dist


# ============================================================================
# VIDEO PROCESSING
# ============================================================================

def process_single_video(video_path: str, max_frames: int = 800, target_fps: int = 25,
                        min_detection_pct: float = 0.3) -> Optional[Dict[str, Any]]:
    """Process a single video and extract all three feature sets."""
    metadata = parse_filename(video_path)
    if metadata is None:
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening: {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_interval = max(1, int(fps / target_fps))
    
    basic_features = []
    extended_features = []
    world_features = []
    timestamps = []
    
    frame_count = 0
    attempted_frames = 0
    detected_frames = 0
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as holistic:
        
        while cap.isOpened() and len(basic_features) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval != 0:
                frame_count += 1
                continue
            
            attempted_frames += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            
            if results.face_landmarks and results.pose_landmarks:
                detected_frames += 1
                
                basic_29, body, nose_2d, norm_dist = extract_basic_features(results, w, h)
                
                left_wrist = body.get('LEFT_WRIST')
                right_wrist = body.get('RIGHT_WRIST')
                left_hand = extract_hand_features(results.left_hand_landmarks, left_wrist, norm_dist, w, h)
                right_hand = extract_hand_features(results.right_hand_landmarks, right_wrist, norm_dist, w, h)
                
                visibility = extract_visibility_features(results)
                move_features = extract_move_distinguishing_features(body, nose_2d, norm_dist)
                
                world = extract_world_landmarks_with_visibility(results)
                if world is None:
                    world = [0.0] * CONFIG.num_world_features
                
                basic_41 = basic_29 + visibility + move_features
                extended_61 = basic_29 + left_hand + right_hand + visibility + move_features
                
                basic_features.append(basic_41)
                extended_features.append(extended_61)
                world_features.append(world)
                timestamps.append(frame_count / fps)
            
            frame_count += 1
    
    cap.release()
    
    detection_pct = detected_frames / attempted_frames if attempted_frames > 0 else 0
    
    if detection_pct < min_detection_pct:
        print(f"Skipping {metadata.filename}: detection {detection_pct:.1%} < {min_detection_pct:.1%}")
        return None
    
    if len(basic_features) == 0:
        print(f"No valid frames in {metadata.filename}")
        return None
    
    return {
        'metadata': metadata.to_dict(),
        'basic': np.array(basic_features, dtype=np.float32),
        'extended': np.array(extended_features, dtype=np.float32),
        'world': np.array(world_features, dtype=np.float32),
        'timestamps': np.array(timestamps, dtype=np.float32),
        'stats': {
            'total_frames': total_frames,
            'attempted_frames': attempted_frames,
            'detected_frames': detected_frames,
            'detection_pct': detection_pct,
            'extracted_frames': len(basic_features)
        }
    }


def process_video_wrapper(args):
    """Wrapper for multiprocessing."""
    video_path, max_frames, target_fps, min_detection_pct = args
    try:
        return process_single_video(video_path, max_frames, target_fps, min_detection_pct)
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None


# ============================================================================
# MAIN COLLECTION
# ============================================================================

def collect_all_features_parallel(
    videofiles: List[str],
    output_dir: str,
    max_frames: int = 800,
    target_fps: int = 25,
    min_detection_pct: float = 0.3,
    num_workers: int = None
) -> Dict[str, Any]:
    """Collect features from all videos and save structured datasets."""
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)
    
    print(f"\n{'='*70}")
    print("STRUCTURED FEATURE EXTRACTION v3.2")
    print(f"{'='*70}")
    print(f"Videos to process: {len(videofiles)}")
    print(f"Workers: {num_workers}")
    print(f"Max frames/video: {max_frames}")
    print(f"Target FPS: {target_fps}")
    print(f"Min detection: {min_detection_pct:.0%}")
    print(f"{'='*70}\n")
    
    args_list = [(v, max_frames, target_fps, min_detection_pct) for v in videofiles]
    
    results = []
    with Pool(num_workers) as pool:
        for result in tqdm(pool.imap(process_video_wrapper, args_list), 
                          total=len(args_list), desc="Processing videos"):
            if result is not None:
                results.append(result)
    
    print(f"\n✓ Processed {len(results)}/{len(videofiles)} videos")
    
    # Organize by training label
    data_by_label = defaultdict(list)
    all_metadata = []
    
    for r in results:
        meta = r['metadata']
        all_metadata.append(meta)
        label = meta['training_label']
        data_by_label[label].append(r)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("SAVING DATASETS")
    print(f"{'='*70}")
    
    def save_dataset(filename: str, feature_key: str, num_features: int, feature_enum):
        dataset = {}
        
        for label in CONFIG.category_labels:
            videos_data = data_by_label.get(label, [])
            
            video_landmarks = [v[feature_key] for v in videos_data]
            video_metadata = [v['metadata'] for v in videos_data]
            video_timestamps = [v['timestamps'] for v in videos_data]
            
            dataset[f'{label}_landmarks'] = np.array(video_landmarks, dtype=object)
            dataset[f'{label}_metadata'] = video_metadata
            dataset[f'{label}_timestamps'] = np.array(video_timestamps, dtype=object)
            dataset[f'{label}_n_videos'] = len(videos_data)
            dataset[f'{label}_n_frames'] = sum(len(v[feature_key]) for v in videos_data)
        
        feature_names = [f.name for f in feature_enum]
        dataset['feature_names'] = feature_names
        
        filepath = os.path.join(output_dir, filename)
        np.savez_compressed(filepath, **dataset)
        
        print(f"\n✓ Saved: {filepath}")
        print(f"  Features: {num_features}")
        for label in CONFIG.category_labels:
            n_vid = dataset[f'{label}_n_videos']
            n_fr = dataset[f'{label}_n_frames']
            print(f"  {label}: {n_vid} videos, {n_fr} frames")
        
        return filepath
    
    save_dataset(CONFIG.npz_basic, 'basic', CONFIG.num_basic_features, FeatureBasicV3)
    save_dataset(CONFIG.npz_extended, 'extended', CONFIG.num_extended_features, FeatureExtendedV3)
    
    # World landmarks
    world_dataset = {}
    for label in CONFIG.category_labels:
        videos_data = data_by_label.get(label, [])
        world_dataset[f'{label}_landmarks'] = np.array([v['world'] for v in videos_data], dtype=object)
        world_dataset[f'{label}_metadata'] = [v['metadata'] for v in videos_data]
        world_dataset[f'{label}_timestamps'] = np.array([v['timestamps'] for v in videos_data], dtype=object)
        world_dataset[f'{label}_n_videos'] = len(videos_data)
        world_dataset[f'{label}_n_frames'] = sum(len(v['world']) for v in videos_data)
    
    world_feature_names = []
    for lm_name in UPPER_BODY_LANDMARKS:
        for dim in ['X', 'Y', 'Z', 'visibility']:
            world_feature_names.append(f"{lm_name}_{dim}")
    world_dataset['feature_names'] = world_feature_names
    
    world_path = os.path.join(output_dir, CONFIG.npz_world)
    np.savez_compressed(world_path, **world_dataset)
    print(f"\n✓ Saved: {world_path}")
    print(f"  Features: {CONFIG.num_world_features}")
    for label in CONFIG.category_labels:
        n_vid = world_dataset[f'{label}_n_videos']
        n_fr = world_dataset[f'{label}_n_frames']
        print(f"  {label}: {n_vid} videos, {n_fr} frames")
    
    # Save metadata JSON
    metadata_path = os.path.join(output_dir, CONFIG.metadata_filename)
    with open(metadata_path, 'w') as f:
        json.dump({
            'all_videos': all_metadata,
            'by_label': {k: [v['metadata'] for v in vlist] for k, vlist in data_by_label.items()},
            'config': {
                'num_basic_features': CONFIG.num_basic_features,
                'num_extended_features': CONFIG.num_extended_features,
                'num_world_features': CONFIG.num_world_features,
                'target_fps': target_fps,
                'max_frames': max_frames,
                'category_labels': CONFIG.category_labels
            }
        }, f, indent=2)
    print(f"\n✓ Metadata: {metadata_path}")
    
    return {
        'results': results,
        'metadata': all_metadata,
        'by_label': data_by_label
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("STRUCTURED FEATURE EXTRACTION v3.2")
    print("="*70)
    print("\nCreates THREE structured datasets:")
    print(f"  1. Basic:    {CONFIG.num_basic_features} features (29 body + 6 vis + 6 move)")
    print(f"  2. Extended: {CONFIG.num_extended_features} features (29 body + 20 hand + 6 vis + 6 move)")
    print(f"  3. World:    {CONFIG.num_world_features} features (23 landmarks × 4)")
    print(f"\nFour categories: {CONFIG.category_labels}")
    print("\nFor plotting, run: python plot_dataset_statistics.py")
    print("="*70 + "\n")
    
    training_dir = os.path.abspath('../TrainingDataRaw/')
    
    if not os.path.exists(training_dir):
        print(f"ERROR: Training folder not found: {training_dir}")
        exit(1)
    
    videofiles = []
    for root, dirs, files in os.walk(training_dir):
        for file in files:
            if file.endswith('.mp4'):
                videofiles.append(os.path.join(root, file))
    
    print(f"Found {len(videofiles)} video files")
    
    if len(videofiles) == 0:
        print("ERROR: No video files found!")
        exit(1)
    
    # Show sample parsing
    print("\nSample filename parsing:")
    for vid in videofiles[:5]:
        meta = parse_filename(vid)
        if meta:
            print(f"  {meta.filename}")
            print(f"    → corpus={meta.corpus}, speaker={meta.speaker}, "
                  f"subtype={meta.subtype} → label={meta.training_label}")
    
    # Process
    result = collect_all_features_parallel(
        videofiles=videofiles,
        output_dir=CONFIG.output_dir,
        max_frames=CONFIG.max_frames_per_video,
        target_fps=CONFIG.target_fps,
        min_detection_pct=CONFIG.min_detection_percentage,
        num_workers=None
    )
    
    print("\n" + "="*70)
    print("ALL DONE!")
    print("="*70)
    print(f"\nOutput directory: {CONFIG.output_dir}")
    print(f"  - {CONFIG.npz_basic}")
    print(f"  - {CONFIG.npz_extended}")
    print(f"  - {CONFIG.npz_world}")
    print(f"  - {CONFIG.metadata_filename}")
    print("\nTo generate plots, run:")
    print("  python R2_plotdatasetoverview.py")
    print("\n")
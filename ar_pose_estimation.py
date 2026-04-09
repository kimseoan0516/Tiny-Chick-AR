"""
HW #4 - Camera Pose Estimation and AR
--------------------------------------
HW #3 (Lens-Distortion-Rectifier)의 캘리브레이션 결과를 이용해
체스보드 영상에서 카메라 자세를 추정하고 3D AR 물체를 렌더링합니다.

AR 물체:
  1. OBJ 3D 모델 렌더링 (MODEL_FILE 지정)
  2. 공중 부양 텍스트 효과
"""

import cv2
import numpy as np
import pickle
import os

# ── 설정 (경로를 상황에 맞게 수정하세요) ──────────────────────────────────────
CALIB_FILE   = '../Lens-Distortion-Rectifier/calibration_result.pkl'
INPUT_VIDEO  = '../Lens-Distortion-Rectifier/results/corrected.avi'
OUTPUT_VIDEO = 'results/ar_result.avi'
DEMO_IMAGE   = 'results/ar_demo.jpg'

BOARD_SIZE   = (7, 7)    # HW3와 동일
SQUARE_SIZE  = 0.025     # HW3와 동일 (2.5 cm)

# corrected.avi(왜곡 보정된 영상)를 사용할 경우 True로 설정
# → new_K 자동 계산, dist=0으로 맞춰 solvePnP 카메라 모델을 영상에 맞춤
USE_UNDISTORTED_VIDEO = True

# 창 없이 처리만 할 때 True로 설정 (imshow 생략)
HEADLESS = True

# 3D 모델 파일 경로 (OBJ). None이면 기하 도형만 렌더링
MODEL_FILE   = 'model/FourthGreen/FourthGreen.obj'
# ─────────────────────────────────────────────────────────────────────────────


# ── 캘리브레이션 로드 ─────────────────────────────────────────────────────────

def load_calibration(path: str):
    """HW #3에서 저장한 .pkl에서 카메라 행렬, 왜곡 계수, 이미지 크기를 반환."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    K, dist, img_size = data['K'], data['dist'], data['img_size']
    print("[INFO] 캘리브레이션 로드 완료")
    print(f"  fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}")
    return K, dist, img_size


# ── OBJ 로더 ─────────────────────────────────────────────────────────────────

def load_mtl(mtl_path: str) -> dict:
    """MTL / LIB 파일에서 재질 정보 파싱. 텍스처 이미지가 있으면 함께 반환."""
    obj_dir   = os.path.dirname(os.path.abspath(mtl_path))
    materials = {}   # {name: {'color': (B,G,R), 'tex': ndarray or None}}
    current   = None
    try:
        with open(mtl_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == 'newmtl':
                    current = ' '.join(parts[1:])
                    materials[current] = {'color': (30, 200, 240), 'tex': None}
                elif parts[0] == 'Kd' and current and len(parts) >= 4:
                    r = min(1.0, float(parts[1]))
                    g = min(1.0, float(parts[2]))
                    b = min(1.0, float(parts[3]))
                    materials[current]['color'] = (int(b*255), int(g*255), int(r*255))
                elif parts[0] == 'map_Kd' and current:
                    # map_Kd 절대경로에서 파일명만 추출해 모델 폴더에서 찾기
                    tex_name  = ' '.join(parts[1:])
                    basename  = os.path.basename(tex_name)
                    base_stem = os.path.splitext(basename)[0]
                    candidates = [
                        os.path.join(obj_dir, basename),
                        os.path.join(obj_dir, base_stem + '.png'),
                        os.path.join(obj_dir, base_stem + '.jpg'),
                        tex_name,   # 절대경로 그대로도 시도
                    ]
                    for p in candidates:
                        if os.path.exists(p):
                            tex = cv2.imread(p)
                            if tex is not None:
                                materials[current]['tex'] = tex
                                print(f"[INFO] 텍스처 로드: {p}")
                                break
    except FileNotFoundError:
        pass
    return materials


def load_obj(path: str):
    """
    OBJ + MTL/LIB 파서 (UV 텍스처 매핑 지원).
    반환: vertices (N,3), faces (M,3), face_colors list[(B,G,R)]
    """
    vertices, uvs = [], []
    raw_faces  = []          # (vert_idxs, uv_idxs, color, tex)
    materials  = {}
    cur_color  = (30, 200, 240)
    cur_tex    = None
    obj_dir    = os.path.dirname(os.path.abspath(path))

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'mtllib':
                base = os.path.splitext(parts[1])[0]
                for try_name in [parts[1], base + '.lib', base + '.mtl']:
                    full = os.path.join(obj_dir, try_name)
                    if os.path.exists(full):
                        materials = load_mtl(full)
                        print(f"[INFO] 재질 파일 로드: {full} ({len(materials)}개)")
                        break
            elif parts[0] == 'usemtl':
                mat_name  = ' '.join(parts[1:])
                mat       = materials.get(mat_name, {'color': (30, 200, 240), 'tex': None})
                cur_color = mat['color']
                cur_tex   = mat['tex']
            elif parts[0] == 'v':
                vertices.append([float(p) for p in parts[1:4]])
            elif parts[0] == 'vt':
                u = float(parts[1])
                v = float(parts[2]) if len(parts) > 2 else 0.0
                uvs.append([u, v])
            elif parts[0] == 'f':
                vi, ti = [], []
                for tok in parts[1:]:
                    sp = tok.split('/')
                    vi.append(int(sp[0]) - 1)
                    ti.append(int(sp[1]) - 1 if len(sp) > 1 and sp[1] else -1)
                for i in range(1, len(vi) - 1):
                    raw_faces.append(([vi[0], vi[i], vi[i+1]],
                                      [ti[0], ti[i], ti[i+1]],
                                      cur_color, cur_tex))

    vertices_np = np.array(vertices, dtype=np.float32)
    uvs_np      = np.array(uvs, dtype=np.float32) if uvs else np.zeros((0, 2), np.float32)

    faces, face_colors = [], []
    for (fvi, fti, color, tex) in raw_faces:
        faces.append(fvi)
        if tex is not None and len(uvs_np) > 0 and all(u >= 0 for u in fti):
            h, w   = tex.shape[:2]
            avg_u  = (uvs_np[fti[0]][0] + uvs_np[fti[1]][0] + uvs_np[fti[2]][0]) / 3.0
            avg_v  = (uvs_np[fti[0]][1] + uvs_np[fti[1]][1] + uvs_np[fti[2]][1]) / 3.0
            px     = int(avg_u * w)    % w
            py     = int((1.0 - avg_v) * h) % h  # OBJ v=0이 아래쪽이므로 반전
            bgr    = tex[py, px]
            face_colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
        else:
            face_colors.append(color)

    faces_np = np.array(faces, dtype=np.int32)
    if not face_colors:
        face_colors = [(30, 200, 240)] * len(faces_np)
    return vertices_np, faces_np, face_colors





def normalize_obj(vertices: np.ndarray, scale: float = 0.08):
    """모델 중심을 원점으로 이동, 최대 크기가 scale이 되도록 정규화."""
    v = vertices.copy()
    v -= v.mean(axis=0)                          # 중심 이동
    v /= np.abs(v).max()                         # [-1, 1] 정규화
    v *= scale                                   # 원하는 크기로 조정
    return v


def render_obj(img, vertices, faces, rvec, tvec, K, dist,
               offset=(0.0, 0.0, 0.0), alpha=0.75, frame_idx=0, face_colors=None):
    """OBJ 삼각형 면을 Painter's algorithm으로 렌더링.
    frame_idx를 이용해 Y축 회전 + Z축 bobbing + 좌우 이동 애니메이션 적용.
    """
    # ── 애니메이션 파라미터 ──
    move_x     = np.sin(frame_idx * 0.05) * SQUARE_SIZE * 2.5   # 좌우 이동 (X축, 일직선)
    
    # 이동 방향에 따라 바라보는 방향 설정
    vel_x = np.cos(frame_idx * 0.05)
    if vel_x > 0:
        spin_deg = -90.0  # 오른쪽 보기 (+X 방향)
    else:
        spin_deg = 90.0   # 왼쪽 보기 (-X 방향)

    # ── Step 1: Y-up OBJ 모델 → AR 공간 직립 (X축 -90° 회전) ──
    # OBJ 파일 기준 Y가 '위'이므로, AR 좌표계(Z=보드 법선 방향)에 세우려면
    # Y → -Z,  Z → Y 로 축 변환 필요 (Rx(-90°))
    Rx = np.array([[1,  0,  0],
                   [0,  0,  1],
                   [0, -1,  0]], dtype=np.float32)
    verts = (Rx @ vertices.T).T

    # ── Step 2: Z축 기준 팽이 회전 ──
    c, s = np.cos(np.radians(spin_deg)), np.sin(np.radians(spin_deg))
    Rz = np.array([[c, -s, 0],
                   [s,  c, 0],
                   [0,  0, 1]], dtype=np.float32)
    verts = (Rz @ verts.T).T

    # ── Step 3: 발바닥(Z 최댓값)을 체스판(Z=0)에 딱 맞게 정렬 ──
    max_z = verts[:, 2].max()
    verts[:, 2] -= max_z

    # 위치 오프셋 + 좌우 이동 적용
    verts += np.array([offset[0] + move_x, offset[1], 0.0], dtype=np.float32)

    # 모든 꽃짓점을 2D로 투영
    pts2d, _ = cv2.projectPoints(verts, rvec, tvec, K, dist)
    pts2d = pts2d.reshape(-1, 2).astype(np.int32)

    # 카메라 좌표계로 변환 (깊이 정렬용)
    R, _ = cv2.Rodrigues(rvec)
    verts_cam = (R @ verts.T).T + tvec.flatten()

    # 각 면의 평균 깊이 계산 후 후방부터 그리기 (Painter's algorithm)
    face_depths = verts_cam[faces, 2].mean(axis=1)
    order = np.argsort(face_depths)[::-1]

    # 라이팅 방향
    light_dir = np.array([0.3, -0.5, -1.0])
    light_dir /= np.linalg.norm(light_dir)

    overlay = img.copy()
    for i in order:
        fi = faces[i]
        tri = pts2d[fi]                          # (3, 2)

        # 법선 벡터 계산 (카메라 공간)
        v0 = verts_cam[fi[0]]
        v1 = verts_cam[fi[1]]
        v2 = verts_cam[fi[2]]
        n  = np.cross(v1 - v0, v2 - v0)
        n_len = np.linalg.norm(n)
        if n_len < 1e-8:
            continue
        n /= n_len

        # 백-페이스 콸링
        if n[2] > 0:
            continue

        # 램버트 음영
        diffuse = max(0.55, float(-np.dot(n, light_dir)))  # ambient 높여서 색상이 밝게
        if face_colors and i < len(face_colors):
            mb, mg, mr = face_colors[i]
            # 전체적으로 색상을 더 밝게 하기 위해 가중치 보정
            color = (min(255, int(mb * diffuse * 1.5)), 
                     min(255, int(mg * diffuse * 1.5)), 
                     min(255, int(mr * diffuse * 1.5)))
        else:
            brightness = int(60 + 180 * diffuse)
            color = (brightness, int(brightness * 0.6), int(brightness * 0.3))

        cv2.fillConvexPoly(overlay, tri, color)
        
        # 색상이 없을 때만 흰 실선을 기름 (재질이 있으면 실선 생략해서 매끄럽게)
        if not face_colors:
            cv2.polylines(overlay, [tri], True, (255, 255, 255), 1)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


# ── Painter's algorithm 기반 면 렌더링 헬퍼 ──────────────────────────────────

def _project(pts3d: np.ndarray, rvec, tvec, K, dist) -> np.ndarray:
    pts2d, _ = cv2.projectPoints(pts3d.astype(np.float32), rvec, tvec, K, dist)
    return pts2d.reshape(-1, 2).astype(np.int32)


def _face_depth(face_pts3d: np.ndarray, rvec, tvec) -> float:
    R, _ = cv2.Rodrigues(rvec)
    pts_cam = (R @ face_pts3d.T).T + tvec.flatten()
    return float(pts_cam[:, 2].mean())


def _draw_faces(img, faces3d: list, colors: list, rvec, tvec, K, dist,
                alpha=0.75, edge_color=(255, 255, 255)):
    """깊이 순 정렬 후 반투명 다각형으로 그리기."""
    depths = [_face_depth(np.array(f, dtype=np.float32), rvec, tvec) for f in faces3d]
    order  = np.argsort(depths)[::-1]           # 먼 면 → 가까운 면

    overlay = img.copy()
    for i in order:
        pts2d = _project(np.array(faces3d[i], dtype=np.float32), rvec, tvec, K, dist)
        cv2.fillConvexPoly(overlay, pts2d, colors[i])
        if edge_color:
            cv2.polylines(overlay, [pts2d], True, edge_color, 2)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


# ── AR 물체 그리기 함수들 ─────────────────────────────────────────────────────

def draw_axes(img, rvec, tvec, K, dist, length: float):
    """XYZ 좌표축 (X=빨강, Y=초록, Z=파랑)."""
    pts = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, -length]])
    p   = _project(pts, rvec, tvec, K, dist)
    orig = tuple(p[0])
    cv2.arrowedLine(img, orig, tuple(p[1]), (0,   0, 220), 3, tipLength=0.3)  # X
    cv2.arrowedLine(img, orig, tuple(p[2]), (0, 220,   0), 3, tipLength=0.3)  # Y
    cv2.arrowedLine(img, orig, tuple(p[3]), (220,  0,   0), 3, tipLength=0.3) # Z
    for pt, label, col in zip(p[1:], ['X', 'Y', 'Z'],
                               [(0, 0, 220), (0, 220, 0), (220, 0, 0)]):
        cv2.putText(img, label, (int(pt[0]) + 5, int(pt[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)





def draw_floating_text(img, rvec, tvec, K, dist, text: str, pos3d, scale=0.9):
    """3D 공간의 특정 위치에 glow 효과 텍스트."""
    p  = _project(np.float32([pos3d]), rvec, tvec, K, dist)
    pt = (int(p[0][0]), int(p[0][1]))
    cv2.putText(img, text, pt, cv2.FONT_HERSHEY_DUPLEX, scale, (  0,   0,   0), 5)
    cv2.putText(img, text, pt, cv2.FONT_HERSHEY_DUPLEX, scale, (  0, 220, 255), 2)


# ── AR 장면 전체 구성 ─────────────────────────────────────────────────────────

def draw_ar_scene(img, rvec, tvec, K, dist, frame_idx: int,
                  obj_verts=None, obj_faces=None, obj_colors=None):
    """체스보드 위에 AR 물체 전체를 배치."""
    s = SQUARE_SIZE

    # 1. 좌표축
    draw_axes(img, rvec, tvec, K, dist, length=s * 1.5)

    if obj_verts is not None and obj_faces is not None:
        # 3D 모델이 있으면 중앙에 렌더링 (Z=0으로 바닥에 붙임)
        cx = s * BOARD_SIZE[0] / 2
        cy = s * BOARD_SIZE[1] / 2
        render_obj(img, obj_verts, obj_faces, rvec, tvec, K, dist,
                   offset=(cx, cy, 0.0), alpha=1.0, frame_idx=frame_idx, face_colors=obj_colors)

    # 5. 공중 텍스트
    draw_floating_text(img, rvec, tvec, K, dist,
                       "AR DEMO", pos3d=[s * 1.5, s * 3.5, -s * 4.5], scale=0.9)


# ── 메인 ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    # ── 캘리브레이션 로드 ──
    if not os.path.exists(CALIB_FILE):
        print(f"[ERROR] 캘리브레이션 파일을 찾을 수 없습니다: {CALIB_FILE}")
        print("  → Lens-Distortion-Rectifier/camera_calibration.py를 먼저 실행하세요.")
        raise SystemExit(1)
    K, dist, img_size = load_calibration(CALIB_FILE)

    # ── corrected.avi 사용 시 effective_K 계산 ────────────────────────────────
    # HW3 distortion_correction.py는 undistort 후 ROI 크롭 → img_size로 리사이즈.
    # 이 크롭+리사이즈가 카메라 행렬을 한 번 더 바꾸므로 정확히 재현해야 한다.
    if USE_UNDISTORTED_VIDEO:
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, img_size, alpha=0, newImgSize=img_size)
        x, y, w_roi, h_roi = roi
        if w_roi > 0 and h_roi > 0:
            sx = img_size[0] / w_roi
            sy = img_size[1] / h_roi
            eff_K = new_K.copy()
            eff_K[0, 0] = new_K[0, 0] * sx              # fx 조정
            eff_K[1, 1] = new_K[1, 1] * sy              # fy 조정
            eff_K[0, 2] = (new_K[0, 2] - x) * sx        # cx 조정
            eff_K[1, 2] = (new_K[1, 2] - y) * sy        # cy 조정
            K = eff_K
        else:
            K = new_K
        dist = np.zeros((1, 5), dtype=np.float32)
        print("[INFO] 왜곡 보정 영상 모드: effective_K 적용, dist=0")
        print(f"  eff_fx={K[0,0]:.2f}  eff_fy={K[1,1]:.2f}  eff_cx={K[0,2]:.2f}  eff_cy={K[1,2]:.2f}")



    # ── 3D 모델 로드 (선택) ──
    obj_verts, obj_faces, obj_colors = None, None, None
    if MODEL_FILE and os.path.exists(MODEL_FILE):
        print(f"[INFO] 3D 모델 로드: {MODEL_FILE}")
        obj_verts, obj_faces, obj_colors = load_obj(MODEL_FILE)
        obj_verts = normalize_obj(obj_verts, scale=SQUARE_SIZE * 2)
        print(f"  vertices={len(obj_verts)}, faces={len(obj_faces)}")
    elif MODEL_FILE:
        print(f"[WARN] 모델 파일 없음 ({MODEL_FILE}) → 기하 도형으로 대체합니다.")

    # ── 체스보드 오브젝트 포인트 (HW3와 동일) ──
    objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # ── 영상 열기 ──
    if not os.path.exists(INPUT_VIDEO):
        print(f"[ERROR] 입력 영상을 찾을 수 없습니다: {INPUT_VIDEO}")
        print("  → INPUT_VIDEO 경로를 수정하세요.")
        raise SystemExit(1)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] 입력 영상: {w}x{h} @ {fps:.1f}fps")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out    = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

    frame_idx      = 0
    pose_count     = 0
    demo_saved     = False
    last_rvec      = None          # 마지막으로 성공한 rvec
    last_tvec      = None          # 마지막으로 성공한 tvec
    coast_frames   = 0             # 체스보드 미검충 프레임 수
    MAX_COAST      = 999999        # 무한정 유지 (단 체스보드가 한 번도 검충된 이후)

    print("[INFO] AR 처리 시작... (ESC 또는 Q 키로 종료)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── 체스보드 코너 검쳙 (SB 우선, 실패 시 클래식으로 폴백) ──
        found_sb, corners_sb = cv2.findChessboardCornersSB(
            gray, BOARD_SIZE,
            cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        )
        if found_sb:
            corners_sub = corners_sb
            found = True
        else:
            found, corners = cv2.findChessboardCorners(
                gray, BOARD_SIZE,
                cv2.CALIB_CB_ADAPTIVE_THRESH
                | cv2.CALIB_CB_NORMALIZE_IMAGE
                | cv2.CALIB_CB_FAST_CHECK,
            )
            if found:
                corners_sub = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria
                )
            else:
                corners_sub = None

        if found and corners_sub is not None:
            ok, rvec, tvec = cv2.solvePnP(objp, corners_sub, K, dist)
            if ok:
                last_rvec, last_tvec = rvec.copy(), tvec.copy()
                coast_frames = 0

        # ── AR 렌더링: 포즈 성공 또는 last-pose coasting ──
        if last_rvec is not None:
            draw_ar_scene(frame, last_rvec, last_tvec, K, dist, frame_idx,
                          obj_verts, obj_faces, obj_colors)
            pose_count += 1

            # 데모 이미지 저장
            demo_frames = [1, 60, 150]
            if pose_count in demo_frames:
                idx = demo_frames.index(pose_count)
                path = DEMO_IMAGE.replace('.jpg', f'_{idx+1}.jpg')
                cv2.imwrite(path, frame)
                print(f"[INFO] 데모 이미지 저장: {path}")
            if not demo_saved:
                cv2.imwrite(DEMO_IMAGE, frame)
                demo_saved = True
                print(f"[INFO] 데모 이미지 저장: {DEMO_IMAGE}")

            coast_frames += 1
        else:
            coast_frames += 1

        # HUD
        status = f"Pose frames: {pose_count}"
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out.write(frame)
        frame_idx += 1

        if not HEADLESS:
            cv2.imshow('HW #4 - AR Pose Estimation', frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):    # ESC or Q
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\n[INFO] 완료!")
    print(f"  포즈 추정 성공 프레임: {pose_count} / {frame_idx}")
    print(f"  AR 영상 저장: {OUTPUT_VIDEO}")
    if demo_saved:
        print(f"  데모 이미지: {DEMO_IMAGE}")

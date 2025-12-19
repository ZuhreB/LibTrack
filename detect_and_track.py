import os
import cv2
import time
import torch
import argparse
from pathlib import Path
import numpy as np
from numpy import random
from random import randint
import torch.backends.cudnn as cudnn

# --- MYSQL KÜTÜPHANESİ ---
import mysql.connector
from datetime import datetime


from collections import Counter

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, \
    check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
    increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, \
    time_synchronized, TracedModel
from utils.download_weights import download

# For SORT tracking
import skimage
from sort import *

# --- MYSQL AYARLARI ---
DB_CONFIG = {
    'user': 'root',
    'password': 'zuhre060',
    'host': 'localhost',
    'database': 'Library_db',
    'raise_on_warnings': True
}


# =========================
#   MYSQL FONKSİYONLARI
# =========================

def init_mysql_table():
    """Veritabanını ve tabloyu otomatik oluşturur, eksik kolonları da tamamlar."""
    try:
        # 1) Veritabanı ismini ayır, önce sadece sunucuya bağlan
        temp_config = DB_CONFIG.copy()
        target_db = temp_config.pop('database')

        cnx_server = mysql.connector.connect(**temp_config)
        cursor_server = cnx_server.cursor()

        try:
            cursor_server.execute(f"CREATE DATABASE IF NOT EXISTS {target_db}")
            cnx_server.commit()
        except mysql.connector.Error as err:
            if err.errno != 1007:  # 1007: DB exists
                raise

        cursor_server.close()
        cnx_server.close()

        # 2) Artık DB ile bağlanıp tabloyu yarat / güncelle
        cnx_db = mysql.connector.connect(**DB_CONFIG)
        cursor_db = cnx_db.cursor()

        # Eğer tablo yoksa, doğru şemayla oluştur
        table_query = """
        CREATE TABLE IF NOT EXISTS person_logs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            record_date DATETIME NOT NULL,
            camera_id VARCHAR(255) NOT NULL,
            person_count INT NOT NULL
        )
        """
        cursor_db.execute(table_query)
        cnx_db.commit()

        # Eski tabloysa ve camera_id kolonu yoksa, ALTER TABLE ile ekle
        try:
            cursor_db.execute("SELECT camera_id FROM person_logs LIMIT 1")
        except mysql.connector.Error as err:
            if err.errno == 1054:  # Unknown column 'camera_id'
                print("camera_id kolonu yok, ALTER TABLE ile ekleniyor...")
                cursor_db.execute(
                    "ALTER TABLE person_logs "
                    "ADD COLUMN camera_id VARCHAR(255) NOT NULL DEFAULT 'unknown'"
                )
                cnx_db.commit()
            else:
                raise

        cursor_db.close()
        cnx_db.close()

        print(f"MySQL: '{target_db}' veritabanı ve person_logs tablo/kolon kontrolü tamam.")

    except mysql.connector.Error as err:
        print(f"MySQL Başlatma Hatası: {err}")
        print("MySQL sunucusu / DB erişimiyle ilgili bir sıkıntı olabilir.")


def save_to_mysql(count, camera_id):
    """Kişi sayısını (0/1) MySQL'e kaydeder (kamera bazlı)."""
    try:
        cnx = mysql.connector.connect(**DB_CONFIG)
        cursor = cnx.cursor()

        add_log = ("INSERT INTO person_logs "
                   "(record_date, camera_id, person_count) "
                   "VALUES (%s, %s, %s)")

        now = datetime.now()
        data_log = (now, camera_id, count)

        cursor.execute(add_log, data_log)
        cnx.commit()

        cursor.close()
        cnx.close()
        print(f"--> MySQL Kayıt: Kamera={camera_id}, Zaman={now}, Kişi(0/1)={count}")

    except mysql.connector.Error as err:
        print(f"MySQL Kayıt Hatası: {err}")


# =========================
#   ÇİZİM FONKSİYONU (Sadeleştirilmiş)
# =========================

def draw_boxes(
        img,
        bbox,
        identities=None,
        categories=None,
        names=None,
        save_with_object_id=False,
        path=None,
        offset=(0, 0),
        box_color=(255, 0, 20)
):
    if identities is None:
        identities = [None] * len(bbox)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), max(0, x2), max(0, y2)

        cat = int(categories[i]) if categories is not None else 0
        obj_id = int(identities[i]) if identities[i] is not None else None

        if obj_id is None:
            continue

        # Cinsiyet kontrolü kalktı, sadece isim ve ID yazıyoruz
        label = f"{names[cat]} id:{obj_id}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        # Kutu çizimi
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), box_color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)

    return img


# =========================
#   ANA DETECT FONKSİYONU
# =========================

def detect(save_img=False):
    # MySQL veritabanını kontrol et / oluştur
    init_mysql_table()

    source, weights, view_img, save_txt, imgsz, trace, colored_trk, save_bbox_dim, save_with_object_id = \
        opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace, \
            opt.colored_trk, opt.save_bbox_dim, opt.save_with_object_id

    save_img = not opt.nosave and not source.endswith('.txt')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Her kamera için ayrı tracker tutan sözlük
    sort_trackers_dict = {}

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if save_txt or save_with_object_id else save_dir).mkdir(parents=True, exist_ok=True)

    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'

    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)

    if trace:
        model = TracedModel(model, device, opt.img_size)
    if half:
        model.half()

    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    names = model.module.names if hasattr(model, 'module') else model.names

    if opt.classes and 0 not in opt.classes:
        print("UYARI: Sadece insan (class 0) aranmalı.")
    elif not opt.classes:
        opt.classes = [0]

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    t0 = time.time()

    # Kamera bazlı occupancy state ve last seen time
    last_person_count = {}  # {camera_id: 0/1}
    occupancy_state = {}  # {camera_id: 0/1}
    last_seen_time = {}  # {camera_id: timestamp}
    OCCUPANCY_TIMEOUT = 10.0  # saniye

    # Her kamera için farklı kutu rengi
    camera_colors = {}
    preset_colors = [
        (255, 0, 0),  # mavi
        (0, 255, 0),  # yeşil
        (0, 0, 255),  # kırmızı
        (255, 255, 0),  # sarı
    ]

    vid_path, vid_writer = None, None

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                   agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        for i, det in enumerate(pred):
            # Her kamera için tracker kontrolü/oluşturma
            if i not in sort_trackers_dict:
                sort_trackers_dict[i] = Sort(max_age=5, min_hits=2, iou_threshold=0.2)

            current_tracker = sort_trackers_dict[i]

            if webcam:
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # Kamera kimliği
            camera_id = str(p)

            # Kamera rengi
            if camera_id not in camera_colors:
                idx = len(camera_colors) % len(preset_colors)
                camera_colors[camera_id] = preset_colors[idx]
            box_color = camera_colors[camera_id]

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + (
                '' if dataset.mode == 'image' else f'_{frame}'
            )

            current_person_count_raw = 0

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # İnsan sayımı
                person_mask = (det[:, -1] == 0)
                n_person = int(person_mask.sum())
                current_person_count_raw = 1 if n_person > 0 else 0

                # Log stringi
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    if int(c) == 0 and n > 1:
                        n = 1
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # SORT güncelleme
                dets_to_sort = np.empty((0, 6))
                for *xyxy, conf, detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack(
                        (dets_to_sort, np.array([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, detclass]))
                    )

                tracked_dets = current_tracker.update(dets_to_sort)

                if len(tracked_dets) > 0:
                    bbox_xyxy = tracked_dets[:, :4]
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]
                    # Sadeleştirilmiş draw_boxes çağrısı
                    draw_boxes(
                        im0,
                        bbox_xyxy,
                        identities,
                        categories,
                        names,
                        save_with_object_id,
                        txt_path,
                        box_color=box_color
                    )
            else:
                tracked_dets = current_tracker.update()

            # ===== 0/1 OCCUPANCY + 10 SANİYE COUNTDOWN MANTIĞI =====
            now_ts = time.time()
            prev_occ = occupancy_state.get(camera_id, 0)
            last_seen = last_seen_time.get(camera_id, None)

            countdown_active = False
            countdown_remain = 0.0

            if current_person_count_raw > 0:
                occupancy_state[camera_id] = 1
                last_seen_time[camera_id] = now_ts
            else:
                if prev_occ == 1:
                    if last_seen is None:
                        last_seen_time[camera_id] = now_ts
                    else:
                        elapsed = now_ts - last_seen
                        if elapsed < OCCUPANCY_TIMEOUT:
                            countdown_active = True
                            countdown_remain = OCCUPANCY_TIMEOUT - elapsed
                        else:
                            occupancy_state[camera_id] = 0

            current_person_count = occupancy_state.get(camera_id, 0)

            # Countdown Overlay
            if countdown_active:
                h, w = im0.shape[:2]
                overlay = im0.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                alpha = 0.35
                im0[:] = cv2.addWeighted(overlay, alpha, im0, 1 - alpha, 0)

                sec_left = int(countdown_remain) + 1
                text = f"LEAVING IN {sec_left}s"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
                cx, cy = w // 2, h // 2
                cv2.putText(
                    im0,
                    text,
                    (cx - tw // 2, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            # --- MYSQL KAYIT ---
            previous_count = last_person_count.get(camera_id, -1)
            if current_person_count != previous_count:
                print(
                    f'[Kamera={camera_id}] {s}Done. '
                    f'RawCount={current_person_count_raw}, Occupancy={current_person_count} '
                    f'({(1E3 * (t2 - t1)):.1f}ms) Inference'
                )
                save_to_mysql(current_person_count, camera_id)
                last_person_count[camera_id] = current_person_count

            # --- GÖRÜNTÜLEME (Cinsiyet Overlay Yok) ---
            if view_img:
                cv2.imshow(str(p), im0)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    raise StopIteration

            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(
                            save_path,
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            fps,
                            (w, h)
                        )
                    vid_writer.write(im0)

    print(f'Done. ({time.time() - t0:.3f}s)')


# =========================
#   MAIN
# =========================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='streams.txt', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--colored-trk', action='store_true', help='assign different color to every tracking id')
    parser.add_argument('--save-bbox-dim', action='store_true', help='save bounding box dimensions')
    parser.add_argument('--save-with-object-id', action='store_true', help='save results with object id')
    parser.add_argument('--db-path', type=str, default='occ.db', help='(Kullanılmıyor)')
    parser.add_argument('--run-name', type=str, default='person_count', help='Tag')

    parser.set_defaults(download=True)
    opt = parser.parse_args()
    print(opt)

    if opt.download and not os.path.exists(''.join(opt.weights)):
        print('Model weights not found. Attempting to download now...')
        download('./')

    with torch.no_grad():
        detect()
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import time
import copy
from collections import deque

import cv2 as cv
import numpy as np
import tensorflow as tf

from utils import CvFpsCalc
from utils import CvDrawText


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument("--file", type=str, default=None)

    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--skip_frame", type=int, default=0)

    parser.add_argument("--model", default='model/EfficientDetD0/saved_model')
    parser.add_argument("--score_th", type=float, default=0.75)

    parser.add_argument("--sign_interval", type=float, default=2.0)
    parser.add_argument("--jutsu_display_time", type=int, default=5)

    parser.add_argument("--use_display_score", type=bool, default=False)
    parser.add_argument("--erase_bbox", type=bool, default=False)
    parser.add_argument("--use_jutsu_lang_en", type=bool, default=False)

    parser.add_argument("--chattering_check", type=int, default=1)

    parser.add_argument("--use_fullscreen", type=bool, default=False)

    args = parser.parse_args()

    return args


def run_inference_single_image(image, inference_func):
    tensor = tf.convert_to_tensor(image)
    output = inference_func(tensor)

    output['num_detections'] = int(output['num_detections'][0])
    output['detection_classes'] = output['detection_classes'][0].numpy()
    output['detection_boxes'] = output['detection_boxes'][0].numpy()
    output['detection_scores'] = output['detection_scores'][0].numpy()
    return output


def main():
    # 引数解析 #################################################################
    args = get_args()

    cap_width = args.width
    cap_height = args.height
    cap_device = args.device
    if args.file is not None:  # 動画ファイルを利用する場合
        cap_device = args.file

    fps = args.fps
    skip_frame = args.skip_frame

    model_path = args.model
    score_th = args.score_th

    sign_interval = args.sign_interval
    jutsu_display_time = args.jutsu_display_time

    use_display_score = args.use_display_score
    erase_bbox = args.erase_bbox
    use_jutsu_lang_en = args.use_jutsu_lang_en

    chattering_check = args.chattering_check

    use_fullscreen = args.use_fullscreen

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデル読み込み ############################################################
    DEFAULT_FUNCTION_KEY = 'serving_default'
    loaded_model = tf.saved_model.load(model_path)
    inference_func = loaded_model.signatures[DEFAULT_FUNCTION_KEY]

    # FPS計測モジュール #########################################################
    cvFpsCalc = CvFpsCalc()

    # フォント読み込み ##########################################################
    # https://opentype.jp/kouzanmouhitufont.htm
    font_path = './utils/font/衡山毛筆フォント.ttf'

    # ラベル読み込み ###########################################################
    with open('setting/labels.csv', encoding='utf8') as f:  # 印
        labels = csv.reader(f)
        labels = [row for row in labels]

    with open('setting/jutsu.csv', encoding='utf8') as f:  # 術
        jutsu = csv.reader(f)
        jutsu = [row for row in jutsu]

    # 印の表示履歴および、検出履歴 ##############################################
    sign_max_display = 18
    sign_max_history = 44
    sign_display_queue = deque(maxlen=sign_max_display)
    sign_history_queue = deque(maxlen=sign_max_history)

    chattering_check_queue = deque(maxlen=chattering_check)
    for index in range(-1, -1 - chattering_check, -1):
        chattering_check_queue.append(index)

    # 術名の言語設定 ###########################################################
    lang_offset = 0
    jutsu_font_size_ratio = sign_max_display
    if use_jutsu_lang_en:
        lang_offset = 1
        jutsu_font_size_ratio = int((sign_max_display / 3) * 4)

    # その他変数初期化 #########################################################
    sign_interval_start = 0  # 印のインターバル開始時間初期化
    jutsu_index = 0  # 術表示名のインデックス
    jutsu_start_time = 0  # 術名表示の開始時間初期化
    frame_count = 0  # フレームナンバーカウンタ

    window_name = 'NARUTO HandSignDetection Ninjutsu Demo'
    if use_fullscreen:
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)

    while True:
        start_time = time.time()

        # カメラキャプチャ #####################################################
        ret, frame = cap.read()
        if not ret:
            continue
        frame_count += 1
        debug_image = copy.deepcopy(frame)

        if (frame_count % (skip_frame + 1)) != 0:
            continue

        # FPS計測 ##############################################################
        fps_result = cvFpsCalc.get()

        # 検出実施 #############################################################
        frame = frame[:, :, [2, 1, 0]]  # BGR2RGB
        image_np_expanded = np.expand_dims(frame, axis=0)
        result_inference = run_inference_single_image(image_np_expanded,
                                                      inference_func)

        # 検出内容の履歴追加 ####################################################
        num_detections = result_inference['num_detections']
        for i in range(num_detections):
            score = result_inference['detection_scores'][i]
            class_id = result_inference['detection_classes'][i].astype(np.int)

            # 検出閾値未満の結果は捨てる
            if score < score_th:
                continue

            # 指定回数以上、同じ印が続いた場合に、印検出とみなす ※瞬間的な誤検出対策
            chattering_check_queue.append(class_id)
            if len(set(chattering_check_queue)) != 1:
                continue

            # 前回と異なる印の場合のみキューに登録
            if len(sign_display_queue) == 0 or \
                sign_display_queue[-1] != class_id:
                sign_display_queue.append(class_id)
                sign_history_queue.append(class_id)
                sign_interval_start = time.time()  # 印の最終検出時間

        # 前回の印検出から指定時間が経過した場合、履歴を消去 ####################
        if (time.time() - sign_interval_start) > sign_interval:
            sign_display_queue.clear()
            sign_history_queue.clear()

        # 術成立判定 #########################################################
        jutsu_index, jutsu_start_time = check_jutsu(
            sign_history_queue,
            labels,
            jutsu,
            jutsu_index,
            jutsu_start_time,
        )

        # キー処理 ###########################################################
        key = cv.waitKey(1)
        if key == 99:  # C：印の履歴を消去
            sign_display_queue.clear()
            sign_history_queue.clear()
        if key == 27:  # ESC：プログラム終了
            break

        # FPS調整 #############################################################
        elapsed_time = time.time() - start_time
        sleep_time = max(0, ((1.0 / fps) - elapsed_time))
        time.sleep(sleep_time)

        # 画面反映 #############################################################
        debug_image = draw_debug_image(
            debug_image,
            font_path,
            fps_result,
            labels,
            result_inference,
            score_th,
            erase_bbox,
            use_display_score,
            jutsu,
            sign_display_queue,
            sign_max_display,
            jutsu_display_time,
            jutsu_font_size_ratio,
            lang_offset,
            jutsu_index,
            jutsu_start_time,
        )
        if use_fullscreen:
            cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN,
                                 cv.WINDOW_FULLSCREEN)
        cv.imshow(window_name, debug_image)
        # cv.moveWindow(window_name, 100, 100)

    cap.release()
    cv.destroyAllWindows()


def check_jutsu(
    sign_history_queue,
    labels,
    jutsu,
    jutsu_index,
    jutsu_start_time,
):
    # 印の履歴から術名をマッチング
    sign_history = ''
    if len(sign_history_queue) > 0:
        for sign_id in sign_history_queue:
            sign_history = sign_history + labels[sign_id][1]
        for index, signs in enumerate(jutsu):
            if sign_history == ''.join(signs[4:]):
                jutsu_index = index
                jutsu_start_time = time.time()  # 術の最終検出時間
                break

    return jutsu_index, jutsu_start_time


def draw_debug_image(
    debug_image,
    font_path,
    fps_result,
    labels,
    result_inference,
    score_th,
    erase_bbox,
    use_display_score,
    jutsu,
    sign_display_queue,
    sign_max_display,
    jutsu_display_time,
    jutsu_font_size_ratio,
    lang_offset,
    jutsu_index,
    jutsu_start_time,
):
    frame_width, frame_height = debug_image.shape[1], debug_image.shape[0]

    # 印のバウンディングボックスの重畳表示(表示オプション有効時) ###################
    if not erase_bbox:
        num_detections = result_inference['num_detections']
        for i in range(num_detections):
            score = result_inference['detection_scores'][i]
            bbox = result_inference['detection_boxes'][i]
            class_id = result_inference['detection_classes'][i].astype(np.int)

            # 検出閾値未満のバウンディングボックスは捨てる
            if score < score_th:
                continue

            x1, y1 = int(bbox[1] * frame_width), int(bbox[0] * frame_height)
            x2, y2 = int(bbox[3] * frame_width), int(bbox[2] * frame_height)

            # バウンディングボックス(長い辺にあわせて正方形を表示)
            x_len = x2 - x1
            y_len = y2 - y1
            square_len = x_len if x_len >= y_len else y_len
            square_x1 = int(((x1 + x2) / 2) - (square_len / 2))
            square_y1 = int(((y1 + y2) / 2) - (square_len / 2))
            square_x2 = square_x1 + square_len
            square_y2 = square_y1 + square_len
            cv.rectangle(debug_image, (square_x1, square_y1),
                         (square_x2, square_y2), (255, 255, 255), 4)
            cv.rectangle(debug_image, (square_x1, square_y1),
                         (square_x2, square_y2), (0, 0, 0), 2)

            # 印の種類
            font_size = int(square_len / 2)
            debug_image = CvDrawText.puttext(
                debug_image, labels[class_id][1],
                (square_x2 - font_size, square_y2 - font_size), font_path,
                font_size, (185, 0, 0))

            # 検出スコア(表示オプション有効時)
            if use_display_score:
                font_size = int(square_len / 8)
                debug_image = CvDrawText.puttext(
                    debug_image, '{:.3f}'.format(score),
                    (square_x1 + int(font_size / 4),
                     square_y1 + int(font_size / 4)), font_path, font_size,
                    (185, 0, 0))

    # ヘッダー作成：FPS #########################################################
    header_image = np.zeros((int(frame_height / 18), frame_width, 3), np.uint8)
    header_image = CvDrawText.puttext(header_image, "FPS:" + str(fps_result),
                                      (5, 0), font_path,
                                      int(frame_height / 20), (255, 255, 255))

    # フッター作成：印の履歴、および、術名表示 ####################################
    footer_image = np.zeros((int(frame_height / 10), frame_width, 3), np.uint8)

    # 印の履歴文字列生成
    sign_display = ''
    if len(sign_display_queue) > 0:
        for sign_id in sign_display_queue:
            sign_display = sign_display + labels[sign_id][1]

    # 術名表示(指定時間描画)
    if lang_offset == 0:
        separate_string = '・'
    else:
        separate_string = '：'
    if (time.time() - jutsu_start_time) < jutsu_display_time:
        if jutsu[jutsu_index][0] == '':  # 属性(火遁等)の定義が無い場合
            jutsu_string = jutsu[jutsu_index][2 + lang_offset]
        else:  # 属性(火遁等)の定義が有る場合
            jutsu_string = jutsu[jutsu_index][0 + lang_offset] + \
                separate_string + jutsu[jutsu_index][2 + lang_offset]
        footer_image = CvDrawText.puttext(
            footer_image, jutsu_string, (5, 0), font_path,
            int(frame_width / jutsu_font_size_ratio), (255, 255, 255))
    # 印表示
    else:
        footer_image = CvDrawText.puttext(footer_image, sign_display, (5, 0),
                                          font_path,
                                          int(frame_width / sign_max_display),
                                          (255, 255, 255))

    # ヘッダーとフッターをデバッグ画像へ結合 ######################################
    debug_image = cv.vconcat([header_image, debug_image])
    debug_image = cv.vconcat([debug_image, footer_image])

    return debug_image


if __name__ == '__main__':
    main()

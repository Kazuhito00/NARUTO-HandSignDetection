#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy

import cv2
import numpy as np
import onnxruntime


class YoloxONNX(object):
    def __init__(
        self,
        model_path='yolox_nano.onnx',
        input_shape=(416, 416),
        class_score_th=0.3,
        with_p6=False,
        providers=[
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    ):
        # 入力サイズ
        self.input_shape = input_shape

        # 閾値
        self.class_score_th = class_score_th
        self.with_p6 = with_p6

        # モデル読み込み
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            providers=providers,
        )

        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_name = self.onnx_session.get_outputs()[0].name

    def inference(self, image):
        temp_image = copy.deepcopy(image)
        image_height, image_width = image.shape[0], image.shape[1]

        # 前処理
        image, ratio = self._preprocess(temp_image, self.input_shape)

        # 推論実施
        results = self.onnx_session.run(
            None,
            {self.input_name: image[None, :, :, :]},
        )

        # 後処理
        bboxes, scores, class_ids = self._postprocess(
            results[0],
            ratio,
            image_width,
            image_height,
        )

        return bboxes, scores, class_ids

    def _preprocess(self, image, input_size, swap=(2, 0, 1)):
        if len(image.shape) == 3:
            padded_image = np.ones(
                (input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_image = np.ones(input_size, dtype=np.uint8) * 114

        ratio = min(input_size[0] / image.shape[0],
                    input_size[1] / image.shape[1])
        resized_image = cv2.resize(
            image,
            (int(image.shape[1] * ratio), int(image.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        )
        resized_image = resized_image.astype(np.uint8)

        padded_image[:resized_image.shape[0], :resized_image.shape[1]] = resized_image
        padded_image = padded_image.transpose(swap)
        padded_image = np.ascontiguousarray(padded_image, dtype=np.float32)

        return padded_image, ratio

    def _postprocess(
        self,
        dets: np.ndarray,
        ratio,
        max_width: int,
        max_height: int,
    ):
        bbox = np.array([])
        score = np.array([])
        class_id = np.array([])
        if dets is not None and dets.shape[0] >= 1:
            class_ids, scores, bboxes = dets[..., 1:2], dets[..., 2:3], dets[..., 3:]
            keep_idx = np.argmax(scores, axis=0)
            class_id = class_ids[keep_idx, ...]
            score = scores[keep_idx, ...]
            bbox = bboxes[keep_idx, ...][0]
            bbox /= ratio
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            bbox[2] = min(bbox[2], max_width)
            bbox[3] = min(bbox[3], max_height)
            bbox = bbox[np.newaxis, :]

        return bbox.astype(np.float32), score.astype(np.float32), class_id.astype(np.int32)

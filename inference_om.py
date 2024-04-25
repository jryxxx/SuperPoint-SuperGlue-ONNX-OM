import cv2
import onnx
import numpy as np
import torch
import math
import random

import sys 
sys.path.append("/home/thtf/test/test_python/acllite")
from acllite.acllite_resource import AclLiteResource
from acllite.acllite_model import AclLiteModel
from acllite.acllite_utils import display_time

class SuperPoint:
    def __init__(self, onnx_model_path):
        self.session = AclLiteModel(onnx_model_path)
        # self.input_name = self.session.get_inputs()[0].name
        # self.output_names = [output.name for output in self.session.get_outputs()]
        # image_width: 320 image_height: 240
        self.resize = (320, 240)
        self.max_keypoints = 500
        self.remove_borders = 4
        self.keypoint_threshold = 0.004

    def preprocess_input(self, image):
        resized_image = image.astype(np.float32) / 255.0
        # 添加 batch 和 channel 维度
        input_data = np.expand_dims(resized_image, axis=(0, 1))
        return input_data

    def find_keypoints(self, scores, threshold=0.004):
        keypoints = []
        new_scores = []
        for idx, score in enumerate(scores):
            if score > self.keypoint_threshold:
                y = idx // self.resize[0]
                x = idx % self.resize[0]
                keypoints.append([x, y])
                new_scores.append(score)
        return keypoints, new_scores

    def normalize_keypoints(self, keypoints, h, w, s):
        keypoints_norm = []
        for kp in keypoints:
            x_norm = (kp[0] - s / 2 + 0.5) / (w * s - s / 2 - 0.5) * 2 - 1
            y_norm = (kp[1] - s / 2 + 0.5) / (h * s - s / 2 - 0.5) * 2 - 1
            keypoints_norm.append([x_norm, y_norm])
        return keypoints_norm

    def clip(self, val, max_val):
        return max(0, min(val, max_val - 1))

    def grid_sample(self, input, grid, dim, h, w):

        output = []
        for g in grid:
            x = (g[0] + 1) * (w - 1) / 2
            y = (g[1] + 1) * (h - 1) / 2

            x_nw = self.clip(int(np.floor(x)), w)
            y_nw = self.clip(int(np.floor(y)), h)

            x_ne = self.clip(x_nw + 1, w)
            y_ne = self.clip(y_nw, h)

            x_sw = self.clip(x_nw, w)
            y_sw = self.clip(y_nw + 1, h)

            x_se = self.clip(x_nw + 1, w)
            y_se = self.clip(y_nw + 1, h)

            nw = (x_se - x) * (y_se - y)
            ne = (x - x_sw) * (y_sw - y)
            sw = (x_ne - x) * (y - y_ne)
            se = (x - x_nw) * (y - y_nw)

            descriptor = []
            for i in range(dim):
                val_nw = input[i, int(y_nw * w + x_nw)]
                val_ne = input[i, int(y_ne * w + x_ne)]
                val_sw = input[i, int(y_sw * w + x_sw)]
                val_se = input[i, int(y_se * w + x_se)]
                descriptor.append(val_nw * nw + val_ne * ne + val_sw * sw + val_se * se)
            output.append(descriptor)
        return output

    def normalize_descriptors(self, descriptors):
        normalized_descriptors = []
        for desc in descriptors:
            norm = np.sqrt(np.dot(desc, desc))
            normalized_desc = [val / norm for val in desc]
            normalized_descriptors.append(normalized_desc)
        return normalized_descriptors

    def sample_descriptors(self, keypoints, descriptors, dim, h, w, s):
        keypoints_norm = self.normalize_keypoints(keypoints, h, w, s)
        output_descriptors = self.grid_sample(descriptors, keypoints_norm, dim, h, w)
        normalized_descriptors = self.normalize_descriptors(output_descriptors)
        return normalized_descriptors

    def sort_indexes(self, data):
        # 创建索引列表，索引范围为 0 到 len(data)-1
        indexes = list(range(len(data)))

        # 使用 lambda 函数根据 data 中元素的值进行降序排序
        indexes.sort(key=lambda i: data[i], reverse=True)

        return indexes

    def top_k_keypoints(self, keypoints, scores):
        k = self.max_keypoints
        if k < len(keypoints) and k != -1:
            # 获取排序后的索引列表
            indexes = sort_indexes(scores)

            # 选取前 k 个关键点和分数
            keypoints_top_k = [keypoints[idx] for idx in indexes[:k]]
            scores_top_k = [scores[idx] for idx in indexes[:k]]

            # 更新 keypoints 和 scores 列表
            keypoints[:] = keypoints_top_k
            scores[:] = scores_top_k
        return keypoints, scores

    def remove_borders_fun(self, keypoints, scores):
        keypoints_selected = []
        scores_selected = []
        border = self.remove_borders
        width, height = self.resize
        for i in range(len(keypoints)):
            x = keypoints[i][0]  # x坐标对应列表中的第一个元素
            y = keypoints[i][1]  # y坐标对应列表中的第二个元素

            # 判断关键点是否在图像边界内
            flag_h = (y >= border) and (y < (height - border))
            flag_w = (x >= border) and (x < (width - border))

            if flag_h and flag_w:
                # 将符合条件的关键点坐标和分数加入选中列表
                keypoints_selected.append([x, y])
                scores_selected.append(scores[i])

        # 更新关键点和分数
        keypoints[:] = keypoints_selected
        scores[:] = scores_selected
        return keypoints, scores

    def inference(self, image):
        input_data = self.preprocess_input(image)
        # outputs = self.session.run(self.output_names, {self.input_name: input_data})
        outputs = self.session.execute([input_data])
        scores = outputs[0].reshape(-1)
        descriptors = outputs[1].reshape(256, -1)

        keypoints, scores = self.find_keypoints(scores, self.keypoint_threshold)
        keypoints, scores = self.remove_borders_fun(keypoints, scores)
        keypoints, scores = self.top_k_keypoints(keypoints, scores)

        descriptors = self.sample_descriptors(
            keypoints,
            descriptors,
            256,
            self.resize[1] / 8,
            self.resize[0] / 8,
            8,
        )

        num_features = len(scores)
        num_keypoints = len(keypoints)
        num_descriptors = len(descriptors)

        # 初始化 features
        features = np.zeros((259, num_features))
        # 将 scores_vec 中的值存储到第一行
        for i in range(num_features):
            features[0, i] = scores[i]

        # 将 keypoints_ 中的值存储到第二行和第三行
        for i in range(1, 3):  # i = 1, 2
            for j in range(num_keypoints):
                features[i, j] = keypoints[j][i - 1]

        # 将 descriptors_ 中的值存储到第四行到第 259 行
        for m in range(3, 259):  # m = 3 to 258
            for n in range(num_descriptors):
                features[m, n] = descriptors[n][m - 3]

        return keypoints, descriptors, features

    def visualize_keypoints(self, image, keypoints, output_path):
        image_display = (
            cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if len(image.shape) == 2
            else image.copy()
        )

        for point in keypoints:
            cv2.circle(image_display, (point[0], point[1]), 1, (0, 0, 255), -1)

        cv2.imwrite(output_path, image_display)
        print(f"Visualization saved to {output_path}")


class SuperGlue:
    def __init__(self, onnx_path):
        self.resize = (320, 240)
        self.session = AclLiteModel(onnx_path)

    def infer(self, features0, features1):

        inputs = [
            self.prepare_keypoints_input(features0),
            self.prepare_scores_input(features0),
            self.prepare_descriptors_input(features0),
            self.prepare_keypoints_input(features1),
            self.prepare_scores_input(features1),
            self.prepare_descriptors_input(features1),
        ]

        outputs = self.session.execute(inputs)
        return outputs

    def where_negative_one(self, flag_data, data, size):
        indices = []

        for i in range(size):
            if flag_data[i] == 1:
                indices.append(data[i])
            else:
                indices.append(-1)

        return indices

    def max_matrix(self, data, h, w, dim):
        if dim == 2:
            values = np.zeros(h - 1, dtype=np.float32)
            indices = np.zeros(h - 1, dtype=np.int32)
            for i in range(h - 1):
                max_value = -np.finfo(np.float32).max
                max_indices = 0
                for j in range(w - 1):
                    if max_value < data[i * w + j]:
                        max_value = data[i * w + j]
                        max_indices = j
                values[i] = max_value
                indices[i] = max_indices
            return indices, values
        elif dim == 1:
            values = np.zeros(w - 1, dtype=np.float32)
            indices = np.zeros(w - 1, dtype=np.int32)
            for i in range(w - 1):
                max_value = -np.finfo(np.float32).max
                max_indices = 0
                for j in range(h - 1):
                    if max_value < data[j * w + i]:
                        max_value = data[j * w + i]
                        max_indices = j
                values[i] = max_value
                indices[i] = max_indices
            return indices, values

    def equal_gather(self, indices0, indices1, size):
        mutual = []
        for i in range(size):
            if indices0[indices1[i]] == i:
                mutual.append(1)
            else:
                mutual.append(0)
        return mutual

    def where_exp(self, flag_data, data, size):
        mscores0 = []
        for i in range(size):
            if flag_data[i] == 1:
                mscores0.append(math.exp(data[i]))
            else:
                mscores0.append(0.0)
        return mscores0

    def where_gather(self, flag_data, indices, mscores0, size):
        mscores1 = []
        for i in range(size):
            if flag_data[i] == 1:
                mscores1.append(mscores0[indices[i]])
            else:
                mscores1.append(0.0)
        return mscores1

    def and_threshold(self, mutual0, mscores0, threshold):
        valid0 = []
        size = len(mscores0)
        for i in range(size):
            if mutual0[i] == 1 and mscores0[i] > threshold:
                valid0.append(1)
            else:
                valid0.append(0)
        return valid0

    def and_gather(self, mutual1, valid0, indices1, size):
        valid1 = []
        size = len(mutual1)
        for i in range(size):
            if mutual1[i] == 1 and valid0[indices1[i]] == 1:
                valid1.append(1)
            else:
                valid1.append(0)
        return valid1

    def decode(self, scores, h, w):
        scores = [element for row in scores for element in row]
        max_indices0, max_values0 = self.max_matrix(scores, h, w, 2)
        max_indices1, max_values1 = self.max_matrix(scores, h, w, 1)
        # debug
        print("max_indices0:", len(max_indices0))
        print("max_indices1:", len(max_indices1))

        mutual0 = self.equal_gather(max_indices1, max_indices0, h - 1)
        mutual1 = self.equal_gather(max_indices0, max_indices1, w - 1)
        # debug
        print("mutual0:", len(mutual0))
        print("mutual1:", len(mutual1))
        mscores0 = self.where_exp(mutual0, max_values0, h - 1)
        mscores1 = self.where_gather(mutual1, max_indices1, mscores0, w - 1)
        valid0 = self.and_threshold(mutual0, mscores0, 0.2)
        valid1 = self.and_gather(mutual1, valid0, max_indices1, w - 1)
        # debug
        print("valid0:", len(valid0))
        print("valid1:", len(valid1))
        indices0 = self.where_negative_one(valid0, max_indices0, h - 1)
        indices1 = self.where_negative_one(valid1, max_indices1, w - 1)
        return indices0, indices1, mscores0, mscores1

    def process_output(self, scores):
        # 从 list 中取出数据
        scores = scores[0][0]
        # debug
        print("scores:", scores.shape)
        h, w = scores.shape
        indices0, indices1, mscores0, mscores1 = self.decode(scores, h, w)
        return indices0, indices1, mscores0, mscores1

    def matching_points(self, features0, features1, outlier_rejection=True):
        width, height = self.resize[0], self.resize[1]
        norm_features0 = self.normalize_keypoints(features0, width, height)
        norm_features1 = self.normalize_keypoints(features1, width, height)

        # Perform keypoints matching logic similar to C++ implementation
        scores = self.infer(norm_features0, norm_features1)
        indices0, indices1, mscores0, mscores1 = self.process_output(scores)

        matches = []
        num_matches = 0
        points0, points1 = [], []
        index = 0

        for i in range(len(indices0)):
            if (
                indices0[i] < len(indices1)
                and indices0[i] >= 0
                and indices1[indices0[i]] == i
            ):
                d = 1.0 - (mscores0[i] + mscores1[indices0[i]]) / 2.0
                matches.append(cv2.DMatch(index, index, d))
                points0.append(cv2.KeyPoint(features0[1, i], features0[2, i], 5))
                points1.append(
                    cv2.KeyPoint(
                        features1[1, indices0[i]], features1[2, indices0[i]], 5
                    )
                )
                index += 1
                num_matches += 1

        return matches, points0, points1

    def normalize_keypoints(self, features, width, height):
        norm_features = features.copy()
        norm_features[1, :] = (features[1, :] - width / 2) / (max(width, height) * 0.7)
        norm_features[2, :] = (features[2, :] - height / 2) / (max(width, height) * 0.7)
        return norm_features


    def prepare_keypoints_input(self, features):
        # keypoints = np.zeros((1, features.shape[1], 2), dtype=np.float32)
        w, h = features.shape
        keypoints = np.zeros((1, 512, 2), dtype=np.float32)
        keypoints[:, : h, 0] = features[1, :]  # x coordinates
        keypoints[:, : h, 1] = features[2, :]  # y coordinates
        return keypoints

    def prepare_scores_input(self, features):
        # edit
        w, h = features.shape
        scores = np.zeros(512, dtype=np.float32)
        scores[: h] = features[0, :].reshape(1, -1).astype(np.float32)
        return scores

    def prepare_descriptors_input(self, features):
        # descriptors = np.zeros((1, 256, features.shape[1]), dtype=np.float32)
        w, h = features.shape
        descriptors = np.zeros((1, 256, 512), dtype=np.float32)
        descriptors[0, :, : h] = features[3:, :].astype(np.float32)
        return descriptors


if __name__ == "__main__":
    # 初始化
    acl_resource = AclLiteResource()
    acl_resource.init()
    # 设置模型路径
    superpoint_onnx_model_path = "/home/thtf/test/test_python/weights/superpoint.om"
    print(superpoint_onnx_model_path)

    # 创建 SuperPoint 实例并加载模型
    superpoint = SuperPoint(superpoint_onnx_model_path)

    # 加载输入图像
    input_image_path_0 = "image/image0.png"
    input_image_0 = cv2.imread(input_image_path_0, cv2.IMREAD_GRAYSCALE)
    input_image_resized_0 = cv2.resize(input_image_0, (320, 240))

    input_image_path_1 = "image/image1.png"
    input_image_1 = cv2.imread(input_image_path_1, cv2.IMREAD_GRAYSCALE)
    input_image_resized_1 = cv2.resize(input_image_1, (320, 240))

    # 进行推理
    keypoints_0, descriptors_0, features_0 = superpoint.inference(input_image_resized_0)
    superpoint.visualize_keypoints(
        input_image_resized_0, keypoints_0, "image/test_1_res.png"
    )
    keypoints_1, descriptors_1, features_1 = superpoint.inference(input_image_resized_1)
    superpoint.visualize_keypoints(
        input_image_resized_1, keypoints_1, "image/test_2_res.png"
    )

    # debug
    print("features_0.shape:", features_0.shape)
    print("features_1.shape:", features_1.shape)

    # 设置模型路径
    superglue_onnx_model_path = "/home/thtf/test/test_python/weights/superglue.om"
    superglue = SuperGlue(superglue_onnx_model_path)

    matches, feature_points0, feature_points1 = superglue.matching_points(
        features_0, features_1
    )

    print("Matching points found:", len(matches))
    # 保存结果
    image0 = cv2.imread(input_image_path_0)
    image0 = cv2.resize(image0, (320, 240))
    image1 = cv2.imread(input_image_path_1)
    image1 = cv2.resize(image1, (320, 240))
    cv2.imwrite("image/test_0_resized.png", image0)
    cv2.imwrite("image/test_1_resized.png", image1)
    # 创建空白的图像用于绘制匹配结果
    match_image = np.zeros(
        (max(image0.shape[0], image1.shape[0]), image0.shape[1] + image1.shape[1], 3),
        dtype=np.uint8,
    )

    # draw resule example 1
    # Draw keypoints and matches
    keypoints_img0 = cv2.drawKeypoints(
        image0, feature_points0, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    keypoints_img1 = cv2.drawKeypoints(
        image1, feature_points1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    match_image[: image0.shape[0], : image0.shape[1]] = keypoints_img0
    match_image[: image1.shape[0], image0.shape[1] :] = keypoints_img1

    for match in matches:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        idx0 = match.queryIdx
        idx1 = match.trainIdx
        (x0, y0) = feature_points0[idx0].pt
        (x1, y1) = feature_points1[idx1].pt
        pt1 = (int(x0), int(y0))
        pt2 = (int(x1) + image0.shape[1], int(y1))
        cv2.line(match_image, pt1, pt2, color, 1)

    cv2.imwrite("image/matches.png", match_image)
    # cv2.imshow("Matches", match_image)
    # cv2.waitKey(0)

    # # draw resule example 2
    # res = cv2.drawMatches(
    #     image0,
    #     feature_points0,
    #     image1,
    #     feature_points0,
    #     matches,
    #     None,
    #     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    # )
    # cv2.imshow("Matches", res)
    # cv2.waitKey(0)

    # 仿射变换矩阵
    # print("feature_points0:", feature_points0[1].pt)
    # print("feature_points1:", feature_points1[1].pt)
    # feature_points0 = np.array([point.pt for point in feature_points0])
    # feature_points1 = np.array([point.pt for point in feature_points1])
    # affine_matrix, _ = cv2.estimateAffine2D(feature_points0, feature_points1)
    # image_transformed = cv2.warpAffine(image0, affine_matrix, (320, 240))
    # cv2.imwrite("image/transformed_image.png", image_transformed)
    # cv2.imshow("Transformed Image", image_transformed)
    # cv2.waitKey(0)

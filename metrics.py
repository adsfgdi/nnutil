import os
import json
import uuid
from typing import Protocol, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from PIL import Image
from shapely.geometry import Polygon as ShapelyPolygon


class Shape(Protocol):
    def iou(self, other: "Shape") -> float: ...
    def union(self, other: "Shape") -> "Shape": ...
    def draw(self, ax, conf: Union[float, None], color: str = "blue"): ...


@dataclass
class Category:
    index: int
    name: str


class Coords:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class Box:
    def __init__(self, top_left: Coords, bottom_right: Coords):
        self.top_left = top_left
        self.bottom_right = bottom_right

    def iou(self, other: "Shape") -> float:
        if not isinstance(other, Box):
            raise Exception(
                "iou can only be calculated between shapes of the same type"
            )

        inter_area = self._intersection_area(other)
        union_area = self._area() + other._area() - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def union(self, other: "Shape") -> "Shape":
        if not isinstance(other, Box):
            raise Exception(
                "union can only be calculated between shapes of the same type"
            )

        return Box(
            top_left=self._top_left_union(other),
            bottom_right=self._bottom_right_union(other),
        )

    def draw(self, ax, conf: Union[float, None], color: str = "blue"):
        width = self.bottom_right.x - self.top_left.x
        height = self.bottom_right.y - self.top_left.y

        rect = patches.Rectangle(
            (self.top_left.x, self.top_left.y),
            width,
            height,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
            alpha=0.7,
        )
        ax.add_patch(rect)

        if conf is not None:
            x = self.top_left.x
            y = self.top_left.y - 5
            bbox_props = dict(boxstyle="round,pad=0.3", fc=color, ec="none", alpha=0.7)
            ax.text(x, y, f"{conf:.2f}", color="white", fontsize=8, bbox=bbox_props)

    def _intersection_area(self, other: "Box") -> float:
        inter_top_left = self._top_left_inter(other)
        inter_bottom_right = self._bottom_right_inter(other)

        inter_box = Box(inter_top_left, inter_bottom_right)
        if not inter_box._is_valid():
            return 0

        return inter_box._area()

    def _top_left_union(self, other: "Box") -> Coords:
        return Coords(
            min(self.top_left.x, other.top_left.x),
            min(self.top_left.y, other.top_left.y),
        )

    def _bottom_right_union(self, other: "Box") -> Coords:
        return Coords(
            max(self.bottom_right.x, other.bottom_right.x),
            max(self.bottom_right.y, other.bottom_right.y),
        )

    def _top_left_inter(self, other: "Box") -> Coords:
        return Coords(
            max(self.top_left.x, other.top_left.x),
            max(self.top_left.y, other.top_left.y),
        )

    def _bottom_right_inter(self, other: "Box") -> Coords:
        return Coords(
            min(self.bottom_right.x, other.bottom_right.x),
            min(self.bottom_right.y, other.bottom_right.y),
        )

    def _area(self) -> float:
        width = self.bottom_right.x - self.top_left.x
        height = self.bottom_right.y - self.top_left.y
        return width * height

    def _is_valid(self) -> bool:
        return (
            self.bottom_right.x > self.top_left.x
            and self.bottom_right.y > self.top_left.y
        )


class Polygon:
    def __init__(self, points: list[tuple[float, float]]):
        polygon = ShapelyPolygon(points)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        if polygon.geom_type == "MultiPolygon":
            largest = max(polygon.geoms, key=lambda g: g.area)  # type:ignore
            polygon = ShapelyPolygon(list(largest.exterior.coords))

        self._polygon = polygon

    def iou(self, other: "Shape") -> float:
        if not isinstance(other, Polygon):
            raise Exception(
                "union can only be calculated between shapes of the same type"
            )

        inter_area = self._intersection_area(other)
        union_area = self._area() + other._area() - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def union(self, other: "Shape") -> "Shape":
        if not isinstance(other, Polygon):
            raise Exception(
                "union can only be calculated between shapes of the same type"
            )

        new_polygon = self._polygon.union(other._polygon)
        if new_polygon.geom_type == "MultiPolygon":
            largest = max(new_polygon.geoms, key=lambda g: g.area)  # type:ignore
            return Polygon(list(largest.exterior.coords))

        return Polygon(list(new_polygon.exterior.coords))  # type:ignore

    def draw(self, ax, conf: Union[float, None], color: str = "blue"):
        x, y = self._polygon.exterior.xy
        path_data = list(zip(x, y))
        path = mpath.Path(path_data, closed=True)
        patch = patches.PathPatch(
            path, facecolor="none", edgecolor=color, lw=2, alpha=0.7
        )
        ax.add_patch(patch)

        if conf is not None and len(x) > 0 and len(y) > 0:
            x_text = x[0]
            y_text = y[0] - 5
            bbox_props = dict(boxstyle="round,pad=0.3", fc=color, ec="none", alpha=0.7)
            ax.text(
                x_text,
                y_text,
                f"{conf:.2f}",
                color="white",
                fontsize=8,
                bbox=bbox_props,
            )

    def _intersection_area(self, other: "Polygon") -> float:
        return self._polygon.intersection(other._polygon).area

    def _area(self) -> float:
        return self._polygon.area

    def is_valid(self) -> bool:
        return not self._polygon.is_empty


class Prediction:
    def __init__(self, id: str, shape: Shape, class_name: str, conf: float):
        self.id = id
        self.shape = shape
        self.conf = conf
        self.class_name = class_name

    def __eq__(self, other):
        if not isinstance(other, Prediction):
            return False
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


class Annotation:
    def __init__(self, id: str, shape: Shape, class_name: str):
        self.id = id
        self.shape = shape
        self.class_name = class_name

    def __eq__(self, other):
        if not isinstance(other, Annotation):
            return False
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


class ConfusionMatrix:
    def __init__(self, tp=0, fp=0, fn=0):
        self.tp = tp
        self.fp = fp
        self.fn = fn

    def accum(self, matrix: "ConfusionMatrix"):
        self.tp += matrix.tp
        self.fp += matrix.fp
        self.fn += matrix.fn

    def F(self, beta: float = 1) -> float:
        precision = self.P()
        recall = self.R()

        if precision + recall == 0:
            return 0

        return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

    def P(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0

    def R(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0

    def __repr__(self):
        return f"P: {self.P() * 100:.2f}%, R: {self.R() * 100:.2f}%, F2: {self.F(2)}"


@dataclass
class AlgoResult:
    tp: list[Prediction]
    fp: list[Prediction]
    fn: list[Annotation]


class DupsAreErrorsAlgo:
    def __init__(self, iou_thres: float):
        self.iou_thres = iou_thres

    def run(
        self, predictions: list[Prediction], annotations: list[Annotation]
    ) -> AlgoResult:
        tp = []
        fp = []

        matched_annotations = set()
        matched_predictions = set()

        for prediction in predictions:
            best_match = self._find_best_match(
                prediction, annotations, matched_annotations
            )
            if best_match:
                tp.append(prediction)
                matched_annotations.add(best_match)
                matched_predictions.add(prediction)
            else:
                fp.append(prediction)

        fn = [anno for anno in annotations if anno not in matched_annotations]

        return AlgoResult(tp=tp, fp=fp, fn=fn)

    def _find_best_match(
        self,
        prediction: Prediction,
        annotations: list[Annotation],
        matched_annotations: set,
    ) -> Union[Annotation, None]:
        best_match = None
        best_iou = 0.0

        for annotation in annotations:
            if annotation in matched_annotations:
                continue

            if prediction.class_name != annotation.class_name:
                continue

            iou = prediction.shape.iou(annotation.shape)
            if iou >= self.iou_thres and iou > best_iou:
                best_iou = iou
                best_match = annotation

        return best_match


class OverlappingAlgo:
    """
    Если одно предсказание покрывает более одной фактической коробки, не засчитывает FN.
    Если более одного предсказания покрывают единственную фактич. коробку, не засчитывает FP.
    """

    def __init__(self, iou_thres: float):
        self.iou_thres = iou_thres

    def run(
        self, predictions: list[Prediction], annotations: list[Annotation]
    ) -> AlgoResult:
        matched_predictions, matched_annotations = self._get_matches(
            predictions, annotations
        )

        # True Positive
        tp = list(matched_predictions)

        # False Positive
        fp = []
        for prediction in predictions:
            if prediction not in matched_predictions:
                fp.append(prediction)

        # False Negative
        fn = []
        for annotation in annotations:
            if annotation not in matched_annotations:
                fn.append(annotation)

        return AlgoResult(tp=tp, fp=fp, fn=fn)

    def _get_matches(
        self, predictions: list[Prediction], annotations: list[Annotation]
    ) -> tuple[set, set]:
        matched_annotations = set()
        matched_predictions = set()

        for prediction in predictions:
            for annotation in annotations:
                if prediction.class_name != annotation.class_name:
                    continue

                if prediction.shape.iou(annotation.shape) >= self.iou_thres:
                    matched_annotations.add(annotation)
                    matched_predictions.add(prediction)

        return matched_predictions, matched_annotations


class Algo(Protocol):
    def run(
        self, predictions: list[Prediction], annotations: list[Annotation]
    ) -> AlgoResult: ...


@dataclass
class Result:
    preds: list[Prediction]
    annotations: list[Annotation]
    image_path: str


class Source(Protocol):
    def load_results(self) -> list[Result]: ...
    def classes(self) -> list[str]: ...


class PredictionDeduplicator:
    def __init__(self, source: Source, iou_thres: float):
        self.source = source
        self.iou_thres = iou_thres

    def load_results(self) -> list[Result]:
        filtered = []
        for res in self.source.load_results():
            res = Result(
                preds=self._drop_duplicates(res.preds),
                annotations=res.annotations,
                image_path=res.image_path,
            )
            filtered.append(res)

        return filtered

    def classes(self) -> list[str]:
        return self.source.classes()

    def _drop_duplicates(self, predictions: list[Prediction]) -> list[Prediction]:
        predictions = sorted(predictions, key=lambda pred: pred.conf, reverse=True)

        unique_preds = []
        for pred in predictions:
            is_duplicate = any(
                pred.class_name == existing_pred.class_name
                and pred.shape.iou(existing_pred.box) >= self.iou_thres
                for existing_pred in unique_preds
            )
            if not is_duplicate:
                unique_preds.append(pred)

        return unique_preds


class PredictionMerger:
    def __init__(self, source: Source, iou_thres: float):
        self.source = source
        self.iou_thres = iou_thres

    def load_results(self) -> list[Result]:
        filtered = []
        for res in self.source.load_results():
            res = Result(
                preds=self._merge_duplicates(res.preds),
                annotations=res.annotations,
                image_path=res.image_path,
            )
            filtered.append(res)

        return filtered

    def classes(self) -> list[str]:
        return self.source.classes()

    def _merge_duplicates(self, predictions: list[Prediction]) -> list[Prediction]:
        predictions = sorted(predictions, key=lambda pred: pred.conf, reverse=True)

        merged_preds = []
        while predictions:
            base_pred = predictions.pop(0)

            overlapping = [base_pred]
            non_overlapping = []

            for pred in predictions:
                if (
                    pred.class_name == base_pred.class_name
                    and base_pred.shape.iou(pred.shape) >= self.iou_thres
                ):
                    overlapping.append(pred)
                else:
                    non_overlapping.append(pred)

            merged_box = self._merge_boxes([pred.shape for pred in overlapping])

            merged_preds.append(
                Prediction(
                    base_pred.id,
                    merged_box,
                    base_pred.class_name,
                    base_pred.conf,
                )
            )

            predictions = non_overlapping

        return merged_preds

    def _merge_boxes(self, boxes: list[Shape]):
        merged_box = boxes[0]
        for box in boxes[1:]:
            merged_box = merged_box.union(box)
        return merged_box


class CachedSource:
    def __init__(self, source: Source):
        self._source = source
        self._cached_results = None

    def load_results(self) -> list[Result]:
        if self._cached_results is None:
            self._cached_results = self._source.load_results()
        return self._cached_results

    def classes(self) -> list[str]:
        return self._source.classes()


class CocoSource:
    def __init__(
        self,
        preds_file_path: Union[str, Path],
        annotations_file_path: Union[str, Path],
        conf_thres: float,
        images_dir_path: Union[str, Path] = "./",
        load_polygons: bool = False,
    ):
        self.preds_file_path = preds_file_path
        self.annotations_file_path = annotations_file_path
        self.conf_thres = conf_thres
        self.images_dir_path = images_dir_path
        self.load_polygons = load_polygons

    def load_results(self) -> list[Result]:
        classes_map = self._get_classes_map()
        preds_by_image = self._load_predictions(classes_map)
        annotations_by_image = self._load_annotations(classes_map)
        pathes_by_image_id = self._load_paths(self.images_dir_path)

        results = []
        for image_id, image_path in pathes_by_image_id.items():
            preds = preds_by_image.get(image_id, [])
            annotations = annotations_by_image.get(image_id, [])
            results.append(
                Result(
                    preds=preds,
                    annotations=annotations,
                    image_path=image_path,
                )
            )

        return results

    def classes(self) -> list[str]:
        return list(self._get_classes_map().values())

    def _get_classes_map(self) -> dict[int, str]:
        with open(self.annotations_file_path, "r") as f:
            data = json.load(f)

        categories = {}
        for item in data["categories"]:
            categories[item["id"]] = item["name"]

        return categories

    def _load_paths(self, images_dir_path: Union[str, Path]) -> dict[int, str]:
        pathes_by_image_id = {}

        with open(self.annotations_file_path, "r") as f:
            data = json.load(f)

        for item in data["images"]:
            image_id = item["id"]
            path = os.path.join(images_dir_path, item["file_name"])
            pathes_by_image_id[image_id] = path

        return pathes_by_image_id

    def _load_annotations(
        self, classes_map: dict[int, str]
    ) -> dict[int, list[Annotation]]:
        with open(self.annotations_file_path, "r") as f:
            data = json.load(f)

        annos_by_image = {}
        for item in data["annotations"]:
            image_id = item["image_id"]
            class_name = classes_map[item["category_id"]]

            if self.load_polygons and "segmentation" in item and item["segmentation"]:
                shape = self._make_polygon(item["segmentation"])
            else:
                x, y, width, height = map(float, item["bbox"])
                top_left = Coords(x, y)
                bottom_right = Coords(x + width, y + height)
                shape = Box(top_left, bottom_right)

            anno = Annotation(
                id=str(uuid.uuid4()),
                shape=shape,
                class_name=class_name,
            )
            annos_by_image.setdefault(image_id, []).append(anno)

        return annos_by_image

    def _load_predictions(
        self, classes_map: dict[int, str]
    ) -> dict[int, list[Prediction]]:
        with open(self.preds_file_path, "r") as f:
            data = json.load(f)

        boxes_by_image = {}
        for item in data:
            conf = item["score"]
            if conf < self.conf_thres:
                continue
            image_id = item["image_id"]
            class_name = classes_map[item["category_id"]]

            if self.load_polygons and "segmentation" in item and item["segmentation"]:
                shape = self._make_polygon(item["segmentation"])
            else:
                x, y, width, height = map(float, item["bbox"])
                top_left = Coords(x, y)
                bottom_right = Coords(x + width, y + height)
                shape = Box(top_left, bottom_right)

            pred = Prediction(
                id=str(uuid.uuid4()),
                shape=shape,
                class_name=class_name,
                conf=conf,
            )
            boxes_by_image.setdefault(image_id, []).append(pred)

        return boxes_by_image

    def _make_polygon(self, segmentation: list[list[float]]) -> Polygon:
        # Предполагается, что всегда ровно один список координат
        seg = segmentation[0]
        coords = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
        return Polygon(coords)


class UltralyticsSource:
    def __init__(
        self,
        preds_dir: str,
        actual_data_dir: str,
        conf_thres: float,
        categories: list[Category],
        load_polygons: bool = False,
    ):
        self.preds_dir = preds_dir
        self.actual_data_dir = actual_data_dir
        self.conf_thres = conf_thres
        self.classes_map = {c.index: c.name for c in categories}
        self.load_polygons = load_polygons

    def load_results(self) -> list[Result]:
        results = []

        annotations_dir = os.path.join(self.actual_data_dir, "labels")
        for filename in os.listdir(annotations_dir):
            if filename.endswith(".txt"):
                ann_file = os.path.join(annotations_dir, filename)
                pred_file = os.path.join(self.preds_dir, filename)

                image_name = filename[:-4] + ".jpeg"
                image_path = os.path.join(self.actual_data_dir, "images", image_name)

                with Image.open(image_path) as img:
                    img_width, img_height = img.size

                annotations = self._load_annotations(ann_file, img_width, img_height)
                predictions = self._load_predictions(pred_file, img_width, img_height)

                res = Result(
                    preds=predictions, annotations=annotations, image_path=image_path
                )
                results.append(res)

        return results

    def classes(self) -> list[str]:
        return list(self.classes_map.values())

    def _load_annotations(
        self, ann_file: str, img_width: int, img_height: int
    ) -> list[Annotation]:
        annotations = []
        with open(ann_file, "r") as f:
            for line in f:
                floats = list(map(float, line.strip().split()))
                class_idx = int(floats[0])

                if not self.load_polygons:
                    if len(floats) != 5:
                        raise Exception("wrong bbox format")

                    x_center, y_center, width, height = floats[1:]
                    shape = self._convert_to_box(
                        x_center, y_center, width, height, img_width, img_height
                    )
                else:
                    if len(floats) < 3:
                        raise Exception("wrong polygons format")
                    shape = self._convert_to_polygon(floats[1:], img_width, img_height)

                anno = Annotation(
                    id=str(uuid.uuid4()),
                    shape=shape,
                    class_name=self.classes_map[class_idx],
                )
                annotations.append(anno)
        return annotations

    def _load_predictions(
        self, pred_file: str, img_width: int, img_height: int
    ) -> list[Prediction]:
        if not os.path.exists(pred_file):
            return []

        predictions = []
        with open(pred_file, "r") as f:
            for line in f:
                floats = list(map(float, line.strip().split()))
                class_idx = int(floats[0])

                if not self.load_polygons:
                    if len(floats) != 6:
                        raise Exception("wrong bbox format")

                    x_center, y_center, width, height, conf = floats[1:]
                    if conf < self.conf_thres:
                        continue

                    shape = self._convert_to_box(
                        x_center, y_center, width, height, img_width, img_height
                    )
                else:
                    if len(floats) < 4:
                        raise Exception("wrong polygons format")

                    conf = floats[-1]
                    if conf < self.conf_thres:
                        continue

                    coords = floats[1:-1]
                    if len(coords) < 2:
                        raise Exception("wrong polygons format")

                    shape = self._convert_to_polygon(coords, img_width, img_height)

                pred = Prediction(
                    id=str(uuid.uuid4()),
                    shape=shape,
                    class_name=self.classes_map[class_idx],
                    conf=conf if self.load_polygons else floats[-1],
                )
                predictions.append(pred)
        return predictions

    def _convert_to_box(
        self,
        x_center: float,
        y_center: float,
        width: float,
        height: float,
        img_width: int,
        img_height: int,
    ) -> Box:
        x_min = (x_center - width / 2) * img_width
        y_min = (y_center - height / 2) * img_height
        x_max = (x_center + width / 2) * img_width
        y_max = (y_center + height / 2) * img_height
        return Box(top_left=Coords(x_min, y_min), bottom_right=Coords(x_max, y_max))

    def _convert_to_polygon(
        self, coords: list[float], img_width: int, img_height: int
    ) -> Polygon:
        points = []
        for i in range(0, len(coords), 2):
            x_norm = coords[i]
            y_norm = coords[i + 1]
            x_abs = x_norm * img_width
            y_abs = y_norm * img_height
            points.append((x_abs, y_abs))
        return Polygon(points)


def _filter_result(res: Result, cls: str):
    return Result(
        preds=[p for p in res.preds if p.class_name == cls],
        annotations=[a for a in res.annotations if a.class_name == cls],
        image_path=res.image_path,
    )


def calculate(algo: Algo, source: Source) -> dict[str, ConfusionMatrix]:
    total = {}
    results = source.load_results()
    classes = source.classes()

    for cls in classes:
        total_per_class = ConfusionMatrix()
        for res in results:
            res_filtered = _filter_result(res, cls)
            algo_res = algo.run(res_filtered.preds, res_filtered.annotations)

            matrix = ConfusionMatrix(
                tp=len(algo_res.tp),
                fp=len(algo_res.fp),
                fn=len(algo_res.fn),
            )
            total_per_class.accum(matrix)

        total[f"{cls}"] = total_per_class

    return total


@dataclass
class SourceInput:
    name: str
    class_name: str
    source: Source


class Displayer:
    def __init__(
        self,
        fp_color: str = "red",
        fn_color: str = "red",
        only_errors: bool = False,
        save_dir: str = "",
    ):
        self.fp_color = fp_color
        self.fn_color = fn_color
        self.save_dir = save_dir
        self.only_errors = only_errors

    def display2(self, algo: Algo, source: Source, limit: Union[int, None] = None):
        self._display_results(
            algo=algo,
            results=source.load_results(),
            limit=limit,
            class_colors=self._generate_class_colors(source.classes()),
            titles=(
                f"Annotations (FN in {self.fn_color})",
                f"Predictions (FP in {self.fp_color})",
                "Combined (by class)",
            ),
        )

    def display(
        self,
        algo: Algo,
        source: Source,
        class_name: str,
        limit: Union[int, None] = None,
    ):
        results = source.load_results()
        classes = source.classes()
        if class_name not in classes:
            raise Exception(f"{class_name} not exists")

        filtered_results = [_filter_result(res, class_name) for res in results]

        self._display_results(
            algo=algo,
            results=filtered_results,
            limit=limit,
            class_colors={class_name: "green"},
            titles=(
                f"Annotations (FN in {self.fn_color})",
                f"Predictions (FP in {self.fp_color})",
                "Combined (Preds in blue)",
            ),
        )

    def _display_results(
        self,
        algo: Algo,
        results: list,
        limit: Union[int, None],
        class_colors: dict,
        titles: tuple,
    ):
        displayed_count = 0

        for res in results:
            if limit is not None and displayed_count >= limit:
                break

            algo_res = algo.run(res.preds, res.annotations)
            if self.only_errors and len(algo_res.fp) + len(algo_res.fn) == 0:
                continue

            with Image.open(res.image_path) as img:
                image = np.array(img)

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
            ax1.imshow(image)
            ax1.set_title(titles[0])
            ax2.imshow(image)
            ax2.set_title(titles[1])
            ax3.imshow(image)
            ax3.set_title(titles[2])

            self._draw_anno_boxes(ax1, res.annotations, algo_res, class_colors)
            self._draw_pred_boxes(ax2, algo_res, class_colors)
            self._draw_anno_and_pred_boxes(
                ax3, res.annotations, algo_res.tp + algo_res.fp
            )

            if self.save_dir:
                self._save(res.image_path)
            else:
                plt.show()

            plt.close(fig)

            displayed_count += 1

    def _save(self, image_path: str):
        os.makedirs(self.save_dir, exist_ok=True)
        plt.savefig(
            f"{self._get_save_path(image_path)}.jpeg",
            bbox_inches="tight",
            pad_inches=0.2,
        )

    def _get_save_path(self, image_path: str):
        name = os.path.splitext(os.path.basename(image_path))[0] + "_res"
        return os.path.join(self.save_dir, name)

    def _draw_anno_boxes(
        self,
        ax,
        annotations: list[Annotation],
        algo_res: AlgoResult,
        colors: dict[str, str],
    ):
        fn_ids = {fn.id for fn in algo_res.fn}

        for ann in annotations:
            color = self.fn_color if ann.id in fn_ids else colors[ann.class_name]
            ann.shape.draw(ax, conf=None, color=color)

    def _draw_pred_boxes(self, ax, algo_res: AlgoResult, colors: dict[str, str]):
        for pred in algo_res.fp:
            pred.shape.draw(ax, conf=pred.conf, color=self.fp_color)
        for pred in algo_res.tp:
            pred.shape.draw(ax, conf=pred.conf, color=colors[pred.class_name])

    def _draw_anno_and_pred_boxes(
        self,
        ax,
        annos: list[Annotation],
        preds: list[Prediction],
        anno_color: str = "green",
        pred_color: str = "blue",
    ):
        for ann in annos:
            ann.shape.draw(ax, conf=None, color=anno_color)
        for pred in preds:
            pred.shape.draw(ax, conf=pred.conf, color=pred_color)

    def _generate_class_colors(self, classes: list[str]) -> dict[str, str]:
        """Генерирует уникальные цвета для каждого класса"""
        if len(classes) <= 10:
            cmap = cm.get_cmap("tab10")
            colors = [mcolors.rgb2hex(cmap(i)) for i in range(len(classes))]
        elif len(classes) <= 20:
            cmap = cm.get_cmap("tab20")
            colors = [mcolors.rgb2hex(cmap(i)) for i in range(len(classes))]
        else:
            cmap = cm.get_cmap("hsv")
            colors = [
                mcolors.rgb2hex(cmap(i / len(classes))) for i in range(len(classes))
            ]

        return dict(zip(classes, colors))

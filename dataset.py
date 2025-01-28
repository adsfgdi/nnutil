import json
import os
import math
import random
from dataclasses import dataclass
from PIL import Image as PILImage
import shutil
from typing import Protocol


@dataclass
class Category:
    index: int
    name: str


class Coords:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"[{self.x} {self.y}]"


class Box:
    def __init__(self, top_left: Coords, bottom_right: Coords):
        self.top_left = top_left
        self.bottom_right = bottom_right

    def __repr__(self) -> str:
        return f"BboxGeometry({self.top_left}, {self.bottom_right})"

    def area(self) -> float:
        width = self.bottom_right.x - self.top_left.x
        height = self.bottom_right.y - self.top_left.y
        return width * height

    def to_yolo_bbox_str(
        self, image_width: float, image_height: float, class_idx: int
    ) -> str:
        x_min = self.top_left.x / image_width
        y_min = self.top_left.y / image_height
        x_max = self.bottom_right.x / image_width
        y_max = self.bottom_right.y / image_height
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        box_width = x_max - x_min
        box_height = y_max - y_min
        return f"{class_idx} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"

    def to_yolo_seg_str(
        self, image_width: float, image_height: float, class_idx: int
    ) -> str:
        x0 = self.top_left.x / image_width
        y0 = self.top_left.y / image_height
        x1 = self.bottom_right.x / image_width
        y1 = self.top_left.y / image_height
        x2 = self.bottom_right.x / image_width
        y2 = self.bottom_right.y / image_height
        x3 = self.top_left.x / image_width
        y3 = self.bottom_right.y / image_height
        coords_str = (
            f"{x0:.6f} {y0:.6f} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f}"
        )
        return f"{class_idx} {coords_str}"

    def to_coco_dict(self, image_id: int, annotation_id: int, category_id: int) -> dict:
        x_min = self.top_left.x
        y_min = self.top_left.y
        width = self.bottom_right.x - self.top_left.x
        height = self.bottom_right.y - self.top_left.y
        return {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x_min, y_min, width, height],
            "area": self.area(),
            "iscrowd": 0,
            "segmentation": [],
        }


class Polygon:
    def __init__(self, points: list[Coords]):
        self.points = points

    def __repr__(self) -> str:
        return f"PolygonGeometry({self.points})"

    def area(self) -> float:
        area = 0.0
        n = len(self.points)
        for i in range(n):
            j = (i + 1) % n
            area += (self.points[i].x * self.points[j].y) - (
                self.points[j].x * self.points[i].y
            )
        return abs(area) / 2.0

    def to_yolo_bbox_str(
        self, image_width: float, image_height: float, class_idx: int
    ) -> str:
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        x_min = min(xs) / image_width
        y_min = min(ys) / image_height
        x_max = max(xs) / image_width
        y_max = max(ys) / image_height
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        box_width = x_max - x_min
        box_height = y_max - y_min
        return f"{class_idx} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"

    def to_yolo_seg_str(
        self, image_width: float, image_height: float, class_idx: int
    ) -> str:
        coords_norm = []
        for p in self.points:
            coords_norm.append(p.x / image_width)
            coords_norm.append(p.y / image_height)
        coords_str = " ".join(f"{c:.6f}" for c in coords_norm)
        return f"{class_idx} {coords_str}"

    def to_coco_dict(self, image_id: int, annotation_id: int, category_id: int) -> dict:
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        width = x_max - x_min
        height = y_max - y_min
        segmentation_coords = []
        for p in self.points:
            segmentation_coords.append(p.x)
            segmentation_coords.append(p.y)
        return {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x_min, y_min, width, height],
            "area": self.area(),
            "iscrowd": 0,
            "segmentation": [segmentation_coords],
        }


class Geometry(Protocol):
    def area(self) -> float: ...

    def to_yolo_bbox_str(
        self, image_width: float, image_height: float, class_idx: int
    ) -> str: ...

    def to_yolo_seg_str(
        self, image_width: float, image_height: float, class_idx: int
    ) -> str: ...

    def to_coco_dict(
        self, image_id: int, annotation_id: int, category_id: int
    ) -> dict: ...


class Annotation:
    def __init__(self, geometry: Geometry, class_name: str):
        self.geometry = geometry
        self.class_name = class_name

    def __repr__(self) -> str:
        return f"Annotation(class={self.class_name}, geometry={self.geometry})"


class Image:
    def __init__(
        self,
        id: str,
        annotations: list[Annotation],
        width: float,
        height: float,
        filename: str,
    ):
        self.id = id
        self.annotations = annotations
        self.width = width
        self.height = height
        self.filename = filename

    def __repr__(self) -> str:
        return f"Image(id={self.id}, annotations={self.annotations})"


class Dataset:
    def __init__(self, images: list[Image] = [], categories: list[Category] = []):
        self.images = images
        self.category_to_index = {c.name: c.index for c in categories}

    def update(self, dataset: "Dataset"):
        """
        Обновляет собственные изображения и аннотации,
        если в новом датасете есть совпадающий id, то обновляем аннотации,
        иначе удаляем такие картинки (которых нет в dataset).
        """
        id_to_image = {image.id: image for image in dataset.images}
        images_to_remove = []
        for image in self.images:
            if image.id in id_to_image:
                image.annotations = id_to_image[image.id].annotations
            else:
                images_to_remove.append(image.id)
        for image_id in images_to_remove:
            self._remove_image(image_id)

    def merge(self, dataset: "Dataset"):
        """
        Объединяет с другим датасетом
        """
        ids = {img.id for img in self.images}
        for image in dataset.images:
            if image.id not in ids:
                self.images.append(image)
                ids.add(image.id)
        for category_name, category_index in dataset.category_to_index.items():
            if category_name not in self.category_to_index:
                self.category_to_index[category_name] = category_index

    def get_missing_images(self, dataset: "Dataset") -> list[Image]:
        """
        Вернёт список изображений, которые отсутствуют в self, но есть в dataset.
        """
        current_ids = {img.id for img in self.images}
        return [img for img in dataset.images if img.id not in current_ids]

    def random_split(self, val_size: float, test_size: float) -> "DatasetSplit":
        """
        Случайным образом делит self.images на train/val/test.
        """
        if val_size + test_size > 1.0:
            raise ValueError("val_size and test_size must sum to 1.0 or less.")
        shuffled_images = random.sample(self.images, len(self.images))
        imgs_num = len(self.images)
        val_num = math.ceil(imgs_num * val_size)
        test_num = math.ceil(imgs_num * test_size)
        val_images = shuffled_images[:val_num]
        test_images = shuffled_images[val_num : val_num + test_num]
        train_images = shuffled_images[val_num + test_num :]
        categories = [
            Category(index=i, name=c) for c, i in self.category_to_index.items()
        ]
        return DatasetSplit(
            train=Dataset(train_images, categories),
            val=Dataset(val_images, categories),
            test=Dataset(test_images, categories),
        )

    def save_yolo_det(self, images_dir: str, save_dir: str, mkdirs=False):
        return self._save_yolo(images_dir, save_dir, mkdirs, False)

    def save_yolo_seg(self, images_dir: str, save_dir: str, mkdirs=False):
        return self._save_yolo(images_dir, save_dir, mkdirs, True)

    def _save_yolo(self, images_dir: str, save_dir: str, mkdirs=False, seg=False):
        """
        Сохранение датасета в YOLO-формате.
        """
        if os.path.exists(save_dir) and len(os.listdir(save_dir)) != 0:
            raise Exception("save dir path exists and not empty")
        imgs_save_dir = os.path.join(save_dir, "images")
        labels_save_dir = os.path.join(save_dir, "labels")
        if mkdirs:
            os.makedirs(imgs_save_dir, exist_ok=True)
            os.makedirs(labels_save_dir, exist_ok=True)
        for image in self.images:
            orig_image_name = os.path.basename(image.filename)
            original_path = os.path.join(images_dir, orig_image_name)
            target_image_path = os.path.join(imgs_save_dir, orig_image_name)
            if os.path.exists(original_path):
                shutil.copy(original_path, target_image_path)
            orig_name_no_ext, _ = os.path.splitext(orig_image_name)
            yolo_label_name = orig_name_no_ext + ".txt"
            yolo_label_path = os.path.join(labels_save_dir, yolo_label_name)
            with open(yolo_label_path, "w") as label_file:
                for annotation in image.annotations:
                    cls_idx = self.category_to_index[annotation.class_name]
                    if seg:
                        line_str = annotation.geometry.to_yolo_seg_str(
                            image.width, image.height, cls_idx
                        )
                    else:
                        line_str = annotation.geometry.to_yolo_bbox_str(
                            image.width, image.height, cls_idx
                        )
                    label_file.write(line_str + "\n")
        print(f"[INFO] Dataset saved: {save_dir}")

    def save_coco(self, save_path: str, mkdirs=False):
        """
        Сохранение датасета в COCO-формате.
        Поддерживаются bbox и полигон (сегментация) через соответствующие методы geometry.
        """
        annotations = []
        images = []
        ann_id = 0
        for image_id, image in enumerate(self.images):
            dumped_image = {
                "id": image_id,
                "width": image.width,
                "height": image.height,
                "file_name": image.filename,
            }
            images.append(dumped_image)
            for anno in image.annotations:
                coco_anno = anno.geometry.to_coco_dict(
                    image_id=image_id,
                    annotation_id=ann_id,
                    category_id=self.category_to_index[anno.class_name],
                )
                ann_id += 1
                annotations.append(coco_anno)
        coco_dataset = {
            "info": {},
            "licenses": [],
            "categories": [
                {"id": v, "name": k} for k, v in self.category_to_index.items()
            ],
            "images": images,
            "annotations": annotations,
        }
        if os.path.exists(save_path):
            backup_path = f"{save_path}.bak"
            os.rename(save_path, backup_path)
            print(f"[INFO] Backup created: {backup_path}")
        if mkdirs:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(coco_dataset, f, indent=2)
            print(f"[INFO] Dataset saved: {save_path}")

    @classmethod
    def from_coco(cls, coco_path: str) -> "Dataset":
        with open(coco_path, "r") as f:
            coco_data = json.load(f)
        categories = [
            Category(index=cat["id"], name=cat["name"])
            for cat in coco_data["categories"]
        ]
        index_to_category = {cat.index: cat.name for cat in categories}
        id_to_image = {}
        for img in coco_data["images"]:
            image_id_str = os.path.basename(img["file_name"]).split(".")[0]
            id_to_image[img["id"]] = Image(
                id=image_id_str,
                annotations=[],
                width=img["width"],
                height=img["height"],
                filename=os.path.basename(img["file_name"]),
            )
        for anno in coco_data["annotations"]:
            image_id = anno["image_id"]
            category_id = anno["category_id"]
            top_left = Coords(anno["bbox"][0], anno["bbox"][1])
            bottom_right = Coords(
                anno["bbox"][0] + anno["bbox"][2],
                anno["bbox"][1] + anno["bbox"][3],
            )
            geometry = Box(top_left, bottom_right)
            class_name = index_to_category[category_id]
            annotation = Annotation(geometry=geometry, class_name=class_name)
            id_to_image[image_id].annotations.append(annotation)
        images = list(id_to_image.values())
        return cls(images=images, categories=categories)

    @classmethod
    def from_yolo_det(cls, images_dir: str, categories: list[Category]) -> "Dataset":
        return cls._from_yolo(images_dir, categories, "detection")

    @classmethod
    def from_yolo_seg(cls, images_dir: str, categories: list[Category]) -> "Dataset":
        return cls._from_yolo(images_dir, categories, "segmentation")

    @classmethod
    def _from_yolo(
        cls, images_dir: str, categories: list[Category], mode: str = "detection"
    ) -> "Dataset":
        """
        mode="detection"  => ожидаем строго 4 координаты (bbox)
        mode="segmentation" => ожидаем полигоны (любое четное кол-во координат)
        """
        images: list[Image] = []
        cat_idx_map = {c.index: c.name for c in categories}

        for image_name in os.listdir(images_dir):
            ext = os.path.splitext(image_name)[1].lower()
            if ext not in [".jpg", ".jpeg", ".png"]:
                continue
            image_path = os.path.join(images_dir, image_name)
            image_width, image_height = PILImage.open(image_path).size

            # Формируем путь к txt (заменяем "images" на "labels" по аналогии с YOLO-структурой)
            yolo_annotations_path = (
                os.path.splitext(image_path)[0].replace("images", "labels") + ".txt"
            )
            annotations: list[Annotation] = []

            if os.path.exists(yolo_annotations_path):
                with open(yolo_annotations_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        cls_idx = int(parts[0])
                        coords = list(map(float, parts[1:]))

                        if cls_idx not in cat_idx_map:
                            raise Exception(f"Unknown class index {cls_idx}")

                        if mode == "detection":
                            if len(coords) != 4:
                                raise Exception(
                                    f"Expected 4 coords for bbox, got {len(coords)}. Line: {line}"
                                )
                            x_center, y_center, box_width, box_height = coords
                            top_left = Coords(
                                x=(x_center - box_width / 2) * image_width,
                                y=(y_center - box_height / 2) * image_height,
                            )
                            bottom_right = Coords(
                                x=(x_center + box_width / 2) * image_width,
                                y=(y_center + box_height / 2) * image_height,
                            )
                            geometry = Box(top_left, bottom_right)

                        elif mode == "segmentation":
                            if len(coords) % 2 != 0:
                                raise Exception(f"Invalid polygon line: {line}")
                            points = []
                            for i in range(0, len(coords), 2):
                                x_abs = coords[i] * image_width
                                y_abs = coords[i + 1] * image_height
                                points.append(Coords(x_abs, y_abs))
                            geometry = Polygon(points)

                        else:
                            raise ValueError(f"Unknown mode: {mode}")

                        class_name = cat_idx_map[cls_idx]
                        annotations.append(
                            Annotation(geometry=geometry, class_name=class_name)
                        )

            images.append(
                Image(
                    id=image_name.split(".")[0],
                    annotations=annotations,
                    width=image_width,
                    height=image_height,
                    filename=image_name,
                )
            )
        return cls(images=images, categories=categories)

    def _remove_image(self, image_id: str):
        self.images = [img for img in self.images if img.id != image_id]


class DatasetSplit:
    def __init__(self, train: Dataset, val: Dataset, test: Dataset):
        self.train = train
        self.val = val
        self.test = test

    def update(self, dataset: "Dataset", split="train"):
        """
        Обновляет текущие разбиения, добавляя новые изображения в split(default="train"),
        удаляя отсутствующие изображения из всех разбиений, и обновляя аннотации.

        Args:
            dataset (Dataset): Новый датасет для обновления разбиений.
        """
        if split not in ["train", "val", "test"]:
            raise Exception('split should be one of ["train", "val", "test"]')
        self.train.update(dataset)
        self.val.update(dataset)
        self.test.update(dataset)
        new_images = self._get_missing_images(dataset)
        if split == "train":
            self.train.images.extend(new_images)
        elif split == "val":
            self.val.images.extend(new_images)
        else:
            self.test.images.extend(new_images)

    def save_coco(self, train_path: str, val_path: str, test_path: str, mkdirs=False):
        self.train.save_coco(train_path, mkdirs=mkdirs)
        self.val.save_coco(val_path, mkdirs=mkdirs)
        self.test.save_coco(test_path, mkdirs=mkdirs)

    @classmethod
    def random_from_yolo_det(
        cls,
        images_dir: str,
        categories: list[Category],
        val_size: float,
        test_size: float,
    ) -> "DatasetSplit":
        return Dataset.from_yolo_det(images_dir, categories).random_split(
            val_size=val_size, test_size=test_size
        )

    @classmethod
    def random_from_yolo_seg(
        cls,
        images_dir: str,
        categories: list[Category],
        val_size: float,
        test_size: float,
    ) -> "DatasetSplit":
        return Dataset.from_yolo_seg(images_dir, categories).random_split(
            val_size=val_size, test_size=test_size
        )

    def save_yolo_seg(
        self,
        train_images_dir: str,
        val_images_dir: str,
        test_images_dir: str,
        train_path: str,
        val_path: str,
        test_path: str,
        mkdirs=False,
    ):
        self.train.save_yolo_seg(train_images_dir, train_path, mkdirs=mkdirs)
        self.val.save_yolo_seg(val_images_dir, val_path, mkdirs=mkdirs)
        self.test.save_yolo_seg(test_images_dir, test_path, mkdirs=mkdirs)

    def save_yolo_det(
        self,
        train_images_dir: str,
        val_images_dir: str,
        test_images_dir: str,
        train_path: str,
        val_path: str,
        test_path: str,
        mkdirs=False,
    ):
        self.train.save_yolo_det(train_images_dir, train_path, mkdirs=mkdirs)
        self.val.save_yolo_det(val_images_dir, val_path, mkdirs=mkdirs)
        self.test.save_yolo_det(test_images_dir, test_path, mkdirs=mkdirs)

    @classmethod
    def from_coco(
        cls, train_path: str, val_path: str, test_path: str
    ) -> "DatasetSplit":
        return cls(
            train=Dataset.from_coco(train_path),
            val=Dataset.from_coco(val_path),
            test=Dataset.from_coco(test_path),
        )

    @classmethod
    def from_yolo_det(
        cls,
        train_images_dir: str,
        val_images_dir: str,
        test_images_dir: str,
        categories: list[Category],
    ) -> "DatasetSplit":
        return cls(
            train=Dataset.from_yolo_det(train_images_dir, categories),
            val=Dataset.from_yolo_det(val_images_dir, categories),
            test=Dataset.from_yolo_det(test_images_dir, categories),
        )

    @classmethod
    def from_yolo_seg(
        cls,
        train_images_dir: str,
        val_images_dir: str,
        test_images_dir: str,
        categories: list[Category],
    ) -> "DatasetSplit":
        return cls(
            train=Dataset.from_yolo_seg(train_images_dir, categories),
            val=Dataset.from_yolo_seg(val_images_dir, categories),
            test=Dataset.from_yolo_seg(test_images_dir, categories),
        )

    def _get_missing_images(self, dataset: "Dataset") -> list[Image]:
        self_dataset = Dataset()
        for ds in [self.train, self.val, self.test]:
            self_dataset.merge(ds)
        return self_dataset.get_missing_images(dataset)

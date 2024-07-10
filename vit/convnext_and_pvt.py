import timm
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import v2 as transforms
from PIL import Image, UnidentifiedImageError
from pycocotools.coco import COCO
import os
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from ranger import Ranger
from ranger.ranger2020 import Ranger as Ranger2020
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou

torch.backends.cudnn.benchmark = False

DEBUG = True

use_segmentation = False  # Set this variable to True for segmentation, False for bounding boxes only


# Paths to the directories and annotation files
which_host = {'macbook': '/Users/mattsloan/Downloads/FLC2019/',
              'linux': '/home/matt/clovers/data/FLC2019/',
              'colab': '/content/flc/1280_flc/',
              'colab_full': '/content/FLC2019/',
              'wsl': '/home/matt/cvt/smaller_flc'}
FLC_ROOT = which_host['wsl']

which_dir_images = {'colab': 'JPEGImages_1280x720',
                    'colab_full': 'JPEGImages',
                    'linux': 'JPEGImages',
                    'macbook:': 'JPEGImages'}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f"Using device: {device}")
num_epochs = 50
training_stats_per_epoch = {'train_loss': [], 'eval_precision': [], 'eval_recall': [], 'eval_f1_score': [], 'avg_iou': [], 'avg_dice': []}

batch_size = 2

state_dict_path = 'convnext_states/faster_rcnn_v2_base_14_loss2.0592.pth'

class PVTBackbone(torch.nn.Module):
    def __init__(self, model_name='pvt_v2_b0', freeze_backbone=False, freeze_percentage=0.0):
        super(PVTBackbone, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, features_only=True, out_indices=[3])

        if freeze_backbone or freeze_percentage > 0:
            layers = list(self.backbone.parameters())
            num_layers = len(layers)
            freeze_layers = int(num_layers * freeze_percentage)
            for param in layers[-freeze_layers:]:
                param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)[0]
        return x

class ConvNeXTBackbone(torch.nn.Module): # theres a nano at 640? tiny 768...pico 512?..small is 768...base 1024
    def __init__(self, model_name='convnextv2_base', freeze_backbone=False, freeze_percentage=0.0):
        super(ConvNeXTBackbone, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, features_only=True, out_indices=[3])

        if freeze_backbone or freeze_percentage > 0:
            layers = list(self.backbone.parameters())
            num_layers = len(layers)
            freeze_layers = int(num_layers * freeze_percentage)
            for param in layers[-freeze_layers:]:
                param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)[0]
        return x

backbone = ConvNeXTBackbone(freeze_backbone=False)
backbone.out_channels = 1024  # Adjust according to the backbone output channels # 

# backbone = PVTBackbone(freeze_backbone=True)#freeze_percentage=0.5)#freeze_backbone=True)#freeze_percentage=0.5)  # Use PVT backbone
# backbone.out_channels = 256# 512  # Adjust according to the PVT output channels : 256 for b0

class CloverDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None, is_positive=True, use_segmentation=True):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.is_positive = is_positive
        self.use_segmentation = use_segmentation

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])

        if not os.path.exists(img_path):
            print(f"Image {img_path} does not exist.")
            return None

        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, UnidentifiedImageError) as e:
            print(f"Error loading image {img_path}: {e}")
            return None

        image, scale_factor = self.resize_image(image, min_size=800, max_size=1333)
        resized_width, resized_height = image.size

        if self.transform:
            image = self.transform(image)

        if self.is_positive:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            boxes = []
            labels = []
            masks = []
            for ann in anns:
                bbox = ann['bbox']
                scaled_bbox = [
                    bbox[0] * scale_factor[0],
                    bbox[1] * scale_factor[1],
                    bbox[2] * scale_factor[0],
                    bbox[3] * scale_factor[1]
                ]
                boxes.append([scaled_bbox[0], scaled_bbox[1], scaled_bbox[0] + scaled_bbox[2], scaled_bbox[1] + scaled_bbox[3]])
                labels.append(ann['category_id'])
                if self.use_segmentation:
                    mask = self.coco.annToMask(ann)
                    mask = Image.fromarray(mask).resize((resized_width, resized_height), Image.NEAREST)
                    mask = torch.as_tensor(np.array(mask), dtype=torch.uint8)
                    masks.append(mask)

            if not boxes:
                if DEBUG: print(f"No valid annotations for image {img_path}")
                return None

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([img_id])}
            if self.use_segmentation:
                masks = torch.stack(masks)
                target['masks'] = masks
        else:
            target = {'boxes': torch.empty((0, 4), dtype=torch.float32), 'labels': torch.empty((0,), dtype=torch.int64), 'image_id': torch.tensor([img_id])}
            if self.use_segmentation:
                target['masks'] = torch.empty((0, resized_height, resized_width), dtype=torch.uint8)

        return image, target

    def resize_image(self, image, min_size=800, max_size=1333):
        w, h = image.size
        scale_factor = min_size / min(w, h)
        if w < h:
            ow = min_size
            oh = int(scale_factor * h)
        else:
            oh = min_size
            ow = int(scale_factor * w)

        if max(ow, oh) > max_size:
            scale_factor = max_size / max(ow, oh)
            ow = int(scale_factor * ow)
            oh = int(scale_factor * oh)

        image = image.resize((ow, oh), Image.BILINEAR)
        return image, (ow / w, oh / h)


train_images_dir = os.path.join(FLC_ROOT, f'trainval/{which_dir_images["colab_full"]}')
train_pos_annotation_file = os.path.join(FLC_ROOT, 'trainval/coco_annotations/instances_trainval_pos.json')
train_neg_annotation_file = os.path.join(FLC_ROOT, 'trainval/coco_annotations/instances_trainval_negs.json')
test_images_dir = os.path.join(FLC_ROOT, f'test/{which_dir_images["colab_full"]}')
test_annotation_file = os.path.join(FLC_ROOT, 'test/coco_annotations/instances_test_pos.json')

class SaltAndPepperNoise:
    def __init__(self, salt_prob=0.0025, pepper_prob=0.0025, max_width=2, max_height=3):
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob
        self.max_width = max_width
        self.max_height = max_height

    def __call__(self, tensor):
        tensor = tensor.clone()

        c, h, w = tensor.shape

        num_salt = int(self.salt_prob * tensor.numel())
        for _ in range(num_salt):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            patch_width = random.randint(1, self.max_width)
            patch_height = random.randint(1, self.max_height)
            tensor[:, max(0, y-patch_height//2):min(h, y+patch_height//2+1), max(0, x-patch_width//2):min(w, x+patch_width//2+1)] = 1.0

        num_pepper = int(self.pepper_prob * tensor.numel())
        for _ in range(num_pepper):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            patch_width = random.randint(1, self.max_width)
            patch_height = random.randint(1, self.max_height)
            tensor[:, max(0, y-patch_height//2):min(h, y+patch_height//2+1), max(0, x-patch_width//2):min(w, x+patch_width//2+1)] = 0.0

        return tensor

class ClampTransform:
    def __init__(self, min_val=0.0, max_val=1.0):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, tensor):
        return torch.clamp(tensor, min=self.min_val, max=self.max_val)

train_transform = transforms.Compose([
    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.01, 4))], p=0.2),
    # transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.RandomApply([SaltAndPepperNoise()], p=0.333),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ClampTransform(min_val=0.0, max_val=1.0),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ClampTransform(min_val=0.0, max_val=1.0),
])

transform = train_transform

pos_dataset = CloverDataset(root_dir=train_images_dir, annotation_file=train_pos_annotation_file, transform=train_transform, is_positive=True, use_segmentation=use_segmentation)
neg_dataset = CloverDataset(root_dir=train_images_dir, annotation_file=train_neg_annotation_file, transform=train_transform, is_positive=False, use_segmentation=use_segmentation)

num_pos_samples = len(pos_dataset)
num_neg_samples = int(num_pos_samples * 1.0)
combined_indices = list(range(len(neg_dataset)))
random.shuffle(combined_indices)
combined_indices = combined_indices[:num_neg_samples]

pos_indices = list(range(num_pos_samples))
all_indices = pos_indices + combined_indices
random.shuffle(all_indices)

class CombinedCloverDataset(Dataset):
    def __init__(self, pos_dataset, neg_dataset, indices):
        self.pos_dataset = pos_dataset
        self.neg_dataset = neg_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        if actual_idx < len(self.pos_dataset):
            return self.pos_dataset[actual_idx]
        else:
            return self.neg_dataset[actual_idx - len(self.pos_dataset)]

combined_dataset = CombinedCloverDataset(pos_dataset, neg_dataset, all_indices)

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return [], []
    return tuple(zip(*batch))

train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size, collate_fn=collate_fn, persistent_workers=False, pin_memory=False)
test_dataset = CloverDataset(root_dir=test_images_dir, annotation_file=test_annotation_file, transform=test_transform, use_segmentation=use_segmentation)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn, persistent_workers=False, pin_memory=False)

rpn_anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),) * len((32, 64, 128, 256, 512))
)

roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=['0'], output_size=7, sampling_ratio=2
)

mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=['0'], output_size=14, sampling_ratio=2
)

if use_segmentation:
    model = MaskRCNN(backbone, num_classes=2,
                     rpn_anchor_generator=rpn_anchor_generator,
                     box_roi_pool=roi_pooler,
                     mask_roi_pool=mask_roi_pooler)
else:
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=2)

model.to(device)
print(model)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = Ranger(params, lr=0.0015, weight_decay=0.005)
# optimizer = torch.optim.Adam(params, lr=0.00015, weight_decay=0.005)

# patience = int(len(train_loader) * 0.33)
# 10% of epochs
patience = int(num_epochs * 0.1)
if patience < 1:
    patience = 1
# patience += 1
if DEBUG: print(f'{patience = }')
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.5)
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=patience, T_mult=2, eta_min=0.00001)

def create_checkered_mask(mask):
    mask = mask.squeeze()
    h, w = mask.shape
    checkered_mask = np.zeros((h, w, 4), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            if (i // 10 % 2 == 0 and j // 10 % 2 == 0) or (i // 10 % 2 == 1 and j // 10 % 2 == 1):
                checkered_mask[i, j] = [1, 0, 0, mask[i, j] * 0.6]
            else:
                checkered_mask[i, j] = [1, 0, 0, 0]

    return checkered_mask

def draw_boxes(image, true_boxes, pred_boxes, true_labels=None, pred_labels=None, true_masks=None, pred_masks=None, epoch=0):
    fig, ax = plt.subplots(1)
    img_np = image.permute(1, 2, 0).cpu().numpy()
    if img_np.max() > 1.0:
        img_np = img_np / 255.0
    ax.imshow(img_np)

    for box in true_boxes:
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2.5, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    for box in pred_boxes:
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='c', facecolor='none', linestyle='--')
        ax.add_patch(rect)

    if true_masks is not None and use_segmentation:
        for mask in true_masks:
            checkered_mask = create_checkered_mask(mask.cpu().numpy())
            ax.imshow(checkered_mask, interpolation='nearest')

    if pred_masks is not None:
        for mask in pred_masks:
            ax.imshow(mask.squeeze(0).cpu().numpy(), cmap='Blues', alpha=0.3)

    handles = [patches.Patch(edgecolor='r', facecolor='none', label='True Box'),
               patches.Patch(edgecolor='c', facecolor='none', linestyle='--', label='Pred Box')]
    
    if use_segmentation:
        handles.append(patches.Patch(facecolor='red', edgecolor='none', alpha=0.3, label='True Mask', hatch='//'))
        handles.append(patches.Patch(facecolor='blue', edgecolor='none', alpha=0.3, label='Pred Mask'))
        
    ax.legend(handles=handles)

    # plt.show()
    plt.savefig(f'/home/matt/cvt/pvt_states/boxes_{epoch}.png')

def custom_loss_function(predictions, targets, device, use_segmentation=True):
    if not any(t['boxes'].numel() > 0 for t in targets):
        return torch.tensor(0.0, device=device)

    batch_size = len(targets)
    extra_pred_penalty = 0
    localization_penalty = 0

    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        true_boxes = target['boxes']

        num_pred = pred_boxes.shape[0]
        num_true = true_boxes.shape[0]
        extra_pred_penalty += max(0, num_pred - num_true) * 2

        if num_pred > 0 and num_true > 0:
            iou_matrix = box_iou(pred_boxes, true_boxes)
            max_ious, _ = torch.max(iou_matrix, dim=1)
            localization_penalty += torch.mean((1 - max_ious) * pred_scores) *1.1

    extra_pred_penalty /= batch_size
    localization_penalty /= batch_size
    total_custom_loss = extra_pred_penalty + localization_penalty

    return total_custom_loss

def filter_predictions_by_score(predictions, score_threshold=0.5, use_segmentation=True):
    filtered_predictions = []
    for pred in predictions:
        scores = pred['scores']
        keep = scores >= score_threshold
        filtered_pred = {
            'boxes': pred['boxes'][keep],
            'labels': pred['labels'][keep],
            'scores': scores[keep]
        }
        if use_segmentation and 'masks' in pred:
            filtered_pred['masks'] = pred['masks'][keep]
        filtered_predictions.append(filtered_pred)
    return filtered_predictions

def train_one_epoch(model, optimizer, data_loader, device, epoch, score_threshold=0.5, use_segmentation=True):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
    scaler = GradScaler()

    for i, (images, targets) in enumerate(progress_bar):
        if not images or not any(t['boxes'].numel() > 0 for t in targets):
            continue

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        if device.type == 'cuda':
            with autocast(device_type='cuda'):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                with torch.no_grad():
                    model.eval()
                    predictions = model(images)
                    filtered_predictions = filter_predictions_by_score(predictions, score_threshold, use_segmentation)
                custom_loss = custom_loss_function(filtered_predictions, targets, device, use_segmentation)
                total_loss = losses + custom_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            with torch.no_grad():
                model.eval()
                predictions = model(images)
                filtered_predictions = filter_predictions_by_score(predictions, score_threshold, use_segmentation)
            custom_loss = custom_loss_function(filtered_predictions, targets, device, use_segmentation)
            total_loss = losses + custom_loss

            total_loss.backward()
            optimizer.step()
        model.train()
        epoch_loss += total_loss.item()
        progress_bar.set_postfix(loss=epoch_loss/(i+1), lr = optimizer.param_groups[0]['lr'])

        percent_to_show = 0.75
        if i % int(len(data_loader) * percent_to_show) == 0 and DEBUG:
            with torch.no_grad():
                model.eval()
                outputs = model(images)
                draw_boxes(images[0], targets[0]['boxes'].cpu(), outputs[0]['boxes'].cpu(), true_masks=targets[0]['masks'] if use_segmentation else None, pred_masks=outputs[0]['masks'] if use_segmentation else None, epoch=epoch)
                model.train()

        # if i > len(data_loader) * 0.1:
        #     current_lr = optimizer.param_groups[0]['lr']
        #     lr_scheduler.step(epoch_loss / (i + 1))
        #     after_lr = optimizer.param_groups[0]['lr']
            
        #     if math.isclose(after_lr, current_lr, rel_tol=0.05):
        #         pass
        #     else:
        #         print(f"\n-------=======Learning rate changed to {after_lr:.6f} from {current_lr:.6f}=======-------\n")
        #         optimizer.param_groups[0]['weight_decay'] = optimizer.param_groups[0]['weight_decay'] * 0.975

    print(f"Epoch: {epoch+1}, Loss: {epoch_loss / len(data_loader)}")
    return epoch_loss / len(data_loader)

def calculate_iou(box1, box2):
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])

    inter_area = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou

def calculate_mAP(outputs, targets):
    metric = MeanAveragePrecision()
    metric.update(outputs, targets)
    return metric.compute()

def calculate_mask_metrics(true_mask, pred_mask):
    true_mask = true_mask.to('cpu', dtype=torch.bool)
    pred_mask = pred_mask.to('cpu', dtype=torch.bool)

    intersection = (true_mask & pred_mask).float().sum()
    union = (true_mask | pred_mask).float().sum()
    iou = intersection / union

    dice = (2 * intersection) / (true_mask.float().sum() + pred_mask.float().sum())

    return iou.item(), dice.item()

def calculate_metrics(outputs, targets, iou_threshold=0.5, use_segmentation=True):
    TP, FP, FN = 0, 0, 0
    total_iou, total_dice, num_masks = 0, 0, 0

    for output, target in zip(outputs, targets):
        pred_boxes = output['boxes'].to('cpu')
        pred_scores = output['scores'].to('cpu')
        true_boxes = target['boxes'].to('cpu')

        pred_labels = torch.zeros(pred_boxes.shape[0], device='cpu')
        true_labels = torch.zeros(true_boxes.shape[0], device='cpu')

        ious = torch.zeros((len(pred_boxes), len(true_boxes)), device='cpu')

        for i, pred_box in enumerate(pred_boxes):
            for j, true_box in enumerate(true_boxes):
                iou = calculate_iou(pred_box, true_box)
                ious[i, j] = iou

        for i in range(len(pred_boxes)):
            max_iou_idx = torch.argmax(ious[i])
            if ious[i, max_iou_idx] > iou_threshold:
                if true_labels[max_iou_idx] == 0:
                    TP += 1
                    true_labels[max_iou_idx] = 1
                else:
                    FP += 1
            else:
                FP += 1

        FN += len(true_boxes) - torch.sum(true_labels)

        if use_segmentation and 'masks' in output and 'masks' in target:
            pred_masks = output['masks'].to('cpu')
            true_masks = target['masks'].to('cpu')
            for pred_mask, true_mask in zip(pred_masks, true_masks):
                iou, dice = calculate_mask_metrics(true_mask, pred_mask)
                total_iou += iou
                total_dice += dice
                num_masks += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    avg_iou = total_iou / num_masks if num_masks > 0 else 0
    avg_dice = total_dice / num_masks if num_masks > 0 else 0

    return precision, recall, avg_iou, avg_dice

def evaluate(model, data_loader, device, use_segmentation=True, batch_size=16):
    model.eval()
    all_outputs = []
    all_targets = []
    batch_outputs = []
    batch_targets = []

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            outputs_cpu = [{k: v.to('cpu') for k, v in output.items()} for output in outputs]
            targets_cpu = [{k: v.to('cpu') for k, v in target.items()} for target in targets]

            batch_outputs.extend(outputs_cpu)
            batch_targets.extend(targets_cpu)

            if len(batch_outputs) >= batch_size:
                precision, recall, avg_iou, avg_dice = calculate_metrics(batch_outputs, batch_targets, use_segmentation=use_segmentation)
                all_outputs.extend(batch_outputs)
                all_targets.extend(batch_targets)
                batch_outputs = []
                batch_targets = []

        # Process remaining batches
        if batch_outputs:
            precision, recall, avg_iou, avg_dice = calculate_metrics(batch_outputs, batch_targets, use_segmentation=use_segmentation)
            all_outputs.extend(batch_outputs)
            all_targets.extend(batch_targets)

    precision, recall, avg_iou, avg_dice = calculate_metrics(all_outputs, all_targets, use_segmentation=use_segmentation)
    
    # Calculate F1 score using precision and recall
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

    return precision, recall, avg_iou, avg_dice, f1

try:
    model.load_state_dict(torch.load(
        state_dict_path,
        map_location=device,
        weights_only=False
    ))
except:
    print(f"Error loading model from {state_dict_path}. Starting from scratch.")
    

for epoch in range(num_epochs):
    
    # if half way through training
    if epoch == num_epochs // 3:
        # unfreezing backbone
        print("Unfreezing backbone")
        
        for param in backbone.backbone.parameters():
            param.requires_grad = True
            
            
    epoch_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, use_segmentation=use_segmentation)
    
    lr_scheduler.step(epoch_loss)

    training_stats_per_epoch['train_loss'].append(epoch_loss)
    
    # every other epoch
    if epoch % 2 == 0 or epoch == 0:
        with torch.no_grad():
            if device.type == 'cuda':
                        with autocast(device_type='cuda'):
                            precision, recall, avg_iou, avg_dice, f1 = evaluate(model, test_loader, device, use_segmentation=use_segmentation)

        print(f"Epoch {epoch+1} Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f} Train Loss: {epoch_loss:.4f}")
        training_stats_per_epoch['eval_precision'].append(precision)
        training_stats_per_epoch['eval_recall'].append(recall)
        training_stats_per_epoch['eval_f1_score'].append(f1)
        training_stats_per_epoch['avg_iou'].append(avg_iou)
        training_stats_per_epoch['avg_dice'].append(avg_dice)
    else:
        # append with the data of previous epoch
        training_stats_per_epoch['eval_precision'].append(training_stats_per_epoch['eval_precision'][-1])
        training_stats_per_epoch['eval_recall'].append(training_stats_per_epoch['eval_recall'][-1])
        training_stats_per_epoch['eval_f1_score'].append(training_stats_per_epoch['eval_f1_score'][-1])
        training_stats_per_epoch['avg_iou'].append(training_stats_per_epoch['avg_iou'][-1])
        training_stats_per_epoch['avg_dice'].append(training_stats_per_epoch['avg_dice'][-1])
        
    if (epoch+1) % 5 == 0 or epoch == 0 or epoch == 1:
        if use_segmentation:
            model_end = 'mask_rcnn'
        else:
            model_end = 'faster_rcnn'
        save_dir = f"/home/matt/cvt/convnext_states/{model_end}_v2_base_{epoch}_loss_{epoch_loss:.4f}.pth"
        torch.save(model.state_dict(), save_dir)
        print(f"Model saved to {save_dir}")
        
    plt.close()
        
# clear plot
plt.clf()

plt.plot(training_stats_per_epoch['train_loss'])
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('/home/matt/cvt/pvt_states/training_loss.png')


plt.plot(training_stats_per_epoch['eval_precision'], label='Precision')
plt.plot(training_stats_per_epoch['eval_recall'], label='Recall')
plt.plot(training_stats_per_epoch['eval_f1_score'], label='F1 Score')
plt.plot(training_stats_per_epoch['avg_iou'], label='Avg IoU')
plt.plot(training_stats_per_epoch['avg_dice'], label='Avg Dice')
plt.title('Evaluation Metrics per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.legend()
plt.savefig('/home/matt/cvt/pvt_states/eval_metrics.png')

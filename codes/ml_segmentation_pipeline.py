"""
COMPLETE ML-ENHANCED 3D ROOM SEGMENTATION PIPELINE
This version ACTUALLY trains the model properly!

Training Process:
1. Loop through ALL areas (Area_1, Area_2, ...)
2. Loop through ALL rooms in each area
3. Extract features and collect training data from ground truth
4. After ALL rooms processed: model.fit(X, y)
5. Save trained model to disk

Usage:
    python ml_segmentation_pipeline.py --mode train    # Train on all rooms
    python ml_segmentation_pipeline.py --mode test     # Test on new rooms
"""

import os
import sys
import numpy as np
import pickle
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing_module import PointCloudPreprocessor
from clustering_module import PointCloudClusterer
from visualization_module import PointCloudVisualizer, export_ply


class MLFeatureExtractor:
    """Extract features from point cloud clusters for ML classification"""
    
    @staticmethod
    def extract_cluster_features(points, cluster_mask):
        """
        Extract 18 geometric features from a cluster.
        
        Returns: dict of features or None if cluster too small
        """
        cluster_points = points[cluster_mask]
        
        if len(cluster_points) < 10:
            return None
        
        features = {}
        
        # 1. Basic statistics
        features['size'] = len(cluster_points)
        centroid = cluster_points.mean(axis=0)
        features['centroid_x'] = centroid[0]
        features['centroid_y'] = centroid[1]
        features['centroid_z'] = centroid[2]
        
        # 2. Bounding box dimensions
        min_bound = cluster_points.min(axis=0)
        max_bound = cluster_points.max(axis=0)
        dimensions = max_bound - min_bound
        features['width'] = dimensions[0]
        features['length'] = dimensions[1]
        features['height'] = dimensions[2]
        features['volume'] = np.prod(dimensions) + 1e-6
        
        # 3. PCA-based shape features
        centered = cluster_points - centroid
        cov_matrix = np.cov(centered.T)
        eigenvalues, _ = np.linalg.eig(cov_matrix)
        eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
        
        if eigenvalues[0] > 1e-6:
            features['planarity'] = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0]
            features['linearity'] = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]
            features['sphericity'] = eigenvalues[2] / eigenvalues[0]
        else:
            features['planarity'] = 0
            features['linearity'] = 0
            features['sphericity'] = 0
        
        # 4. Density
        features['density'] = len(cluster_points) / features['volume']
        
        # 5. Shape ratios
        features['vertical_ratio'] = dimensions[2] / (dimensions[0] + dimensions[1] + 1e-6)
        features['aspect_xy'] = dimensions[0] / (dimensions[1] + 1e-6)
        features['aspect_xz'] = dimensions[0] / (dimensions[2] + 1e-6)
        
        # 6. Height information
        features['height_above_floor'] = min_bound[2]
        features['height_range'] = dimensions[2]
        
        return features


class MLSemanticClassifier:
    """Machine Learning classifier for semantic segmentation"""
    
    # Label mapping
    LABELS = {
        'floor': 0, 'ceiling': 1, 'wall': 2, 'door': 3, 'window': 4,
        'table': 5, 'chair': 6, 'sofa': 7, 'bed': 8, 'bookcase': 9,
        'board': 10, 'clutter': 11, 'beam': 12, 'column': 13, 'unknown': 14
    }
    
    LABEL_NAMES = {v: k for k, v in LABELS.items()}
    
    LABEL_COLORS = {
        'floor': [139, 69, 19], 'ceiling': [173, 216, 230], 'wall': [255, 200, 100],
        'door': [139, 90, 43], 'window': [135, 206, 250], 'table': [210, 180, 140],
        'chair': [255, 69, 0], 'sofa': [255, 140, 0], 'bed': [255, 192, 203],
        'bookcase': [160, 82, 45], 'board': [245, 245, 220], 'clutter': [169, 169, 169],
        'beam': [128, 128, 128], 'column': [105, 105, 105], 'unknown': [192, 192, 192]
    }
    
    def __init__(self, model_path='trained_model.pkl'):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        self.model_path = model_path
        
        # Training data accumulation
        self.training_features = []
        self.training_labels = []
    
    def object_name_to_label(self, object_name):
        """Convert object filename to label"""
        name_lower = object_name.lower()
        
        if 'floor' in name_lower:
            return 'floor'
        elif 'ceiling' in name_lower:
            return 'ceiling'
        elif 'wall' in name_lower:
            return 'wall'
        elif 'door' in name_lower:
            return 'door'
        elif 'window' in name_lower:
            return 'window'
        elif 'table' in name_lower:
            return 'table'
        elif 'chair' in name_lower:
            return 'chair'
        elif 'sofa' in name_lower or 'couch' in name_lower:
            return 'sofa'
        elif 'bed' in name_lower:
            return 'bed'
        elif 'bookcase' in name_lower or 'shelf' in name_lower:
            return 'bookcase'
        elif 'board' in name_lower or 'whiteboard' in name_lower:
            return 'board'
        elif 'beam' in name_lower:
            return 'beam'
        elif 'column' in name_lower or 'pillar' in name_lower:
            return 'column'
        elif 'clutter' in name_lower:
            return 'clutter'
        else:
            return 'unknown'
    
    def add_training_sample(self, features_dict, label_name):
        """Add a single training sample"""
        if label_name not in self.LABELS:
            label_name = 'unknown'
        
        self.training_features.append(features_dict)
        self.training_labels.append(self.LABELS[label_name])
    
    def train(self):
        """Train the model on ALL accumulated training data"""
        if len(self.training_features) < 20:
            print(f"⚠️  Not enough training data: {len(self.training_features)} samples (need 20+)")
            return False
        
        print(f"\n{'='*70}")
        print("🎓 TRAINING ML MODEL")
        print(f"{'='*70}")
        print(f"Training samples: {len(self.training_features)}")
        
        # Get feature names
        if self.feature_names is None:
            self.feature_names = list(self.training_features[0].keys())
        
        # Convert to arrays
        X = np.array([[f[name] for name in self.feature_names] 
                      for f in self.training_features])
        y = np.array(self.training_labels)
        
        # Show label distribution
        label_counts = Counter(y)
        print("\n📊 Label distribution:")
        for label_id, count in sorted(label_counts.items()):
            label_name = self.LABEL_NAMES[label_id]
            percentage = count / len(y) * 100
            print(f"  {label_name:15s}: {count:5d} samples ({percentage:5.2f}%)")
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("\n🔄 Training Random Forest...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"\n✓ Training complete!")
        print(f"  Train accuracy: {train_score*100:.2f}%")
        print(f"  Test accuracy: {test_score*100:.2f}%")
        
        # Feature importance
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:5]
        
        print("\n🔝 Top 5 important features:")
        for i, idx in enumerate(indices):
            print(f"  {i+1}. {self.feature_names[idx]:20s}: {importances[idx]:.4f}")
        
        # Classification report
        y_pred = self.model.predict(X_test_scaled)
        print("\n📈 Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=[self.LABEL_NAMES[i] for i in sorted(set(y_test))],
                                   zero_division=0))
        
        self.is_trained = True
        
        # Save model
        self.save_model()
        
        print(f"{'='*70}\n")
        return True
    
    def save_model(self):
        """Save trained model to disk"""
        if not self.is_trained:
            print("⚠️  Model not trained yet, cannot save")
            return
        
        data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'label_mapping': self.LABELS,
            'training_samples': len(self.training_features)
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"💾 Model saved to: {self.model_path}")
    
    def load_model(self):
        """Load pre-trained model from disk"""
        if not os.path.exists(self.model_path):
            print(f"⚠️  Model file not found: {self.model_path}")
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
            
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.is_trained = True
            
            print(f"✓ Loaded model from: {self.model_path}")
            print(f"  Trained on {data.get('training_samples', 'unknown')} samples")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict(self, features_dict):
        """Predict label for a cluster"""
        if not self.is_trained:
            return self.LABELS['unknown']
        
        X = np.array([[features_dict[name] for name in self.feature_names]])
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        return prediction


class MLTrainingPipeline:
    """Complete ML training pipeline"""
    
    def __init__(self, model_path='trained_model.pkl'):
        self.preprocessor = PointCloudPreprocessor()
        self.clusterer = PointCloudClusterer()
        self.classifier = MLSemanticClassifier(model_path=model_path)
        self.visualizer = PointCloudVisualizer()
        
        self.rooms_processed = 0
        self.samples_collected = 0
    
    def collect_training_data_from_room(self, anno_path, room_name, voxel_size=0.02, eps=0.05):
        """
        Process ONE room and collect training data from ground truth.
        Does NOT train the model yet - just collects data.
        """
        print(f"\n{'─'*70}")
        print(f"📂 Processing: {room_name}")
        print(f"{'─'*70}")
        
        # 1. Load data with ground truth labels
        if not os.path.exists(anno_path):
            print(f"❌ Path not found: {anno_path}")
            return False
        
        object_files = sorted([f for f in os.listdir(anno_path) if f.endswith('.txt')])
        
        all_points = []
        all_colors = []
        all_object_ids = []
        object_names = []
        
        for idx, obj_file in enumerate(object_files):
            obj_path = os.path.join(anno_path, obj_file)
            object_name = obj_file.replace('.txt', '')
            
            try:
                data = np.loadtxt(obj_path)
                if len(data.shape) == 1:
                    data = data.reshape(1, -1)
                
                points = data[:, :3]
                colors = data[:, 3:6] if data.shape[1] >= 6 else np.ones((len(points), 3)) * 128
                
                all_points.append(points)
                all_colors.append(colors)
                all_object_ids.extend([idx] * len(points))
                object_names.append(object_name)
                
            except Exception as e:
                print(f"  ⚠️  Error loading {obj_file}: {e}")
        
        if not all_points:
            print("❌ No data loaded")
            return False
        
        points = np.vstack(all_points)
        colors = np.vstack(all_colors)
        object_ids = np.array(all_object_ids)
        
        print(f"  Loaded: {len(points):,} points from {len(object_files)} objects")
        
        # 2. Preprocess
        processed_points, processed_colors = self.preprocessor.preprocess(
            points, colors,
            use_statistical=True,
            use_radius=False,
            voxel_size=voxel_size,
            stat_k=20,
            stat_std=2.0
        )
        
        # 3. For each original object, extract features and add to training
        # (We use original objects as "clusters" since we have ground truth)
        samples_this_room = 0
        
        for idx, obj_name in enumerate(object_names):
            # Find points belonging to this object
            mask = object_ids == idx
            
            if np.sum(mask) < 10:
                continue
            
            # Extract features from this object
            features = MLFeatureExtractor.extract_cluster_features(points, mask)
            
            if features is not None:
                # Get label from object name
                label_name = self.classifier.object_name_to_label(obj_name)
                
                # Add to training data
                self.classifier.add_training_sample(features, label_name)
                samples_this_room += 1
        
        self.rooms_processed += 1
        self.samples_collected += samples_this_room
        
        print(f"  ✓ Collected {samples_this_room} training samples")
        print(f"  Total: {self.samples_collected} samples from {self.rooms_processed} rooms")
        
        return True
    
    def train_on_dataset(self, dataset_path, max_rooms=None):
        """
        MAIN TRAINING FUNCTION
        
        1. Loop through all areas
        2. Loop through all rooms
        3. Collect training data from each room
        4. After ALL rooms: train the model
        5. Save model to disk
        """
        print(f"\n{'#'*70}")
        print(f"# BATCH TRAINING ON ENTIRE DATASET")
        print(f"# Dataset: {dataset_path}")
        print(f"{'#'*70}\n")
        
        # Find all rooms
        all_rooms = []
        for area in sorted(os.listdir(dataset_path)):
            area_path = os.path.join(dataset_path, area)
            if not os.path.isdir(area_path) or not area.startswith('Area'):
                continue
            
            for room in sorted(os.listdir(area_path)):
                room_path = os.path.join(area_path, room)
                anno_path = os.path.join(room_path, 'Annotations')
                
                if os.path.exists(anno_path):
                    all_rooms.append({
                        'area': area,
                        'room': room,
                        'anno_path': anno_path,
                        'name': f"{area}_{room}"
                    })
        
        total_rooms = len(all_rooms)
        if max_rooms:
            all_rooms = all_rooms[:max_rooms]
            print(f"📊 Found {total_rooms} rooms, processing first {max_rooms}")
        else:
            print(f"📊 Found {total_rooms} rooms, processing ALL")
        
        print(f"\n{'='*70}")
        print("PHASE 1: COLLECTING TRAINING DATA FROM ALL ROOMS")
        print(f"{'='*70}\n")
        
        # Loop through ALL rooms and collect data
        for idx, room_info in enumerate(all_rooms):
            print(f"\n[{idx+1}/{len(all_rooms)}] Processing: {room_info['name']}")
            
            self.collect_training_data_from_room(
                room_info['anno_path'],
                room_info['name'],
                voxel_size=0.02,
                eps=0.05
            )
        
        # Now train the model on ALL collected data
        print(f"\n{'='*70}")
        print("PHASE 2: TRAINING MODEL ON COLLECTED DATA")
        print(f"{'='*70}")
        
        success = self.classifier.train()
        
        if success:
            print(f"\n{'#'*70}")
            print(f"# ✓ TRAINING COMPLETE!")
            print(f"# Rooms processed: {self.rooms_processed}")
            print(f"# Training samples: {self.samples_collected}")
            print(f"# Model saved to: {self.classifier.model_path}")
            print(f"{'#'*70}\n")
        else:
            print("\n❌ Training failed!")
        
        return success
    
    def test_on_room(self, anno_path, room_name, output_dir="ml_output"):
        """Test trained model on a new room"""
        if not self.classifier.is_trained:
            print("❌ Model not trained! Run training first.")
            return False
        
        print(f"\n{'#'*70}")
        print(f"# TESTING ON: {room_name}")
        print(f"{'#'*70}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and preprocess
        object_files = sorted([f for f in os.listdir(anno_path) if f.endswith('.txt')])
        
        all_points = []
        all_colors = []
        
        for obj_file in object_files:
            data = np.loadtxt(os.path.join(anno_path, obj_file))
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            all_points.append(data[:, :3])
            all_colors.append(data[:, 3:6] if data.shape[1] >= 6 else np.ones((len(data), 3)) * 128)
        
        points = np.vstack(all_points)
        colors = np.vstack(all_colors)
        
        print(f"Loaded {len(points):,} points")
        
        # Preprocess
        processed_points, processed_colors = self.preprocessor.preprocess(
            points, colors, use_statistical=True, voxel_size=0.02
        )
        
        # Cluster
        cluster_labels, _ = self.clusterer.dbscan_clustering(
            processed_points, eps=0.05, min_samples=10
        )
        
        # Predict labels
        semantic_labels = np.full(len(processed_points), 
                                  self.classifier.LABELS['unknown'], 
                                  dtype=np.int32)
        
        unique_clusters = set(cluster_labels)
        unique_clusters.discard(-1)
        
        print(f"\nPredicting labels for {len(unique_clusters)} clusters...")
        
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            features = MLFeatureExtractor.extract_cluster_features(processed_points, mask)
            
            if features:
                predicted_label = self.classifier.predict(features)
                semantic_labels[mask] = predicted_label
        
        # Visualize
        semantic_colors = np.zeros((len(semantic_labels), 3), dtype=np.uint8)
        for label_name, label_value in self.classifier.LABELS.items():
            mask = semantic_labels == label_value
            semantic_colors[mask] = self.classifier.LABEL_COLORS[label_name]
        
        # Export
        output_path = os.path.join(output_dir, f"{room_name}_segmented.ply")
        export_ply(processed_points, semantic_colors, output_path)
        
        # Statistics
        print("\n📊 Segmentation Results:")
        for label_name, label_value in sorted(self.classifier.LABELS.items(), key=lambda x: x[1]):
            count = np.sum(semantic_labels == label_value)
            if count > 0:
                percentage = count / len(semantic_labels) * 100
                print(f"  {label_name:15s}: {count:8,} points ({percentage:5.2f}%)")
        
        print(f"\n✓ Saved to: {output_path}")
        return True


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ML-Enhanced 3D Room Segmentation')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test'],
                       help='Mode: train (train model) or test (test on new room)')
    parser.add_argument('--dataset', type=str, 
                       default=r"E:\3D-semantic\3D-Room-Scene-Semantic-Segmentation\dataset\Stanford3dDataset_v1.2_Aligned_Version",
                       help='Path to dataset')
    parser.add_argument('--max_rooms', type=int, default=None,
                       help='Max rooms for training (None = all rooms)')
    parser.add_argument('--model', type=str, default='trained_model.pkl',
                       help='Path to model file')
    parser.add_argument('--test_room', type=str, default=None,
                       help='Room to test (format: Area_1/office_1)')
    
    args = parser.parse_args()
    
    pipeline = MLTrainingPipeline(model_path=args.model)
    
    if args.mode == 'train':
        print("\n🎓 TRAINING MODE")
        print("="*70)
        
        pipeline.train_on_dataset(
            args.dataset,
            max_rooms=args.max_rooms
        )
        
    elif args.mode == 'test':
        print("\n🧪 TESTING MODE")
        print("="*70)
        
        # Load trained model
        if not pipeline.classifier.load_model():
            print("❌ No trained model found! Run training first:")
            print("   python ml_segmentation_pipeline.py --mode train")
            sys.exit(1)
        
        if args.test_room:
            anno_path = os.path.join(args.dataset, args.test_room, "Annotations")
            room_name = args.test_room.replace('/', '_')
            pipeline.test_on_room(anno_path, room_name)
        else:
            print("Please specify --test_room, e.g.: --test_room Area_1/office_1")
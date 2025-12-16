"""
SIMPLE TRAINING SCRIPT
Just run this file - it will train the model on your entire dataset!

Steps:
1. Loops through ALL areas (Area_1, Area_2, ...)
2. Loops through ALL rooms in each area
3. Collects training data from every room
4. Trains the model on ALL collected data
5. Saves trained_model.pkl

Usage:
    python simple_train.py
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_segmentation_pipeline import MLTrainingPipeline


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                  ML ROOM SEGMENTATION TRAINING                   ║
║                                                                  ║
║  This script will:                                              ║
║  1. Loop through ALL areas (Area_1, Area_2, ...)               ║
║  2. Loop through ALL rooms in each area                        ║
║  3. Extract features from each room                            ║
║  4. Train Random Forest model on ALL data                      ║
║  5. Save trained_model.pkl                                     ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # ========================================================================
    # CONFIGURATION - CHANGE THIS TO YOUR DATASET PATH
    # ========================================================================
    DATASET_PATH = r"E:\3D-semantic\3D-Room-Scene-Semantic-Segmentation\dataset\Stanford3dDataset_v1.2_Aligned_Version"
    
    # How many rooms to use for training?
    # None = ALL rooms (recommended for best accuracy)
    # 50 = First 50 rooms (faster, good for testing)
    MAX_ROOMS = None  # Change to 50 for quick testing
    
    MODEL_PATH = "trained_model.pkl"
    # ========================================================================
    
    # Verify dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"❌ ERROR: Dataset not found at: {DATASET_PATH}")
        print("\nPlease update DATASET_PATH in this script to your actual dataset location.")
        print("Example:")
        print('  DATASET_PATH = r"C:\\path\\to\\Stanford3dDataset_v1.2_Aligned_Version"')
        sys.exit(1)
    
    print(f"✓ Dataset found: {DATASET_PATH}\n")
    
    # Ask user confirmation
    if MAX_ROOMS:
        print(f"⚠️  Training on first {MAX_ROOMS} rooms (quick mode)")
    else:
        print("📚 Training on ALL rooms (full dataset)")
    
    response = input("\nProceed with training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        sys.exit(0)
    
    # Initialize pipeline
    print("\n🚀 Starting training pipeline...\n")
    pipeline = MLTrainingPipeline(model_path=MODEL_PATH)
    
    # Train on dataset
    success = pipeline.train_on_dataset(
        dataset_path=DATASET_PATH,
        max_rooms=MAX_ROOMS
    )
    
    if success:
        print("\n" + "="*70)
        print("✅ SUCCESS! Model training complete!")
        print("="*70)
        print(f"\nModel saved to: {MODEL_PATH}")
        print(f"Rooms processed: {pipeline.rooms_processed}")
        print(f"Training samples: {pipeline.samples_collected}")
        print("\nYou can now use this model to segment new rooms!")
        print("\nNext step: Run test script to see it in action:")
        print("  python simple_test.py")
    else:
        print("\n❌ Training failed. Check error messages above.")


if __name__ == "__main__":
    main()
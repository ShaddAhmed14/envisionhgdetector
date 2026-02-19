#!/usr/bin/env python3
"""
Test script for the current EnvisionHGDetector setup.
Tests configuration, model availability, and detector initialization.
"""

import os
import sys


def test_current_setup():
    """Test the current installation and identify what needs updating."""
    try:
        from envisionhgdetector import Config, GestureDetector
        
        print("=" * 60)
        print("EnvisionHGDetector Current Setup Analysis")
        print("=" * 60)
        
        # Test basic imports
        print("\n1. Testing imports...")
        try:
            from envisionhgdetector import GestureModel, make_model
            print("   [OK] GestureModel and make_model imported")
        except ImportError as e:
            print(f"   [FAIL] Original model imports failed: {e}")
        
        # Test config
        print("\n2. Testing Config...")
        config = Config()
        print(f"   [OK] Config initialized")
        print(f"   - Feature set: {config.feature_set}")
        print(f"   - Num features: {config.num_original_features}")
        print(f"   - Sequence length: {config.seq_length}")
        print(f"   - CNN model: {config.cnn_model_filename}")
        print(f"   - LightGBM model: {config.lightgbm_model_filename}")
        
        # Check model availability
        print("\n3. Checking model availability...")
        cnn_available = config.is_model_available("cnn")
        lgbm_available = config.is_model_available("lightgbm")
        print(f"   CNN available: {cnn_available}")
        print(f"   LightGBM available: {lgbm_available}")
        print(f"   Available models: {config.available_models}")
        
        if cnn_available:
            print(f"   CNN path: {config.weights_path}")
        if lgbm_available:
            print(f"   LightGBM path: {config.lightgbm_weights_path}")
        
        # Test default thresholds
        print("\n4. Default thresholds...")
        thresholds = config.default_thresholds
        for name, value in thresholds.items():
            print(f"   {name}: {value}")
        
        # Test detector with different model types
        print("\n5. Testing GestureDetector initialization...")
        
        # Test CNN
        if cnn_available:
            try:
                detector_cnn = GestureDetector(model_type="cnn")
                print("   [OK] GestureDetector with model_type='cnn'")
            except Exception as e:
                print(f"   [FAIL] CNN detector failed: {e}")
        else:
            print("   [SKIP] CNN model not available")
        
        # Test LightGBM
        if lgbm_available:
            try:
                detector_lgbm = GestureDetector(model_type="lightgbm")
                print("   [OK] GestureDetector with model_type='lightgbm'")
            except Exception as e:
                print(f"   [FAIL] LightGBM detector failed: {e}")
        else:
            print("   [SKIP] LightGBM model not available")
        
        # Test Combined
        if cnn_available and lgbm_available:
            try:
                detector_combined = GestureDetector(model_type="combined")
                print("   [OK] GestureDetector with model_type='combined'")
            except Exception as e:
                print(f"   [FAIL] Combined detector failed: {e}")
        else:
            print("   [SKIP] Combined model requires both CNN and LightGBM")
        
        # Check for RealtimeGestureDetector
        print("\n6. Testing RealtimeGestureDetector...")
        try:
            from envisionhgdetector import RealtimeGestureDetector
            detector_realtime = RealtimeGestureDetector(confidence_threshold=0.5)
            print("   [OK] RealtimeGestureDetector available")
            print(f"   - Confidence threshold: {detector_realtime.confidence_threshold}")
            print(f"   - Gesture labels: {detector_realtime.model.gesture_labels}")
        except ImportError:
            print("   [FAIL] RealtimeGestureDetector not available")
        except Exception as e:
            print(f"   [FAIL] RealtimeGestureDetector error: {e}")
        
        # Check package location and files
        import envisionhgdetector
        package_path = os.path.dirname(envisionhgdetector.__file__)
        print(f"\n7. Package location: {package_path}")
        
        # List Python files in package
        print("\n8. Package files:")
        try:
            files = os.listdir(package_path)
            py_files = sorted([f for f in files if f.endswith('.py')])
            for file in py_files:
                print(f"   - {file}")
        except Exception as e:
            print(f"   Error listing files: {e}")
        
        # Check model directory
        print("\n9. Model directory:")
        model_dir = os.path.join(package_path, 'model')
        if os.path.exists(model_dir):
            try:
                model_files = os.listdir(model_dir)
                for file in sorted(model_files):
                    filepath = os.path.join(model_dir, file)
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    print(f"   - {file} ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"   Error listing model files: {e}")
        else:
            print("   [FAIL] Model directory not found")
        
        # Check for new module files
        print("\n10. Checking for required modules...")
        required_modules = [
            'config.py',
            'detector.py', 
            'model_cnn.py',
            'model_lightgbm.py',
            'model_combined.py',
            'label_video.py',
            'label_video_combined.py',
        ]
        for module in required_modules:
            module_path = os.path.join(package_path, module)
            if os.path.exists(module_path):
                print(f"   [OK] {module}")
            else:
                print(f"   [MISSING] {module}")
        
        print("\n" + "=" * 60)
        print("Analysis complete!")
        print("=" * 60)
        
        return True
        
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_label_encoder_order():
    """Test that LightGBM label order is correctly handled."""
    print("\n" + "=" * 60)
    print("LightGBM Label Encoder Order Test")
    print("=" * 60)
    
    try:
        from sklearn.preprocessing import LabelEncoder
        
        # Simulate what happens during training
        le = LabelEncoder()
        le.fit(["NoGesture", "Gesture"])
        
        print("\nLabelEncoder behavior:")
        print(f"  Input labels: ['NoGesture', 'Gesture']")
        print(f"  Sorted classes: {list(le.classes_)}")
        print(f"  Index 0 = {le.classes_[0]}")
        print(f"  Index 1 = {le.classes_[1]}")
        
        print("\nIMPORTANT: sklearn LabelEncoder sorts alphabetically!")
        print("  - Model output index 0 = Gesture (not NoGesture)")
        print("  - Model output index 1 = NoGesture (not Gesture)")
        print("\nMake sure inference code handles this correctly.")
        
        return True
    except Exception as e:
        print(f"[FAIL] Label encoder test failed: {e}")
        return False


def show_usage_examples():
    """Show usage examples for different detector types."""
    print("\n" + "=" * 60)
    print("Usage Examples")
    print("=" * 60)
    
    print("""
# CNN-only detector (3-class: NoGesture, Gesture, Move)
from envisionhgdetector import GestureDetector

detector = GestureDetector(
    model_type="cnn",
    motion_threshold=0.7,
    gesture_threshold=0.5,
    min_gap_s=0.2,
    min_length_s=0.3
)
detector.process_folder(input_folder, output_folder)

# LightGBM-only detector (2-class: NoGesture, Gesture)
detector = GestureDetector(
    model_type="lightgbm",
    gesture_threshold=0.6,  # LightGBM may need higher threshold
    min_gap_s=0.2,
    min_length_s=0.3
)
detector.process_folder(input_folder, output_folder)

# Combined detector (both models, dual-panel video output)
detector = GestureDetector(
    model_type="combined",
    cnn_motion_threshold=0.7,
    cnn_gesture_threshold=0.5,
    lgbm_threshold=0.6,  # Higher threshold for LightGBM
    min_gap_s=0.2,
    min_length_s=0.3
)
detector.process_folder(input_folder, output_folder)

# Real-time webcam detection (LightGBM-based)
from envisionhgdetector import RealtimeGestureDetector

detector = RealtimeGestureDetector(
    confidence_threshold=0.6,
    min_gap_s=0.2,
    min_length_s=0.3
)
raw_results, segments = detector.process_webcam()
""")


def main():
    """Run the setup analysis."""
    print("Analyzing EnvisionHGDetector setup...\n")
    
    success = test_current_setup()
    test_label_encoder_order()
    show_usage_examples()
    
    if success:
        print("\n[OK] Analysis completed successfully!")
    else:
        print("\n[FAIL] Analysis failed. Please check your installation.")


if __name__ == "__main__":
    main()
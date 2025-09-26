"""
Script to fix TensorFlow GPU detection issue
Uninstalls CPU-only TensorFlow and installs GPU-enabled version
"""
import subprocess
import sys

def run_command(command, description):
    """Run a command and print the result"""
    print(f"\n🔧 {description}...")
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                               capture_output=True, text=True)
        print(f"✅ Success: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e.stderr}")
        return False

def main():
    print("🚀 Fixing TensorFlow GPU Detection Issue")
    print("=" * 50)
    
    # Step 1: Uninstall existing TensorFlow
    run_command("pip uninstall tensorflow -y", "Uninstalling CPU-only TensorFlow")
    
    # Step 2: Install TensorFlow with GPU support
    run_command("pip install tensorflow[and-cuda]", "Installing TensorFlow with GPU support")
    
    # Step 3: Verify installation
    print("\n🔍 Verifying TensorFlow GPU installation...")
    verify_code = '''
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"GPU devices: {len(tf.config.list_physical_devices('GPU'))}")
if tf.config.list_physical_devices('GPU'):
    print("✅ GPU detected successfully!")
    for device in tf.config.list_physical_devices('GPU'):
        print(f"   - {device}")
else:
    print("❌ No GPU devices found")
'''
    
    try:
        result = subprocess.run([sys.executable, "-c", verify_code], 
                               capture_output=True, text=True, timeout=30)
        print(result.stdout)
        if result.stderr:
            print(f"Warnings: {result.stderr}")
    except Exception as e:
        print(f"Error during verification: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 TensorFlow GPU setup complete!")
    print("You can now run your full N-BaLoT experiment with GPU acceleration.")

if __name__ == "__main__":
    main()
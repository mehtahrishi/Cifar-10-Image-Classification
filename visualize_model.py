from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

# List of model filenames (change these based on your saved models)
models_to_visualize = ["cnn_model.h5", "cnn_model_v2.h5"]  

for model_name in models_to_visualize:
    try:
        # Load the model
        model = load_model(model_name)
        
        # Generate and save the architecture diagram
        plot_model(model, to_file=f"{model_name}_architecture.png", show_shapes=True, show_layer_names=True)
        print(f"Model architecture saved as {model_name}_architecture.png")
    
    except Exception as e:
        print(f"Error loading {model_name}: {e}")

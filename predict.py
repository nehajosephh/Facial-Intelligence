from main import predict_age
import os

def main():
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create an images folder if it doesn't exist
    images_dir = os.path.join(current_dir, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        print(f"Created 'images' folder at: {images_dir}")
        print("Please place your images in this folder")
        return
    
    # Get list of images in the folder
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("No images found in the 'images' folder")
        print("Please add some images to the folder and try again")
        return
    
    print("\nAvailable images:")
    for i, image in enumerate(image_files, 1):
        print(f"{i}. {image}")
    
    # Let user choose an image
    while True:
        try:
            choice = int(input("\nEnter the number of the image to predict age (or 0 to exit): "))
            if choice == 0:
                break
            if 1 <= choice <= len(image_files):
                selected_image = image_files[choice - 1]
                image_path = os.path.join(images_dir, selected_image)
                
                # Optional: Ask if user wants to use trained model
                use_trained = input("Do you want to use trained model weights? (y/n): ").lower() == 'y'
                model_path = "best_age_model.pth" if use_trained else None
                
                print("\nPredicting age...")
                predict_age(image_path, model_path)
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

if __name__ == "__main__":
    print("Age Prediction Tool")
    print("==================")
    main() 
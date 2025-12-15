from itpt.models import get_list, get_model

def main():
    models = get_list()
    print("Available models:")
    for name in models:
        print(f" - {name}")

    if not models:
        print("No models available.")
        return

    model_name = models[0]
    print(f"\nUsing model: {model_name}")

    model = get_model(model_name)

    model.load()

    image_path = "example.png"

    print(f"Converting image: {image_path}")
    tree = model.convert(image_path)

    print("\nResulting Newick tree:")
    print(tree)

if __name__ == "__main__":
    main()

from itpt.models import get_list, get_model

def run_example():
    model_names = get_list()
    print("Available models:")
    for name in model_names:
        print(f" - {name}")

    if not model_names:
        print("No models available.")
        return

    model_name = model_names[0]
    print(f"\nUsing model: {model_name}")

    model = get_model(model_name)
    model.load()

    image_path = "example.png"

    print(f"\nConverting image: {image_path}")
    newick = model.convert(image_path)

    print("\nResulting Newick tree:")
    print(newick.to_string())

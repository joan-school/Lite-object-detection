import tensorflow as tf

def convert(model_name):
    print(f"Converting {model_name}...")

    saved_model_dir = f"models/experts/{model_name}/exported_model/saved_model"

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    # 🔥 FIX HERE
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    output_path = f"models/experts/{model_name}/{model_name}.tflite"

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"{model_name} saved at {output_path}")

convert("kitchen")
convert("display")
convert("climate")
import onnx

inp = r"C:\Users\Nicholas\Downloads\onnx_out\stroke_type_classifier.onnx" #INSERT FILE NAME HERE
out = r"C:\Users\Nicholas\Downloads\onnx_out\stroke_type_classifier_single.onnx" #INSERT FILE NAME HERE

m = onnx.load(inp, load_external_data=True)
onnx.save_model(m, out, save_as_external_data=False)

print("Saved single-file ONNX", out)

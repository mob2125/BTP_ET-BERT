import os

# Define the command template
command_template = (
    "python3 vizualize/viz.py --load_model_path models/finetuned_model_2.bin --vocab_path models/encryptd_vocab_all.txt --test_path datasets/ISCX_data/nolabel_test_dataset.tsv --prediction_path datasets/prediction.tsv --labels_num 17 --embedding word_pos_seg --encoder transformer --mask fully_visible --html_file vizualize_layer_{layer}.html --include_layers {layer}"
)

# Loop through layers 0 to 11 and run the command for each
for layer in range(12):
    command = command_template.format(layer=layer)
    print(f"Running command for layer {layer}: {command}")
    os.system(command)

print("Visualization generation complete.")


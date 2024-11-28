## **Project Scope: Reproducing and Distilling ET-BERT for Campus Network Data**

**Project Goal:**

To reproduce the ET-BERT model for the specific domain of university network data and subsequently distill it into a smaller, more computationally efficient model, while maintaining or improving its performance on relevant tasks.

**Scope:**

1. **Data Acquisition and Preprocessing:**
   * **Data Collection:** Gather a suitable dataset of university network data.
   * **Data Transformation:** Convert the data into a format suitable for training the ET-BERT model, such as tokenized sequences and corresponding labels.

2. **ET-BERT Model Reproduction:**
   * **Repository Cloning:** Clone the official ET-BERT GitHub repository and set up the necessary environment.
   * **Hyperparameter Tuning:** Experiment with different hyperparameter values (e.g., learning rate, batch size, number of epochs) to optimize the model's performance.
   * **Training and Evaluation:** Train the ET-BERT model on the prepared dataset using appropriate metrics (e.g., accuracy, F1-score) to evaluate its performance on tasks like link prediction, node classification, or community detection.

3. **Model Distillation:**
   * **Teacher-Student Setup:** Establish a teacher-student framework where the original ET-BERT model (teacher) guides the training of a smaller, distilled model (student).
   * **Knowledge Transfer Techniques:** Explore various knowledge transfer techniques used for BERT model (e.g., knowledge distillation, attention transfer, parameter sharing) to transfer the teacher's knowledge to the student.
   * **Distilled Model Evaluation:** Evaluate the performance of the distilled model on the same tasks as the original ET-BERT model, comparing their accuracy and computational efficiency.

4. **Analysis and Reporting:**
   * **Performance Comparison:** Analyze the performance differences between the original and distilled ET-BERT models, considering factors like accuracy, computational cost, and inference time.
   * **Documentation and Reporting:** Create comprehensive documentation and reports detailing the project methodology, results, and conclusions.

**Pretraining:**
    To pretrain the model run the following code:

    ```
       python3 pre-training/pretrain.py --dataset_path dataset.pt --vocab_path models/encryptd_vocab.txt --output_model_path models/pre-trained_model.bin --world_size 2 --gpu_ranks 0 1 --total_steps 500000 --save_checkpoint_steps 10000 --batch_size 32 --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert
    ```

**Fine-tuning:**
    To fine-tune the model run the following code:
    ```
    python3 fine-tuning/run_classifier.py --pretrained_model_path models/pre-trained_model.bin --vocab_path models/encryptd_vocab.txt --train_path datasets/ISCX_data/train_dataset.tsv --dev_path datasets/ISCX_data/valid_dataset.tsv --test_path datasets/ISCX_data/test_dataset.tsv --epochs_num 10 --batch_size 32 --embedding word_pos_seg --encoder transformer --mask fully_visible --seq_length 128 --learning_rate 2e-5
    ```
    
**Distillation:**
    To distill the model run the following code:
    ```
   python3 distillation/distill.py --teacher_model_path models/pretrained.bin --dataset_path dataset.pt --vocab_path models/encryptd_vocab.txt --output_model_path models/distilled_model.bin --world_size 2 --gpu_ranks 0 1 --total_steps 500000 --save_checkpoint_steps 10000 --batch_size 32 --embedding word_pos_seg --encoder transformer --mask fully_visible --target tinybert
    ```

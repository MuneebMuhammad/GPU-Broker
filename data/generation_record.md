# Generation Record

- **Generation model:** GPT-6 Astra
- **Generation date:** 12 October 2026

## Prompt

Can you help me create a total of 60 test cases for my dataset. I want to include the popular workflows for LLMs only. I have attached an example dataset with three cases. Can you create total 60 cases that I can then use to test my GPU Broker system.

| input_text                                                                                                                                                             | model_name   | parameter_count | dataset_size_gb | sequence_length | batch_size | training_method | precision | number_of_layers | hidden_size | epochs | training_time_constraint_hours | workload_type |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | --------------- | --------------- | --------------- | ---------- | --------------- | --------- | ---------------- | ----------- | ------ | ------------------------------ | ------------- |
| I want to fine-tune Llama 7B with LoRA on a 30 GB dataset using sequence length 2048.                                                                                  | Llama 7B     | 7B              | 30              | 2048            | null       | LoRA            | null      | null             | null        | null   | null                           | training      |
| I want to fine-tune Llama 3.2 3B using LoRA on a 50 GB text dataset. I plan to use a sequence length of 2048, batch size of 8, BF16 precision, and train for 3 epochs. | Llama 3.2 3B | 3B              | 50              | 2048            | 8          | LoRA            | BF16      | null             | null        | 3      | null                           | training      |
| I want to run inference on Llama 3 70B                                                                                                                                 | Llama 3 70B  | 70B             | null            | null            | null       | null            | null      | null             | null        | null   | null                           | inference     |
